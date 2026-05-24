"""Classical baseline models for anomaly detection.

Implements five baselines that follow the same ``fit`` / ``anomaly_scores``
interface as the QAE models:
  - ``__init__(seed=42, ...)``
  - ``fit(X, ...) → list[float]``
  - ``anomaly_scores(X) → np.ndarray`` of shape ``(n_samples,)``

Four baselines consume 2D PCA features ``(n_samples, 6)``:
  MatchedAutoencoder, FullAutoencoder, IsolationForestDetector, OneClassSVMDetector.

One baseline consumes raw 3D windowed time series ``(n_windows, 128, 5)``:
  LSTMAutoencoder.

Higher anomaly score = more anomalous (consistent with QAE convention).

Reference: R005 (MatchedAutoencoder), R006 (FullAutoencoder),
           R007 (LSTMAutoencoder), R008 (IsolationForest), R009 (OneClassSVM).
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

logger = logging.getLogger(__name__)

# Auto-detect CUDA
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

__all__ = [
    "count_trainable_parameters",
    "MatchedAutoencoder",
    "FullAutoencoder",
    "LSTMAutoencoder",
    "IsolationForestDetector",
    "OneClassSVMDetector",
]


def count_trainable_parameters(model: object) -> int | None:
    """Count trainable parameters for parametric models.

    Non-parametric sklearn wrappers return ``None``.  QAE models expose their
    PennyLane parameter tensor through ``weights``.
    """
    if hasattr(model, "num_trainable_parameters"):
        value = getattr(model, "num_trainable_parameters")
        return int(value() if callable(value) else value)
    torch_model = getattr(model, "_model", None)
    if isinstance(torch_model, nn.Module):
        return int(sum(p.numel() for p in torch_model.parameters() if p.requires_grad))
    weights = getattr(model, "weights", None)
    if weights is not None:
        return int(np.asarray(weights).size)
    return None


# ------------------------------------------------------------------
# PyTorch autoencoder base
# ------------------------------------------------------------------

class _AutoencoderBase:
    """Shared training / scoring logic for PyTorch autoencoders."""

    _model: nn.Module  # subclasses must set this

    def fit(
        self,
        X_train: np.ndarray,
        n_epochs: int = 200,
        batch_size: int = 32,
        verbose: bool = True,
    ) -> list[float]:
        """Train the autoencoder on normal data.

        Parameters
        ----------
        X_train : np.ndarray
            Training features, shape ``(n_samples, 6)`` in [0, π].
        n_epochs : int
            Number of training epochs (default 200).
        batch_size : int
            Mini-batch size (default 32).
        verbose : bool
            Print progress every 10 epochs (default True).

        Returns
        -------
        list[float]
            Per-epoch mean loss values.
        """
        self._model.train()
        self._model.to(_DEVICE)
        X_t = torch.tensor(X_train, dtype=torch.float32, device=_DEVICE)
        dataset = torch.utils.data.TensorDataset(X_t)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(self._seed),
        )
        criterion = nn.MSELoss()
        loss_history: list[float] = []

        for epoch in range(n_epochs):
            epoch_losses: list[float] = []
            for (batch,) in loader:
                batch = batch.to(_DEVICE)
                self._optimizer.zero_grad()
                recon = self._model(batch)
                loss = criterion(recon, batch)
                loss.backward()
                self._optimizer.step()
                epoch_losses.append(loss.item())

            epoch_mean = float(np.mean(epoch_losses))
            loss_history.append(epoch_mean)

            if verbose and (epoch % 10 == 0 or epoch == n_epochs - 1):
                logger.info(
                    "Epoch %03d/%03d  loss=%.6f", epoch, n_epochs, epoch_mean,
                )

        return loss_history

    @torch.no_grad()
    def anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """Per-sample MSE reconstruction error.

        Parameters
        ----------
        X : np.ndarray
            Input features, shape ``(n_samples, 6)``.

        Returns
        -------
        np.ndarray
            1-D array of shape ``(n_samples,)``.  Higher = more anomalous.
        """
        self._model.eval()
        self._model.to(_DEVICE)
        X_t = torch.tensor(X, dtype=torch.float32)

        # Batch scoring to avoid CUDA OOM on large arrays
        batch_size = 4096
        mse_parts: list[torch.Tensor] = []
        for i in range(0, len(X_t), batch_size):
            batch = X_t[i : i + batch_size].to(_DEVICE)
            recon = self._model(batch)
            mse = ((batch - recon) ** 2).mean(dim=1)
            mse_parts.append(mse.cpu())
        return torch.cat(mse_parts).numpy()


# ------------------------------------------------------------------
# MatchedAutoencoder — 100 parameters, mirrors QAE capacity
# ------------------------------------------------------------------

class _MatchedNetwork(nn.Module):
    """6→5→4→5→6, no bias, sigmoid activations, output scaled to [0, π].  100 parameters."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 5, bias=False),
            nn.Sigmoid(),
            nn.Linear(5, 4, bias=False),
            nn.Sigmoid(),
            nn.Linear(4, 5, bias=False),
            nn.Sigmoid(),
            nn.Linear(5, 6, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) * np.pi


class MatchedAutoencoder(_AutoencoderBase):
    """Classical autoencoder with ~same parameter count as the QAE (100).

    Architecture: ``Linear(6,5) → σ → Linear(5,4) → σ → Linear(4,5) → σ → Linear(5,6) → σ``
    No bias on any layer.  6×5 + 5×4 + 4×5 + 5×6 = 100 parameters.

    Parameters
    ----------
    seed : int
        RNG seed (default 42).
    lr : float
        Adam learning rate (default 0.001).
    """

    def __init__(self, seed: int = 42, lr: float = 0.001) -> None:
        self._seed = seed
        torch.manual_seed(seed)
        self._model = _MatchedNetwork()
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=lr)


# ------------------------------------------------------------------
# FullAutoencoder — 3006 parameters, classical advantage
# ------------------------------------------------------------------

class _FullNetwork(nn.Module):
    """6→48→24→48→6, with bias, ReLU hidden + sigmoid output scaled to [0, π].  3006 parameters."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 48, bias=True),
            nn.ReLU(),
            nn.Linear(48, 24, bias=True),
            nn.ReLU(),
            nn.Linear(24, 48, bias=True),
            nn.ReLU(),
            nn.Linear(48, 6, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) * np.pi


class FullAutoencoder(_AutoencoderBase):
    """Classical autoencoder with generous capacity (3006 parameters).

    Architecture: ``Linear(6,48) → ReLU → Linear(48,24) → ReLU → Linear(24,48) → ReLU → Linear(48,6) → σ``
    With bias on all layers.  (6×48+48)+(48×24+24)+(24×48+48)+(48×6+6) = 3006.

    Parameters
    ----------
    seed : int
        RNG seed (default 42).
    lr : float
        Adam learning rate (default 0.001).
    """

    def __init__(self, seed: int = 42, lr: float = 0.001) -> None:
        self._seed = seed
        torch.manual_seed(seed)
        self._model = _FullNetwork()
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=lr)


# ------------------------------------------------------------------
# sklearn wrappers
# ------------------------------------------------------------------

class IsolationForestDetector:
    """Isolation Forest wrapper with QAE-compatible interface.

    Wraps ``sklearn.ensemble.IsolationForest(n_estimators=100)``.

    **Score convention:** sklearn's ``decision_function`` returns higher
    values for *normal* samples.  We negate to match the QAE convention
    (higher = more anomalous).

    Parameters
    ----------
    seed : int
        RNG seed for ``random_state`` (default 42).
    n_estimators : int
        Number of trees (default 100).
    """

    def __init__(self, seed: int = 42, n_estimators: int = 100) -> None:
        self._seed = seed
        self._model = IsolationForest(
            n_estimators=n_estimators,
            random_state=seed,
        )

    def fit(self, X: np.ndarray, **kwargs) -> list[float]:
        """Fit the Isolation Forest.

        Returns
        -------
        list[float]
            Empty list (no iterative training).
        """
        self._model.fit(X)
        return []

    def anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """Anomaly scores (higher = more anomalous).

        Returns
        -------
        np.ndarray
            1-D array of shape ``(n_samples,)``.
        """
        return -self._model.decision_function(X)


class OneClassSVMDetector:
    """One-Class SVM wrapper with QAE-compatible interface.

    Wraps ``sklearn.svm.OneClassSVM(kernel='rbf', gamma='scale', nu=0.1)``.

    **Score convention:** Negated ``decision_function`` — higher = more
    anomalous.

    Parameters
    ----------
    seed : int
        Accepted for API compatibility (OneClassSVM has no random_state).
    """

    def __init__(self, seed: int = 42) -> None:
        self._seed = seed
        self._model = OneClassSVM(kernel="rbf", gamma="scale", nu=0.1)

    def fit(self, X: np.ndarray, **kwargs) -> list[float]:
        """Fit the One-Class SVM.

        Returns
        -------
        list[float]
            Empty list (no iterative training).
        """
        self._model.fit(X)
        return []

    def anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """Anomaly scores (higher = more anomalous).

        Returns
        -------
        np.ndarray
            1-D array of shape ``(n_samples,)``.
        """
        return -self._model.decision_function(X)


# ------------------------------------------------------------------
# LSTMAutoencoder — raw windowed time series (n_windows, 128, 5)
# ------------------------------------------------------------------

class _LSTMAutoEncoderNetwork(nn.Module):
    """Encoder-decoder LSTM with teacher forcing.

    Encoder LSTM compresses ``(seq_len, input_size)`` into a latent hidden
    state. Decoder LSTM reconstructs the original sequence using teacher
    forcing (original input as decoder input, encoder final hidden state
    as initial hidden state).

    Parameters
    ----------
    input_size : int
        Number of features per timestep (default 5 for raw sensors).
    hidden_size : int
        LSTM hidden / cell dimension (default 16).
    """

    def __init__(self, input_size: int = 5, hidden_size: int = 16) -> None:
        super().__init__()
        self.encoder = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=1, batch_first=True,
        )
        self.decoder = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=1, batch_first=True,
        )
        self.output_proj = nn.Linear(hidden_size, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with teacher forcing.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(batch, seq_len, input_size)``.

        Returns
        -------
        torch.Tensor
            Reconstructed sequence, same shape as *x*.
        """
        # Encode — only need final hidden/cell state
        _, (h_n, c_n) = self.encoder(x)

        # Decode with teacher forcing: input is original sequence
        dec_out, _ = self.decoder(x, (h_n, c_n))

        # Project decoder hidden states to sensor space
        return self.output_proj(dec_out)


class LSTMAutoencoder:
    """LSTM autoencoder operating on raw windowed time series.

    Accepts 3D input ``(n_windows, seq_len, n_sensors)`` — typically
    ``(n_windows, 128, 5)`` from :meth:`FeatureEngineer.extract_windows`.
    This is fundamentally different from the 2D PCA baselines.

    The model normalises each sensor channel to zero mean / unit variance
    using training-set statistics (stored for inference).

    Architecture: Encoder LSTM(5→16) → Decoder LSTM(5→16) + Linear(16→5),
    teacher forcing during training.

    Reference: R007.

    Parameters
    ----------
    seed : int
        RNG seed (default 42).
    hidden_size : int
        LSTM hidden dimension (default 16).
    learning_rate : float
        Adam learning rate (default 0.001).
    """

    def __init__(
        self,
        seed: int = 42,
        hidden_size: int = 16,
        learning_rate: float = 0.001,
    ) -> None:
        self._seed = seed
        self._hidden_size = hidden_size
        self._learning_rate = learning_rate

        torch.manual_seed(seed)
        self._model = _LSTMAutoEncoderNetwork(
            input_size=5, hidden_size=hidden_size,
        )
        self._optimizer = torch.optim.Adam(
            self._model.parameters(), lr=learning_rate,
        )
        # Per-sensor normalisation stats — set during fit()
        self._sensor_mean: np.ndarray | None = None
        self._sensor_std: np.ndarray | None = None

    # -- helpers -------------------------------------------------------

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        """Apply stored per-sensor standardisation."""
        assert self._sensor_mean is not None, (
            "Model not fitted — call fit() before anomaly_scores()"
        )
        return (X - self._sensor_mean) / self._sensor_std

    # -- public API ----------------------------------------------------

    def fit(
        self,
        X_3d: np.ndarray,
        n_epochs: int = 50,
        batch_size: int = 32,
        verbose: bool = True,
    ) -> list[float]:
        """Train on normal windowed data.

        Parameters
        ----------
        X_3d : np.ndarray
            Training windows, shape ``(n_windows, seq_len, n_sensors)``.
        n_epochs : int
            Training epochs (default 50).
        batch_size : int
            Mini-batch size (default 32).
        verbose : bool
            Log progress every 10 epochs.

        Returns
        -------
        list[float]
            Per-epoch mean loss.
        """
        # Compute per-sensor normalisation from training data
        # X_3d shape: (n_windows, seq_len, n_sensors)
        self._sensor_mean = X_3d.mean(axis=(0, 1))  # (n_sensors,)
        self._sensor_std = X_3d.std(axis=(0, 1))     # (n_sensors,)
        # Guard against constant sensors
        self._sensor_std = np.where(
            self._sensor_std < 1e-8, 1.0, self._sensor_std,
        )

        X_norm = self._normalize(X_3d)
        X_t = torch.tensor(X_norm, dtype=torch.float32)
        dataset = torch.utils.data.TensorDataset(X_t)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(self._seed),
        )

        criterion = nn.MSELoss()
        loss_history: list[float] = []

        self._model.to(_DEVICE)
        self._model.train()
        for epoch in range(n_epochs):
            epoch_losses: list[float] = []
            for (batch,) in loader:
                batch = batch.to(_DEVICE)
                self._optimizer.zero_grad()
                recon = self._model(batch)
                loss = criterion(recon, batch)
                loss.backward()
                self._optimizer.step()
                epoch_losses.append(loss.item())

            epoch_mean = float(np.mean(epoch_losses))
            loss_history.append(epoch_mean)

            if verbose and (epoch % 10 == 0 or epoch == n_epochs - 1):
                logger.info(
                    "LSTM-AE Epoch %03d/%03d  loss=%.6f",
                    epoch, n_epochs, epoch_mean,
                )

        return loss_history

    @torch.no_grad()
    def anomaly_scores(self, X_3d: np.ndarray) -> np.ndarray:
        """Per-window mean MSE reconstruction error.

        Parameters
        ----------
        X_3d : np.ndarray
            Input windows, shape ``(n_windows, seq_len, n_sensors)``.

        Returns
        -------
        np.ndarray
            1-D array of shape ``(n_windows,)``.  Higher = more anomalous.
        """
        self._model.eval()
        self._model.to(_DEVICE)
        X_norm = self._normalize(X_3d)
        X_t = torch.tensor(X_norm, dtype=torch.float32)

        # Batch scoring to avoid CUDA OOM on large anomaly arrays
        batch_size = 1024
        mse_parts: list[torch.Tensor] = []
        for i in range(0, len(X_t), batch_size):
            batch = X_t[i : i + batch_size].to(_DEVICE)
            recon = self._model(batch)
            mse = ((batch - recon) ** 2).mean(dim=(1, 2))
            mse_parts.append(mse.cpu())
        return torch.cat(mse_parts).numpy()
