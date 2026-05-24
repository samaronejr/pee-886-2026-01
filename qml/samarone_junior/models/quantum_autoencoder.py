"""Quantum Autoencoder with EfficientSU2 ansatz.

Implements two QAE loss variants for anomaly detection:
  1. Trash fidelity — loss = 1 − P(|00⟩ on trash qubits)
  2. Reconstruction error — loss = MSE(cos(x), decoded ⟨Z⟩ values)

Both use 6 qubits (4 latent + 2 trash), angle encoding via RY gates,
and a custom EfficientSU2 ansatz with RY + RZ rotations and linear
CNOT entanglement.

Reference: R003 — QAE architecture, R004 — reconstruction-error variant.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Ansatz
# ------------------------------------------------------------------

def efficient_su2(
    weights: pnp.tensor,
    wires: Sequence[int],
) -> None:
    """EfficientSU2: RY + RZ per qubit, linear CNOT entanglement, per layer.

    Parameters
    ----------
    weights : pnp.tensor
        Shape ``(n_layers, n_qubits, 2)`` — trainable rotation angles.
    wires : Sequence[int]
        Qubit indices to act on.
    """
    n_layers = weights.shape[0]
    n_qubits = len(wires)
    for layer in range(n_layers):
        for i, w in enumerate(wires):
            qml.RY(weights[layer, i, 0], wires=w)
            qml.RZ(weights[layer, i, 1], wires=w)
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[wires[i], wires[i + 1]])


# ------------------------------------------------------------------
# Trash-fidelity QAE
# ------------------------------------------------------------------

class QAETrashFidelity:
    """Quantum autoencoder with trash-fidelity loss.

    The autoencoder uses 6 qubits — 4 latent (wires 0–3) and 2 trash
    (wires 4–5). Input features are angle-encoded via RY gates, then
    processed through a custom EfficientSU2 ansatz.

    Loss = 1 − P(|00⟩ on trash qubits). The encoder drives trash qubits
    toward |0⟩, compressing information into the latent register.

    Parameters
    ----------
    n_qubits : int
        Total number of qubits (default 6).
    n_trash : int
        Number of trash qubits (default 2).
    n_layers : int
        Ansatz depth (default 4).
    learning_rate : float
        Adam step size (default 0.01).
    seed : int
        RNG seed for reproducibility (default 42).
    """

    def __init__(
        self,
        n_qubits: int = 6,
        n_trash: int = 2,
        n_layers: int = 4,
        learning_rate: float = 0.01,
        seed: int = 42,
    ) -> None:
        self.n_qubits = n_qubits
        self.n_trash = n_trash
        self.n_latent = n_qubits - n_trash
        self.n_layers = n_layers
        self.learning_rate = learning_rate
        self._seed = seed

        # Local RNG for reproducible, order-independent weight initialization
        rng = np.random.default_rng(seed)

        self._dev = qml.device("default.qubit", wires=n_qubits)

        # 48 params for default config: 4 layers × 6 qubits × 2 rotations
        self._weights = pnp.array(
            rng.uniform(-0.1, 0.1, (n_layers, n_qubits, 2)),
            requires_grad=True,
        )

        # Trash qubit indices
        self._trash_wires = list(range(n_qubits - n_trash, n_qubits))

        # Build the QNode — backprop for statevector simulator
        @qml.qnode(self._dev, diff_method="backprop")
        def _circuit(features, weights):
            qml.AngleEmbedding(features, wires=range(n_qubits), rotation="Y")
            efficient_su2(weights, wires=range(n_qubits))
            return qml.probs(wires=self._trash_wires)

        self._circuit = _circuit

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def weights(self) -> pnp.tensor:
        """Current trainable parameters."""
        return self._weights

    @property
    def num_trainable_parameters(self) -> int:
        """Number of trainable circuit parameters."""
        return int(np.asarray(self._weights).size)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

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
            Training features, shape ``(n_samples, n_qubits)`` in [0, π].
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
        opt = qml.AdamOptimizer(stepsize=self.learning_rate)
        n_samples = len(X_train)
        loss_history: list[float] = []
        shuffle_rng = np.random.default_rng(self._seed)

        for epoch in range(n_epochs):
            # Shuffle training data each epoch (local RNG for isolation)
            perm = shuffle_rng.permutation(n_samples)
            X_shuffled = X_train[perm]

            epoch_losses: list[float] = []

            for start in range(0, n_samples, batch_size):
                batch = X_shuffled[start : start + batch_size]

                # Cost function: closure captures batch, takes only weights
                def cost_fn(w):
                    total = 0.0
                    for sample in batch:
                        probs = self._circuit(sample, w)
                        total += probs[0]  # P(|00⟩) on trash qubits
                    return 1.0 - total / len(batch)

                self._weights, cost_val = opt.step_and_cost(
                    cost_fn, self._weights
                )
                epoch_losses.append(float(cost_val))

            epoch_mean = float(np.mean(epoch_losses))
            loss_history.append(epoch_mean)

            if verbose and (epoch % 10 == 0 or epoch == n_epochs - 1):
                logger.info(
                    "Epoch %03d/%03d  loss=%.4f", epoch, n_epochs, epoch_mean
                )

        return loss_history

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly scores for input samples.

        Score = 1 − P(|00⟩ on trash qubits). Higher means more anomalous.

        Parameters
        ----------
        X : np.ndarray
            Input features, shape ``(n_samples, n_qubits)`` in [0, π].

        Returns
        -------
        np.ndarray
            1-D array of shape ``(n_samples,)``.
        """
        scores = np.empty(len(X), dtype=np.float64)
        for i, sample in enumerate(X):
            probs = self._circuit(sample, self._weights)
            scores[i] = 1.0 - float(probs[0])
        return scores


# ------------------------------------------------------------------
# Reconstruction-error QAE
# ------------------------------------------------------------------

class QAEReconstruction:
    """Quantum autoencoder with reconstruction-error loss.

    Same architecture as :class:`QAETrashFidelity` (6 qubits, 4 latent +
    2 trash, EfficientSU2 ansatz) but trained with a full observable
    reconstruction loss:

    1. Encoder maps input to latent + trash register.
    2. Trash qubits [4, 5] are **measured and reset** to |0⟩ via
       mid-circuit measurement (MCM).
    3. The **adjoint** (inverse) of the encoder decodes the latent state.
    4. Loss = MSE(cos(x_i), ⟨Z_i⟩) over all six reconstructed wires —
       perfect reconstruction of RY(x)|0⟩ gives ⟨Z_i⟩ = cos(x_i).

    Uses ``diff_method='best'`` (parameter-shift) because backprop is
    incompatible with mid-circuit measurements.

    Parameters
    ----------
    n_qubits : int
        Total number of qubits (default 6).
    n_trash : int
        Number of trash qubits (default 2).
    n_layers : int
        Ansatz depth (default 4).
    learning_rate : float
        Adam step size (default 0.01).
    seed : int
        RNG seed for reproducibility (default 42).
    """

    def __init__(
        self,
        n_qubits: int = 6,
        n_trash: int = 2,
        n_layers: int = 4,
        learning_rate: float = 0.01,
        seed: int = 42,
    ) -> None:
        self.n_qubits = n_qubits
        self.n_trash = n_trash
        self.n_latent = n_qubits - n_trash
        self.n_layers = n_layers
        self.learning_rate = learning_rate
        self._seed = seed

        # Local RNG for reproducible, order-independent weight initialization
        rng = np.random.default_rng(seed)

        # Wireless device — MCM reset internally allocates auxiliary wires
        self._dev = qml.device("default.qubit")

        # 48 params for default config: 4 layers × 6 qubits × 2 rotations
        self._weights = pnp.array(
            rng.uniform(-0.1, 0.1, (n_layers, n_qubits, 2)),
            requires_grad=True,
        )

        self._trash_wires = list(range(n_qubits - n_trash, n_qubits))

        # QNode with parameter-shift (MCM incompatible with backprop)
        @qml.qnode(self._dev, diff_method="best")
        def _circuit(features, weights):
            qml.AngleEmbedding(features, wires=range(n_qubits), rotation="Y")
            efficient_su2(weights, wires=range(n_qubits))
            # Mid-circuit measurement + reset of trash qubits to |0⟩
            for tw in self._trash_wires:
                qml.measure(tw, reset=True)
            # Decode via inverse encoder
            qml.adjoint(efficient_su2)(weights, wires=range(n_qubits))
            # After decoding, all wires are reconstructed output wires.
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self._circuit = _circuit

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def weights(self) -> pnp.tensor:
        """Current trainable parameters."""
        return self._weights

    @property
    def num_trainable_parameters(self) -> int:
        """Number of trainable circuit parameters."""
        return int(np.asarray(self._weights).size)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,
        n_epochs: int = 200,
        batch_size: int = 32,
        verbose: bool = True,
    ) -> list[float]:
        """Train the autoencoder on normal data.

        Loss = MSE(cos(x), ⟨Z⟩) over all reconstructed wires. Perfect
        reconstruction of RY(x_i)|0⟩ yields ⟨Z_i⟩ = cos(x_i).

        Parameters
        ----------
        X_train : np.ndarray
            Training features, shape ``(n_samples, n_qubits)`` in [0, π].
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
        opt = qml.AdamOptimizer(stepsize=self.learning_rate)
        n_samples = len(X_train)
        loss_history: list[float] = []
        shuffle_rng = np.random.default_rng(self._seed)

        for epoch in range(n_epochs):
            perm = shuffle_rng.permutation(n_samples)
            X_shuffled = X_train[perm]

            epoch_losses: list[float] = []

            for start in range(0, n_samples, batch_size):
                batch = X_shuffled[start : start + batch_size]

                def cost_fn(w):
                    total_mse = pnp.array(0.0)
                    for sample in batch:
                        expvals = self._circuit(sample, w)
                        targets = pnp.cos(sample[: self.n_qubits])
                        expvals_stacked = pnp.stack(expvals)
                        total_mse = total_mse + pnp.mean(
                            (targets - expvals_stacked) ** 2
                        )
                    return total_mse / len(batch)

                self._weights, cost_val = opt.step_and_cost(
                    cost_fn, self._weights,
                )
                epoch_losses.append(float(cost_val))

            epoch_mean = float(np.mean(epoch_losses))
            loss_history.append(epoch_mean)

            if verbose and (epoch % 10 == 0 or epoch == n_epochs - 1):
                logger.info(
                    "Epoch %03d/%03d  loss=%.4f", epoch, n_epochs, epoch_mean,
                )

        return loss_history

    def reconstruct_observables(self, X: np.ndarray) -> np.ndarray:
        """Return reconstructed Pauli-Z observables for all output wires."""
        outputs = np.empty((len(X), self.n_qubits), dtype=np.float64)
        weights_np = np.array(self._weights)
        for i, sample in enumerate(X):
            expvals = self._circuit(sample, weights_np)
            outputs[i] = np.array([float(v) for v in expvals], dtype=np.float64)
        return outputs

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly scores for input samples.

        Score = MSE(cos(x), ⟨Z⟩). Higher means worse reconstruction
        (more anomalous).

        Uses a separate no-diff circuit for faster inference (no
        parameter-shift overhead).

        Parameters
        ----------
        X : np.ndarray
            Input features, shape ``(n_samples, n_qubits)`` in [0, π].

        Returns
        -------
        np.ndarray
            1-D array of shape ``(n_samples,)``.
        """
        # Build a lightweight inference-only circuit (no diff overhead)
        n_qubits = self.n_qubits
        trash_wires = self._trash_wires
        dev = qml.device("default.qubit")

        @qml.qnode(dev, diff_method=None)
        def _infer(features, weights):
            qml.AngleEmbedding(features, wires=range(n_qubits), rotation="Y")
            efficient_su2(weights, wires=range(n_qubits))
            for tw in trash_wires:
                qml.measure(tw, reset=True)
            qml.adjoint(efficient_su2)(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        weights_np = np.array(self._weights)
        scores = np.empty(len(X), dtype=np.float64)
        for i, sample in enumerate(X):
            expvals = _infer(sample, weights_np)
            targets = np.cos(sample[:n_qubits])
            expvals_np = np.array([float(v) for v in expvals])
            scores[i] = float(np.mean((targets - expvals_np) ** 2))
        return scores
