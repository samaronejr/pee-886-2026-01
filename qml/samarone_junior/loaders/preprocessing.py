"""Feature engineering pipeline for quantum angle encoding.

Transforms raw sensor time-series into [0, π]-normalised PCA vectors
suitable for parameterised quantum circuits.

Pipeline: sliding windows → statistical features → PCA → MinMaxScaler.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.stats import kurtosis as _scipy_kurtosis
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


class FeatureEngineer:
    """Sliding-window feature extraction with PCA reduction to [0, π].

    Parameters
    ----------
    n_components : int
        Number of PCA components (default 6).
    window_size : int
        Length of each sliding window in timesteps (default 128).
    stride : int
        Step between consecutive windows (default 64).
    """

    def __init__(
        self,
        n_components: int = 6,
        window_size: int = 128,
        stride: int = 64,
    ) -> None:
        self.n_components = n_components
        self.window_size = window_size
        self.stride = stride
        self.pca: PCA | None = None
        self.scaler: MinMaxScaler | None = None

    # ------------------------------------------------------------------
    # Windowing
    # ------------------------------------------------------------------

    def extract_windows(self, data: np.ndarray) -> np.ndarray:
        """Slide a fixed-size window over *data*.

        Parameters
        ----------
        data : np.ndarray
            2-D array of shape ``(n_timesteps, n_sensors)``.

        Returns
        -------
        np.ndarray
            3-D array of shape ``(n_windows, window_size, n_sensors)``.
            Empty ``(0, window_size, n_sensors)`` when *data* is shorter
            than one window.
        """
        n_timesteps, n_sensors = data.shape
        if n_timesteps < self.window_size:
            return np.empty((0, self.window_size, n_sensors), dtype=data.dtype)

        starts = np.arange(0, n_timesteps - self.window_size + 1, self.stride)
        windows = np.stack([data[s : s + self.window_size] for s in starts])
        return windows

    # ------------------------------------------------------------------
    # Statistical features
    # ------------------------------------------------------------------

    def compute_features(self, windows: np.ndarray) -> np.ndarray:
        """Compute per-sensor statistics for each window.

        For every window and every sensor column the following four
        statistics are computed: **mean**, **std**, **range** (max − min),
        and **kurtosis** (excess, Fisher definition).  Constant-value
        windows produce ``kurtosis = 0.0`` instead of ``NaN``.

        Parameters
        ----------
        windows : np.ndarray
            3-D array ``(n_windows, window_size, n_sensors)``.

        Returns
        -------
        np.ndarray
            2-D array ``(n_windows, n_sensors * 4)``.
        """
        n_windows, _, n_sensors = windows.shape

        means = windows.mean(axis=1)                       # (n_win, n_sens)
        stds = windows.std(axis=1)                         # (n_win, n_sens)
        ranges = windows.max(axis=1) - windows.min(axis=1) # (n_win, n_sens)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            kurts = _scipy_kurtosis(windows, axis=1, fisher=True)  # (n_win, n_sens)

        # Constant windows yield NaN kurtosis — replace with 0.0
        kurts = np.nan_to_num(kurts, nan=0.0)

        # Interleave: [s0_mean, s0_std, s0_range, s0_kurt, s1_mean, ...]
        features = np.empty((n_windows, n_sensors * 4), dtype=np.float64)
        for i in range(n_sensors):
            features[:, i * 4 + 0] = means[:, i]
            features[:, i * 4 + 1] = stds[:, i]
            features[:, i * 4 + 2] = ranges[:, i]
            features[:, i * 4 + 3] = kurts[:, i]

        return features

    # ------------------------------------------------------------------
    # PCA + scaling
    # ------------------------------------------------------------------

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        """Fit PCA and MinMaxScaler, return [0, π]-normalised components.

        Parameters
        ----------
        features : np.ndarray
            2-D array ``(n_samples, n_features)``.

        Returns
        -------
        np.ndarray
            2-D array ``(n_samples, n_components)`` with values in [0, π].
        """
        self.pca = PCA(n_components=self.n_components)
        pca_out = self.pca.fit_transform(features)

        self.scaler = MinMaxScaler(feature_range=(0, np.pi))
        return np.clip(self.scaler.fit_transform(pca_out), 0, np.pi)

    @staticmethod
    def clip_report(scaled_to_pi_raw: np.ndarray) -> dict[str, object]:
        """Compute clipping diagnostics before applying ``np.clip``."""
        raw = np.asarray(scaled_to_pi_raw, dtype=float)
        if raw.size == 0:
            n_components = raw.shape[1] if raw.ndim == 2 else 0
            return {
                "clip_low_rate": 0.0,
                "clip_high_rate": 0.0,
                "clip_any_window_rate": 0.0,
                "clip_rate_by_component": [0.0] * n_components,
            }
        low = raw < 0.0
        high = raw > np.pi
        any_clip = (low | high).any(axis=1)
        return {
            "clip_low_rate": float(low.mean()),
            "clip_high_rate": float(high.mean()),
            "clip_any_window_rate": float(any_clip.mean()),
            "clip_rate_by_component": (low | high).mean(axis=0).astype(float).tolist(),
        }

    def fit_transform_with_clip_report(self, features: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
        """Fit PCA/scaler and return transformed features plus clip diagnostics."""
        self.pca = PCA(n_components=self.n_components)
        pca_out = self.pca.fit_transform(features)
        self.scaler = MinMaxScaler(feature_range=(0, np.pi))
        raw_scaled = self.scaler.fit_transform(pca_out)
        return np.clip(raw_scaled, 0, np.pi), self.clip_report(raw_scaled)

    def transform(self, features: np.ndarray) -> np.ndarray:
        """Apply the already-fitted pipeline to new data.

        Parameters
        ----------
        features : np.ndarray
            2-D array ``(n_samples, n_features)``.

        Returns
        -------
        np.ndarray
            2-D array ``(n_samples, n_components)`` with values in [0, π].

        Raises
        ------
        RuntimeError
            If called before :meth:`fit_transform`.
        """
        if self.pca is None or self.scaler is None:
            raise RuntimeError(
                "FeatureEngineer has not been fitted yet. "
                "Call fit_transform() first."
            )
        return np.clip(self.scaler.transform(self.pca.transform(features)), 0, np.pi)

    def transform_with_clip_report(self, features: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
        """Apply fitted pipeline and return transformed features plus clip diagnostics."""
        if self.pca is None or self.scaler is None:
            raise RuntimeError(
                "FeatureEngineer has not been fitted yet. "
                "Call fit_transform() first."
            )
        raw_scaled = self.scaler.transform(self.pca.transform(features))
        return np.clip(raw_scaled, 0, np.pi), self.clip_report(raw_scaled)
