"""Evaluation metrics for anomaly detection: threshold calibration, binary classification metrics, and statistical significance."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def calibrate_threshold(calibration_scores: np.ndarray, percentile: float = 95.0) -> float:
    """Return a calibration-only percentile as an anomaly threshold.

    Parameters
    ----------
    calibration_scores : np.ndarray
        Array of anomaly scores from the held-out calibration-normal split.
    percentile : float, default 95.0
        Percentile value in [0, 100].

    Returns
    -------
    float
        The threshold value.
    """
    scores = np.asarray(calibration_scores, dtype=float).ravel()
    if scores.size == 0:
        raise ValueError("Cannot calibrate threshold from an empty calibration set.")
    if np.isnan(scores).any():
        raise ValueError("Cannot calibrate threshold from NaN calibration scores.")
    return float(np.percentile(scores, percentile))


def compute_binary_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Compute binary classification metrics from continuous anomaly scores.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth binary labels (0 = normal, 1 = anomaly).
    y_score : np.ndarray
        Continuous anomaly scores (higher → more anomalous).
    threshold : float
        Decision threshold; predictions are ``(y_score > threshold).astype(int)``.

    Returns
    -------
    dict[str, float]
        Keys: ``f1``, ``auc_roc``, ``auc_pr``, ``precision``, ``recall``.
        AUC values are ``NaN`` when *y_true* contains only one class (graceful
        degradation instead of crashing).
    """
    y_pred = (y_score > threshold).astype(int)

    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))

    # AUC metrics require both classes in y_true; degrade to NaN otherwise.
    try:
        auc_roc = float(roc_auc_score(y_true, y_score))
    except ValueError:
        auc_roc = float("nan")

    try:
        auc_pr = float(average_precision_score(y_true, y_score))
    except ValueError:
        auc_pr = float("nan")

    return {
        "f1": f1,
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "precision": precision,
        "recall": recall,
    }


def compute_significance(
    df: pd.DataFrame,
    model_a: str,
    model_b: str,
    metric: str = "f1",
) -> dict[str, float]:
    """Wilcoxon signed-rank test on paired (seed, fold, event_class) observations.

    Parameters
    ----------
    df : pd.DataFrame
        Results DataFrame with columns: model, seed, fold, event_class, and *metric*.
    model_a, model_b : str
        Model names to compare.
    metric : str
        Metric column to test (default ``"f1"``).

    Returns
    -------
    dict[str, float]
        Keys: ``statistic``, ``p_value``, ``n_pairs``, ``mean_diff``.
        Returns ``p_value=NaN`` when fewer than 6 paired observations exist
        (Wilcoxon requires at least ~6 non-zero differences).
    """
    key_cols = ["seed", "fold", "event_class"]
    a = df[df["model"] == model_a].set_index(key_cols)[metric]
    b = df[df["model"] == model_b].set_index(key_cols)[metric]
    paired = a.align(b, join="inner")
    va, vb = paired[0].dropna(), paired[1].dropna()
    common = va.index.intersection(vb.index)
    va, vb = va.loc[common], vb.loc[common]

    n_pairs = len(va)
    mean_diff = float((va - vb).mean())

    if n_pairs < 6 or (va == vb).all():
        return {
            "statistic": float("nan"),
            "p_value": float("nan"),
            "n_pairs": n_pairs,
            "mean_diff": mean_diff,
        }

    stat, p_val = wilcoxon(va.values, vb.values)
    return {
        "statistic": float(stat),
        "p_value": float(p_val),
        "n_pairs": n_pairs,
        "mean_diff": mean_diff,
    }


def _block_metric_frame(
    df: pd.DataFrame,
    metric: str,
    block_cols: tuple[str, ...] = ("seed", "fold"),
) -> pd.DataFrame:
    """Aggregate metric rows to paired experimental blocks."""
    required = {"model", metric, *block_cols}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns for block aggregation: {missing}")
    return (
        df.groupby(["model", *block_cols], as_index=False)
        .agg(**{metric: (metric, "mean"), "n_event_rows": (metric, "size")})
    )


def compute_block_significance(
    df: pd.DataFrame,
    model_a: str,
    model_b: str,
    metric: str = "f1",
    block_cols: tuple[str, ...] = ("seed", "fold"),
    min_pairs: int = 6,
) -> dict[str, float | int | str]:
    """Wilcoxon signed-rank test after aggregating to seed-fold blocks.

    This is the primary inference path for the QAE/3W report.  It avoids
    treating event-class rows inside the same seed/fold block as independent
    observations.
    """
    block_df = _block_metric_frame(df, metric=metric, block_cols=block_cols)
    a = block_df[block_df["model"] == model_a].set_index(list(block_cols))[metric]
    b = block_df[block_df["model"] == model_b].set_index(list(block_cols))[metric]
    va, vb = a.align(b, join="inner")
    va, vb = va.dropna(), vb.dropna()
    common = va.index.intersection(vb.index)
    va, vb = va.loc[common], vb.loc[common]

    n_pairs = len(va)
    diffs = va - vb
    mean_diff = float(diffs.mean()) if n_pairs else float("nan")
    blocking = "_".join(block_cols)

    if n_pairs < min_pairs or (n_pairs > 0 and (va == vb).all()):
        return {
            "statistic": float("nan"),
            "p_value": float("nan"),
            "n_pairs": n_pairs,
            "n_blocks": n_pairs,
            "mean_diff": mean_diff,
            "blocking": blocking,
            "metric": f"{metric}_mean",
        }

    stat, p_val = wilcoxon(va.values, vb.values)
    return {
        "statistic": float(stat),
        "p_value": float(p_val),
        "n_pairs": n_pairs,
        "n_blocks": n_pairs,
        "mean_diff": mean_diff,
        "blocking": blocking,
        "metric": f"{metric}_mean",
    }


def holm_adjust(p_values: list[float] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return Holm-adjusted p-values and reject flags for alpha=0.05."""
    p = np.asarray(p_values, dtype=float)
    adjusted = np.full_like(p, np.nan, dtype=float)
    finite_mask = np.isfinite(p)
    finite_indices = np.flatnonzero(finite_mask)
    if len(finite_indices) == 0:
        return adjusted, np.zeros_like(p, dtype=bool)

    order = finite_indices[np.argsort(p[finite_mask])]
    m = len(order)
    running_max = 0.0
    for rank, idx in enumerate(order, start=1):
        raw_adj = (m - rank + 1) * p[idx]
        running_max = max(running_max, raw_adj)
        adjusted[idx] = min(running_max, 1.0)
    reject = np.isfinite(adjusted) & (adjusted <= 0.05)
    return adjusted, reject


def compute_confidence_intervals(
    df: pd.DataFrame,
    model: str,
    metric: str = "f1",
    alpha: float = 0.05,
) -> dict[str, float]:
    """Bootstrap confidence interval for a model's mean metric value.

    Parameters
    ----------
    df : pd.DataFrame
        Results DataFrame with columns: model and *metric*.
    model : str
        Model name to compute CI for.
    metric : str
        Metric column (default ``"f1"``).
    alpha : float
        Significance level (default 0.05 for 95% CI).

    Returns
    -------
    dict[str, float]
        Keys: ``mean``, ``ci_lower``, ``ci_upper``, ``n_obs``.
    """
    values = df.loc[df["model"] == model, metric].dropna().values
    n = len(values)

    if n == 0:
        return {
            "mean": float("nan"),
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
            "n_obs": 0,
        }

    mean_val = float(values.mean())

    # Percentile bootstrap (10000 resamples)
    rng = np.random.default_rng(42)
    n_boot = 10000
    boot_means = rng.choice(values, size=(n_boot, n), replace=True).mean(axis=1)

    ci_lower = float(np.percentile(boot_means, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))

    return {
        "mean": mean_val,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n_obs": n,
    }


def compute_block_confidence_intervals(
    df: pd.DataFrame,
    model: str,
    metric: str = "f1",
    block_cols: tuple[str, ...] = ("seed", "fold"),
    alpha: float = 0.05,
    n_boot: int = 10000,
    seed: int = 42,
) -> dict[str, float | int | str]:
    """Block-bootstrap confidence interval for a model's mean metric."""
    model_df = df.loc[df["model"] == model, [*block_cols, metric]].dropna()
    if model_df.empty:
        return {
            "mean": float("nan"),
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
            "n_obs": 0,
            "n_blocks": 0,
            "blocking": "_".join(block_cols),
        }

    grouped = [group[metric].to_numpy(dtype=float) for _, group in model_df.groupby(list(block_cols))]
    n_blocks = len(grouped)
    values = model_df[metric].to_numpy(dtype=float)
    mean_val = float(values.mean())

    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_boot, dtype=float)
    for idx in range(n_boot):
        sampled = rng.integers(0, n_blocks, size=n_blocks)
        boot_values = np.concatenate([grouped[i] for i in sampled])
        boot_means[idx] = float(boot_values.mean())

    return {
        "mean": mean_val,
        "ci_lower": float(np.percentile(boot_means, 100 * alpha / 2)),
        "ci_upper": float(np.percentile(boot_means, 100 * (1 - alpha / 2))),
        "n_obs": int(len(values)),
        "n_blocks": int(n_blocks),
        "blocking": "_".join(block_cols),
    }
