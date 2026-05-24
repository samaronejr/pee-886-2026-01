#!/usr/bin/env python
"""CLI experiment runner: file-level KFold CV across all 7 anomaly-detection models.

Orchestrates the full evaluation pipeline:
1. Load class-0 normal files from 3W dataset.
2. For each seed × fold: train on normal data, calibrate threshold, score anomaly classes 1–9.
3. Write structured JSONL rows to an output file.

Usage
-----
    python scripts/samarone_junior/run_experiment.py \\
        --data-path data/samarone_junior/3w \\
        --output-dir data/samarone_junior/results \\
        --models IsolationForestDetector \\
        --n-folds 2 --seeds 42 --max-train-windows 200 --verbose
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata as importlib_metadata
import json
import logging
import platform
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

# ---------------------------------------------------------------------------
# Make the qml package importable when running from the repo root
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PKG_ROOT = _SCRIPT_DIR.parent.parent  # pee-886-2026-01/
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from qml.samarone_junior.loaders import ThreeWLoader  # noqa: E402
from qml.samarone_junior.loaders.preprocessing import FeatureEngineer  # noqa: E402
from qml.samarone_junior.evaluation import calibrate_threshold, compute_binary_metrics  # noqa: E402
from qml.samarone_junior.models import (  # noqa: E402
    QAETrashFidelity,
    QAEReconstruction,
    MatchedAutoencoder,
    FullAutoencoder,
    LSTMAutoencoder,
    IsolationForestDetector,
    OneClassSVMDetector,
    count_trainable_parameters,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, type] = {
    "QAETrashFidelity": QAETrashFidelity,
    "QAEReconstruction": QAEReconstruction,
    "MatchedAutoencoder": MatchedAutoencoder,
    "FullAutoencoder": FullAutoencoder,
    "LSTMAutoencoder": LSTMAutoencoder,
    "IsolationForestDetector": IsolationForestDetector,
    "OneClassSVMDetector": OneClassSVMDetector,
}

PARAM_COUNTS: dict[str, int | None] = {
    "QAETrashFidelity": 48,
    "QAEReconstruction": 48,
    "MatchedAutoencoder": 100,
    "FullAutoencoder": 3006,
    "LSTMAutoencoder": 3029,
    "IsolationForestDetector": None,
    "OneClassSVMDetector": None,
}

EVENT_NAMES: dict[int, str] = ThreeWLoader.EVENT_NAMES

# Anomaly classes to evaluate against.
ANOMALY_CLASSES = list(range(1, 10))


class WindowLabelPolicy(str, Enum):
    """Window-level label aggregation policies."""

    CENTER = "center"
    MAJORITY = "majority"
    ANY = "any"
    FRACTION = "fraction"


@dataclass(frozen=True)
class FoldSplit:
    """File-level normal split for one seed/fold."""

    seed: int
    fold: int
    train_normal_files: list[Path]
    calibration_normal_files: list[Path]
    test_normal_files: list[Path]


@dataclass
class WindowBatch:
    """Windows plus optional window-level labels and diagnostics."""

    windows: np.ndarray
    window_labels: list[str]
    diagnostics: list[dict[str, object]]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalise_event_label(value: object) -> int | None:
    """Map 3W class/transient labels to event labels; missing stays ``None``."""
    if pd.isna(value):
        return None
    label = int(value)
    if label >= 100:
        label -= 100
    return label


def assign_window_label(
    labels: np.ndarray,
    event_class: int,
    policy: WindowLabelPolicy | str = WindowLabelPolicy.MAJORITY,
    min_labeled_fraction: float = 0.80,
) -> str:
    """Assign a binary event label to one window from point-level labels."""
    policy = WindowLabelPolicy(policy)
    if not 0.0 <= min_labeled_fraction <= 1.0:
        raise ValueError("min_labeled_fraction must be between 0.0 and 1.0")
    normalised = [_normalise_event_label(value) for value in labels]
    valid = np.array([value is not None for value in normalised], dtype=bool)
    if valid.mean() < min_labeled_fraction:
        return "unlabeled"

    y = np.array([value for value in normalised if value is not None], dtype=int)
    if len(y) == 0:
        return "unlabeled"
    if policy == WindowLabelPolicy.CENTER:
        center = _normalise_event_label(labels[len(labels) // 2])
        if center is None:
            return "unlabeled"
        return "anomaly" if center == event_class else "normal_or_other"
    if policy == WindowLabelPolicy.MAJORITY:
        return "anomaly" if np.mean(y == event_class) > 0.5 else "normal_or_other"
    if policy == WindowLabelPolicy.ANY:
        return "anomaly" if np.any(y == event_class) else "normal_or_other"
    if policy == WindowLabelPolicy.FRACTION:
        return "anomaly" if np.mean(y == event_class) >= min_labeled_fraction else "normal_or_other"
    raise ValueError(f"Unknown window label policy: {policy}")


def _make_fold_split(
    *,
    normal_files: list[Path],
    fold_train_indices: np.ndarray,
    fold_test_indices: np.ndarray,
    seed: int,
    fold: int,
    calibration_fraction: float,
) -> FoldSplit:
    """Split fold-training normal files into train and calibration files."""
    train_pool = np.array(fold_train_indices, dtype=int)
    if len(train_pool) < 2:
        raise ValueError("At least two fold-training normal files are required for calibration split")
    rng = np.random.RandomState(seed + 10_000 * fold)
    shuffled = train_pool.copy()
    rng.shuffle(shuffled)
    n_calibration = max(1, int(round(calibration_fraction * len(shuffled))))
    n_calibration = min(n_calibration, len(shuffled) - 1)
    calibration_idx = sorted(shuffled[:n_calibration].tolist())
    train_idx = sorted(shuffled[n_calibration:].tolist())
    test_idx = sorted(np.array(fold_test_indices, dtype=int).tolist())
    return FoldSplit(
        seed=seed,
        fold=fold,
        train_normal_files=[normal_files[i] for i in train_idx],
        calibration_normal_files=[normal_files[i] for i in calibration_idx],
        test_normal_files=[normal_files[i] for i in test_idx],
    )


def _window_starts(n_rows: int, fe: FeatureEngineer) -> np.ndarray:
    if n_rows < fe.window_size:
        return np.array([], dtype=int)
    return np.arange(0, n_rows - fe.window_size + 1, fe.stride, dtype=int)


def _label_counts(labels: list[str]) -> dict[str, int]:
    return {
        "n_positive_windows": labels.count("anomaly") + labels.count("anomaly_file_level"),
        "n_normal_or_other_windows": labels.count("normal_or_other"),
        "n_unlabeled_windows": labels.count("unlabeled"),
    }


def _load_files_and_extract_window_batch(
    loader: ThreeWLoader,
    file_paths: list[Path],
    fe: FeatureEngineer,
    *,
    split: str,
    event_class: int | None = None,
    label_policy: WindowLabelPolicy | str = WindowLabelPolicy.MAJORITY,
    min_labeled_fraction: float = 0.80,
) -> WindowBatch:
    """Load files, extract per-file windows, and keep window-label diagnostics."""
    window_batches: list[np.ndarray] = []
    all_labels: list[str] = []
    diagnostics: list[dict[str, object]] = []
    sensors = loader.SENSORS
    for path in file_paths:
        try:
            try:
                df = pd.read_parquet(path, columns=sensors + ["class"])
                point_labels = df["class"].to_numpy()
            except Exception:
                df = pd.read_parquet(path, columns=sensors)
                point_labels = None
        except Exception as exc:
            logger.warning("Failed to read %s: %s", path, exc)
            continue
        if df.empty:
            logger.warning("Empty DataFrame from %s — skipping", path)
            continue

        sensor_valid = df[sensors].notna().all(axis=1).to_numpy()
        data = df.loc[sensor_valid, sensors].to_numpy(dtype=np.float64)
        labels = point_labels[sensor_valid] if point_labels is not None else None
        if len(data) == 0:
            logger.info("No complete sensor rows in %s after dropna — skipping", path)
            continue

        windows = fe.extract_windows(data)
        if len(windows) == 0:
            logger.info("No windows extracted from %s — file shorter than window size", path)
            continue

        starts = _window_starts(len(data), fe)
        file_labels: list[str] = []
        if event_class is None:
            file_labels = ["normal"] * len(windows)
        elif labels is None:
            file_labels = ["anomaly_file_level"] * len(windows)
        else:
            for start in starts:
                stop = start + fe.window_size
                file_labels.append(
                    assign_window_label(
                        labels[start:stop],
                        event_class,
                        policy=label_policy,
                        min_labeled_fraction=min_labeled_fraction,
                    )
                )

        counts = _label_counts(file_labels)
        diagnostics.append(
            {
                "split": split,
                "event_class": event_class if event_class is not None else 0,
                "source_file": str(path),
                "n_windows": len(windows),
                "label_policy": "normal" if event_class is None else str(WindowLabelPolicy(label_policy).value),
                **counts,
                "positive_fraction": counts["n_positive_windows"] / len(windows),
            }
        )
        window_batches.append(windows)
        all_labels.extend(file_labels)

    if not window_batches:
        return WindowBatch(
            windows=np.empty((0, fe.window_size, len(sensors)), dtype=np.float64),
            window_labels=[],
            diagnostics=diagnostics,
        )
    return WindowBatch(
        windows=np.concatenate(window_batches, axis=0),
        window_labels=all_labels,
        diagnostics=diagnostics,
    )


def _load_files_and_extract_windows(
    loader: ThreeWLoader,
    file_paths: list[Path],
    fe: FeatureEngineer,
) -> np.ndarray:
    """Load Parquet files and extract sliding windows per physical instance.

    File-level cross-validation only remains valid if each sliding window comes
    from a single Parquet instance.  Concatenating files before windowing creates
    artificial bridge windows at file boundaries, so this helper windows each
    file independently and concatenates the resulting 3-D window arrays.

    Returns
    -------
    np.ndarray
        3-D array ``(n_windows, window_size, n_sensors)`` or empty.
    """
    return _load_files_and_extract_window_batch(
        loader,
        file_paths,
        fe,
        split="unannotated",
    ).windows


def _append_csv_rows(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    """Append diagnostics rows, creating a header when needed."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def _clip_diagnostic_row(
    *,
    seed: int,
    fold: int,
    split: str,
    event_class: int,
    n_windows: int,
    report: dict[str, object],
) -> dict[str, object]:
    row: dict[str, object] = {
        "seed": seed,
        "fold": fold,
        "split": split,
        "event_class": event_class,
        "n_windows": n_windows,
        "clip_low_rate": report["clip_low_rate"],
        "clip_high_rate": report["clip_high_rate"],
        "clip_any_window_rate": report["clip_any_window_rate"],
    }
    for idx, value in enumerate(report.get("clip_rate_by_component", [])):
        row[f"component_{idx}_rate"] = value
    return row


def _git_commit() -> str | None:
    """Return the current submodule commit when git metadata is available."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=_PKG_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _package_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _model_hyperparameters(model: object) -> dict[str, object]:
    """Return JSON-safe model hyperparameters for result provenance."""
    params: dict[str, object] = {"seed": getattr(model, "seed", None)}
    if hasattr(model, "num_trainable_parameters"):
        params["num_trainable_parameters"] = getattr(model, "num_trainable_parameters")
    if hasattr(model, "n_qubits"):
        params["n_qubits"] = getattr(model, "n_qubits")
    if hasattr(model, "n_latent"):
        params["n_latent"] = getattr(model, "n_latent")
    if hasattr(model, "n_trash"):
        params["n_trash"] = getattr(model, "n_trash")
    if hasattr(model, "n_layers"):
        params["n_layers"] = getattr(model, "n_layers")

    wrapped = getattr(model, "_model", None)
    if hasattr(wrapped, "get_params"):
        params.update(wrapped.get_params(deep=False))
    return {key: value for key, value in params.items() if value is not None}


def _write_run_manifest(
    *,
    path: Path,
    args: argparse.Namespace,
    loader: ThreeWLoader,
    model_names: list[str],
    selected_folds: list[int],
    selected_events: list[int],
) -> None:
    """Write run provenance before metrics are streamed."""
    class_counts: dict[str, int] = {}
    example_columns: list[str] = []
    for class_label in range(10):
        try:
            files = loader.list_instances(class_label)
        except FileNotFoundError:
            files = []
        class_counts[str(class_label)] = len(files)
        if not example_columns and files:
            try:
                example_columns = list(pd.read_parquet(files[0]).columns)
            except Exception:
                example_columns = []

    manifest = {
        "run_created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "software_versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit-learn": _package_version("scikit-learn"),
            "pennylane": _package_version("PennyLane"),
            "torch": _package_version("torch"),
        },
        "dataset": {
            "root": str(loader.data_path),
            "version": "3W v2.0.0 (local parquet mirror)",
            "class_file_counts": class_counts,
            "detected_columns_example": example_columns,
            "detected_label_columns": [
                column for column in ("class", "state") if column in example_columns
            ],
        },
        "experiment": {
            "models": model_names,
            "seeds": args.seeds,
            "n_folds": args.n_folds,
            "selected_folds": selected_folds,
            "event_classes": selected_events,
            "threshold_source": "calibration_normal",
            "threshold_percentile": args.threshold_percentile,
            "threshold_quantile": args.threshold_percentile / 100.0,
            "calibration_fraction": args.calibration_fraction,
            "label_policy": args.label_policy,
            "min_labeled_fraction": args.min_labeled_fraction,
            "window_size": 128,
            "stride": 64,
            "n_pca_components": 6,
            "max_train_windows": args.max_train_windows,
            "qae_epochs": args.qae_epochs,
            "classical_epochs": args.classical_epochs,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _resolve_output_path(args: argparse.Namespace) -> Path:
    """Resolve the JSONL output path and enforce explicit overwrite safety."""
    output_file = getattr(args, "output_file", None)
    if output_file:
        output_path = Path(output_file)
    else:
        output_path = Path(args.output_dir) / "metrics.jsonl"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not getattr(args, "overwrite", False):
        raise FileExistsError(
            f"Output file already exists: {output_path}. "
            "Use --overwrite or choose --output-file/--output-dir for a new run."
        )
    return output_path


def _resolve_int_selection(
    values: list[int] | None,
    default_values: list[int] | range,
    valid_values: list[int] | range,
    label: str,
) -> list[int]:
    """Return a validated, de-duplicated integer selection.

    ``run_experiment`` writes one row per identity key
    ``(model, seed, fold, event_class)``.  Duplicate fold or event selectors
    would therefore create duplicate JSONL rows, so selections are de-duplicated
    while preserving user order.
    """
    selected = list(default_values if values is None else values)
    valid = set(valid_values)
    invalid = [value for value in selected if value not in valid]
    if invalid:
        expected = ", ".join(map(str, valid_values))
        raise ValueError(f"Invalid {label} value(s): {invalid}. Expected one of: {expected}")

    deduped: list[int] = []
    seen: set[int] = set()
    for value in selected:
        if value not in seen:
            deduped.append(value)
            seen.add(value)

    if not deduped:
        raise ValueError(f"At least one {label} must be selected")
    return deduped


def _subsample_windows(
    windows: np.ndarray,
    max_windows: int | None,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Randomly subsample windows if *max_windows* is set and exceeded."""
    if max_windows is None or len(windows) <= max_windows:
        return windows
    indices = rng.choice(len(windows), size=max_windows, replace=False)
    indices.sort()
    return windows[indices]


def _is_lstm(model_name: str) -> bool:
    return model_name == "LSTMAutoencoder"


# ---------------------------------------------------------------------------
# Core experiment
# ---------------------------------------------------------------------------


def run_experiment(args: argparse.Namespace) -> None:
    """Main experiment loop."""
    loader = ThreeWLoader(data_path=args.data_path)

    # Resolve output path with overwrite guard.
    try:
        output_path = _resolve_output_path(args)
    except FileExistsError as exc:
        logger.error(str(exc))
        raise SystemExit(2) from exc

    # Determine which models to run
    if args.models == ["all"]:
        model_names = list(MODEL_REGISTRY.keys())
    else:
        model_names = args.models
        for mn in model_names:
            if mn not in MODEL_REGISTRY:
                logger.error("Unknown model: %s. Available: %s", mn, list(MODEL_REGISTRY.keys()))
                sys.exit(1)

    try:
        selected_folds = _resolve_int_selection(
            getattr(args, "folds", None),
            range(args.n_folds),
            range(args.n_folds),
            "fold",
        )
        selected_events = _resolve_int_selection(
            getattr(args, "event_classes", None),
            ANOMALY_CLASSES,
            ANOMALY_CLASSES,
            "event_class",
        )
    except ValueError as exc:
        logger.error(str(exc))
        raise SystemExit(2) from exc
    selected_fold_set = set(selected_folds)
    logger.info("Selected folds: %s", selected_folds)
    logger.info("Selected event classes: %s", selected_events)

    # Load class-0 (normal) file list
    normal_files = loader.list_instances(0)
    n_normals = len(normal_files)
    logger.info("Loaded %d class-0 (Normal) file paths", n_normals)

    # Choose epoch counts per model category
    qae_epochs = args.qae_epochs
    classical_epochs = args.classical_epochs
    diagnostics_dir = Path(args.diagnostics_dir)
    clipping_diagnostics_path = diagnostics_dir / "clipping_diagnostics.csv"
    label_diagnostics_path = diagnostics_dir / "window_label_diagnostics.csv"
    manifest_path = Path(args.manifest_path) if args.manifest_path else diagnostics_dir / "run_manifest.json"
    for diagnostics_path in (clipping_diagnostics_path, label_diagnostics_path):
        if diagnostics_path.exists() and getattr(args, "overwrite", False):
            diagnostics_path.unlink()
    if manifest_path.exists() and getattr(args, "overwrite", False):
        manifest_path.unlink()

    _write_run_manifest(
        path=manifest_path,
        args=args,
        loader=loader,
        model_names=model_names,
        selected_folds=selected_folds,
        selected_events=selected_events,
    )

    clip_fields = [
        "seed", "fold", "split", "event_class", "n_windows",
        "clip_low_rate", "clip_high_rate", "clip_any_window_rate",
        "component_0_rate", "component_1_rate", "component_2_rate",
        "component_3_rate", "component_4_rate", "component_5_rate",
    ]
    label_fields = [
        "split", "event_class", "source_file", "n_windows", "label_policy",
        "n_positive_windows", "n_normal_or_other_windows", "n_unlabeled_windows",
        "positive_fraction",
    ]

    # Open output file (write mode)
    with open(output_path, "w") as fout:
        for seed in args.seeds:
            kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=seed)
            rng = np.random.RandomState(seed)

            for fold_idx, (train_idx, test_idx) in enumerate(kf.split(range(n_normals))):
                if fold_idx not in selected_fold_set:
                    continue

                logger.info(
                    "=== seed=%d  fold=%d/%d  train_files=%d  test_files=%d ===",
                    seed, fold_idx, args.n_folds, len(train_idx), len(test_idx),
                )

                try:
                    split = _make_fold_split(
                        normal_files=normal_files,
                        fold_train_indices=train_idx,
                        fold_test_indices=test_idx,
                        seed=seed,
                        fold=fold_idx,
                        calibration_fraction=args.calibration_fraction,
                    )
                except ValueError as exc:
                    logger.warning("Invalid split seed=%d fold=%d: %s", seed, fold_idx, exc)
                    continue

                train_files = split.train_normal_files
                calibration_files = split.calibration_normal_files
                test_files = split.test_normal_files

                fe = FeatureEngineer(n_components=6, window_size=128, stride=64)

                train_batch = _load_files_and_extract_window_batch(
                    loader, train_files, fe, split="train_normal",
                )
                train_windows_3d = train_batch.windows
                if len(train_windows_3d) == 0:
                    logger.warning("No training windows for seed=%d fold=%d — skipping", seed, fold_idx)
                    continue

                # Apply subsampling
                train_windows_3d = _subsample_windows(train_windows_3d, args.max_train_windows, rng)

                # Compute 2D features: statistical features → PCA (fit on train)
                train_features_raw = fe.compute_features(train_windows_3d)
                train_features_2d, train_clip = fe.fit_transform_with_clip_report(train_features_raw)

                calibration_batch = _load_files_and_extract_window_batch(
                    loader, calibration_files, fe, split="calibration_normal",
                )
                calibration_windows_3d = calibration_batch.windows
                if len(calibration_windows_3d) == 0:
                    logger.warning("No calibration windows for seed=%d fold=%d — skipping", seed, fold_idx)
                    continue
                calibration_features_raw = fe.compute_features(calibration_windows_3d)
                calibration_features_2d, calibration_clip = fe.transform_with_clip_report(calibration_features_raw)

                # Load test-normal windows
                test_batch = _load_files_and_extract_window_batch(
                    loader, test_files, fe, split="test_normal",
                )
                test_normal_windows_3d = test_batch.windows

                # Compute 2D features for test normals (transform only, no fit!)
                if len(test_normal_windows_3d) > 0:
                    test_normal_features_raw = fe.compute_features(test_normal_windows_3d)
                    test_normal_features_2d, test_clip = fe.transform_with_clip_report(test_normal_features_raw)
                else:
                    test_normal_features_2d = np.empty((0, 6), dtype=np.float64)
                    test_clip = FeatureEngineer.clip_report(test_normal_features_2d)

                _append_csv_rows(
                    clipping_diagnostics_path,
                    [
                        _clip_diagnostic_row(
                            seed=seed, fold=fold_idx, split="train_normal", event_class=0,
                            n_windows=len(train_windows_3d), report=train_clip,
                        ),
                        _clip_diagnostic_row(
                            seed=seed, fold=fold_idx, split="calibration_normal", event_class=0,
                            n_windows=len(calibration_windows_3d), report=calibration_clip,
                        ),
                        _clip_diagnostic_row(
                            seed=seed, fold=fold_idx, split="test_normal", event_class=0,
                            n_windows=len(test_normal_windows_3d), report=test_clip,
                        ),
                    ],
                    clip_fields,
                )
                _append_csv_rows(
                    label_diagnostics_path,
                    train_batch.diagnostics + calibration_batch.diagnostics + test_batch.diagnostics,
                    label_fields,
                )

                # ---- Per-model training & scoring ----
                for model_name in model_names:
                    logger.info("  Model: %s (seed=%d, fold=%d)", model_name, seed, fold_idx)

                    model_cls = MODEL_REGISTRY[model_name]

                    # Determine epoch count
                    if model_name.startswith("QAE"):
                        n_epochs = qae_epochs
                    else:
                        n_epochs = classical_epochs

                    # Instantiate model
                    model = model_cls(seed=seed)

                    # Select training input based on model type
                    if _is_lstm(model_name):
                        train_input = train_windows_3d
                        calibration_input = calibration_windows_3d
                    else:
                        train_input = train_features_2d
                        calibration_input = calibration_features_2d

                    # Fit
                    t_start = time.time()
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", category=RuntimeWarning)
                            model.fit(
                                train_input,
                                n_epochs=n_epochs,
                                batch_size=32,
                                verbose=args.verbose,
                            )
                    except Exception as exc:
                        logger.error(
                            "  fit() failed for %s seed=%d fold=%d: %s",
                            model_name, seed, fold_idx, exc,
                        )
                        continue
                    train_time = time.time() - t_start
                    logger.info("  Trained %s in %.1fs", model_name, train_time)

                    # Score calibration-normal data for threshold calibration
                    try:
                        calibration_scores = model.anomaly_scores(calibration_input)
                        if np.isnan(calibration_scores).any():
                            nan_count = int(np.isnan(calibration_scores).sum())
                            logger.warning(
                                "  %d NaN in calibration scores for %s seed=%d fold=%d — skipping",
                                nan_count, model_name, seed, fold_idx,
                            )
                            continue
                    except Exception as exc:
                        logger.error("  anomaly_scores(calibration_normal) failed for %s: %s", model_name, exc)
                        continue

                    # Score test normals
                    if _is_lstm(model_name):
                        test_normal_input = test_normal_windows_3d
                    else:
                        test_normal_input = test_normal_features_2d

                    if len(test_normal_input) > 0:
                        try:
                            test_normal_scores = model.anomaly_scores(test_normal_input)
                            if np.isnan(test_normal_scores).any():
                                nan_count = int(np.isnan(test_normal_scores).sum())
                                logger.warning(
                                    "  %d NaN in test-normal scores for %s seed=%d fold=%d — skipping",
                                    nan_count, model_name, seed, fold_idx,
                                )
                                continue
                        except Exception as exc:
                            logger.error("  anomaly_scores(test_normal) failed for %s: %s", model_name, exc)
                            continue
                    else:
                        test_normal_scores = np.array([], dtype=np.float64)

                    n_test_normal = len(test_normal_scores)
                    n_calibration_normal = len(calibration_scores)

                    threshold = calibrate_threshold(calibration_scores, args.threshold_percentile)

                    # ---- Score each anomaly class ----
                    for event_class in selected_events:
                        event_name = EVENT_NAMES.get(event_class, f"Class{event_class}")
                        logger.info("    Event %d: %s", event_class, event_name)

                        try:
                            anomaly_files = loader.list_instances(event_class)
                        except FileNotFoundError:
                            logger.warning("    Class directory %d not found — skipping", event_class)
                            continue

                        if not anomaly_files:
                            logger.warning("    No files for class %d — skipping", event_class)
                            continue

                        # Load anomaly windows and keep only positive event windows.
                        anomaly_batch = _load_files_and_extract_window_batch(
                            loader,
                            anomaly_files,
                            fe,
                            split="anomaly_eval",
                            event_class=event_class,
                            label_policy=args.label_policy,
                            min_labeled_fraction=args.min_labeled_fraction,
                        )
                        _append_csv_rows(label_diagnostics_path, anomaly_batch.diagnostics, label_fields)

                        anomaly_candidate_windows_3d = anomaly_batch.windows
                        positive_mask = np.array(
                            [label in {"anomaly", "anomaly_file_level"} for label in anomaly_batch.window_labels],
                            dtype=bool,
                        )
                        anomaly_windows_3d = anomaly_candidate_windows_3d[positive_mask]

                        if len(anomaly_windows_3d) == 0:
                            logger.warning(
                                "    Zero positive windows for class %d after window labeling — skipping",
                                event_class,
                            )
                            continue

                        # Choose input representation
                        if _is_lstm(model_name):
                            anomaly_input = anomaly_windows_3d
                        else:
                            anomaly_features_raw = fe.compute_features(anomaly_windows_3d)
                            anomaly_input, anomaly_clip = fe.transform_with_clip_report(anomaly_features_raw)

                        if _is_lstm(model_name):
                            anomaly_clip = FeatureEngineer.clip_report(np.empty((0, 6), dtype=np.float64))
                        _append_csv_rows(
                            clipping_diagnostics_path,
                            [
                                _clip_diagnostic_row(
                                    seed=seed, fold=fold_idx, split="anomaly_eval",
                                    event_class=event_class, n_windows=len(anomaly_windows_3d),
                                    report=anomaly_clip,
                                )
                            ],
                            clip_fields,
                        )

                        # Score anomaly windows
                        try:
                            anomaly_scores = model.anomaly_scores(anomaly_input)
                            if np.isnan(anomaly_scores).any():
                                nan_count = int(np.isnan(anomaly_scores).sum())
                                logger.warning(
                                    "    %d NaN in anomaly scores for %s class=%d — skipping",
                                    nan_count, model_name, event_class,
                                )
                                continue
                        except Exception as exc:
                            logger.error(
                                "    anomaly_scores() failed for %s class=%d: %s",
                                model_name, event_class, exc,
                            )
                            continue

                        n_test_anomaly = len(anomaly_scores)

                        # Build binary test set: normal(0) vs anomaly(1)
                        y_true = np.concatenate([
                            np.zeros(n_test_normal, dtype=int),
                            np.ones(n_test_anomaly, dtype=int),
                        ])
                        y_score = np.concatenate([
                            test_normal_scores,
                            anomaly_scores,
                        ])

                        # Compute metrics
                        metrics = compute_binary_metrics(y_true, y_score, threshold)

                        # Build JSONL row
                        row = {
                            "model": model_name,
                            "seed": seed,
                            "fold": fold_idx,
                            "event_class": event_class,
                            "event_name": event_name,
                            "n_params": count_trainable_parameters(model),
                            "threshold_percentile": args.threshold_percentile,
                            "threshold_source": "calibration_normal",
                            "threshold_quantile": args.threshold_percentile / 100.0,
                            "threshold_value": float(threshold),
                            "n_train": len(train_input),
                            "n_train_normal_files": len(train_files),
                            "n_calibration_normal_files": len(calibration_files),
                            "n_test_normal_files": len(test_files),
                            "n_calibration_normal_windows": n_calibration_normal,
                            "calibration_fraction": args.calibration_fraction,
                            "n_test_normal": n_test_normal,
                            "n_test_anomaly": n_test_anomaly,
                            "n_anomaly_eval_files": len(anomaly_files),
                            "n_anomaly_candidate_windows": len(anomaly_candidate_windows_3d),
                            "n_anomaly_positive_windows": int(positive_mask.sum()),
                            "n_anomaly_unlabeled_windows": anomaly_batch.window_labels.count("unlabeled"),
                            "n_anomaly_normal_or_other_windows": anomaly_batch.window_labels.count("normal_or_other"),
                            "label_policy": args.label_policy,
                            "min_labeled_fraction": args.min_labeled_fraction,
                            "clipping_diagnostics_path": str(clipping_diagnostics_path),
                            "window_label_diagnostics_path": str(label_diagnostics_path),
                            "run_manifest_path": str(manifest_path),
                            "model_hyperparameters": _model_hyperparameters(model),
                            "train_time_s": round(train_time, 3),
                            **{k: (None if (isinstance(v, float) and np.isnan(v)) else v)
                               for k, v in metrics.items()},
                        }
                        if model_name == "QAEReconstruction":
                            row.update(
                                {
                                    "qae_reconstruction_loss_scope": "full_6_qubit_observable_mse",
                                    "qae_reconstruction_target": "cos_angle_embedding",
                                    "uses_mid_circuit_measurement": True,
                                    "uses_trash_reset": True,
                                    "decoder": "adjoint",
                                }
                            )

                        fout.write(json.dumps(row) + "\n")
                        fout.flush()

                        logger.info(
                            "    → F1=%.3f  AUC-ROC=%s  n_test=%d+%d",
                            metrics["f1"],
                            f'{metrics["auc_roc"]:.3f}' if not np.isnan(metrics["auc_roc"]) else "NaN",
                            n_test_normal,
                            n_test_anomaly,
                        )

    logger.info("Results written to %s", output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run anomaly-detection experiment across models/folds/seeds.",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/samarone_junior/3w",
        help="Path to 3W dataset root (default: data/samarone_junior/3w)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/samarone_junior/results",
        help="Directory for JSONL output (default: data/samarone_junior/results)",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help=(
            "Exact JSONL output path. Overrides --output-dir/metrics.jsonl "
            "when provided."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing JSONL output file.",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["all"],
        help="Model names (space-separated) or 'all' (default: all)",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Number of KFold splits (default: 5)",
    )
    parser.add_argument(
        "--folds",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Optional subset of fold indices to run, zero-based. "
            "Defaults to all folds. Useful for Slurm array retries."
        ),
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 123, 456, 789, 1024, 2023, 2024, 2025, 2026, 7777],
        help=(
            "RNG seeds (default: 42 123 456 789 1024 "
            "2023 2024 2025 2026 7777)"
        ),
    )
    parser.add_argument(
        "--event-classes",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Optional subset of anomaly event classes to score (1–9). "
            "Defaults to all anomaly classes. Useful for partial fold retries."
        ),
    )
    parser.add_argument(
        "--threshold-percentile",
        type=float,
        default=95.0,
        help="Percentile for threshold calibration (default: 95)",
    )
    parser.add_argument(
        "--calibration-fraction",
        type=float,
        default=0.20,
        help=(
            "Fraction of each fold's normal training files held out for "
            "threshold calibration (default: 0.20)"
        ),
    )
    parser.add_argument(
        "--label-policy",
        type=str,
        choices=[policy.value for policy in WindowLabelPolicy],
        default=WindowLabelPolicy.MAJORITY.value,
        help=(
            "Window label aggregation policy for anomaly files: center, majority, "
            "any, or fraction (default: majority)"
        ),
    )
    parser.add_argument(
        "--min-labeled-fraction",
        type=float,
        default=0.80,
        help=(
            "Minimum non-missing point-label fraction required for window labels; "
            "also used as the positive fraction cutoff by --label-policy fraction "
            "(default: 0.80)"
        ),
    )
    parser.add_argument(
        "--diagnostics-dir",
        type=str,
        default="data/samarone_junior/artifacts",
        help=(
            "Directory for clipping and window-label diagnostics CSV files "
            "(default: data/samarone_junior/artifacts)"
        ),
    )
    parser.add_argument(
        "--manifest-path",
        type=str,
        default=None,
        help=(
            "Optional path for run_manifest.json "
            "(default: <diagnostics-dir>/run_manifest.json)"
        ),
    )
    parser.add_argument(
        "--max-train-windows",
        type=int,
        default=None,
        help="Cap training windows per fold (default: no cap)",
    )
    parser.add_argument(
        "--qae-epochs",
        type=int,
        default=200,
        help="Training epochs for QAE models (default: 200)",
    )
    parser.add_argument(
        "--classical-epochs",
        type=int,
        default=200,
        help="Training epochs for classical models (default: 200)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable INFO-level logging",
    )
    args = parser.parse_args(argv)
    if not 0.0 <= args.calibration_fraction <= 1.0:
        parser.error("--calibration-fraction must be between 0.0 and 1.0")
    if not 0.0 <= args.min_labeled_fraction <= 1.0:
        parser.error("--min-labeled-fraction must be between 0.0 and 1.0")
    if not 0.0 <= args.threshold_percentile <= 100.0:
        parser.error("--threshold-percentile must be between 0.0 and 100.0")
    return args


def main(argv: list[str] | None = None) -> None:
    """Entry point."""
    args = parse_args(argv)

    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    run_experiment(args)


if __name__ == "__main__":
    main()
