#!/usr/bin/env python3
"""Validate and merge experiment JSONL metrics.

The experiment writes one row per ``(model, seed, fold, event_class)``.  These
helpers keep smoke tests, merge jobs, and full rerun gates honest by detecting
schema drift, duplicate result keys, and incomplete grids before figures or
paper claims are updated.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence


REQUIRED_RESULT_KEYS: tuple[str, ...] = (
    "model",
    "seed",
    "fold",
    "event_class",
    "event_name",
    "n_params",
    "threshold_percentile",
    "threshold_value",
    "n_train",
    "n_test_normal",
    "n_test_anomaly",
    "train_time_s",
    "f1",
    "auc_roc",
    "auc_pr",
    "precision",
    "recall",
)

CORRECTED_METHODOLOGY_KEYS: tuple[str, ...] = (
    "threshold_source",
    "threshold_quantile",
    "n_train_normal_files",
    "n_calibration_normal_files",
    "n_test_normal_files",
    "n_calibration_normal_windows",
    "calibration_fraction",
    "n_anomaly_eval_files",
    "n_anomaly_candidate_windows",
    "n_anomaly_positive_windows",
    "n_anomaly_unlabeled_windows",
    "n_anomaly_normal_or_other_windows",
    "label_policy",
    "min_labeled_fraction",
    "clipping_diagnostics_path",
    "window_label_diagnostics_path",
    "run_manifest_path",
    "model_hyperparameters",
)

RESULT_KEY_COLUMNS: tuple[str, ...] = ("model", "seed", "fold", "event_class")
DEFAULT_EVENT_CLASSES: tuple[int, ...] = tuple(range(1, 10))
DEFAULT_CLASS_LABELS: tuple[int, ...] = tuple(range(10))


class ResultValidationError(ValueError):
    """Raised when a JSONL result file fails integrity validation."""


def _as_set(values: Sequence[int | str] | None) -> set[int | str] | None:
    return set(values) if values is not None else None


def expected_row_count(
    models: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
    n_folds: int | None = None,
    event_classes: Sequence[int] | None = None,
) -> int | None:
    """Return the configured grid size, or ``None`` if any dimension is open."""
    if models is None or seeds is None or n_folds is None:
        return None
    events = tuple(event_classes) if event_classes is not None else DEFAULT_EVENT_CLASSES
    return len(models) * len(seeds) * int(n_folds) * len(events)


def load_jsonl(path: str | Path) -> list[dict]:
    """Load a JSONL metrics file with line-numbered parse errors."""
    path = Path(path)
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ResultValidationError(
                    f"{path}:{line_no}: invalid JSON: {exc.msg}"
                ) from exc
            if not isinstance(row, dict):
                raise ResultValidationError(f"{path}:{line_no}: row is not an object")
            rows.append(row)
    return rows


def validate_rows(
    rows: Sequence[dict],
    *,
    path: str | Path = "<rows>",
    models: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
    n_folds: int | None = None,
    event_classes: Sequence[int] | None = None,
    expected_rows: int | None = None,
    require_corrected_methodology: bool = False,
) -> dict[str, int]:
    """Validate result rows and return a compact summary."""
    path = str(path)
    model_set = _as_set(models)
    seed_set = _as_set(seeds)
    event_set = _as_set(event_classes if event_classes is not None else None)

    required = set(REQUIRED_RESULT_KEYS)
    if require_corrected_methodology:
        required.update(CORRECTED_METHODOLOGY_KEYS)
    keys: list[tuple] = []
    for idx, row in enumerate(rows, start=1):
        missing = sorted(required - set(row))
        if missing:
            raise ResultValidationError(f"{path}:{idx}: missing required keys: {missing}")

        key = tuple(row[col] for col in RESULT_KEY_COLUMNS)
        keys.append(key)

        model, seed, fold, event_class = key
        if model_set is not None and model not in model_set:
            raise ResultValidationError(f"{path}:{idx}: unexpected model {model!r}")
        if seed_set is not None and seed not in seed_set:
            raise ResultValidationError(f"{path}:{idx}: unexpected seed {seed!r}")
        if n_folds is not None and not (0 <= int(fold) < int(n_folds)):
            raise ResultValidationError(f"{path}:{idx}: fold {fold!r} outside 0..{n_folds - 1}")
        if event_set is not None and event_class not in event_set:
            raise ResultValidationError(f"{path}:{idx}: unexpected event_class {event_class!r}")

    duplicates = [key for key, count in Counter(keys).items() if count > 1]
    if duplicates:
        preview = ", ".join(map(str, duplicates[:5]))
        more = "" if len(duplicates) <= 5 else f" (+{len(duplicates) - 5} more)"
        raise ResultValidationError(f"{path}: duplicate result keys: {preview}{more}")

    configured_rows = expected_row_count(models, seeds, n_folds, event_classes)
    target_rows = expected_rows if expected_rows is not None else configured_rows
    if target_rows is not None and len(rows) != target_rows:
        raise ResultValidationError(
            f"{path}: expected {target_rows} rows, found {len(rows)}"
        )

    return {
        "rows": len(rows),
        "unique_keys": len(keys),
        "models": len({row["model"] for row in rows}),
        "seeds": len({row["seed"] for row in rows}),
        "folds": len({row["fold"] for row in rows}),
        "event_classes": len({row["event_class"] for row in rows}),
    }


def validate_result_file(
    path: str | Path,
    *,
    models: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
    n_folds: int | None = None,
    event_classes: Sequence[int] | None = None,
    expected_rows: int | None = None,
    require_corrected_methodology: bool = False,
) -> dict[str, int]:
    """Load and validate a JSONL metrics file."""
    rows = load_jsonl(path)
    return validate_rows(
        rows,
        path=path,
        models=models,
        seeds=seeds,
        n_folds=n_folds,
        event_classes=event_classes,
        expected_rows=expected_rows,
        require_corrected_methodology=require_corrected_methodology,
    )


def merge_result_files(
    input_paths: Iterable[str | Path],
    output_path: str | Path,
    *,
    overwrite: bool = False,
    models: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
    n_folds: int | None = None,
    event_classes: Sequence[int] | None = None,
    expected_rows: int | None = None,
    require_corrected_methodology: bool = False,
) -> dict[str, int]:
    """Validate inputs together, then write a deterministic merged JSONL file."""
    inputs = [Path(p) for p in input_paths]
    if not inputs:
        raise ResultValidationError("no input files provided for merge")
    for path in inputs:
        if not path.is_file():
            raise ResultValidationError(f"missing input file: {path}")

    output = Path(output_path)
    if output.exists() and not overwrite:
        raise ResultValidationError(
            f"output file already exists: {output}; pass --overwrite to replace it"
        )

    rows: list[dict] = []
    for path in inputs:
        rows.extend(load_jsonl(path))

    summary = validate_rows(
        rows,
        path="merged inputs",
        models=models,
        seeds=seeds,
        n_folds=n_folds,
        event_classes=event_classes,
        expected_rows=expected_rows,
        require_corrected_methodology=require_corrected_methodology,
    )

    rows_sorted = sorted(
        rows,
        key=lambda row: (
            str(row["model"]),
            int(row["seed"]),
            int(row["fold"]),
            int(row["event_class"]),
        ),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows_sorted:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    return summary


def validate_dataset_layout(
    data_path: str | Path,
    *,
    class_labels: Sequence[int] = DEFAULT_CLASS_LABELS,
) -> dict[str, int]:
    """Validate the documented 3W layout and report zero-byte Parquet files."""
    root = Path(data_path)
    if not root.is_dir():
        raise ResultValidationError(f"dataset root not found: {root}")

    n_parquet = 0
    zero_byte: list[Path] = []
    for label in class_labels:
        class_dir = root / str(label)
        if not class_dir.is_dir():
            raise ResultValidationError(f"missing class directory: {class_dir}")
        files = sorted(class_dir.glob("*.parquet"))
        n_parquet += len(files)
        zero_byte.extend(path for path in files if path.stat().st_size == 0)

    if zero_byte:
        preview = ", ".join(str(path) for path in zero_byte[:5])
        more = "" if len(zero_byte) <= 5 else f" (+{len(zero_byte) - 5} more)"
        raise ResultValidationError(f"zero-byte Parquet files: {preview}{more}")

    return {"class_dirs": len(class_labels), "parquet_files": n_parquet, "zero_byte": 0}


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", type=Path, help="JSONL files to validate")
    parser.add_argument("--check-data", type=Path, help="Validate 3W dataset class directories")
    parser.add_argument("--merge", nargs="+", type=Path, help="Input JSONL files to merge")
    parser.add_argument("--output", type=Path, help="Merged JSONL output path")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing --output")
    parser.add_argument("--models", nargs="+", help="Expected model names")
    parser.add_argument("--seeds", nargs="+", type=int, help="Expected seeds")
    parser.add_argument("--n-folds", type=int, help="Expected number of folds")
    parser.add_argument(
        "--event-classes",
        nargs="+",
        type=int,
        default=None,
        help="Expected anomaly event classes (default: validate row count with 1..9)",
    )
    parser.add_argument("--expected-rows", type=int, help="Explicit expected row count")
    parser.add_argument(
        "--require-corrected-methodology",
        action="store_true",
        help="Require split, threshold-source, label-policy, and diagnostics metadata keys.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        if args.check_data is not None:
            summary = validate_dataset_layout(args.check_data)
            print(f"DATA OK: {args.check_data}: {summary}")
        elif args.merge:
            if args.output is None:
                raise ResultValidationError("--output is required with --merge")
            summary = merge_result_files(
                args.merge,
                args.output,
                overwrite=args.overwrite,
                models=args.models,
                seeds=args.seeds,
                n_folds=args.n_folds,
                event_classes=args.event_classes,
                expected_rows=args.expected_rows,
                require_corrected_methodology=args.require_corrected_methodology,
            )
            print(f"MERGE OK: {summary} -> {args.output}")
        else:
            if not args.inputs:
                raise ResultValidationError("provide at least one JSONL input")
            for path in args.inputs:
                summary = validate_result_file(
                    path,
                    models=args.models,
                    seeds=args.seeds,
                    n_folds=args.n_folds,
                    event_classes=args.event_classes,
                    expected_rows=args.expected_rows,
                    require_corrected_methodology=args.require_corrected_methodology,
                )
                print(f"VALID: {path}: {summary}")
    except ResultValidationError as exc:
        print(f"INVALID: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
