#!/usr/bin/env python
"""Targeted regression tests for the QAE/3W remediation gate.

Run from ``pee-886-2026-01`` with::

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python scripts/tests/run_targeted_tests.py

The suite is intentionally unittest-based and uses only temporary synthetic
DataFrames/Parquet files, never the real 3W dataset.
"""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from qml.samarone_junior.loaders import FeatureEngineer, ThreeWLoader
from scripts.samarone_junior.run_experiment import _load_files_and_extract_windows


SENSORS = ThreeWLoader.SENSORS
def _sensor_frame(value: float, n_rows: int, class_label: int = 0) -> pd.DataFrame:
    data = {sensor: np.full(n_rows, value, dtype=np.float64) for sensor in SENSORS}
    data["class"] = np.full(n_rows, class_label, dtype=np.int64)
    return pd.DataFrame(data)


def _write_parquet(path: Path, df: pd.DataFrame) -> Path:
    df.to_parquet(path, index=False)
    return path


def _result_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "model": "IsolationForestDetector",
        "seed": 42,
        "fold": 0,
        "event_class": 1,
        "event_name": "Abrupt BSW Increase",
        "n_params": None,
        "threshold_percentile": 95.0,
        "threshold_value": 0.5,
        "n_train": 10,
        "n_test_normal": 4,
        "n_test_anomaly": 6,
        "train_time_s": 0.01,
        "precision": 0.8,
        "recall": 1.0,
        "f1": 0.8888888889,
        "auc_roc": 0.95,
        "auc_pr": 0.93,
    }
    row.update(overrides)
    return row


def _corrected_methodology_fields() -> dict[str, object]:
    return {
        "threshold_source": "calibration_normal",
        "threshold_quantile": 0.95,
        "n_train_normal_files": 8,
        "n_calibration_normal_files": 2,
        "n_test_normal_files": 3,
        "n_calibration_normal_windows": 5,
        "calibration_fraction": 0.20,
        "n_anomaly_eval_files": 4,
        "n_anomaly_candidate_windows": 12,
        "n_anomaly_positive_windows": 6,
        "n_anomaly_unlabeled_windows": 2,
        "n_anomaly_normal_or_other_windows": 4,
        "label_policy": "majority",
        "min_labeled_fraction": 0.80,
        "clipping_diagnostics_path": "data/samarone_junior/artifacts/clipping_diagnostics.csv",
        "window_label_diagnostics_path": "data/samarone_junior/artifacts/window_label_diagnostics.csv",
        "run_manifest_path": "data/samarone_junior/artifacts/run_manifest.json",
        "model_hyperparameters": {"n_estimators": 100, "contamination": "auto"},
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> Path:
    with path.open("w", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row) + "\n")
    return path


class WindowExtractionTests(unittest.TestCase):
    def test_extracts_windows_per_file_without_cross_file_bridge_rows(self) -> None:
        """Returned windows contain rows from exactly one source file."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = _write_parquet(root / "WELL-00001.parquet", _sensor_frame(1.0, 8))
            second = _write_parquet(root / "WELL-00002.parquet", _sensor_frame(2.0, 8))

            fe = FeatureEngineer(n_components=2, window_size=6, stride=4)
            windows = _load_files_and_extract_windows(ThreeWLoader(str(root)), [first, second], fe)

        self.assertEqual(windows.shape, (2, 6, len(SENSORS)))
        for window in windows:
            unique_values = set(np.unique(window[:, 0]).tolist())
            self.assertEqual(len(unique_values), 1, f"Bridge window detected: {unique_values}")

    def test_skips_short_files_without_dropping_valid_file_windows(self) -> None:
        """Files shorter than the window size contribute zero windows."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            short = _write_parquet(root / "WELL-short.parquet", _sensor_frame(1.0, 5))
            valid = _write_parquet(root / "WELL-valid.parquet", _sensor_frame(2.0, 6))

            fe = FeatureEngineer(n_components=2, window_size=6, stride=3)
            windows = _load_files_and_extract_windows(ThreeWLoader(str(root)), [short, valid], fe)

        self.assertEqual(windows.shape, (1, 6, len(SENSORS)))
        self.assertTrue(np.all(windows[:, :, 0] == 2.0))


class FeatureEngineerTests(unittest.TestCase):
    def test_fit_transform_and_transform_return_six_clipped_angle_features(self) -> None:
        """FeatureEngineer outputs six PCA angles clipped to [0, pi]."""
        rng = np.random.RandomState(123)
        fe = FeatureEngineer(n_components=6, window_size=4, stride=2)
        train_features = rng.normal(loc=0.0, scale=1.0, size=(12, 10))
        extreme_features = rng.normal(loc=1_000.0, scale=500.0, size=(7, 10))

        train_angles = fe.fit_transform(train_features)
        transformed_angles = fe.transform(extreme_features)

        self.assertEqual(train_angles.shape, (12, 6))
        self.assertEqual(transformed_angles.shape, (7, 6))
        self.assertTrue(np.all(train_angles >= 0.0))
        self.assertTrue(np.all(train_angles <= math.pi))
        self.assertTrue(np.all(transformed_angles >= 0.0))
        self.assertTrue(np.all(transformed_angles <= math.pi))


class ResultIntegrityTests(unittest.TestCase):
    def test_detects_duplicate_model_seed_fold_event_keys(self) -> None:
        """Duplicate result identity keys fail validation with a clear error."""
        from scripts.samarone_junior.result_integrity import validate_result_file

        with tempfile.TemporaryDirectory() as tmp:
            path = _write_jsonl(Path(tmp) / "metrics.jsonl", [_result_row(), _result_row()])

            with self.assertRaisesRegex(ValueError, "duplicate|Duplicate"):
                validate_result_file(path)

    def test_validates_required_schema_and_expected_row_count(self) -> None:
        """A complete one-row model/seed/fold/event grid passes validation."""
        from scripts.samarone_junior.result_integrity import validate_result_file

        with tempfile.TemporaryDirectory() as tmp:
            path = _write_jsonl(Path(tmp) / "metrics.jsonl", [_result_row()])

            report = validate_result_file(
                path,
                models=["IsolationForestDetector"],
                seeds=[42],
                n_folds=1,
                event_classes=[1],
            )

        self.assertIsNotNone(report)

    def test_missing_schema_key_fails_validation(self) -> None:
        """Rows missing a run_experiment metric key fail schema validation."""
        from scripts.samarone_junior.result_integrity import validate_result_file

        row = _result_row()
        row.pop("auc_pr")
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_jsonl(Path(tmp) / "metrics.jsonl", [row])

            with self.assertRaisesRegex(ValueError, "auc_pr|required|schema"):
                validate_result_file(path)

    def test_corrected_methodology_schema_can_be_required(self) -> None:
        """Promotion gates can require corrected split/label/diagnostic metadata."""
        from scripts.samarone_junior.result_integrity import validate_result_file

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            legacy = _write_jsonl(root / "legacy.jsonl", [_result_row()])
            corrected = _write_jsonl(
                root / "corrected.jsonl",
                [_result_row(**_corrected_methodology_fields())],
            )

            with self.assertRaisesRegex(ValueError, "threshold_source|diagnostics|missing"):
                validate_result_file(legacy, require_corrected_methodology=True)
            report = validate_result_file(
                corrected,
                models=["IsolationForestDetector"],
                seeds=[42],
                n_folds=1,
                event_classes=[1],
                require_corrected_methodology=True,
            )

        self.assertEqual(report["rows"], 1)

    def test_expected_row_count_mismatch_fails_validation(self) -> None:
        """Configured expected row grids must match the JSONL row count."""
        from scripts.samarone_junior.result_integrity import validate_result_file

        with tempfile.TemporaryDirectory() as tmp:
            path = _write_jsonl(Path(tmp) / "metrics.jsonl", [_result_row()])

            with self.assertRaisesRegex(ValueError, "row count|expected"):
                validate_result_file(
                    path,
                    models=["IsolationForestDetector"],
                    seeds=[42],
                    n_folds=2,
                    event_classes=[1],
                )

    def test_merge_result_files_validates_missing_inputs(self) -> None:
        """Merge validation fails before writing when a per-model file is missing."""
        from scripts.samarone_junior.result_integrity import merge_result_files

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, "missing input"):
                merge_result_files([root / "missing.jsonl"], root / "merged.jsonl")

    def test_merge_result_files_writes_deterministic_valid_output(self) -> None:
        """Valid per-model rows merge to a schema-checked JSONL output."""
        from scripts.samarone_junior.result_integrity import merge_result_files, validate_result_file

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = _write_jsonl(root / "a.jsonl", [_result_row(model="IsolationForestDetector")])
            second = _write_jsonl(
                root / "b.jsonl",
                [_result_row(model="MatchedAutoencoder", n_params=100)],
            )
            output = root / "merged.jsonl"

            summary = merge_result_files(
                [second, first],
                output,
                models=["IsolationForestDetector", "MatchedAutoencoder"],
                seeds=[42],
                n_folds=1,
                event_classes=[1],
            )
            validate_result_file(
                output,
                models=["IsolationForestDetector", "MatchedAutoencoder"],
                seeds=[42],
                n_folds=1,
                event_classes=[1],
            )

        self.assertEqual(summary["rows"], 2)


class TheoryMethodologyRegressionTests(unittest.TestCase):
    def test_fold_split_creates_disjoint_train_calibration_and_test_files(self) -> None:
        """Normal files are split into disjoint train/calibration/test partitions."""
        from scripts.samarone_junior.run_experiment import _make_fold_split

        files = [Path(f"normal_{idx}.parquet") for idx in range(10)]

        split = _make_fold_split(
            normal_files=files,
            fold_train_indices=np.arange(8),
            fold_test_indices=np.array([8, 9]),
            seed=42,
            fold=3,
            calibration_fraction=0.25,
        )

        train = set(split.train_normal_files)
        calibration = set(split.calibration_normal_files)
        test = set(split.test_normal_files)
        self.assertTrue(train)
        self.assertTrue(calibration)
        self.assertTrue(test)
        self.assertTrue(train.isdisjoint(calibration))
        self.assertTrue(train.isdisjoint(test))
        self.assertTrue(calibration.isdisjoint(test))

    def test_threshold_uses_only_calibration_scores_and_rejects_empty_input(self) -> None:
        """Extreme test-normal scores cannot influence a calibration-only threshold."""
        from qml.samarone_junior.evaluation import calibrate_threshold

        calibration_scores = np.array([0, 1, 2, 3, 4], dtype=float)
        test_normal_scores = np.array([1000, 2000, 3000], dtype=float)

        threshold = calibrate_threshold(calibration_scores, percentile=95.0)

        self.assertLess(threshold, test_normal_scores.min())
        with self.assertRaisesRegex(ValueError, "empty|calibration"):
            calibrate_threshold(np.array([], dtype=float), percentile=95.0)

    def test_window_label_majority_policy_marks_unlabeled_windows(self) -> None:
        """Window-level labels are explicit and do not assume whole-file positives."""
        from scripts.samarone_junior.run_experiment import WindowLabelPolicy, assign_window_label

        self.assertEqual(
            assign_window_label(np.array([2, 2, 2, 0]), 2, WindowLabelPolicy.MAJORITY),
            "anomaly",
        )
        self.assertEqual(
            assign_window_label(np.array([2, 0, 0, 0]), 2, WindowLabelPolicy.MAJORITY),
            "normal_or_other",
        )
        self.assertEqual(
            assign_window_label(
                np.array([np.nan, np.nan, 2, 2], dtype=float),
                2,
                WindowLabelPolicy.MAJORITY,
                min_labeled_fraction=0.80,
            ),
            "unlabeled",
        )

    def test_block_aware_significance_uses_seed_fold_blocks(self) -> None:
        """Primary paired tests aggregate event rows to seed-fold blocks."""
        from qml.samarone_junior.evaluation.metrics import compute_block_significance

        rows: list[dict[str, object]] = []
        for seed in (1, 2):
            for fold in (0, 1):
                for event_class in (1, 2, 3):
                    rows.append(_result_row(model="ModelA", seed=seed, fold=fold, event_class=event_class, f1=0.8))
                    rows.append(_result_row(model="ModelB", seed=seed, fold=fold, event_class=event_class, f1=0.6))
        result = compute_block_significance(pd.DataFrame(rows), "ModelA", "ModelB", metric="f1")

        self.assertEqual(result["blocking"], "seed_fold")
        self.assertEqual(result["n_pairs"], 4)
        self.assertAlmostEqual(result["mean_diff"], 0.2)

    def test_trainable_parameter_counts_match_reported_low_capacity_models(self) -> None:
        """Capacity claims come from trainable model parameters, not hand math."""
        from qml.samarone_junior.models import (
            FullAutoencoder,
            LSTMAutoencoder,
            MatchedAutoencoder,
            count_trainable_parameters,
        )

        self.assertEqual(count_trainable_parameters(MatchedAutoencoder(seed=0)), 100)
        self.assertEqual(count_trainable_parameters(FullAutoencoder(seed=0)), 3006)
        self.assertEqual(count_trainable_parameters(LSTMAutoencoder(seed=0)), 3029)

    def test_qae_reconstruction_observes_all_reconstructed_wires(self) -> None:
        """Full reconstruction scores compare six observables with six targets."""
        from qml.samarone_junior.models import QAEReconstruction

        model = QAEReconstruction(seed=0)
        x = np.random.default_rng(0).uniform(0.0, np.pi, size=(2, 6))

        observables = model.reconstruct_observables(x)
        scores = model.anomaly_scores(x)

        self.assertEqual(observables.shape, (2, 6))
        self.assertTrue(np.isfinite(observables).all())
        self.assertEqual(scores.shape, (2,))
        self.assertTrue(np.isfinite(scores).all())


class ThresholdHistogramTests(unittest.TestCase):
    def test_threshold_histogram_data_deduplicates_event_rows_per_fold_threshold(self) -> None:
        """One threshold shared by nine event rows is counted once."""
        from qml.samarone_junior.visualization.figures import _threshold_histogram_data

        rows = [
            {
                "model": "IsolationForestDetector",
                "seed": 42,
                "fold": 0,
                "event_class": event_class,
                "threshold_value": 0.75,
            }
            for event_class in range(1, 10)
        ]
        rows.append(
            {
                "model": "IsolationForestDetector",
                "seed": 42,
                "fold": 1,
                "event_class": 1,
                "threshold_value": 0.80,
            }
        )

        histogram_df = _threshold_histogram_data(pd.DataFrame(rows))

        self.assertEqual(len(histogram_df), 2)
        self.assertCountEqual(histogram_df["threshold_value"].tolist(), [0.75, 0.80])


class FigureSmokeTests(unittest.TestCase):
    def test_publication_figure_functions_create_nonempty_pdfs(self) -> None:
        """All figure functions render against synthetic or data-free inputs."""
        from qml.samarone_junior.visualization import figures

        rows: list[dict[str, object]] = []
        for model in ("IsolationForestDetector", "MatchedAutoencoder"):
            for fold in (0, 1):
                for event_class in (1, 2):
                    rows.append(
                        {
                            "model": model,
                            "seed": 42,
                            "fold": fold,
                            "event_class": event_class,
                            "event_name": figures.EVENT_NAMES[event_class],
                            "threshold_value": 0.5 + 0.1 * fold,
                            "f1": 0.7 + 0.05 * fold,
                            "auc_roc": 0.8,
                            "auc_pr": 0.75,
                        }
                    )
        df = pd.DataFrame(rows)

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            generated = [
                figures.fig_qae_circuit(out / "fig_qae_circuit.pdf"),
                figures.fig_preprocessing_pipeline(out / "fig_preprocessing_pipeline.pdf"),
                figures.fig_roc_curves(df, out / "fig_roc_curves.pdf"),
                figures.fig_pr_curves(df, out / "fig_pr_curves.pdf"),
                figures.fig_threshold_histograms(df, out / "fig_threshold_histograms.pdf"),
                figures.fig_f1_boxplots(df, out / "fig_f1_boxplots.pdf"),
                figures.fig_param_vs_f1(df, out / "fig_param_vs_f1.pdf"),
                figures.fig_event_heatmap(df, out / "fig_event_heatmap.pdf"),
            ]

            for pdf in generated:
                self.assertTrue(pdf.is_file(), f"Missing figure output: {pdf}")
                self.assertGreater(pdf.stat().st_size, 0, f"Empty figure output: {pdf}")


class DatasetLayoutTests(unittest.TestCase):
    def test_documented_dataset_root_layout_lists_class_parquet_files(self) -> None:
        """A data root containing direct 0..9 class directories is supported."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for class_label in range(10):
                (root / str(class_label)).mkdir()
            target = _write_parquet(root / "0" / "WELL-00001.parquet", _sensor_frame(0.0, 2))

            files = ThreeWLoader(str(root)).list_instances(0)

        self.assertEqual(files, [target])

    def test_data_health_check_reports_class_dirs_and_zero_byte_files(self) -> None:
        """Dataset health check catches zero-byte Parquet placeholders."""
        from scripts.samarone_junior.result_integrity import validate_dataset_layout

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for class_label in range(10):
                (root / str(class_label)).mkdir()
            (root / "0" / "empty.parquet").touch()

            with self.assertRaisesRegex(ValueError, "zero-byte"):
                validate_dataset_layout(root)

    def test_unsupported_dataset_layout_fails_with_class_directory_message(self) -> None:
        """Pointing at a parent of dataset/0..9 fails with an actionable message."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "dataset" / "0").mkdir(parents=True)

            with self.assertRaisesRegex(FileNotFoundError, "Class directory not found"):
                ThreeWLoader(str(root)).list_instances(0)


class OutputPathTests(unittest.TestCase):
    def test_default_seed_grid_matches_corrected_full_rerun_contract(self) -> None:
        """Default CLI seeds align with the 3150-row corrected baseline grid."""
        from scripts.samarone_junior.result_integrity import expected_row_count
        from scripts.samarone_junior.run_experiment import MODEL_REGISTRY, parse_args

        args = parse_args([])

        self.assertEqual(len(args.seeds), 10)
        self.assertEqual(
            expected_row_count(
                list(MODEL_REGISTRY),
                args.seeds,
                args.n_folds,
            ),
            3150,
        )

    def test_resolve_output_path_rejects_existing_metrics_without_overwrite(self) -> None:
        """Existing metrics.jsonl cannot be overwritten by default."""
        from scripts.samarone_junior.run_experiment import _resolve_output_path

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            existing = output_dir / "metrics.jsonl"
            existing.write_text("stale\n", encoding="utf-8")
            args = SimpleNamespace(output_dir=str(output_dir), output_file=None, overwrite=False)

            with self.assertRaisesRegex(FileExistsError, "overwrite|exists"):
                _resolve_output_path(args)

    def test_resolve_output_path_allows_existing_metrics_with_overwrite(self) -> None:
        """Explicit overwrite resolves to the existing metrics path."""
        from scripts.samarone_junior.run_experiment import _resolve_output_path

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            existing = output_dir / "metrics.jsonl"
            existing.write_text("stale\n", encoding="utf-8")
            args = SimpleNamespace(output_dir=str(output_dir), output_file=None, overwrite=True)

            resolved = _resolve_output_path(args)

        self.assertEqual(resolved, existing)


class ExperimentSubsetTests(unittest.TestCase):
    def test_cli_accepts_fold_and_event_class_subsets_for_array_retries(self) -> None:
        """Retry jobs can target one seed/fold and only missing event classes."""
        from scripts.samarone_junior.run_experiment import parse_args

        args = parse_args([
            "--models",
            "QAETrashFidelity",
            "--seeds",
            "2026",
            "--folds",
            "2",
            "--event-classes",
            "5",
            "6",
            "7",
            "8",
            "9",
        ])

        self.assertEqual(args.models, ["QAETrashFidelity"])
        self.assertEqual(args.seeds, [2026])
        self.assertEqual(args.folds, [2])
        self.assertEqual(args.event_classes, [5, 6, 7, 8, 9])

    def test_subset_resolution_deduplicates_to_prevent_duplicate_result_keys(self) -> None:
        """Duplicate CLI selectors are collapsed before writing JSONL rows."""
        from scripts.samarone_junior.run_experiment import _resolve_int_selection

        selected = _resolve_int_selection([2, 2, 4, 2], range(5), range(5), "fold")

        self.assertEqual(selected, [2, 4])

    def test_subset_resolution_rejects_invalid_folds_and_event_classes(self) -> None:
        """Invalid retry selectors fail before expensive model training starts."""
        from scripts.samarone_junior.run_experiment import _resolve_int_selection

        with self.assertRaisesRegex(ValueError, "fold"):
            _resolve_int_selection([5], range(5), range(5), "fold")

        with self.assertRaisesRegex(ValueError, "event_class"):
            _resolve_int_selection([0], list(range(1, 10)), list(range(1, 10)), "event_class")


class ResultSummaryTests(unittest.TestCase):
    def test_result_summary_formatters_are_stable_for_reports(self) -> None:
        """Summary helper formatting is deterministic for paper-ready values."""
        from scripts.samarone_junior.summarize_results import fmt_float, fmt_p_value, fmt_p_value_tex

        self.assertEqual(fmt_float(0.123456), "0.123")
        self.assertEqual(fmt_p_value(3.313963614400204e-07), "3.31 × 10^-7")
        self.assertEqual(fmt_p_value_tex(3.313963614400204e-07), r"3.31 \times 10^{-7}")

    def test_result_summary_latex_table_renders_model_rows(self) -> None:
        """Generated LaTeX snippets include ranked models and escaped row endings."""
        from scripts.samarone_junior.summarize_results import SummaryTables, render_latex_table

        summary = pd.DataFrame(
            [
                {
                    "rank": 1,
                    "model": "QAETrashFidelity",
                    "n_params": 48,
                    "f1": 0.6,
                    "ci_lower": 0.5,
                    "ci_upper": 0.7,
                    "auc_roc": 0.8,
                },
                {
                    "rank": 2,
                    "model": "MatchedAutoencoder",
                    "n_params": 100,
                    "f1": 0.4,
                    "ci_lower": 0.3,
                    "ci_upper": 0.5,
                    "auc_roc": 0.7,
                },
            ]
        )
        tables = SummaryTables(
            validation={"rows": 2, "unique_keys": 2, "models": 2, "seeds": 1, "folds": 1, "event_classes": 1},
            model_summary=summary,
            per_event_f1=pd.DataFrame(),
            qae_trash_comparisons=pd.DataFrame(),
            qae_recon_comparisons=pd.DataFrame(),
        )

        tex = render_latex_table(tables)

        self.assertIn(r"\textbf{QAETrashFidelity}", tex)
        self.assertIn("MatchedAutoencoder", tex)
        self.assertIn(r"$0,600$", tex)
        self.assertIn(r"\\", tex)

    def test_result_summary_holm_adjustment_marks_familywise_rejections(self) -> None:
        """Summary comparisons carry Holm-adjusted p-values for paper claims."""
        from scripts.samarone_junior.summarize_results import _with_holm_adjustment

        comparisons = pd.DataFrame({"p_value": [0.001, 0.02, 0.50]})

        adjusted = _with_holm_adjustment(comparisons)

        self.assertIn("p_holm", adjusted)
        self.assertIn("holm_reject_0.05", adjusted)
        self.assertTrue(bool(adjusted.loc[0, "holm_reject_0.05"]))

    def test_result_summary_provenance_and_qae_objective_text_follow_metrics(self) -> None:
        """Generated narrative uses supplied provenance and does not hard-code stale QAE claims."""
        from scripts.samarone_junior.summarize_results import ResultProvenance, SummaryTables, render_markdown

        summary = pd.DataFrame(
            [
                {
                    "rank": 1,
                    "model": "LSTMAutoencoder",
                    "f1": 0.8722,
                    "auc_roc": 0.9892,
                    "auc_pr": 0.9520,
                    "precision": 0.88,
                    "recall": 0.87,
                    "ci_lower": 0.84,
                    "ci_upper": 0.90,
                    "ci_width": 0.06,
                    "n_params": 3029,
                },
                {
                    "rank": 2,
                    "model": "FullAutoencoder",
                    "f1": 0.8560,
                    "auc_roc": 0.9800,
                    "auc_pr": 0.8756,
                    "precision": 0.86,
                    "recall": 0.85,
                    "ci_lower": 0.82,
                    "ci_upper": 0.89,
                    "ci_width": 0.07,
                    "n_params": 3006,
                },
                {
                    "rank": 3,
                    "model": "IsolationForestDetector",
                    "f1": 0.7761,
                    "auc_roc": 0.9454,
                    "auc_pr": 0.8563,
                    "precision": 0.78,
                    "recall": 0.77,
                    "ci_lower": 0.72,
                    "ci_upper": 0.82,
                    "ci_width": 0.10,
                    "n_params": None,
                },
                {
                    "rank": 4,
                    "model": "OneClassSVMDetector",
                    "f1": 0.6970,
                    "auc_roc": 0.7523,
                    "auc_pr": 0.8126,
                    "precision": 0.70,
                    "recall": 0.69,
                    "ci_lower": 0.64,
                    "ci_upper": 0.75,
                    "ci_width": 0.11,
                    "n_params": None,
                },
                {
                    "rank": 5,
                    "model": "QAEReconstruction",
                    "f1": 0.6263,
                    "auc_roc": 0.9045,
                    "auc_pr": 0.8352,
                    "precision": 0.63,
                    "recall": 0.62,
                    "ci_lower": 0.57,
                    "ci_upper": 0.68,
                    "ci_width": 0.11,
                    "n_params": 54,
                },
                {
                    "rank": 6,
                    "model": "QAETrashFidelity",
                    "f1": 0.6032,
                    "auc_roc": 0.8890,
                    "auc_pr": 0.8248,
                    "precision": 0.61,
                    "recall": 0.60,
                    "ci_lower": 0.55,
                    "ci_upper": 0.66,
                    "ci_width": 0.11,
                    "n_params": 48,
                },
                {
                    "rank": 7,
                    "model": "MatchedAutoencoder",
                    "f1": 0.4926,
                    "auc_roc": 0.7810,
                    "auc_pr": 0.7874,
                    "precision": 0.50,
                    "recall": 0.49,
                    "ci_lower": 0.43,
                    "ci_upper": 0.55,
                    "ci_width": 0.12,
                    "n_params": 100,
                },
            ]
        )
        model_columns = summary["model"].tolist()
        per_event = pd.DataFrame(
            [
                {
                    "event_class": 6,
                    "event_name": "Rapid Productivity Loss",
                    **{model: float(summary.loc[summary["model"] == model, "f1"].iloc[0]) for model in model_columns},
                }
            ]
        )
        trash = pd.DataFrame(
            [
                {
                    "baseline": "MatchedAutoencoder",
                    "comparison": "QAETrashFidelity vs. MatchedAutoencoder",
                    "mean_diff": 0.111,
                    "n_blocks": 50,
                    "p_value": 0.001,
                    "p_holm": 0.00541,
                    "holm_reject_0.05": True,
                },
                {
                    "baseline": "OneClassSVMDetector",
                    "comparison": "QAETrashFidelity vs. OneClassSVMDetector",
                    "mean_diff": -0.094,
                    "n_blocks": 50,
                    "p_value": 0.02,
                    "p_holm": 0.08,
                    "holm_reject_0.05": False,
                },
                {
                    "baseline": "IsolationForestDetector",
                    "comparison": "QAETrashFidelity vs. IsolationForestDetector",
                    "mean_diff": -0.173,
                    "n_blocks": 50,
                    "p_value": 0.001,
                    "p_holm": 0.004,
                    "holm_reject_0.05": True,
                },
                {
                    "baseline": "FullAutoencoder",
                    "comparison": "QAETrashFidelity vs. FullAutoencoder",
                    "mean_diff": -0.253,
                    "n_blocks": 50,
                    "p_value": 0.0001,
                    "p_holm": 0.0004,
                    "holm_reject_0.05": True,
                },
                {
                    "baseline": "LSTMAutoencoder",
                    "comparison": "QAETrashFidelity vs. LSTMAutoencoder",
                    "mean_diff": -0.269,
                    "n_blocks": 50,
                    "p_value": 0.00001,
                    "p_holm": 0.00005,
                    "holm_reject_0.05": True,
                },
                {
                    "baseline": "QAEReconstruction",
                    "comparison": "QAETrashFidelity vs. QAEReconstruction",
                    "mean_diff": -0.023,
                    "n_blocks": 50,
                    "p_value": 0.097,
                    "p_holm": 0.194,
                    "holm_reject_0.05": False,
                },
            ]
        )
        recon = pd.DataFrame(
            [
                {
                    "baseline": "LSTMAutoencoder",
                    "comparison": "QAEReconstruction vs. LSTMAutoencoder",
                    "mean_diff": -0.246,
                    "n_blocks": 50,
                    "p_value": 0.00001,
                    "p_holm": 0.00005,
                    "holm_reject_0.05": True,
                },
                {
                    "baseline": "FullAutoencoder",
                    "comparison": "QAEReconstruction vs. FullAutoencoder",
                    "mean_diff": -0.230,
                    "n_blocks": 50,
                    "p_value": 0.00002,
                    "p_holm": 0.00008,
                    "holm_reject_0.05": True,
                },
                {
                    "baseline": "MatchedAutoencoder",
                    "comparison": "QAEReconstruction vs. MatchedAutoencoder",
                    "mean_diff": 0.134,
                    "n_blocks": 50,
                    "p_value": 0.0005,
                    "p_holm": 0.00285,
                    "holm_reject_0.05": True,
                },
                {
                    "baseline": "QAETrashFidelity",
                    "comparison": "QAEReconstruction vs. QAETrashFidelity",
                    "mean_diff": 0.023,
                    "n_blocks": 50,
                    "p_value": 0.097,
                    "p_holm": 0.194,
                    "holm_reject_0.05": False,
                },
            ]
        )
        tables = SummaryTables(
            validation={"rows": 3150, "unique_keys": 3150, "models": 7, "seeds": 10, "folds": 5, "event_classes": 9},
            model_summary=summary,
            per_event_f1=per_event,
            qae_trash_comparisons=trash,
            qae_recon_comparisons=recon,
        )

        report = render_markdown(
            tables,
            results_path=Path("data/samarone_junior/results/metrics.jsonl"),
            provenance=ResultProvenance(
                run_id="theoryfix_20260511T044609Z",
                validation_date="2026-05-13",
                remote_path="/home/samarone.lima/qml_project/results/theoryfix_20260511T044609Z/",
                hpc_jobs="28656, 28657, 28658",
                recovery_note="QAE rowfill recovered missing rows.",
                checksum="ce4ab88b",
            ),
        )

        self.assertIn("theoryfix_20260511T044609Z", report)
        self.assertIn("28656, 28657, 28658", report)
        self.assertIn("QAEReconstruction has the higher mean F1", report)
        self.assertIn("not statistically significant", report)
        self.assertNotIn("corrected_20260502T230443Z", report)
        self.assertNotIn("QAEReconstruction ranks last", report)
        self.assertNotIn("QAETrashFidelity outperforms QAEReconstruction", report)


class ImportSmokeTests(unittest.TestCase):
    def test_key_public_apis_import_without_side_effects(self) -> None:
        """Critical public APIs remain importable from package and script modules."""
        from qml.samarone_junior.evaluation import (
            calibrate_threshold,
            compute_binary_metrics,
            compute_block_significance,
        )
        from qml.samarone_junior.models import (
            FullAutoencoder,
            IsolationForestDetector,
            LSTMAutoencoder,
            MatchedAutoencoder,
            OneClassSVMDetector,
            QAEReconstruction,
            QAETrashFidelity,
            count_trainable_parameters,
        )
        from qml.samarone_junior.visualization import figures
        from scripts.samarone_junior import run_experiment

        self.assertTrue(callable(calibrate_threshold))
        self.assertTrue(callable(compute_binary_metrics))
        self.assertTrue(callable(compute_block_significance))
        self.assertTrue(callable(count_trainable_parameters))
        self.assertTrue(callable(figures.fig_threshold_histograms))
        self.assertIn("IsolationForestDetector", run_experiment.MODEL_REGISTRY)
        for cls in (
            QAETrashFidelity,
            QAEReconstruction,
            MatchedAutoencoder,
            FullAutoencoder,
            LSTMAutoencoder,
            IsolationForestDetector,
            OneClassSVMDetector,
        ):
            self.assertTrue(callable(cls))


def main() -> None:
    suite = unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
