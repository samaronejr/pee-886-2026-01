"""Standalone import smoke test for the qml package and key public APIs."""

from __future__ import annotations

import sys


def _import(name: str):
    print(f"Attempting to import {name!r}...")
    module = __import__(name, fromlist=["*"])
    print(f"Imported {name!r} successfully.")
    return module


def test_imports() -> None:
    try:
        _import("qml")

        loaders = _import("qml.samarone_junior.loaders")
        assert hasattr(loaders, "ThreeWLoader")
        assert hasattr(loaders, "FeatureEngineer")

        models = _import("qml.samarone_junior.models")
        for name in (
            "QAETrashFidelity",
            "QAEReconstruction",
            "MatchedAutoencoder",
            "FullAutoencoder",
            "LSTMAutoencoder",
            "IsolationForestDetector",
            "OneClassSVMDetector",
        ):
            assert hasattr(models, name), f"Missing public model API: {name}"

        evaluation = _import("qml.samarone_junior.evaluation")
        assert hasattr(evaluation, "calibrate_threshold")
        assert hasattr(evaluation, "compute_binary_metrics")

        figures = _import("qml.samarone_junior.visualization.figures")
        assert hasattr(figures, "fig_threshold_histograms")

        runner = _import("scripts.samarone_junior.run_experiment")
        assert hasattr(runner, "MODEL_REGISTRY")
        assert hasattr(runner, "_load_files_and_extract_windows")

        summarizer = _import("scripts.samarone_junior.summarize_results")
        assert hasattr(summarizer, "compute_summary_tables")
        assert hasattr(summarizer, "render_markdown")

    except ImportError as exc:
        print(f"Import Error: {exc}")
        sys.exit(1)
    except AttributeError as exc:
        print(f"Attribute Error (possible missing __all__): {exc}")
        sys.exit(1)
    except AssertionError as exc:
        print(f"Public API assertion failed: {exc}")
        sys.exit(1)
    except Exception as exc:
        print(f"Unexpected error during import: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    test_imports()
