#!/usr/bin/env python3
"""Generate all 8 publication figures from experiment results.

Usage (from pee-886-2026-01/)::

    python scripts/samarone_junior/generate_figures.py \
        --results data/samarone_junior/results/metrics.jsonl \
        --output-dir data/samarone_junior/figures

Reads the JSONL metrics file and produces 8 PDF figures in the output
directory. Creates the output directory if it does not exist.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

# Ensure project root is on sys.path so we can import the qml package
_project_root = Path(__file__).resolve().parents[2]  # scripts/samarone_junior -> pee-886-2026-01
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from qml.samarone_junior.visualization.figures import (  # noqa: E402
    fig_event_heatmap,
    fig_f1_boxplots,
    fig_param_vs_f1,
    fig_pr_curves,
    fig_preprocessing_pipeline,
    fig_qae_circuit,
    fig_roc_curves,
    fig_threshold_histograms,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_results(path: Path) -> pd.DataFrame:
    """Load JSONL results into a DataFrame with NaN handling."""
    df = pd.read_json(path, lines=True)
    logger.info("Loaded %d rows from %s (%d unique models)", len(df), path, df["model"].nunique())
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate publication figures from JSONL results.")
    parser.add_argument("--results", required=True, type=Path, help="Path to metrics.jsonl")
    parser.add_argument("--output-dir", required=True, type=Path, help="Output directory for PDFs")
    args = parser.parse_args()

    if not args.results.exists():
        logger.error("Results file not found: %s", args.results)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = load_results(args.results)

    t0 = time.perf_counter()

    # -- Figures that don't need the DataFrame --
    fig_qae_circuit(args.output_dir / "fig_qae_circuit.pdf")
    fig_preprocessing_pipeline(args.output_dir / "fig_preprocessing_pipeline.pdf")

    # -- Figures from experiment metrics --
    fig_roc_curves(df, args.output_dir / "fig_roc_curves.pdf")
    fig_pr_curves(df, args.output_dir / "fig_pr_curves.pdf")
    fig_threshold_histograms(df, args.output_dir / "fig_threshold_histograms.pdf")
    fig_f1_boxplots(df, args.output_dir / "fig_f1_boxplots.pdf")
    fig_param_vs_f1(df, args.output_dir / "fig_param_vs_f1.pdf")
    fig_event_heatmap(df, args.output_dir / "fig_event_heatmap.pdf")

    elapsed = time.perf_counter() - t0
    n_pdfs = len(list(args.output_dir.glob("*.pdf")))
    logger.info("Generated %d PDF figures in %.1fs → %s", n_pdfs, elapsed, args.output_dir)


if __name__ == "__main__":
    main()
