"""Evaluation subpackage — threshold calibration and binary classification metrics."""

from qml.samarone_junior.evaluation.metrics import (
    calibrate_threshold,
    compute_block_confidence_intervals,
    compute_block_significance,
    compute_binary_metrics,
    compute_confidence_intervals,
    compute_significance,
    holm_adjust,
)

__all__ = [
    "calibrate_threshold",
    "compute_block_confidence_intervals",
    "compute_block_significance",
    "compute_binary_metrics",
    "compute_confidence_intervals",
    "compute_significance",
    "holm_adjust",
]
