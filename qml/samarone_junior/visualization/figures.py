"""Publication-quality figure generation for QML anomaly-detection paper.

Eight figure functions, each producing a single PDF. All functions that
operate on experiment data take a pandas DataFrame loaded from the JSONL
results file. PennyLane is imported lazily — only ``fig_qae_circuit``
needs it, so the module remains importable without PennyLane installed.

Usage (CLI)::

    python scripts/samarone_junior/generate_figures.py \
        --results data/samarone_junior/results/metrics.jsonl \
        --output-dir data/samarone_junior/figures

Reference: R018 — parameter-count vs. F1 analysis.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless rendering before any pyplot import

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

PARAM_COUNTS: dict[str, int | None] = {
    "QAETrashFidelity": 48,
    "QAEReconstruction": 48,
    "MatchedAutoencoder": 100,
    "FullAutoencoder": 3006,
    "LSTMAutoencoder": 3029,
    "IsolationForestDetector": None,
    "OneClassSVMDetector": None,
}

EVENT_NAMES: dict[int, str] = {
    1: "Abrupt BSW Increase",
    2: "Spurious DHSV Closure",
    3: "Severe Slugging",
    4: "Flow Instability",
    5: "Rapid Productivity Loss",
    6: "Quick Restriction in PCK",
    7: "Scaling in PCK",
    8: "Hydrate in Production Line",
    9: "Hydrate in Service Line",
}

# Short labels for axes
_MODEL_SHORT: dict[str, str] = {
    "QAETrashFidelity": "QAE-Trash",
    "QAEReconstruction": "QAE-Recon",
    "MatchedAutoencoder": "Matched-AE",
    "FullAutoencoder": "Full-AE",
    "LSTMAutoencoder": "LSTM-AE",
    "IsolationForestDetector": "IF",
    "OneClassSVMDetector": "OC-SVM",
}

# Color palette keyed by model short name
_PALETTE = sns.color_palette("husl", n_colors=len(_MODEL_SHORT))
_MODEL_COLORS: dict[str, tuple] = {
    short: _PALETTE[i] for i, short in enumerate(_MODEL_SHORT.values())
}


def _short(model: str) -> str:
    """Return short display label for a model name."""
    return _MODEL_SHORT.get(model, model)


# ------------------------------------------------------------------
# Publication defaults
# ------------------------------------------------------------------

def _apply_style() -> None:
    """Apply consistent publication styling."""
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.family": "serif",
        "pdf.fonttype": 42,       # TrueType fonts in PDF (LaTeX-friendly)
        "ps.fonttype": 42,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
    })


# ------------------------------------------------------------------
# Figure 1 — QAE circuit diagram
# ------------------------------------------------------------------

def fig_qae_circuit(output_path: str | Path) -> Path:
    """Draw the QAE trash-fidelity circuit and save as PDF.

    Requires PennyLane (imported lazily).
    """
    output_path = Path(output_path)
    _apply_style()

    try:
        import pennylane as qml
        from pennylane import numpy as pnp
    except ImportError as exc:
        raise ImportError(
            "PennyLane is required for fig_qae_circuit. "
            "Install with: pip install pennylane"
        ) from exc

    # Build a lightweight QAE instance for drawing
    from qml.samarone_junior.models.quantum_autoencoder import QAETrashFidelity

    model = QAETrashFidelity(n_qubits=6, n_layers=4, n_trash=2)
    dummy_features = pnp.zeros(6)

    fig, ax = qml.draw_mpl(model._circuit)(dummy_features, model.weights)
    ax.set_title("QAE Trash-Fidelity Circuit (6 qubits, 4 layers)", fontsize=11)
    fig.tight_layout()
    fig.savefig(str(output_path), format="pdf", bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path)
    return output_path


# ------------------------------------------------------------------
# Figure 2 — Preprocessing pipeline schematic
# ------------------------------------------------------------------

def fig_preprocessing_pipeline(output_path: str | Path) -> Path:
    """Draw the preprocessing pipeline schematic as a block diagram."""
    output_path = Path(output_path)
    _apply_style()

    fig, ax = plt.subplots(figsize=(12, 2.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2)
    ax.axis("off")

    stages = [
        ("Raw\n5-sensor\nSignals", 0.5),
        ("Sliding\nWindows\n(W=128)", 2.5),
        ("4 Stats/sensor\n→ 20-D\nFeatures", 4.5),
        ("PCA\n→ 6-D", 6.5),
        ("[0, π]\nScaling", 8.5),
    ]

    box_w, box_h = 1.6, 1.4
    y_center = 1.0

    for label, x_center in stages:
        bbox = mpatches.FancyBboxPatch(
            (x_center - box_w / 2, y_center - box_h / 2),
            box_w,
            box_h,
            boxstyle="round,pad=0.1",
            facecolor="#4C72B0",
            edgecolor="#2b4570",
            linewidth=1.5,
            alpha=0.85,
        )
        ax.add_patch(bbox)
        ax.text(
            x_center, y_center, label,
            ha="center", va="center", fontsize=9,
            color="white", fontweight="bold",
        )

    # Arrows between boxes
    for i in range(len(stages) - 1):
        x_start = stages[i][1] + box_w / 2 + 0.05
        x_end = stages[i + 1][1] - box_w / 2 - 0.05
        ax.annotate(
            "",
            xy=(x_end, y_center),
            xytext=(x_start, y_center),
            arrowprops=dict(
                arrowstyle="-|>",
                color="#333333",
                lw=2,
                connectionstyle="arc3,rad=0",
            ),
        )

    ax.set_title("Data Preprocessing Pipeline", fontsize=12, pad=10)
    fig.tight_layout()
    fig.savefig(str(output_path), format="pdf", bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path)
    return output_path


# ------------------------------------------------------------------
# Figure 3 — AUC-ROC comparison (bar chart)
# ------------------------------------------------------------------

def fig_roc_curves(df: pd.DataFrame, output_path: str | Path) -> Path:
    """Bar chart of mean AUC-ROC per model, grouped by event class."""
    output_path = Path(output_path)
    _apply_style()

    agg = (
        df.groupby(["model", "event_class", "event_name"])["auc_roc"]
        .mean()
        .reset_index()
    )
    agg["model_short"] = agg["model"].map(_short)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        data=agg,
        x="event_class",
        y="auc_roc",
        hue="model_short",
        ax=ax,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_xlabel("Event Class")
    ax.set_ylabel("Mean AUC-ROC")
    ax.set_title("AUC-ROC by Model and Event Class")
    ax.set_ylim(0, 1.05)
    ax.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(str(output_path), format="pdf", bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path)
    return output_path


# ------------------------------------------------------------------
# Figure 4 — AUC-PR comparison (bar chart)
# ------------------------------------------------------------------

def fig_pr_curves(df: pd.DataFrame, output_path: str | Path) -> Path:
    """Bar chart of mean AUC-PR per model, grouped by event class."""
    output_path = Path(output_path)
    _apply_style()

    agg = (
        df.groupby(["model", "event_class", "event_name"])["auc_pr"]
        .mean()
        .reset_index()
    )
    agg["model_short"] = agg["model"].map(_short)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        data=agg,
        x="event_class",
        y="auc_pr",
        hue="model_short",
        ax=ax,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_xlabel("Event Class")
    ax.set_ylabel("Mean AUC-PR")
    ax.set_title("AUC-PR by Model and Event Class")
    ax.set_ylim(0, 1.05)
    ax.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(str(output_path), format="pdf", bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path)
    return output_path


# ------------------------------------------------------------------
# Figure 5 — Threshold value distributions
# ------------------------------------------------------------------

def _threshold_histogram_data(df: pd.DataFrame) -> pd.DataFrame:
    """Return one threshold row per model/seed/fold/value combination.

    The experiment repeats the same calibrated threshold across the 9 anomaly
    event rows for each ``(model, seed, fold)``.  Histograms should visualize
    calibrated thresholds, not multiply counts by the number of event classes.
    """
    required = {"model", "threshold_value"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"Missing required threshold columns: {missing}")

    dedupe_cols = [
        col for col in ("model", "seed", "fold", "threshold_value")
        if col in df.columns
    ]
    return (
        df.dropna(subset=["threshold_value"])
        .drop_duplicates(subset=dedupe_cols)
        .copy()
    )


def fig_threshold_histograms(df: pd.DataFrame, output_path: str | Path) -> Path:
    """Histogram of threshold_value across folds/seeds, per model."""
    output_path = Path(output_path)
    _apply_style()

    df_plot = _threshold_histogram_data(df)
    models = df_plot["model"].unique()
    n_models = len(models)

    fig, axes = plt.subplots(1, n_models, figsize=(4 * n_models, 4), sharey=True)
    if n_models == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        subset = df_plot[df_plot["model"] == model]
        ax.hist(
            subset["threshold_value"],
            bins=20,
            color=_MODEL_COLORS.get(_short(model), "#4C72B0"),
            edgecolor="white",
            alpha=0.8,
        )
        ax.set_title(_short(model), fontsize=10)
        ax.set_xlabel("Threshold Value")

    axes[0].set_ylabel("Count")
    fig.suptitle("Threshold Value Distribution by Model", fontsize=12)
    fig.tight_layout()
    fig.savefig(str(output_path), format="pdf", bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path)
    return output_path


# ------------------------------------------------------------------
# Figure 6 — F1 boxplots
# ------------------------------------------------------------------

def fig_f1_boxplots(df: pd.DataFrame, output_path: str | Path) -> Path:
    """Boxplots of F1 across folds/seeds, grouped by model and event class."""
    output_path = Path(output_path)
    _apply_style()

    df_plot = df.copy()
    df_plot["model_short"] = df_plot["model"].map(_short)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(
        data=df_plot,
        x="model_short",
        y="f1",
        hue="event_name",
        ax=ax,
        linewidth=0.8,
        fliersize=3,
    )
    ax.set_xlabel("Model")
    ax.set_ylabel("F1 Score")
    ax.set_title("F1 Score Distribution by Model and Event Type")
    ax.set_ylim(-0.05, 1.1)
    ax.legend(
        title="Event", bbox_to_anchor=(1.02, 1), loc="upper left",
        fontsize=7, title_fontsize=8,
    )
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(str(output_path), format="pdf", bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path)
    return output_path


# ------------------------------------------------------------------
# Figure 7 — Parameter count vs. F1 scatter (R018)
# ------------------------------------------------------------------

def fig_param_vs_f1(df: pd.DataFrame, output_path: str | Path) -> Path:
    """Scatter plot: x = n_params (log), y = mean F1 per model.

    Non-parametric models (n_params=NaN) are placed at x=1 and annotated.
    """
    output_path = Path(output_path)
    _apply_style()

    # Mean F1 per model across all folds/seeds/events. Prefer generated
    # result-schema counts over the legacy fallback constants.
    agg = df.groupby("model")["f1"].mean().reset_index()
    if "n_params" in df.columns:
        observed_counts = (
            df.dropna(subset=["n_params"])
            .groupby("model")["n_params"]
            .first()
            .to_dict()
        )
    else:
        observed_counts = {}
    agg["n_params"] = agg["model"].map(lambda model: observed_counts.get(model, PARAM_COUNTS.get(model)))
    agg["model_short"] = agg["model"].map(_short)

    # Separate parametric and non-parametric
    param_mask = agg["n_params"].notna()
    parametric = agg[param_mask].copy()
    nonparam = agg[~param_mask].copy()
    nonparam["n_params"] = 1  # place at x=1 for visibility

    fig, ax = plt.subplots(figsize=(7, 5))

    # Parametric models
    if not parametric.empty:
        ax.scatter(
            parametric["n_params"], parametric["f1"],
            s=100, zorder=5, edgecolor="black", linewidth=0.8,
            label="Parametric",
        )
        for _, row in parametric.iterrows():
            ax.annotate(
                row["model_short"],
                (row["n_params"], row["f1"]),
                textcoords="offset points", xytext=(8, 4),
                fontsize=8,
            )

    # Non-parametric models at x=1
    if not nonparam.empty:
        ax.scatter(
            nonparam["n_params"], nonparam["f1"],
            s=100, marker="D", zorder=5, edgecolor="black", linewidth=0.8,
            color="orange", label="Non-parametric (at x=1)",
        )
        for _, row in nonparam.iterrows():
            ax.annotate(
                row["model_short"],
                (row["n_params"], row["f1"]),
                textcoords="offset points", xytext=(8, 4),
                fontsize=8,
            )

    ax.set_xscale("log")
    ax.set_xlabel("Number of Parameters (log scale)")
    ax.set_ylabel("Mean F1 Score")
    ax.set_title("Parameter Count vs. Mean F1 Score")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(str(output_path), format="pdf", bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path)
    return output_path


# ------------------------------------------------------------------
# Figure 8 — Event-type heatmap
# ------------------------------------------------------------------

def fig_event_heatmap(df: pd.DataFrame, output_path: str | Path) -> Path:
    """Heatmap: rows = event classes, columns = models, cells = mean F1."""
    output_path = Path(output_path)
    _apply_style()

    pivot = df.pivot_table(
        index="event_name",
        columns="model",
        values="f1",
        aggfunc="mean",
    )
    # Rename columns to short labels
    pivot.columns = [_short(c) for c in pivot.columns]

    # Sort rows by event class number
    event_order = [EVENT_NAMES[i] for i in sorted(EVENT_NAMES) if EVENT_NAMES[i] in pivot.index]
    pivot = pivot.reindex(event_order)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        vmin=0,
        vmax=1,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Mean F1"},
    )
    ax.set_title("Mean F1 by Event Type and Model")
    ax.set_xlabel("Model")
    ax.set_ylabel("Event Type")
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    fig.tight_layout()
    fig.savefig(str(output_path), format="pdf", bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path)
    return output_path


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

__all__ = [
    "fig_qae_circuit",
    "fig_preprocessing_pipeline",
    "fig_roc_curves",
    "fig_pr_curves",
    "fig_threshold_histograms",
    "_threshold_histogram_data",
    "fig_f1_boxplots",
    "fig_param_vs_f1",
    "fig_event_heatmap",
    "PARAM_COUNTS",
    "EVENT_NAMES",
]
