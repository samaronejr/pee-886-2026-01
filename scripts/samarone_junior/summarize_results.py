#!/usr/bin/env python3
"""Summarize validated QAE/3W experiment metrics for reports and papers.

The script intentionally derives tables from ``metrics.jsonl`` instead of
hard-coded numbers, reducing the risk of stale paper claims after HPC reruns.
"""

from __future__ import annotations

import argparse
import math
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd

# Ensure project root is on sys.path when run from any working directory.
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from qml.samarone_junior.evaluation.metrics import (  # noqa: E402
    compute_block_confidence_intervals,
    compute_block_significance,
    holm_adjust,
)
from scripts.samarone_junior.result_integrity import validate_result_file  # noqa: E402

EXPECTED_MODELS: list[str] = [
    "IsolationForestDetector",
    "OneClassSVMDetector",
    "MatchedAutoencoder",
    "FullAutoencoder",
    "LSTMAutoencoder",
    "QAETrashFidelity",
    "QAEReconstruction",
]
EXPECTED_SEEDS: list[int] = [42, 123, 456, 789, 1024, 2023, 2024, 2025, 2026, 7777]
EXPECTED_FOLDS = 5
EXPECTED_EVENT_CLASSES: list[int] = list(range(1, 10))

MODEL_FAMILIES: dict[str, str] = {
    "QAETrashFidelity": "Quantum (trash-qubit fidelity)",
    "QAEReconstruction": "Quantum (MCM + adjoint decoder)",
    "MatchedAutoencoder": "Classical AE (parameter-matched)",
    "FullAutoencoder": "Classical AE (high capacity)",
    "LSTMAutoencoder": "Classical recurrent AE",
    "IsolationForestDetector": "Tree-based anomaly detector",
    "OneClassSVMDetector": "Kernel one-class detector",
}

MODEL_INPUTS: dict[str, str] = {
    "QAETrashFidelity": "6-D PCA angles",
    "QAEReconstruction": "6-D PCA angles",
    "MatchedAutoencoder": "6-D PCA angles",
    "FullAutoencoder": "6-D PCA angles",
    "LSTMAutoencoder": "raw windows (128 × 5)",
    "IsolationForestDetector": "6-D PCA features",
    "OneClassSVMDetector": "6-D PCA features",
}

PAPER_MODEL_LABELS: dict[str, str] = {
    "LSTMAutoencoder": "LSTMAutoencoder",
    "FullAutoencoder": "FullAutoencoder",
    "IsolationForestDetector": "IsolationForestDetector",
    "OneClassSVMDetector": "OneClassSVMDetector",
    "QAETrashFidelity": r"\textbf{QAETrashFidelity}",
    "MatchedAutoencoder": "MatchedAutoencoder",
    "QAEReconstruction": "QAEReconstruction",
}


@dataclass(frozen=True)
class SummaryTables:
    validation: dict[str, int]
    model_summary: pd.DataFrame
    per_event_f1: pd.DataFrame
    qae_trash_comparisons: pd.DataFrame
    qae_recon_comparisons: pd.DataFrame


@dataclass(frozen=True)
class ResultProvenance:
    """Human-facing provenance included in generated result reports."""

    run_id: str = "unknown-run"
    validation_date: str = "unknown date"
    remote_path: str | None = None
    hpc_jobs: str | None = None
    recovery_note: str | None = None
    checksum: str | None = None


def fmt_float(value: float, digits: int = 3) -> str:
    """Format finite floats to fixed decimals; NaN becomes an em dash."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return f"{float(value):.{digits}f}"


def fmt_ci(low: float, high: float) -> str:
    """Format a 95% confidence interval for Markdown tables."""
    return f"[{fmt_float(low)}, {fmt_float(high)}]"


def fmt_p_value(value: float) -> str:
    """Format p-values compactly for Markdown."""
    if value is None or math.isnan(float(value)):
        return "NaN"
    value = float(value)
    if value == 0.0:
        return "< 1e-300"
    exponent = math.floor(math.log10(abs(value)))
    mantissa = value / (10 ** exponent)
    return f"{mantissa:.2f} × 10^{exponent}"


def fmt_p_value_tex(value: float) -> str:
    """Format p-values for LaTeX math mode."""
    if value is None or math.isnan(float(value)):
        return r"\mathrm{NaN}"
    value = float(value)
    if value == 0.0:
        return r"<10^{-300}"
    exponent = math.floor(math.log10(abs(value)))
    mantissa = value / (10 ** exponent)
    return rf"{mantissa:.2f} \times 10^{{{exponent}}}"


def fmt_params(value: object) -> str:
    """Render trainable parameter counts."""
    try:
        if value is None or pd.isna(value):
            return "—"
        return str(int(value))
    except (TypeError, ValueError):
        return "—"


def _load_validated_results(path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    validation = validate_result_file(
        path,
        models=EXPECTED_MODELS,
        seeds=EXPECTED_SEEDS,
        n_folds=EXPECTED_FOLDS,
        event_classes=EXPECTED_EVENT_CLASSES,
    )
    df = pd.read_json(path, lines=True)
    return df, validation


def _model_param_count(df: pd.DataFrame, model: str) -> object:
    values = df.loc[df["model"] == model, "n_params"].dropna().unique()
    if len(values) == 0:
        return None
    return values[0]


def _with_holm_adjustment(comparisons: pd.DataFrame) -> pd.DataFrame:
    """Attach Holm-adjusted p-values within one comparison family."""
    comparisons = comparisons.copy()
    adjusted, reject = holm_adjust(comparisons["p_value"].to_numpy(dtype=float))
    comparisons["p_holm"] = adjusted
    comparisons["holm_reject_0.05"] = reject
    return comparisons


def compute_summary_tables(results_path: Path) -> SummaryTables:
    """Compute all tables used by the result report."""
    df, validation = _load_validated_results(results_path)

    metric_means = (
        df.groupby("model")[["f1", "auc_roc", "auc_pr", "precision", "recall"]]
        .mean()
        .reset_index()
    )
    ci_rows = []
    for model in metric_means["model"]:
        ci = compute_block_confidence_intervals(df, model, metric="f1")
        ci_rows.append({"model": model, **ci, "n_params": _model_param_count(df, model)})
    ci_df = pd.DataFrame(ci_rows)
    model_summary = metric_means.merge(ci_df, on="model", how="left")
    model_summary = model_summary.sort_values("f1", ascending=False).reset_index(drop=True)
    model_summary.insert(0, "rank", range(1, len(model_summary) + 1))
    model_summary["ci_width"] = model_summary["ci_upper"] - model_summary["ci_lower"]

    per_event_f1 = (
        df.pivot_table(
            index=["event_class", "event_name"],
            columns="model",
            values="f1",
            aggfunc="mean",
        )
        .reset_index()
        .sort_values("event_class")
    )

    qae_trash_baselines = [
        "MatchedAutoencoder",
        "OneClassSVMDetector",
        "IsolationForestDetector",
        "FullAutoencoder",
        "LSTMAutoencoder",
        "QAEReconstruction",
    ]
    trash_rows = []
    for baseline in qae_trash_baselines:
        stat = compute_block_significance(df, "QAETrashFidelity", baseline, metric="f1")
        trash_rows.append({"comparison": f"QAETrashFidelity vs. {baseline}", "baseline": baseline, **stat})
    qae_trash_comparisons = _with_holm_adjustment(pd.DataFrame(trash_rows))

    recon_rows = []
    for baseline in [m for m in model_summary["model"] if m != "QAEReconstruction"]:
        stat = compute_block_significance(df, "QAEReconstruction", baseline, metric="f1")
        recon_rows.append({"comparison": f"QAEReconstruction vs. {baseline}", "baseline": baseline, **stat})
    qae_recon_comparisons = _with_holm_adjustment(pd.DataFrame(recon_rows))

    return SummaryTables(
        validation=validation,
        model_summary=model_summary,
        per_event_f1=per_event_f1,
        qae_trash_comparisons=qae_trash_comparisons,
        qae_recon_comparisons=qae_recon_comparisons,
    )


def _verdict_for_trash(row: pd.Series) -> str:
    baseline = row["baseline"]
    delta = float(row["mean_diff"])
    significant = bool(row.get("holm_reject_0.05", False))
    if baseline == "MatchedAutoencoder":
        if delta > 0 and significant:
            return "QAE wins at comparable capacity"
        if delta > 0:
            return "QAE higher, not significant"
    if baseline == "QAEReconstruction":
        if not significant:
            return "No significant QAE-objective difference"
        return "Trash-fidelity QAE wins" if delta > 0 else "Reconstruction QAE wins"
    if not significant:
        return "No significant difference"
    return "Baseline wins" if delta < 0 else "QAE wins"


def _qae_objective_sentence(summary: pd.DataFrame, trash_vs_recon: pd.Series) -> str:
    """Describe QAETrashFidelity vs QAEReconstruction from actual sign/significance."""
    trash_f1 = float(summary.loc[summary["model"] == "QAETrashFidelity", "f1"].iloc[0])
    recon_f1 = float(summary.loc[summary["model"] == "QAEReconstruction", "f1"].iloc[0])
    delta_trash_minus_recon = float(trash_vs_recon["mean_diff"])
    p_holm = float(trash_vs_recon["p_holm"])
    significant = bool(trash_vs_recon.get("holm_reject_0.05", False))
    if delta_trash_minus_recon > 0:
        direction = "QAETrashFidelity has the higher mean F1"
    elif delta_trash_minus_recon < 0:
        direction = "QAEReconstruction has the higher mean F1"
    else:
        direction = "Both QAE objectives have the same mean F1"
    significance = "significant" if significant else "not statistically significant"
    return (
        f"**QAE objective choice:** {direction} "
        f"(QAETrashFidelity {fmt_float(trash_f1)} vs. QAEReconstruction {fmt_float(recon_f1)}; "
        f"QAETrashFidelity − QAEReconstruction Δ = {delta_trash_minus_recon:+.3f}, "
        f"Holm p = {fmt_p_value(p_holm)}), but the paired block difference is {significance}."
    )


def _render_reconstruction_paragraph(
    summary: pd.DataFrame,
    recon: pd.DataFrame,
    trash_vs_recon: pd.Series,
) -> list[str]:
    """Render rank/significance-aware QAEReconstruction discussion."""
    recon_rank = int(summary.loc[summary["model"] == "QAEReconstruction", "rank"].iloc[0])
    recon_f1 = float(summary.loc[summary["model"] == "QAEReconstruction", "f1"].iloc[0])
    matched = recon.loc[recon["baseline"] == "MatchedAutoencoder"].iloc[0]
    trash = recon.loc[recon["baseline"] == "QAETrashFidelity"].iloc[0]
    strongest_loss = recon.sort_values("mean_diff").iloc[0]

    if float(trash["mean_diff"]) > 0:
        qae_sentence = (
            f"Against QAETrashFidelity, QAEReconstruction has higher mean F1 "
            f"(Δ = {float(trash['mean_diff']):+.3f}; Holm p = {fmt_p_value(trash['p_holm'])})."
        )
    elif float(trash["mean_diff"]) < 0:
        qae_sentence = (
            f"Against QAETrashFidelity, QAEReconstruction has lower mean F1 "
            f"(Δ = {float(trash['mean_diff']):+.3f}; Holm p = {fmt_p_value(trash['p_holm'])})."
        )
    else:
        qae_sentence = (
            f"Against QAETrashFidelity, QAEReconstruction ties in mean F1 "
            f"(Holm p = {fmt_p_value(trash['p_holm'])})."
        )
    if not bool(trash.get("holm_reject_0.05", False)):
        qae_sentence += " This QAE-objective difference is not significant after Holm correction."

    return [
        f"QAEReconstruction ranks {recon_rank} with mean F1 = {fmt_float(recon_f1)}.",
        (
            f"It outperforms the parameter-matched MatchedAutoencoder by "
            f"Δ = {float(matched['mean_diff']):+.3f} mean F1 "
            f"(Holm p = {fmt_p_value(matched['p_holm'])})."
        ),
        qae_sentence,
        (
            f"Its largest deficit is against {strongest_loss['baseline']} "
            f"(QAEReconstruction − baseline Δ = {float(strongest_loss['mean_diff']):+.3f}, "
            f"Holm p = {fmt_p_value(strongest_loss['p_holm'])})."
        ),
    ]


def _render_wrapped_command(command: str, args: Sequence[str]) -> list[str]:
    """Render a multi-line shell command with continuations."""
    lines = [command, *(f"    {arg}" for arg in args)]
    return [f"{line} \\" for line in lines[:-1]] + [lines[-1]]


def render_markdown(
    tables: SummaryTables,
    *,
    results_path: Path,
    provenance: ResultProvenance | None = None,
    checksum: str | None = None,
) -> str:
    """Render the full RESULTS.md document."""
    if provenance is None:
        provenance = ResultProvenance(checksum=checksum)
    elif checksum and provenance.checksum is None:
        provenance = ResultProvenance(
            run_id=provenance.run_id,
            validation_date=provenance.validation_date,
            remote_path=provenance.remote_path,
            hpc_jobs=provenance.hpc_jobs,
            recovery_note=provenance.recovery_note,
            checksum=checksum,
        )

    summary = tables.model_summary
    per_event = tables.per_event_f1
    trash = tables.qae_trash_comparisons
    recon = tables.qae_recon_comparisons

    best_model = str(summary.iloc[0]["model"])
    best_f1 = float(summary.iloc[0]["f1"])
    qae_vs_matched = trash.loc[trash["baseline"] == "MatchedAutoencoder"].iloc[0]
    qae_vs_lstm = trash.loc[trash["baseline"] == "LSTMAutoencoder"].iloc[0]
    qae_vs_full = trash.loc[trash["baseline"] == "FullAutoencoder"].iloc[0]
    trash_vs_recon = trash.loc[trash["baseline"] == "QAEReconstruction"].iloc[0]
    summarize_args = [
        "--results data/samarone_junior/results/metrics.jsonl",
        "--output-md data/samarone_junior/results/RESULTS.md",
        "--output-tex-table data/samarone_junior/results/generated_ranking_table.tex",
        "--output-stats-blocked data/samarone_junior/results/stats_blocked.csv",
        "--output-stats-holm data/samarone_junior/results/stats_blocked_holm.csv",
        f"--run-id {shlex.quote(provenance.run_id)}",
        f"--validation-date {shlex.quote(provenance.validation_date)}",
    ]
    if provenance.remote_path:
        summarize_args.append(f"--remote-path {shlex.quote(provenance.remote_path)}")
    if provenance.hpc_jobs:
        summarize_args.append(f"--hpc-jobs {shlex.quote(provenance.hpc_jobs)}")
    if provenance.recovery_note:
        summarize_args.append(f"--recovery-note {shlex.quote(provenance.recovery_note)}")
    if provenance.checksum:
        summarize_args.append(f"--checksum {provenance.checksum}")

    lines: list[str] = [
        "# Results — QAE vs. Classical Baselines on Petrobras 3W",
        "",
        f"> **Status:** Corrected full HPC rerun `{provenance.run_id}` validated on {provenance.validation_date}.",
        "",
        "## Validation and Provenance",
        "",
        f"- **Canonical metrics:** `{results_path}`",
        f"- **Rows / unique keys:** {tables.validation['rows']} / {tables.validation['unique_keys']}",
        f"- **Grid:** {tables.validation['models']} models × {tables.validation['seeds']} seeds × {tables.validation['folds']} folds × {tables.validation['event_classes']} event classes = 3150 rows",
    ]
    if provenance.remote_path:
        lines.append(f"- **Remote source:** `{provenance.remote_path}`")
    if provenance.hpc_jobs:
        lines.append(f"- **HPC jobs:** {provenance.hpc_jobs}")
    if provenance.recovery_note:
        lines.append(f"- **Recovery note:** {provenance.recovery_note}")
    if provenance.checksum:
        lines.append(f"- **SHA-256:** `{provenance.checksum}`")
    lines.extend([
        "",
        "## Experimental Setup",
        "",
        "- **Dataset:** Petrobras 3W v2.0.0 — 5 sensors (P-PDG, P-TPT, T-TPT, P-MON-CKP, T-JUS-CKP), 10 event classes (0 = Normal, 1–9 = anomaly types), 42 real wells.",
        "- **Feature pipeline:** per-file sliding windows (W=128, stride=64) → 4 statistics/sensor → PCA(6) → MinMaxScaler → clip to [0, π].",
        "- **Evaluation:** file-level 5-fold cross-validation × 10 seeds × 9 anomaly classes = 450 observations per model, 3150 total rows.",
        "- **Threshold:** 95th percentile of held-out calibration-normal scores.",
        "- **Training:** 200 epochs classical, 50 epochs QAE; max 500 training windows per fold.",
        "",
        "## Models Compared",
        "",
        "| Model | Family | Params | Input |",
        "|-------|--------|--------|-------|",
    ])
    for model in EXPECTED_MODELS:
        params = fmt_params(_model_param_count_from_summary(summary, model))
        lines.append(f"| {model} | {MODEL_FAMILIES[model]} | {params} | {MODEL_INPUTS[model]} |")

    lines.extend([
        "",
        "## Aggregate Performance",
        "",
        "Mean metrics are aggregated across 10 seeds × 5 folds × 9 anomaly classes (450 observations/model). Confidence intervals use a seed-fold block bootstrap over 50 blocks/model.",
        "",
        "| Rank | Model | F1 | 95% CI | CI width | AUC-ROC | AUC-PR | Precision | Recall |",
        "|------|-------|----|--------|----------|---------|--------|-----------|--------|",
    ])
    for _, row in summary.iterrows():
        model = str(row["model"])
        model_label = f"**{model}**" if model == "QAETrashFidelity" else model
        lines.append(
            "| {rank} | {model} | {f1} | {ci} | {ci_width} | {auc_roc} | {auc_pr} | {precision} | {recall} |".format(
                rank=int(row["rank"]),
                model=model_label,
                f1=fmt_float(row["f1"]),
                ci=fmt_ci(row["ci_lower"], row["ci_upper"]),
                ci_width=fmt_float(row["ci_width"]),
                auc_roc=fmt_float(row["auc_roc"]),
                auc_pr=fmt_float(row["auc_pr"]),
                precision=fmt_float(row["precision"]),
                recall=fmt_float(row["recall"]),
            )
        )

    lines.extend([
        "",
        "## Statistical Significance",
        "",
        "Wilcoxon signed-rank tests first aggregate event rows within each `(seed, fold)` block, then compare the 50 paired seed-fold blocks. Reported adjusted p-values use Holm correction within each comparison family.",
        "",
        "### QAETrashFidelity vs. baselines",
        "",
        "| Comparison | Δ mean F1 | Blocks | p-value | Holm p | Verdict |",
        "|------------|----------:|-------:|--------:|-------:|---------|",
    ])
    for _, row in trash.iterrows():
        lines.append(
            f"| vs. {row['baseline']} | {float(row['mean_diff']):+.3f} | {int(row['n_blocks'])} | {fmt_p_value(row['p_value'])} | {fmt_p_value(row['p_holm'])} | {_verdict_for_trash(row)} |"
        )

    recon_paragraph = _render_reconstruction_paragraph(summary, recon, trash_vs_recon)
    lines.extend([
        "",
        "### QAEReconstruction vs. all models",
        "",
        *recon_paragraph,
        "",
        "## Per-Event-Class Analysis",
        "",
        "Mean F1 per event class; each cell aggregates 10 seeds × 5 folds = 50 observations.",
        "",
    ])
    ordered_models = list(summary["model"])
    header = "| Event | Name | Best | " + " | ".join(ordered_models) + " |"
    sep = "|------:|------|------|" + "---:|" * len(ordered_models)
    lines.extend([header, sep])
    for _, row in per_event.iterrows():
        event_class = int(row["event_class"])
        event_name = str(row["event_name"])
        best = max(ordered_models, key=lambda model: row[model])
        values = " | ".join(fmt_float(row[model]) for model in ordered_models)
        lines.append(f"| {event_class} | {event_name} | {best} | {values} |")

    event6 = per_event.loc[per_event["event_class"] == 6].iloc[0]
    event6_best = max(ordered_models, key=lambda model: event6[model])
    lines.extend([
        "",
        "## Key Findings",
        "",
        f"1. **Best aggregate model:** {best_model} leads with mean F1 = {fmt_float(best_f1)}.",
        f"2. **Parameter-efficiency result:** QAETrashFidelity (48 parameters) beats the parameter-matched MatchedAutoencoder (100 parameters) by Δ = {float(qae_vs_matched['mean_diff']):+.3f} mean F1 (Holm p = {fmt_p_value(qae_vs_matched['p_holm'])}). This is the defensible quantum-efficiency claim; it is not a broad win over stronger classical baselines.",
        f"3. **Capacity dominates raw performance:** QAETrashFidelity trails FullAutoencoder by Δ = {float(qae_vs_full['mean_diff']):+.3f} and LSTMAutoencoder by Δ = {float(qae_vs_lstm['mean_diff']):+.3f}.",
        f"4. {_qae_objective_sentence(summary, trash_vs_recon)}",
        f"5. **Event 6 remains difficult:** `{event6['event_name']}` has the lowest broad performance; only {event6_best} reaches F1 = {fmt_float(event6[event6_best])}.",
        "",
        "## Limitations",
        "",
        "- **LSTM-AE input asymmetry:** LSTM-AE uses raw 3D temporal windows `(n, 128, 5)`, while the other models use 6-D PCA summaries. Treat the LSTM-AE result as a temporal-aware upper bound, not a like-for-like architecture comparison.",
        "- **Training-window cap:** QAE simulation cost required capping training windows at 500 per fold; this also constrains classical baselines in this benchmark.",
        "- **Simulator-only QAE:** Results are CPU simulator results and do not include hardware noise or queue-time constraints.",
        "",
        "## Reproducibility",
        "",
        "```bash",
        "cd pee-886-2026-01",
        "python scripts/samarone_junior/result_integrity.py \\",
        "    data/samarone_junior/results/metrics.jsonl \\",
        "    --models IsolationForestDetector OneClassSVMDetector MatchedAutoencoder FullAutoencoder LSTMAutoencoder QAETrashFidelity QAEReconstruction \\",
        "    --seeds 42 123 456 789 1024 2023 2024 2025 2026 7777 \\",
        "    --n-folds 5 \\",
        "    --require-corrected-methodology",
        *_render_wrapped_command("python scripts/samarone_junior/summarize_results.py", summarize_args),
        "python scripts/samarone_junior/generate_figures.py \\",
        "    --results data/samarone_junior/results/metrics.jsonl \\",
        "    --output-dir data/samarone_junior/figures",
        "```",
        "",
        "This file is generated by `scripts/samarone_junior/summarize_results.py`; rerun that script after any new metrics file is promoted.",
        "",
    ])
    return "\n".join(lines)


def _model_param_count_from_summary(summary: pd.DataFrame, model: str) -> object:
    rows = summary.loc[summary["model"] == model, "n_params"]
    return None if rows.empty else rows.iloc[0]


def render_latex_table(tables: SummaryTables) -> str:
    """Render the ranking table body for the paper results section."""
    lines = [
        r"\begin{table}[h!]",
        r"\centering",
        r"\caption{Ranking dos modelos por F1 médio sobre $10 \times 5 \times 9 = 450$ observações por modelo, com intervalo de confiança \textit{bootstrap} em blocos (semente--fold) de 95\%.}",
        r"\label{tab:ranking}",
        r"\begin{tabular}{rlrrrr}",
        r"\toprule",
        r"\textbf{\#} & \textbf{Modelo} & \textbf{Parâmetros} & \textbf{F1} & \textbf{IC 95\%} & \textbf{AUC-ROC} \\",
        r"\midrule",
    ]
    for _, row in tables.model_summary.iterrows():
        model = str(row["model"])
        params = fmt_params(row["n_params"])
        params_tex = "---" if params == "—" else f"${params}$"
        lines.append(
            "{rank} & {model} & {params} & ${f1}$ & $[{lo};\\, {hi}]$ & ${auc}$ \\\\".format(
                rank=int(row["rank"]),
                model=PAPER_MODEL_LABELS.get(model, model),
                params=params_tex,
                f1=fmt_float(row["f1"]).replace(".", ","),
                lo=fmt_float(row["ci_lower"]).replace(".", ","),
                hi=fmt_float(row["ci_upper"]).replace(".", ","),
                auc=fmt_float(row["auc_roc"]).replace(".", ","),
            )
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def comparison_stats_frame(tables: SummaryTables) -> pd.DataFrame:
    """Return one CSV-ready block-aware comparison table."""
    trash = tables.qae_trash_comparisons.copy()
    trash.insert(0, "comparison_family", "QAETrashFidelity")
    recon = tables.qae_recon_comparisons.copy()
    recon.insert(0, "comparison_family", "QAEReconstruction")
    return pd.concat([trash, recon], ignore_index=True, sort=False)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True, help="Validated metrics.jsonl input")
    parser.add_argument("--output-md", type=Path, help="Write full Markdown report")
    parser.add_argument("--output-tex-table", type=Path, help="Write LaTeX ranking table snippet")
    parser.add_argument("--output-stats-blocked", type=Path, help="Write block-aware comparison stats CSV")
    parser.add_argument("--output-stats-holm", type=Path, help="Write block-aware Holm-adjusted stats CSV")
    parser.add_argument("--checksum", help="Optional SHA-256 checksum to include in Markdown")
    parser.add_argument("--run-id", default="unknown-run", help="Canonical HPC/local run identifier")
    parser.add_argument("--validation-date", default="unknown date", help="Date when the promoted results were validated")
    parser.add_argument("--remote-path", help="Optional remote source path for the promoted metrics")
    parser.add_argument("--hpc-jobs", help="Optional HPC job provenance string")
    parser.add_argument("--recovery-note", help="Optional note describing row recovery or merge handling")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    tables = compute_summary_tables(args.results)
    provenance = ResultProvenance(
        run_id=args.run_id,
        validation_date=args.validation_date,
        remote_path=args.remote_path,
        hpc_jobs=args.hpc_jobs,
        recovery_note=args.recovery_note,
        checksum=args.checksum,
    )
    if args.output_md:
        write_text(args.output_md, render_markdown(tables, results_path=args.results, provenance=provenance))
    else:
        print(render_markdown(tables, results_path=args.results, provenance=provenance))
    if args.output_tex_table:
        write_text(args.output_tex_table, render_latex_table(tables))
    stats = comparison_stats_frame(tables)
    if args.output_stats_blocked:
        raw_cols = [col for col in stats.columns if col not in {"p_holm", "holm_reject_0.05"}]
        write_csv(args.output_stats_blocked, stats[raw_cols])
    if args.output_stats_holm:
        write_csv(args.output_stats_holm, stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
