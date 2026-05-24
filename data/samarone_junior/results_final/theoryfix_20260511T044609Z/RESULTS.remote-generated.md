# Results — QAE vs. Classical Baselines on Petrobras 3W

> **Status:** Corrected full HPC rerun validated on 2026-05-10. Results are based on `corrected_20260502T230443Z`, with QAE missing-slice array recovery `corrected_20260502T230443Z_qae_array2_20260510T031632` and merge job `24627`.

## Validation and Provenance

- **Canonical metrics:** `/results/metrics.jsonl`
- **Rows / unique keys:** 3150 / 3150
- **Grid:** 7 models × 10 seeds × 5 folds × 9 event classes = 3150 rows
- **HPC jobs:** base run `corrected_20260502T230443Z`; retry run `corrected_20260502T230443Z_qae_retry1_20260502T232204`; array job `24626`; merge job `24627`.

## Experimental Setup

- **Dataset:** Petrobras 3W v2.0.0 — 5 sensors (P-PDG, P-TPT, T-TPT, P-MON-CKP, T-JUS-CKP), 10 event classes (0 = Normal, 1–9 = anomaly types), 42 real wells.
- **Feature pipeline:** per-file sliding windows (W=128, stride=64) → 4 statistics/sensor → PCA(6) → MinMaxScaler → clip to [0, π].
- **Evaluation:** file-level 5-fold cross-validation × 10 seeds × 9 anomaly classes = 450 observations per model, 3150 total rows.
- **Threshold:** 95th percentile of held-out calibration-normal scores.
- **Training:** 200 epochs classical, 50 epochs QAE; max 500 training windows per fold.

## Models Compared

| Model | Family | Params | Input |
|-------|--------|--------|-------|
| IsolationForestDetector | Tree-based anomaly detector | — | 6-D PCA features |
| OneClassSVMDetector | Kernel one-class detector | — | 6-D PCA features |
| MatchedAutoencoder | Classical AE (parameter-matched) | 100 | 6-D PCA angles |
| FullAutoencoder | Classical AE (high capacity) | 3006 | 6-D PCA angles |
| LSTMAutoencoder | Classical recurrent AE | 3029 | raw windows (128 × 5) |
| QAETrashFidelity | Quantum (trash-qubit fidelity) | 48 | 6-D PCA angles |
| QAEReconstruction | Quantum (MCM + adjoint decoder) | 48 | 6-D PCA angles |

## Aggregate Performance

Mean metrics are aggregated across 10 seeds × 5 folds × 9 anomaly classes (450 observations/model). Confidence intervals use a seed-fold block bootstrap over 50 blocks/model.

| Rank | Model | F1 | 95% CI | CI width | AUC-ROC | AUC-PR | Precision | Recall |
|------|-------|----|--------|----------|---------|--------|-----------|--------|
| 1 | LSTMAutoencoder | 0.872 | [0.869, 0.876] | 0.007 | 0.989 | 0.952 | 0.869 | 0.971 |
| 2 | FullAutoencoder | 0.856 | [0.851, 0.861] | 0.009 | 0.980 | 0.876 | 0.864 | 0.921 |
| 3 | IsolationForestDetector | 0.776 | [0.754, 0.797] | 0.043 | 0.945 | 0.856 | 0.858 | 0.734 |
| 4 | OneClassSVMDetector | 0.697 | [0.690, 0.704] | 0.015 | 0.752 | 0.813 | 0.857 | 0.699 |
| 5 | QAEReconstruction | 0.626 | [0.575, 0.675] | 0.100 | 0.905 | 0.835 | 0.817 | 0.557 |
| 6 | **QAETrashFidelity** | 0.603 | [0.550, 0.651] | 0.101 | 0.889 | 0.825 | 0.806 | 0.523 |
| 7 | MatchedAutoencoder | 0.493 | [0.441, 0.541] | 0.100 | 0.781 | 0.787 | 0.764 | 0.409 |

## Statistical Significance

Wilcoxon signed-rank tests first aggregate event rows within each `(seed, fold)` block, then compare the 50 paired seed-fold blocks. Reported adjusted p-values use Holm correction within each comparison family.

### QAETrashFidelity vs. baselines

| Comparison | Δ mean F1 | Blocks | p-value | Holm p | Verdict |
|------------|----------:|-------:|--------:|-------:|---------|
| vs. MatchedAutoencoder | +0.111 | 50 | 2.71 × 10^-3 | 5.41 × 10^-3 | QAE wins at comparable capacity |
| vs. OneClassSVMDetector | -0.094 | 50 | 1.48 × 10^-3 | 4.45 × 10^-3 | Baseline wins |
| vs. IsolationForestDetector | -0.173 | 50 | 6.34 × 10^-9 | 2.53 × 10^-8 | Baseline wins |
| vs. FullAutoencoder | -0.253 | 50 | 3.55 × 10^-15 | 1.78 × 10^-14 | Baseline wins |
| vs. LSTMAutoencoder | -0.269 | 50 | 1.78 × 10^-15 | 1.07 × 10^-14 | Baseline wins |
| vs. QAEReconstruction | -0.023 | 50 | 1.94 × 10^-1 | 1.94 × 10^-1 | Baseline wins |

### QAEReconstruction vs. all models

QAEReconstruction ranks last with mean F1 = 0.626. It loses significantly to QAETrashFidelity (Δ = -0.023 for QAETrashFidelity − QAEReconstruction, Holm p = 1.94 × 10^-1) and to every classical baseline.
Its largest gap is against LSTMAutoencoder (QAEReconstruction − baseline Δ = -0.246, Holm p = 1.07 × 10^-14).

## Per-Event-Class Analysis

Mean F1 per event class; each cell aggregates 10 seeds × 5 folds = 50 observations.

| Event | Name | Best | LSTMAutoencoder | FullAutoencoder | IsolationForestDetector | OneClassSVMDetector | QAEReconstruction | QAETrashFidelity | MatchedAutoencoder |
|------:|------|------|---:|---:|---:|---:|---:|---:|---:|
| 1 | Abrupt BSW Increase | LSTMAutoencoder | 0.996 | 0.989 | 0.923 | 0.746 | 0.640 | 0.719 | 0.489 |
| 2 | Spurious DHSV Closure | LSTMAutoencoder | 0.944 | 0.940 | 0.925 | 0.932 | 0.629 | 0.667 | 0.658 |
| 3 | Severe Slugging | LSTMAutoencoder | 0.994 | 0.991 | 0.912 | 0.858 | 0.763 | 0.695 | 0.508 |
| 4 | Flow Instability | LSTMAutoencoder | 0.821 | 0.759 | 0.725 | 0.551 | 0.623 | 0.576 | 0.242 |
| 5 | Rapid Productivity Loss | LSTMAutoencoder | 0.998 | 0.994 | 0.836 | 0.581 | 0.579 | 0.560 | 0.428 |
| 6 | Quick Restriction in PCK | LSTMAutoencoder | 0.118 | 0.061 | 0.001 | 0.086 | 0.006 | 0.002 | 0.000 |
| 7 | Scaling in PCK | LSTMAutoencoder | 0.996 | 0.988 | 0.856 | 0.932 | 0.856 | 0.768 | 0.715 |
| 8 | Hydrate in Production Line | LSTMAutoencoder | 0.988 | 0.988 | 0.867 | 0.866 | 0.841 | 0.732 | 0.786 |
| 9 | Hydrate in Service Line | LSTMAutoencoder | 0.996 | 0.995 | 0.940 | 0.721 | 0.701 | 0.711 | 0.605 |

## Key Findings

1. **Best aggregate model:** LSTMAutoencoder leads with mean F1 = 0.872.
2. **Parameter-efficiency result:** QAETrashFidelity (48 parameters) beats the parameter-matched MatchedAutoencoder (100 parameters) by Δ = +0.111 mean F1 (Holm p = 5.41 × 10^-3). This is the defensible quantum-efficiency claim; it is not a broad win over stronger classical baselines.
3. **Capacity dominates raw performance:** QAETrashFidelity trails FullAutoencoder by Δ = -0.253 and LSTMAutoencoder by Δ = -0.269.
4. **QAE objective choice matters:** QAETrashFidelity outperforms QAEReconstruction by Δ = -0.023 mean F1 (Holm p = 1.94 × 10^-1).
5. **Event 6 remains difficult:** `Quick Restriction in PCK` has the lowest broad performance; only LSTMAutoencoder reaches F1 = 0.118.

## Limitations

- **LSTM-AE input asymmetry:** LSTM-AE uses raw 3D temporal windows `(n, 128, 5)`, while the other models use 6-D PCA summaries. Treat the LSTM-AE result as a temporal-aware upper bound, not a like-for-like architecture comparison.
- **Training-window cap:** QAE simulation cost required capping training windows at 500 per fold; this also constrains classical baselines in this benchmark.
- **Simulator-only QAE:** Results are CPU simulator results and do not include hardware noise or queue-time constraints.

## Reproducibility

```bash
cd pee-886-2026-01
python scripts/samarone_junior/result_integrity.py \
    data/samarone_junior/results/metrics.jsonl \
    --models IsolationForestDetector OneClassSVMDetector MatchedAutoencoder FullAutoencoder LSTMAutoencoder QAETrashFidelity QAEReconstruction \
    --seeds 42 123 456 789 1024 2023 2024 2025 2026 7777 \
    --n-folds 5 \
    --require-corrected-methodology
python scripts/samarone_junior/summarize_results.py \
    --results data/samarone_junior/results/metrics.jsonl \
    --output-md data/samarone_junior/results/RESULTS.md \
    --output-tex-table data/samarone_junior/results/generated_ranking_table.tex \
    --output-stats-blocked data/samarone_junior/results/stats_blocked.csv \
    --output-stats-holm data/samarone_junior/results/stats_blocked_holm.csv
python scripts/samarone_junior/generate_figures.py \
    --results data/samarone_junior/results/metrics.jsonl \
    --output-dir data/samarone_junior/figures
```

This file is generated by `scripts/samarone_junior/summarize_results.py`; rerun that script after any new metrics file is promoted.
