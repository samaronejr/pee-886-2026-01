# Corrected Result Provenance

Canonical local metrics:

- `data/samarone_junior/results/metrics.jsonl`
- SHA-256: `ce4ab88b8d70e7cb989e12e163bf5e93149de70a278d387bd47586d2d9b86747`
- Rows: `3150`
- Unique keys: `3150`
- Grid: `7 models × 10 seeds × 5 folds × 9 anomaly event classes`
- Methodology: calibration-normal thresholds, file-level splits, explicit anomaly-window labeling, 6-wire QAE reconstruction scoring, seed-fold blocked inference.

Remote source on LPS:

- Host: `LPS_loginServer`
- Remote directory: `/home/samarone.lima/qml_project/results/theoryfix_20260511T044609Z/`
- Canonical remote file: `/home/samarone.lima/qml_project/results/theoryfix_20260511T044609Z/metrics.jsonl`
- Run ID: `theoryfix_20260511T044609Z`
- Rowfill recovery: Slurm array job `28656` completed missing QAE slices with concurrency throttled to `%10`.
- QAE merge job: `28657`.
- Final merge/summarize job: `28658`.
- Remote final merge timestamp: `2026-05-13 18:47 -03`.

Promoted local quarantine copy:

- Directory: `data/samarone_junior/results_final/theoryfix_20260511T044609Z/`
- Copied artifacts:
  - `metrics.jsonl`
  - `RESULTS.remote-generated.md`
  - `generated_ranking_table.remote-generated.tex`
  - `stats_blocked.remote-generated.csv`
  - `stats_blocked_holm.remote-generated.csv`
  - `SHA256SUMS.txt`

Local validation command:

```bash
python scripts/samarone_junior/result_integrity.py \
    data/samarone_junior/results/metrics.jsonl \
    --models IsolationForestDetector OneClassSVMDetector MatchedAutoencoder FullAutoencoder LSTMAutoencoder QAETrashFidelity QAEReconstruction \
    --seeds 42 123 456 789 1024 2023 2024 2025 2026 7777 \
    --n-folds 5 \
    --require-corrected-methodology
```

Observed validation output:

```text
VALID: data/samarone_junior/results/metrics.jsonl: {'rows': 3150, 'unique_keys': 3150, 'models': 7, 'seeds': 10, 'folds': 5, 'event_classes': 9}
```

Notes:

- `theoryfix_20260511T044609Z` is the only promoted numerical source for final manuscript claims.
- Earlier `corrected_20260502T230443Z*` runs and partial QAE retry outputs are audit history only, not canonical local results.
- `RESULTS.md`, `generated_ranking_table.tex`, `stats_blocked*.csv`, and figures are generated from the canonical local metrics file.
