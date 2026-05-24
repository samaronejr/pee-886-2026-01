# Final corrected result quarantine copy

This directory preserves the validated LPS/HPC output bundle for the corrected
full rerun `theoryfix_20260511T044609Z`.

## Canonical use

- Use `../results/metrics.jsonl` as the canonical local metrics file for code,
  figures, and manuscript generation.
- Use this directory as an immutable provenance/quarantine copy of the remote
  artifacts that were promoted into the canonical local results directory.
- Do not mix files from older `corrected_*`, smoke, partial, or rowfill-only
  runs into final claims.

## Validation anchor

- Metrics SHA-256:
  `ce4ab88b8d70e7cb989e12e163bf5e93149de70a278d387bd47586d2d9b86747`
- Expected grid: `7 models × 10 seeds × 5 folds × 9 anomaly classes = 3150`
  unique `(model, seed, fold, event_class)` rows.
- Remote source:
  `/home/samarone.lima/qml_project/results/theoryfix_20260511T044609Z/`
- Recovery history: rowfill array `28656` (throttled `%10`), QAE merge `28657`,
  final merge/summarize `28658`.

Validate the canonical promoted file from the repository root with:

```bash
cd pee-886-2026-01
python scripts/samarone_junior/result_integrity.py \
    data/samarone_junior/results/metrics.jsonl \
    --models IsolationForestDetector OneClassSVMDetector MatchedAutoencoder FullAutoencoder LSTMAutoencoder QAETrashFidelity QAEReconstruction \
    --seeds 42 123 456 789 1024 2023 2024 2025 2026 7777 \
    --n-folds 5 \
    --require-corrected-methodology
```
