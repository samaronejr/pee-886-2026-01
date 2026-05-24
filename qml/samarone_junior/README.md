# 👤 Student Space — Samarone Junior

This directory contains a Quantum Machine Learning implementation for anomaly detection on the Petrobras 3W dataset, developed as part of the PEE-886 course at COPPE/UFRJ.

## 📜 Contribution Rules (Reminder)
1. **Naming Convention**: All files must use `snake_case` (lowercase letters and underscores). Ex: `my_model.py`.
2. **Location**: Work only within the `samarone_junior` workspace folders (📁 `qml/samarone_junior/`, 📁 `notebooks/samarone_junior/`, 📁 `scripts/samarone_junior/`, 📁 `data/samarone_junior/`).
3. **Dependencies**: Add new packages only to the `requirements.txt` file at the root of the repository.

## 🛠️ About this Implementation

**Quantum Autoencoder for Anomaly Detection on the Petrobras 3W Dataset**

This project benchmarks Quantum Autoencoder (QAE) variants against classical anomaly-detection baselines on multivariate oil-well time series. The corrected results support a narrower parameter-efficiency claim against the low-capacity MatchedAutoencoder, not a broad quantum advantage over stronger classical baselines.

### 🚀 Technologies and Architecture

**Core Libraries:**

| Library | Version | Role |
|---|---|---|
| [PennyLane](https://pennylane.ai/) | 0.44 | Quantum circuit simulation and QAE training |
| PyTorch | — | Neural-network baselines (Matched AE, Full AE, LSTM AE) |
| scikit-learn | — | Isolation Forest, One-Class SVM, preprocessing |
| NumPy / SciPy | — | Feature engineering and statistical metrics |

**Package Structure:**

```
qml/samarone_junior/
├── __init__.py
├── loaders/                   # Data loading and feature engineering
│   ├── three_w_loader.py      # ThreeWLoader — reads 3W Parquet files by class
│   └── preprocessing.py       # FeatureEngineer — sliding-window statistical features
├── models/                    # All anomaly-detection models
│   ├── quantum_autoencoder.py # QAETrashFidelity, QAEReconstruction, efficient_su2
│   └── baselines.py           # MatchedAutoencoder, FullAutoencoder, LSTMAutoencoder,
│                              # IsolationForestDetector, OneClassSVMDetector
├── trainer/                   # Training orchestration (future extension)
├── evaluation/                # Threshold calibration and binary metrics
│   └── metrics.py             # calibrate_threshold(), compute_binary_metrics()
└── visualization/             # Publication figure generation
    └── figures.py             # 8 figure functions (ROC, precision-recall, heatmaps, etc.)
```

**Models Compared:**

| Model | Type | Description |
|---|---|---|
| `QAETrashFidelity` | Quantum | 48-parameter QAE with trash-qubit fidelity as anomaly score |
| `QAEReconstruction` | Quantum | 48-parameter QAE with reconstruction error as anomaly score |
| `MatchedAutoencoder` | Classical | 100-parameter low-capacity autoencoder comparator (not parameter-equal to the 48-parameter QAEs) |
| `FullAutoencoder` | Classical | Larger autoencoder (unconstrained capacity) |
| `LSTMAutoencoder` | Classical | LSTM-based sequence autoencoder |
| `IsolationForestDetector` | Classical | Isolation Forest (non-parametric) |
| `OneClassSVMDetector` | Classical | One-Class SVM with RBF kernel (non-parametric) |

### 📖 Usage Instructions

**1. Import the module:**

```python
from qml.samarone_junior.loaders import ThreeWLoader, FeatureEngineer
from qml.samarone_junior.models import QAETrashFidelity, IsolationForestDetector
from qml.samarone_junior.evaluation import calibrate_threshold, compute_binary_metrics
```

**2. Run the full experiment pipeline:**

```bash
cd pee-886-2026-01
python scripts/samarone_junior/run_experiment.py \
    --data-path data/samarone_junior/3w \
    --output-dir data/samarone_junior/results \
    --n-folds 5 --seeds 42 123 456 789 1024 2023 2024 2025 2026 7777 \
    --classical-epochs 200 --qae-epochs 50 \
    --max-train-windows 500 --threshold-percentile 95 \
    --overwrite \
    --verbose
```

Key CLI options:
- `--data-path` — path to the 3W dataset root directory containing class directories `0/` through `9/`
- `--output-dir` — where to write the JSONL results file
- `--output-file` — exact JSONL output path, overriding `--output-dir/metrics.jsonl`
- `--overwrite` — allow replacing an existing JSONL output file
- `--models` — subset of models to run (default: all 7)
- `--n-folds` — number of cross-validation folds (default: 5)
- `--folds` — zero-based fold subset for resumable SLURM array retries
- `--seeds` — random seeds for reproducibility
- `--event-classes` — anomaly class subset (`1` through `9`) for partial fold retries
- `--max-train-windows` — cap training windows per fold (useful for quick tests)

The full corrected baseline uses `7 models × 10 seeds × 5 folds × 9 anomaly
classes = 3150` result rows. Treat earlier outputs as pre-fix/stale until a
corrected JSONL file passes `result_integrity.py` validation.

Quick smoke run before expensive HPC jobs:

```bash
cd pee-886-2026-01
python scripts/samarone_junior/run_experiment.py \
    --data-path data/samarone_junior/3w \
    --output-dir data/samarone_junior/results_smoke \
    --models IsolationForestDetector \
    --n-folds 2 \
    --seeds 42 \
    --max-train-windows 50 \
    --overwrite \
    --verbose
python scripts/samarone_junior/result_integrity.py \
    data/samarone_junior/results_smoke/metrics.jsonl \
    --models IsolationForestDetector --seeds 42 --n-folds 2
```

For the corrected full HPC run, promote the validated JSONL to
`data/samarone_junior/results/metrics.jsonl`, validate it with
`result_integrity.py`, then regenerate the report tables with:

```bash
python scripts/samarone_junior/summarize_results.py \
    --results data/samarone_junior/results/metrics.jsonl \
    --output-md data/samarone_junior/results/RESULTS.md \
    --output-tex-table data/samarone_junior/results/generated_ranking_table.tex \
    --output-stats-blocked data/samarone_junior/results/stats_blocked.csv \
    --output-stats-holm data/samarone_junior/results/stats_blocked_holm.csv \
    --run-id theoryfix_20260511T044609Z \
    --validation-date 2026-05-13 \
    --remote-path /home/samarone.lima/qml_project/results/theoryfix_20260511T044609Z/ \
    --hpc-jobs 'rowfill array `28656`; QAE merge `28657`; final merge/summarize `28658`' \
    --recovery-note 'QAE missing rows recovered with rowfill array after scheduler cancellations; final merge validated 3150 unique rows.' \
    --checksum ce4ab88b8d70e7cb989e12e163bf5e93149de70a278d387bd47586d2d9b86747
```

Final corrected-result interpretation:

- LSTMAutoencoder is the best aggregate model (mean F1 = 0.872); FullAutoencoder is second (0.856).
- QAEReconstruction (0.626) and QAETrashFidelity (0.603) outperform MatchedAutoencoder (0.493), supporting only a low-capacity parameter-efficiency claim.
- QAEReconstruction has higher mean F1 than QAETrashFidelity, but the paired seed-fold Holm-adjusted comparison is not significant (p = 0.194).
- Do **not** claim broad quantum advantage over the stronger classical baselines.

**3. Generate publication figures:**

```bash
cd pee-886-2026-01
python scripts/samarone_junior/generate_figures.py \
    --results data/samarone_junior/results/metrics.jsonl \
    --output-dir data/samarone_junior/figures
```

This produces 8 PDF figures in the output directory, ready for LaTeX inclusion.

**4. Explore the notebooks:**

Six annotated Jupyter notebooks are provided in `pee-886-2026-01/notebooks/samarone_junior/`:

| Notebook | Topic |
|---|---|
| `01_data_exploration.ipynb` | 3W dataset structure, class distribution, sensor signals |
| `02_preprocessing.ipynb` | Sliding-window feature engineering pipeline |
| `03_qae_training.ipynb` | Quantum Autoencoder circuit, training loop, loss curves |
| `04_classical_baselines.ipynb` | Classical model training and threshold calibration |
| `05_results_comparison.ipynb` | Cross-model metric comparison and statistical analysis |
| `06_figure_generation.ipynb` | Publication figure generation walkthrough |

### 📂 Dataset Setup

This project uses the [Petrobras 3W dataset](https://github.com/petrobras/3W):

1. Clone or download the 3W repository:
   ```bash
   git clone https://github.com/petrobras/3W.git data/samarone_junior/3w
   ```
2. The path passed to `--data-path` must directly contain class directories `0/` through `9/` with Parquet files under each class.
3. Class 0 (Normal Operation) files use the `WELL-` prefix; simulated instances use `SIMULATED_`; drawn instances use `DRAWN_`.
4. For this repository layout, point `--data-path` to `data/samarone_junior/3w` when running from `pee-886-2026-01/`.

### 📚 References

1. **Quantum Autoencoders:** Romero, J., Olson, J. P., & Aspuru-Guzik, A. (2017). Quantum autoencoders for efficient compression of quantum data. *Quantum Science and Technology*, 2(4), 045001. [arXiv:1612.02806](https://arxiv.org/abs/1612.02806)

2. **3W Dataset:** Vargas, R. E. V., Munaro, C. J., Ciarelli, P. M., Medeiros, A. G., Amaral, B. G., Barrionuevo, D. C., ... & Ribeiro, T. J. (2019). A realistic and public dataset with rare undesirable real events in oil wells. *Journal of Petroleum Science and Engineering*, 181, 106223. [DOI:10.1016/j.petrol.2019.106223](https://doi.org/10.1016/j.petrol.2019.106223)

3. **PennyLane:** Bergholm, V., Izaac, J., Schuld, M., et al. (2022). PennyLane: Automatic differentiation of hybrid quantum-classical computations. [arXiv:1811.04968](https://arxiv.org/abs/1811.04968) — [Documentation](https://docs.pennylane.ai/)

4. **Anomaly Detection with QAE:** Ngairangbam, V. S., Spannowsky, M., & Sussman, M. (2022). Anomaly detection in high-energy physics using a quantum autoencoder. *Physical Review D*, 105(9), 095004. [arXiv:2112.04958](https://arxiv.org/abs/2112.04958)
