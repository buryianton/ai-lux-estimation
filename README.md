# AI-Based Indoor Illuminance Estimation from Smartphone Images

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository contains the code for the paper:

> **"AI-Based Estimation of Indoor Illuminance from Smartphone Photographs: A Pilot Study for Occupational Health Applications"**  
> Anton Buryi, [Journal], [Year]  
> DOI: [to be added upon publication]

## Overview

We trained machine-learning models to estimate indoor illuminance (lux) from fixed-geometry smartphone photographs, targeting occupational health screening applications where routine workplace illuminance measurements are mandatory.

**Key results (v4 dataset, 5,814 images):**

| Surface condition | Best model | MAPE (session split) | MAPE (physical-point split) |
|---|---|---|---|
| White paper (5pt) | ExtraTrees | 4.13% | 3.81% |
| White paper (all labels) | ExtraTrees | 3.61% | 5.20% |
| Table surfaces (5pt) | ExtraTrees | 16.42% | 16.15% |
| White + tables (5pt) | ExtraTrees | 6.01% | 10.08% |
| Threshold classification (AUC) | ExtraTrees | ≥0.986 | ≥0.996 |

## Repository Structure

```
ai-lux-estimation/
├── features/
│   └── extract_features.py          # Core feature extraction pipeline
├── models/
│   ├── train_extratrees.py          # ExtraTrees regression
│   ├── train_xgboost.py             # XGBoost regression
│   └── threshold_classification.py  # Pass/fail compliance screening
├── evaluation/
│   ├── grouped_split.py             # Session and physical-point grouped splits
│   └── metrics.py                   # MAPE, MAE, RMSE, R², threshold metrics
├── notebooks/
│   ├── 01_benchmark_all_models.ipynb      # Full model comparison
│   ├── 02_threshold_classification.ipynb  # Compliance screening analysis
│   └── 03_feature_importance.ipynb        # Feature importance analysis
├── data/
│   └── README.md                    # Dataset description and access instructions
├── results/
│   └── summary_table.md             # All reported results
├── CHANGELOG.md
├── requirements.txt
└── README.md
```

## Dataset

The dataset (5,814 photographs with lux annotations) is hosted on Zenodo:

- **Dataset DOI:** [https://doi.org/10.5281/zenodo.20499312](https://doi.org/10.5281/zenodo.20499312)
- **Master annotation CSV:** `annotation_master_latest.csv` (5,814 rows × 23 columns)
- **Master feature CSV:** `feature_master_latest.csv` (5,814 rows × 133 columns)
- See `data/README.md` for full dataset description.

**Camera:** Samsung Galaxy A16  
**Fixed settings:** ISO 50, exposure 1/50 s, white balance 4000 K, 45° tilt, 31 cm height  
**Lux meter:** Lutron LX-101A  
**Illuminance range:** 2–2,280 lux (central measurements); 1–2,280 lux (all measured points)

## Installation

```bash
git clone https://github.com/buryianton/ai-lux-estimation.git
cd ai-lux-estimation
pip install -r requirements.txt
```

## Quick Start

### 1. Train and evaluate the best model (ExtraTrees on white paper)

```python
from models.train_extratrees import train_evaluate_extratrees

results = train_evaluate_extratrees(
    csv_path='path/to/feature_master_latest.csv',
    subset='white_paper',
    split_strategy='session',
    log_space=False
)
print(f"MAPE: {results['metrics']['MAPE']:.2f}%")
```

### 2. Run the full benchmark (reproduces paper tables)

```bash
python models/train_extratrees.py --csv path/to/feature_master_latest.csv --all
python models/train_xgboost.py --csv path/to/feature_master_latest.csv --all
```

### 3. Threshold classification (compliance screening)

```python
from models.threshold_classification import train_threshold_classifier

results = train_threshold_classifier(
    csv_path='path/to/feature_master_latest.csv',
    threshold_lux=500,
    subset='white_paper',
    split_strategy='session'
)
print(f"AUC: {results['metrics']['AUC']:.4f}")
```

### 4. Run all threshold configurations

```bash
python models/threshold_classification.py \
    --csv path/to/feature_master_latest.csv --all
```

## Reproducing Paper Results

All results can be reproduced by running the notebooks in order:

```bash
# In Google Colab (recommended — data is on Drive):
# Open notebooks/01_benchmark_all_models.ipynb
# Set csv_path to your local copy of feature_master_latest.csv
# Run all cells
```

## Citation

If you use this code or dataset, please cite:

```bibtex
@article{buryi2025lux,
  title   = {AI-Based Estimation of Indoor Illuminance from Smartphone Photographs:
             A Pilot Study for Occupational Health Applications},
  author  = {Buryi, Anton},
  journal = {[Journal name]},
  year    = {2025},
  doi     = {[to be added]}
}
```

Dataset citation:
```bibtex
@dataset{buryi2025dataset,
  title   = {AI-Based Surface Illuminance Estimation Dataset:
             Smartphone Image Database with Lux Meter Ground Truth},
  author  = {Buryi, Anton},
  year    = {2025},
  doi     = {10.5281/zenodo.20499312},
  url     = {https://doi.org/10.5281/zenodo.20499312}
}
```

## License

MIT License — see [LICENSE](LICENSE) for details.

## Contact

Anton Buryi — GitHub: [@buryianton](https://github.com/buryianton)

