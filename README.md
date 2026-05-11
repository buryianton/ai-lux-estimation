# AI-Based Indoor Illuminance Estimation from Smartphone Images

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository contains the code for the paper:

> **"AI-Based Estimation of Indoor Illuminance from Smartphone Photographs: A Pilot Study for Occupational Health Applications"**  
> Anton Buryy, [Journal], 2026  
> DOI: [to be added upon publication]

## Overview

We trained machine-learning models to estimate indoor illuminance (lux) from fixed-geometry smartphone photographs, targeting occupational health screening applications in Russia where routine workplace illuminance measurements are mandatory.

**Key results:**
| Surface condition | Best model | MAPE (session split) | MAPE (physical-point split) |
|---|---|---|---|
| White paper | ExtraTrees | 3.28% | 4.55% |
| Table surfaces | ExtraTrees + log | 6.85% | 8.55% (XGBoost + log) |
| White + tables | XGBoost + log | 4.87% | 8.72% (ExtraTrees + log) |
| Threshold classification (AUC) | Random Forest | >0.99 | >0.99 |

## Repository Structure

```
ai-lux-estimation/
├── features/
│   └── extract_features.py       # Core feature extraction pipeline
├── models/
│   ├── train_extratrees.py       # ExtraTrees + log-space regression
│   ├── train_xgboost.py          # XGBoost + log-space regression
│   └── threshold_classification.py  # Pass/fail compliance screening
├── evaluation/
│   ├── grouped_split.py          # Session and physical-point grouped splits
│   └── metrics.py                # MAPE, MAE, RMSE, R², threshold metrics
├── notebooks/
│   ├── 01_benchmark_all_models.ipynb      # Full model comparison
│   ├── 02_threshold_classification.ipynb  # Compliance screening analysis
│   └── 03_feature_importance.ipynb        # Feature importance analysis
├── data/
│   └── README.md                 # Dataset description and access instructions
├── results/
│   └── summary_table.md          # All reported results
├── requirements.txt
└── README.md
```

## Dataset

The dataset (5,814 photographs with lux annotations) is hosted on Google Drive and is **not** included in this repository due to size constraints.

- **Dataset link:** [Google Drive](https://drive.google.com/drive/folders/18vgAnvUA8uLOCT_fZ1HK85zBR4VqMzUk)
- **Master feature CSV:** `AI_Lux_Project/Experiments_4/optionB_master_enriched37_with_pointid_grid5.csv`
- See `data/README.md` for full dataset description.

**Camera:** Samsung Galaxy A16  
**Fixed settings:** ISO 50, exposure 1/20 s, white balance 4000 K, 45° tilt, 31 cm height  
**Lux meter:** TES-1330A Digital Lux Meter  
**Lux range:** 48–1,833 lux (mean = 608, SD = 295)

## Installation

```bash
git clone https://github.com/buryianton/ai-lux-estimation.git
cd ai-lux-estimation
pip install -r requirements.txt
```

## Quick Start

### 1. Feature extraction from a CSV with image paths

```python
from features.extract_features import extract_all_features
import pandas as pd

df = pd.read_csv('your_annotations.csv')
features = extract_all_features(df, image_col='image_path')
```

### 2. Train and evaluate the best model (ExtraTrees + log on white paper)

```python
from models.train_extratrees import train_evaluate_extratrees
from evaluation.grouped_split import grouped_split

results = train_evaluate_extratrees(
    csv_path='path/to/feature_master.csv',
    subset='white_paper',
    split_strategy='session',
    log_space=True
)
print(f"MAPE: {results['MAPE']:.2f}%")
```

### 3. Run the full benchmark (reproduces Table 1 and Table 2 from the paper)

```bash
python models/train_extratrees.py --all
python models/train_xgboost.py --all
```

### 4. Threshold classification (compliance screening)

```python
from models.threshold_classification import train_threshold_classifier

results = train_threshold_classifier(
    csv_path='path/to/feature_master.csv',
    threshold_lux=300,   # Russian office standard
    subset='white_plus_tables'
)
print(f"AUC: {results['AUC']:.4f}, Accuracy: {results['accuracy']:.4f}")
```

## Reproducing Paper Results

All results in the paper can be reproduced by running the notebooks in order:

```bash
# In Google Colab (recommended — data is on Drive):
# Open notebooks/01_benchmark_all_models.ipynb
# Set PROJECT_ROOT = '/content/drive/MyDrive/AI_Lux_Project'
# Run all cells
```

## Citation

If you use this code or dataset, please cite:

```bibtex
@article{buryy2025lux,
  title   = {AI-Based Estimation of Indoor Illuminance from Smartphone Photographs:
             A Pilot Study for Occupational Health Applications},
  author  = {Buryy, Anton},
  journal = {[Journal name]},
  year    = {2025},
  doi     = {[to be added]}
}
```

## License

MIT License — see [LICENSE](LICENSE) for details.

## Contact

Anton Buryy — GitHub: [@buryianton](https://github.com/buryianton)
