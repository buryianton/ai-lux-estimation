"""
Evaluation metrics for illuminance estimation.

Primary metric: MAPE (Mean Absolute Percentage Error) — directly interpretable
as percentage deviation from true lux, aligned with occupational health standards.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    roc_auc_score, accuracy_score, balanced_accuracy_score,
    confusion_matrix, classification_report
)
from typing import Dict, Union


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Percentage Error (%).

    Excludes zero values from the denominator. The primary metric used
    throughout this study because it is scale-independent and directly
    interpretable in the context of occupational lighting standards.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true > 0
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def regression_metrics(y_true: np.ndarray,
                       y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute all regression metrics used in the paper.

    Returns
    -------
    dict with keys: MAPE, MAE, RMSE, R2
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        'MAPE': round(mape(y_true, y_pred), 3),
        'MAE':  round(mean_absolute_error(y_true, y_pred), 2),
        'RMSE': round(np.sqrt(mean_squared_error(y_true, y_pred)), 2),
        'R2':   round(r2_score(y_true, y_pred), 4),
    }


def classification_metrics(y_true: np.ndarray,
                            y_pred_proba: np.ndarray,
                            threshold: float = 0.5) -> Dict[str, float]:
    """
    Compute classification metrics for threshold-based compliance screening.

    Parameters
    ----------
    y_true : array of 0/1 labels (1 = above lux threshold, 0 = below)
    y_pred_proba : predicted probability of class 1
    threshold : decision threshold on predicted probability

    Returns
    -------
    dict with keys: AUC, accuracy, balanced_accuracy, sensitivity, specificity
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred_proba = np.asarray(y_pred_proba, dtype=float)
    y_pred_class = (y_pred_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_class, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        'AUC':               round(roc_auc_score(y_true, y_pred_proba), 4),
        'accuracy':          round(accuracy_score(y_true, y_pred_class), 4),
        'balanced_accuracy': round(balanced_accuracy_score(y_true, y_pred_class), 4),
        'sensitivity':       round(sensitivity, 4),
        'specificity':       round(specificity, 4),
    }


def mape_by_lux_bin(y_true: np.ndarray,
                    y_pred: np.ndarray,
                    bins: list = None) -> pd.DataFrame:
    """
    Compute MAPE broken down by lux range bins.

    Useful for understanding model accuracy across the measurement range,
    particularly around occupational health thresholds (200, 300, 500 lux).

    Parameters
    ----------
    bins : list of bin edges (default: [0,100,200,300,500,750,1000,9999])
    """
    if bins is None:
        bins = [0, 100, 200, 300, 500, 750, 1000, 9999]

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    labels = [f'{bins[i]}-{bins[i+1]}' for i in range(len(bins)-1)]
    labels[-1] = f'>{bins[-2]}'

    bin_idx = np.digitize(y_true, bins) - 1
    bin_idx = np.clip(bin_idx, 0, len(labels) - 1)

    rows = []
    for i, label in enumerate(labels):
        mask = bin_idx == i
        n = mask.sum()
        if n == 0:
            continue
        ape = np.abs((y_true[mask] - y_pred[mask]) / np.clip(y_true[mask], 1e-6, None)) * 100
        rows.append({
            'lux_bin':    label,
            'n':          int(n),
            'MAPE':       round(float(np.mean(ape)), 2),
            'median_APE': round(float(np.median(ape)), 2),
            'max_APE':    round(float(np.max(ape)), 2),
        })

    return pd.DataFrame(rows)


def print_results_table(results: pd.DataFrame) -> None:
    """Pretty-print a results DataFrame sorted by MAPE."""
    cols = ['subset', 'split', 'model', 'log_space', 'MAPE', 'MAE', 'R2', 'train_n', 'test_n']
    cols = [c for c in cols if c in results.columns]
    print(results.sort_values(['subset', 'split', 'MAPE'])[cols].to_string(index=False))
