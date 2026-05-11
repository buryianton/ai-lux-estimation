"""
Threshold-based illuminance classification for occupational health compliance screening.

Background
----------
Russian workplace standard SP 52.13330.2016 specifies minimum illuminance levels
for different task categories (e.g., 300 lux for general office work, 500 lux for
fine visual tasks). Rather than estimating the exact lux value, a binary classifier
can determine whether illuminance is above or below the required threshold — a
simpler and more reliable task that is directly actionable for compliance screening.

Results from this study
-----------------------
  Threshold = 200 lux, white_plus_tables, physical-point split:
    AUC = 0.996, accuracy = 0.97, balanced accuracy = 0.93  (n_test = 422)

  Threshold = 500 lux, white_paper, session split:
    AUC = 1.000, accuracy = 0.995, balanced accuracy = 0.994  (n_test = 220)

  Threshold = 200 lux (5-point mean), white_paper, session split:
    AUC ≈ 1.000, accuracy = 1.000  (n_test = 31 sessions)

Why classification outperforms regression for compliance
--------------------------------------------------------
Regression must estimate the exact lux value despite reflectance confounding.
Classification only needs to determine which side of the threshold the scene falls on,
which is a much lower-information requirement. Near the threshold, errors matter;
far from it, the model is very reliable. AUC > 0.99 means the model correctly
separates "pass" from "fail" scenes with near-perfect reliability.

Usage
-----
    from models.threshold_classification import train_threshold_classifier

    results = train_threshold_classifier(
        csv_path='path/to/feature_master.csv',
        threshold_lux=300,
        subset='white_plus_tables',
        split_strategy='physical_point_id'
    )
    print(f"AUC: {results['metrics']['AUC']:.4f}")
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import RocCurveDisplay

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.grouped_split import (
    grouped_split, get_feature_columns, get_subset
)
from evaluation.metrics import classification_metrics


# ── Hyperparameters ───────────────────────────────────────────────────────────

RF_PARAMS = dict(
    n_estimators=300,
    class_weight='balanced',   # important: threshold imbalance varies by lux level
    random_state=42,
    n_jobs=-1,
)

# Thresholds of occupational health relevance (lux)
STANDARD_THRESHOLDS = {
    200:  "Minimum for casual visual tasks (SP 52.13330)",
    300:  "General office work (SP 52.13330, category IV)",
    500:  "Fine visual tasks / reading (SP 52.13330, category III)",
    750:  "Precision work (SP 52.13330, category II)",
}


# ── Core function ─────────────────────────────────────────────────────────────

def train_threshold_classifier(csv_path: str,
                               threshold_lux: float = 300,
                               subset: str = 'white_plus_tables',
                               split_strategy: str = 'physical_point_id',
                               target_col: str = 'target_lux',
                               surface_col: str = 'surface_group',
                               test_size: float = 0.2,
                               random_state: int = 42,
                               model_type: str = 'random_forest',
                               verbose: bool = True) -> dict:
    """
    Train a binary classifier: above vs. below a lux threshold.

    Parameters
    ----------
    threshold_lux : float
        Lux threshold for binary label (1 = above, 0 = below).
    subset : str
        'white_paper', 'table', 'white_plus_tables', or 'all'.
    split_strategy : str
        'session' or 'physical_point_id'.
    model_type : str
        'random_forest' or 'extra_trees'.

    Returns
    -------
    dict with keys: model, feature_cols, metrics, y_test, y_pred_proba,
                    df_train, df_test, threshold_lux
    """
    df = pd.read_csv(csv_path, low_memory=False)
    df_sub = get_subset(df, subset, surface_col)

    split_col = {'session': 'session', 'physical_point_id': 'physical_point_id'}[split_strategy]
    feature_cols = get_feature_columns(df_sub, target_col=target_col)

    cols_needed = feature_cols + [target_col, split_col]
    df_clean = df_sub[[c for c in cols_needed if c in df_sub.columns]].dropna()

    # Create binary labels
    df_clean = df_clean.copy()
    df_clean['label'] = (df_clean[target_col] >= threshold_lux).astype(int)

    n_pos = df_clean['label'].sum()
    n_neg = len(df_clean) - n_pos
    if verbose:
        print(f"Class balance: {n_pos} above / {n_neg} below {threshold_lux} lux "
              f"({100*n_pos/len(df_clean):.1f}% positive)")

    df_train, df_test = grouped_split(df_clean, split_col, test_size, random_state)

    feat_cols_present = [c for c in feature_cols if c in df_train.columns]
    X_train = df_train[feat_cols_present].values
    X_test  = df_test[feat_cols_present].values
    y_train = df_train['label'].values
    y_test  = df_test['label'].values

    # Model
    if model_type == 'extra_trees':
        model = ExtraTreesClassifier(**RF_PARAMS)
    else:
        model = RandomForestClassifier(**RF_PARAMS)

    model.fit(X_train, y_train)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    metrics = classification_metrics(y_test, y_pred_proba)

    if verbose:
        desc = STANDARD_THRESHOLDS.get(int(threshold_lux), f"{threshold_lux} lux")
        print(f"\n{model_type} | {subset} | split={split_strategy} | "
              f"threshold={threshold_lux} lux ({desc})")
        print(f"  AUC={metrics['AUC']:.4f}  "
              f"accuracy={metrics['accuracy']:.4f}  "
              f"balanced_accuracy={metrics['balanced_accuracy']:.4f}  "
              f"sensitivity={metrics['sensitivity']:.4f}  "
              f"specificity={metrics['specificity']:.4f}")
        print(f"  [train={len(df_train)}, test={len(df_test)}]")

    return {
        'model':         model,
        'feature_cols':  feat_cols_present,
        'metrics':       metrics,
        'y_test':        y_test,
        'y_pred_proba':  y_pred_proba,
        'df_train':      df_train,
        'df_test':       df_test,
        'threshold_lux': threshold_lux,
        'subset':        subset,
        'split':         split_strategy,
    }


def plot_roc_curve(result: dict, save_path: str = None) -> None:
    """Plot ROC curve for the threshold classifier."""
    from sklearn.metrics import RocCurveDisplay
    fig, ax = plt.subplots(figsize=(6, 6))
    RocCurveDisplay.from_predictions(
        result['y_test'], result['y_pred_proba'],
        name=f"threshold={result['threshold_lux']} lux",
        ax=ax
    )
    auc = result['metrics']['AUC']
    ax.set_title(
        f"ROC Curve — Pass/Fail Classification at {result['threshold_lux']} lux\n"
        f"{result['subset']}, split={result['split']}, AUC={auc:.4f}",
        fontsize=10
    )
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()


def run_all_thresholds(csv_path: str, save_dir: str = None) -> pd.DataFrame:
    """
    Run threshold classification for all standard lux thresholds and subsets.
    Reproduces the threshold classification results from the paper.
    """
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    configs = [
        # (threshold, subset, split)
        (200, 'white_plus_tables', 'physical_point_id'),  # main result
        (200, 'white_plus_tables', 'session'),
        (500, 'white_paper',       'session'),
        (500, 'white_paper',       'physical_point_id'),
        (300, 'white_plus_tables', 'session'),
        (300, 'white_plus_tables', 'physical_point_id'),
    ]

    print("=" * 90)
    print("Threshold classification — all configurations")
    print("=" * 90)

    rows = []
    for threshold, subset, split in configs:
        r = train_threshold_classifier(
            csv_path=csv_path,
            threshold_lux=threshold,
            subset=subset,
            split_strategy=split,
            verbose=True
        )
        rows.append({
            'threshold_lux': threshold,
            'subset': subset,
            'split': split,
            'train_n': len(r['df_train']),
            'test_n':  len(r['df_test']),
            **r['metrics']
        })

    results_df = pd.DataFrame(rows)
    print("\nFull classification results:")
    cols = ['threshold_lux', 'subset', 'split', 'AUC', 'accuracy',
            'balanced_accuracy', 'sensitivity', 'specificity', 'train_n', 'test_n']
    print(results_df[cols].to_string(index=False))

    if save_dir:
        results_df.to_csv(f"{save_dir}/threshold_classification_results.csv", index=False)

    return results_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Threshold-based illuminance compliance classification'
    )
    parser.add_argument('--csv',       required=True)
    parser.add_argument('--threshold', type=float, default=300,
                        help='Lux threshold (default: 300)')
    parser.add_argument('--subset',    default='white_plus_tables')
    parser.add_argument('--split',     default='physical_point_id')
    parser.add_argument('--all',       action='store_true',
                        help='Run all standard threshold configurations')
    parser.add_argument('--roc',       action='store_true',
                        help='Plot ROC curve')
    parser.add_argument('--save-dir',  default=None)
    args = parser.parse_args()

    if args.all:
        run_all_thresholds(args.csv, save_dir=args.save_dir)
    else:
        result = train_threshold_classifier(
            csv_path=args.csv,
            threshold_lux=args.threshold,
            subset=args.subset,
            split_strategy=args.split,
            verbose=True
        )
        if args.roc:
            save_path = (f"{args.save_dir}/roc_{args.subset}_{int(args.threshold)}lux.png"
                         if args.save_dir else None)
            plot_roc_curve(result, save_path=save_path)
