"""
ExtraTrees regressor with log-space regression for indoor illuminance estimation.

This is the best-performing model in the study:
  - White paper:        MAPE = 3.28% (session split), 4.55% (physical-point split)
  - Table surfaces:     MAPE = 6.85% (session split, log-space), 8.74% (physical-point split)
  - White + tables:     MAPE = 4.95% (session split, log-space)

Log-space regression
--------------------
Training on log1p(lux) rather than raw lux values consistently reduces MAPE on
table surfaces by ~0.4 percentage points. The improvement is because lux spans
almost two orders of magnitude (48–1833 lux), and a multiplicative error model
(log-space) better matches the physical measurement structure than an additive one.

Grouped splits
--------------
We use session-based and physical-point-based grouped splits (see evaluation/grouped_split.py)
to prevent data leakage. A simple random split would give unrealistically low error
because repeated shots of the same scene are nearly identical.

Usage
-----
    # As a module:
    from models.train_extratrees import train_evaluate_extratrees

    results = train_evaluate_extratrees(
        csv_path='path/to/feature_master.csv',
        subset='white_paper',
        split_strategy='session',
        log_space=False
    )

    # As a script (reproduces all paper results):
    python models/train_extratrees.py --csv path/to/feature_master.csv --all
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import ExtraTreesRegressor

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.grouped_split import (
    grouped_split, prepare_xy, get_feature_columns, get_subset
)
from evaluation.metrics import regression_metrics, print_results_table


# ── Default hyperparameters ───────────────────────────────────────────────────
# These are fixed (not tuned) — ExtraTrees is remarkably robust to hyperparameter
# choice on tabular data of this size.

ET_PARAMS = dict(
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
)

SPLIT_COLS = {
    'session':           'session',
    'physical_point_id': 'physical_point_id',
}

SUBSETS = ['white_paper', 'table', 'white_plus_tables']


# ── Core training/evaluation function ────────────────────────────────────────

def train_evaluate_extratrees(csv_path: str,
                              subset: str = 'white_paper',
                              split_strategy: str = 'session',
                              log_space: bool = False,
                              target_col: str = 'target_lux',
                              surface_col: str = 'surface_group',
                              test_size: float = 0.2,
                              random_state: int = 42,
                              verbose: bool = True) -> dict:
    """
    Train and evaluate an ExtraTrees regressor on the illuminance dataset.

    Parameters
    ----------
    csv_path : str
        Path to the master feature CSV (e.g. optionB_master_enriched37_with_pointid_grid5.csv).
    subset : str
        Surface subset: 'white_paper', 'table', 'white_plus_tables', or 'all'.
    split_strategy : str
        'session' or 'physical_point_id'.
    log_space : bool
        Train on log1p(lux) and back-transform predictions.
    target_col : str
        Name of the lux target column.
    surface_col : str
        Name of the surface group column.

    Returns
    -------
    dict with keys: model, feature_cols, metrics (MAPE, MAE, RMSE, R2),
                    y_test, y_pred, df_train, df_test
    """
    # Load data
    df = pd.read_csv(csv_path, low_memory=False)
    df_sub = get_subset(df, subset, surface_col)

    if len(df_sub) < 20:
        raise ValueError(f"Too few rows in subset '{subset}': {len(df_sub)}")

    split_col = SPLIT_COLS[split_strategy]
    if split_col not in df_sub.columns:
        raise ValueError(f"Split column '{split_col}' not found in CSV.")

    feature_cols = get_feature_columns(df_sub, target_col=target_col)

    # Drop rows with missing features or target
    cols_needed = feature_cols + [target_col, split_col]
    df_clean = df_sub[[c for c in cols_needed if c in df_sub.columns]].dropna()

    # Grouped split
    df_train, df_test = grouped_split(df_clean, split_col, test_size, random_state)

    X_train, X_test, y_train, y_test, y_train_fit = prepare_xy(
        df_train, df_test, feature_cols, target_col, log_space
    )

    # Train
    model = ExtraTreesRegressor(**ET_PARAMS)
    model.fit(X_train, y_train_fit)

    # Predict + back-transform
    y_pred = model.predict(X_test)
    if log_space:
        y_pred = np.expm1(y_pred)

    metrics = regression_metrics(y_test, y_pred)

    if verbose:
        tag = '(log)' if log_space else '     '
        print(f"ExtraTrees {tag} | {subset:<20} | split={split_strategy:<20} | "
              f"MAPE={metrics['MAPE']:6.2f}%  MAE={metrics['MAE']:6.1f}  "
              f"R²={metrics['R2']:.4f}  "
              f"[train={len(df_train)}, test={len(df_test)}]")

    return {
        'model':        model,
        'feature_cols': feature_cols,
        'metrics':      metrics,
        'y_test':       y_test,
        'y_pred':       y_pred,
        'df_train':     df_train,
        'df_test':      df_test,
        'log_space':    log_space,
        'subset':       subset,
        'split':        split_strategy,
    }


def plot_predicted_vs_actual(result: dict,
                             save_path: str = None,
                             title: str = None) -> None:
    """Scatter plot of predicted vs. actual lux values."""
    y_test = result['y_test']
    y_pred = result['y_pred']
    metrics = result['metrics']

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, y_pred, alpha=0.5, s=20, color='steelblue', edgecolors='none')

    lims = [
        min(y_test.min(), y_pred.min()) * 0.92,
        max(y_test.max(), y_pred.max()) * 1.05,
    ]
    ax.plot(lims, lims, 'r--', lw=1.5, label='Perfect prediction')
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel('Measured lux (ground truth)', fontsize=12)
    ax.set_ylabel('Predicted lux', fontsize=12)

    log_tag = ' + log-space' if result['log_space'] else ''
    t = title or (
        f"ExtraTrees{log_tag} — {result['subset']} "
        f"(split={result['split']})\n"
        f"MAPE={metrics['MAPE']:.2f}%  "
        f"MAE={metrics['MAE']:.1f} lux  "
        f"R²={metrics['R2']:.4f}  "
        f"n_test={len(y_test)}"
    )
    ax.set_title(t, fontsize=10)
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()


# ── Batch run (reproduce all paper results) ───────────────────────────────────

def run_all(csv_path: str, save_dir: str = None) -> pd.DataFrame:
    """
    Reproduce all ExtraTrees results reported in the paper.
    Tests all combinations of subset × split × log_space.
    """
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    all_results = []
    print("=" * 90)
    print("ExtraTrees full benchmark (reproduces paper Tables 1 & 2)")
    print("=" * 90)

    for subset in SUBSETS:
        for split in ['session', 'physical_point_id']:
            for log in [False, True]:
                try:
                    r = train_evaluate_extratrees(
                        csv_path=csv_path,
                        subset=subset,
                        split_strategy=split,
                        log_space=log,
                        verbose=True
                    )
                    row = {
                        'model': 'ExtraTrees',
                        'subset': subset,
                        'split': split,
                        'log_space': log,
                        'train_n': len(r['df_train']),
                        'test_n':  len(r['df_test']),
                        **r['metrics']
                    }
                    all_results.append(row)

                    if save_dir and split == 'session' and log:
                        plot_predicted_vs_actual(
                            r,
                            save_path=f"{save_dir}/scatter_{subset}_{split}_log.png"
                        )
                except Exception as e:
                    print(f"  ⚠ Skipped {subset}/{split}/log={log}: {e}")

    results_df = pd.DataFrame(all_results)
    print("\n" + "=" * 90)
    print("SUMMARY (best per subset+split):")
    print("=" * 90)
    best = results_df.loc[results_df.groupby(['subset', 'split'])['MAPE'].idxmin()]
    print_results_table(best)

    if save_dir:
        results_df.to_csv(f"{save_dir}/extratrees_results.csv", index=False)
        print(f"\nSaved full results → {save_dir}/extratrees_results.csv")

    return results_df


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train and evaluate ExtraTrees for indoor illuminance estimation'
    )
    parser.add_argument('--csv',      required=True,
                        help='Path to master feature CSV')
    parser.add_argument('--subset',   default='white_paper',
                        choices=['white_paper', 'table', 'white_plus_tables', 'all'])
    parser.add_argument('--split',    default='session',
                        choices=['session', 'physical_point_id'])
    parser.add_argument('--log',      action='store_true',
                        help='Use log-space regression')
    parser.add_argument('--all',      action='store_true',
                        help='Run all subsets/splits (reproduces paper results)')
    parser.add_argument('--save-dir', default=None,
                        help='Directory to save plots and result CSVs')
    args = parser.parse_args()

    if args.all:
        run_all(args.csv, save_dir=args.save_dir)
    else:
        result = train_evaluate_extratrees(
            csv_path=args.csv,
            subset=args.subset,
            split_strategy=args.split,
            log_space=args.log,
            verbose=True
        )
        if args.save_dir:
            plot_predicted_vs_actual(
                result,
                save_path=f"{args.save_dir}/scatter_{args.subset}_{args.split}.png"
            )
