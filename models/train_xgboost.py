"""
XGBoost regressor with log-space regression for indoor illuminance estimation.

XGBoost wins the strictest generalization test:
  Physical-point split, table surfaces:  MAPE = 8.55% (log-space)
  Physical-point split, white + tables:  MAPE = 9.15% (log-space)
  Session split, white + tables:         MAPE = 4.87% (log-space)  ← best overall

XGBoost vs ExtraTrees
---------------------
ExtraTrees wins on session-based splits for most subsets. XGBoost wins on
physical-point splits for table and mixed surfaces, suggesting slightly better
generalization to unseen measurement locations. The difference is small
(0.2–0.6 pp MAPE) and both models should be reported in the paper.

LightGBM was also benchmarked and consistently underperformed ExtraTrees and
XGBoost on dataset sizes < 500 images per subset, likely due to leaf-wise
growth overfitting. LightGBM is not recommended for this dataset size.

Usage
-----
    python models/train_xgboost.py --csv path/to/feature_master.csv --all
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import xgboost as xgb

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.grouped_split import (
    grouped_split, prepare_xy, get_feature_columns, get_subset
)
from evaluation.metrics import regression_metrics, print_results_table


# ── Hyperparameters ───────────────────────────────────────────────────────────
# Chosen via empirical comparison; regularization (reg_alpha, reg_lambda)
# is important for generalization on small subsets.

XGB_PARAMS = dict(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    n_jobs=-1,
    verbosity=0,
)

SUBSETS = ['white_paper', 'table', 'white_plus_tables']


# ── Core function ─────────────────────────────────────────────────────────────

def train_evaluate_xgboost(csv_path: str,
                           subset: str = 'white_paper',
                           split_strategy: str = 'session',
                           log_space: bool = False,
                           target_col: str = 'target_lux',
                           surface_col: str = 'surface_group',
                           test_size: float = 0.2,
                           random_state: int = 42,
                           verbose: bool = True) -> dict:
    """
    Train and evaluate an XGBoost regressor.

    See train_evaluate_extratrees() in train_extratrees.py for parameter docs.
    """
    df = pd.read_csv(csv_path, low_memory=False)
    df_sub = get_subset(df, subset, surface_col)

    split_col = {'session': 'session', 'physical_point_id': 'physical_point_id'}[split_strategy]
    feature_cols = get_feature_columns(df_sub, target_col=target_col)

    cols_needed = feature_cols + [target_col, split_col]
    df_clean = df_sub[[c for c in cols_needed if c in df_sub.columns]].dropna()

    df_train, df_test = grouped_split(df_clean, split_col, test_size, random_state)
    X_train, X_test, y_train, y_test, y_train_fit = prepare_xy(
        df_train, df_test, feature_cols, target_col, log_space
    )

    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(X_train, y_train_fit)

    y_pred = model.predict(X_test)
    if log_space:
        y_pred = np.expm1(y_pred)

    metrics = regression_metrics(y_test, y_pred)

    if verbose:
        tag = '(log)' if log_space else '     '
        print(f"XGBoost    {tag} | {subset:<20} | split={split_strategy:<20} | "
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


def feature_importance_plot(result: dict,
                            top_n: int = 30,
                            save_path: str = None) -> pd.Series:
    """Plot XGBoost feature importance (gain) for the top N features."""
    model = result['model']
    feature_cols = result['feature_cols']

    importances = pd.Series(
        model.feature_importances_, index=feature_cols
    ).sort_values(ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(9, 8))
    importances.plot.barh(ax=ax, color='steelblue')
    ax.invert_yaxis()
    ax.set_xlabel('Feature importance (gain)')
    ax.set_title(
        f'XGBoost — Top {top_n} Features\n'
        f"({result['subset']}, split={result['split']})",
        fontweight='bold'
    )
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()

    print(f"\nTop 10 features:")
    print(importances.head(10).to_string())
    return importances


def run_all(csv_path: str, save_dir: str = None) -> pd.DataFrame:
    """Run all XGBoost experiments (reproduces paper results)."""
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    all_results = []
    print("=" * 90)
    print("XGBoost full benchmark")
    print("=" * 90)

    for subset in SUBSETS:
        for split in ['session', 'physical_point_id']:
            for log in [False, True]:
                try:
                    r = train_evaluate_xgboost(
                        csv_path=csv_path,
                        subset=subset,
                        split_strategy=split,
                        log_space=log,
                        verbose=True
                    )
                    all_results.append({
                        'model': 'XGBoost',
                        'subset': subset,
                        'split': split,
                        'log_space': log,
                        'train_n': len(r['df_train']),
                        'test_n':  len(r['df_test']),
                        **r['metrics']
                    })
                except Exception as e:
                    print(f"  ⚠ Skipped {subset}/{split}/log={log}: {e}")

    results_df = pd.DataFrame(all_results)
    print("\nSUMMARY (best per subset+split):")
    best = results_df.loc[results_df.groupby(['subset', 'split'])['MAPE'].idxmin()]
    print_results_table(best)

    if save_dir:
        results_df.to_csv(f"{save_dir}/xgboost_results.csv", index=False)

    return results_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv',      required=True)
    parser.add_argument('--subset',   default='white_paper',
                        choices=['white_paper', 'table', 'white_plus_tables', 'all'])
    parser.add_argument('--split',    default='session',
                        choices=['session', 'physical_point_id'])
    parser.add_argument('--log',      action='store_true')
    parser.add_argument('--all',      action='store_true')
    parser.add_argument('--importance', action='store_true',
                        help='Plot feature importance for the given subset/split')
    parser.add_argument('--save-dir', default=None)
    args = parser.parse_args()

    if args.all:
        run_all(args.csv, save_dir=args.save_dir)
    else:
        result = train_evaluate_xgboost(
            csv_path=args.csv,
            subset=args.subset,
            split_strategy=args.split,
            log_space=args.log,
            verbose=True
        )
        if args.importance:
            save_path = f"{args.save_dir}/xgb_importance_{args.subset}.png" \
                        if args.save_dir else None
            feature_importance_plot(result, save_path=save_path)
