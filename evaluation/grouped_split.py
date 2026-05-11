"""
Grouped train-test split strategies for illuminance estimation evaluation.

Two strategies are used to prevent data leakage from repeated measurements:

1. Session split: all images from the same measurement session go entirely
   to train OR test. Prevents leakage from repeated shots of the same scene.

2. Physical-point split: all images at the same spatial measurement location
   (physical_point_id) go entirely to train OR test. The strictest evaluation —
   tests generalization to new measurement positions.

Why grouped splits matter
--------------------------
A simple random 80/20 split would leak information: if photo #1 and photo #2
are taken at the same physical point under the same lamp at slightly different
camera positions, their lux values are nearly identical. A random split puts
#1 in train and #2 in test, causing unrealistically low test error.
Grouped splits prevent this entirely.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from typing import Tuple, Optional


def grouped_split(df: pd.DataFrame,
                  group_col: str,
                  test_size: float = 0.2,
                  random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a DataFrame such that all rows with the same group value are
    entirely in train or entirely in test.

    Parameters
    ----------
    df : pd.DataFrame
        Input data (must contain group_col).
    group_col : str
        Column to group by (e.g. 'session' or 'physical_point_id').
    test_size : float
        Fraction of groups to hold out for testing.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    (df_train, df_test) : Tuple[pd.DataFrame, pd.DataFrame]
    """
    groups = df[group_col].values
    n_groups = len(np.unique(groups))

    if n_groups < 3:
        raise ValueError(
            f"Need at least 3 unique groups in '{group_col}' for a grouped split, "
            f"got {n_groups}."
        )

    splitter = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    train_idx, test_idx = next(splitter.split(df, groups=groups))
    return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()


def prepare_xy(df_train: pd.DataFrame,
               df_test: pd.DataFrame,
               feature_cols: list,
               target_col: str,
               log_space: bool = False):
    """
    Extract X and y arrays from train/test DataFrames.

    Parameters
    ----------
    log_space : bool
        If True, return log1p-transformed y_train for training and
        the original y_test for evaluation. Predictions must be
        back-transformed with np.expm1() before computing metrics.

    Returns
    -------
    X_train, X_test, y_train, y_test, y_train_fit
        y_train     = original training targets (for reference)
        y_train_fit = targets used for model.fit() (log-transformed if log_space=True)
        y_test      = original test targets (always in original scale)
    """
    feat_cols_present = [c for c in feature_cols if c in df_train.columns]

    X_train = df_train[feat_cols_present].values
    X_test  = df_test[feat_cols_present].values
    y_train = df_train[target_col].values.astype(float)
    y_test  = df_test[target_col].values.astype(float)

    if log_space:
        y_train_fit = np.log1p(np.clip(y_train, 1e-3, None))
    else:
        y_train_fit = y_train

    return X_train, X_test, y_train, y_test, y_train_fit


def get_feature_columns(df: pd.DataFrame,
                        target_col: str = 'target_lux',
                        exclude_extra: Optional[list] = None) -> list:
    """
    Identify numeric feature columns by excluding metadata and target columns.

    Parameters
    ----------
    exclude_extra : list, optional
        Additional columns to exclude beyond the standard set.

    Returns
    -------
    list of feature column names
    """
    # Standard non-feature columns in the AI Lux project dataset
    STANDARD_EXCLUDE = {
        target_col,
        'surface_group', 'surface_type', 'surface', 'surface_raw',
        'surface_norm', 'surface_group_norm', 'family_group',
        'session', 'physical_point_id', 'point', 'label_kind',
        'image_path', 'filename', 'table_type', 'table_base',
        'is_main_transfer_surface',
        # Individual lux readings at each point (not to be used as features —
        # they would constitute direct data leakage of the target variable)
        'lux_C', 'lux_UL', 'lux_UR', 'lux_LR', 'lux_LL',
        # Pixel coordinates of ROI centers (not image-derived features)
        'x_C', 'y_C', 'x_UL', 'y_UL', 'x_UR', 'y_UR',
        'x_LR', 'y_LR', 'x_LL', 'y_LL',
    }

    if exclude_extra:
        STANDARD_EXCLUDE.update(exclude_extra)

    feature_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in STANDARD_EXCLUDE
    ]
    return feature_cols


def get_subset(df: pd.DataFrame,
               subset: str,
               surface_col: str = 'surface_group') -> pd.DataFrame:
    """
    Filter DataFrame to a named surface subset.

    Parameters
    ----------
    subset : str
        One of: 'white_paper', 'table', 'white_plus_tables', 'colored_paper', 'all'
    """
    if subset == 'all':
        return df.copy()
    if subset == 'white_plus_tables':
        return df[df[surface_col].isin(['white_paper', 'table'])].copy()
    return df[df[surface_col] == subset].copy()
