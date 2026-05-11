# Results Summary

All results reported in the paper, reproduced by running `models/train_extratrees.py --all`
and `models/train_xgboost.py --all`.

## Table 1. Model performance on white-paper surfaces

| Model | Log-space | Split | MAPE (%) | MAE (lux) | R² | n_train | n_test |
|---|---|---|---|---|---|---|---|
| ExtraTrees | No | session | **3.28** | 20.5 | 0.9927 | 538 | 135 |
| ExtraTrees | Yes | session | 3.34 | 20.7 | 0.9925 | 538 | 135 |
| XGBoost | Yes | session | 3.42 | 21.1 | 0.9929 | 538 | 135 |
| XGBoost | No | session | 3.45 | 21.8 | 0.9915 | 538 | 135 |
| LightGBM | Yes | session | 3.46 | 20.7 | 0.9933 | 538 | 135 |
| RandomForest | Yes | session | 3.68 | 23.3 | 0.9905 | 538 | 135 |
| RandomForest | No | session | 3.77 | 23.5 | 0.9905 | 538 | 135 |
| LightGBM | No | session | 4.29 | 25.0 | 0.9896 | 538 | 135 |
| ExtraTrees | No | physical_point_id | **4.55** | 34.7 | 0.9852 | 520 | 153 |
| XGBoost | Yes | physical_point_id | 4.59 | 31.9 | 0.9862 | 520 | 153 |
| ExtraTrees | Yes | physical_point_id | 4.67 | 35.1 | 0.9847 | 520 | 153 |
| RandomForest | No | physical_point_id | 4.81 | 36.3 | 0.9840 | 520 | 153 |
| XGBoost | No | physical_point_id | 4.84 | 35.5 | 0.9838 | 520 | 153 |
| LightGBM | No | physical_point_id | 4.87 | 34.5 | 0.9857 | 520 | 153 |
| RandomForest | Yes | physical_point_id | 4.89 | 36.7 | 0.9831 | 520 | 153 |
| LightGBM | Yes | physical_point_id | 5.11 | 34.9 | 0.9839 | 520 | 153 |

## Table 2. Model performance on table-only surfaces

| Model | Log-space | Split | MAPE (%) | MAE (lux) | R² | n_train | n_test |
|---|---|---|---|---|---|---|---|
| ExtraTrees | Yes | session | **6.85** | 39.8 | 0.9788 | 374 | 94 |
| XGBoost | Yes | session | 7.01 | 41.9 | 0.9778 | 374 | 94 |
| ExtraTrees | No | session | 7.24 | 38.7 | 0.9780 | 374 | 94 |
| LightGBM | No | session | 7.74 | 48.1 | 0.9671 | 374 | 94 |
| LightGBM | Yes | session | 8.07 | 49.6 | 0.9706 | 374 | 94 |
| XGBoost | No | session | 8.19 | 48.4 | 0.9662 | 374 | 94 |
| RandomForest | Yes | session | 8.50 | 51.5 | 0.9683 | 374 | 94 |
| RandomForest | No | session | 8.92 | 50.9 | 0.9665 | 374 | 94 |
| XGBoost | Yes | physical_point_id | **8.55** | 71.4 | 0.9404 | 367 | 101 |
| ExtraTrees | No | physical_point_id | 8.74 | 68.9 | 0.9454 | 367 | 101 |
| ExtraTrees | Yes | physical_point_id | 8.84 | 69.0 | 0.9443 | 367 | 101 |
| RandomForest | No | physical_point_id | 8.98 | 72.6 | 0.9351 | 367 | 101 |
| XGBoost | No | physical_point_id | 9.10 | 71.0 | 0.9460 | 367 | 101 |
| LightGBM | No | physical_point_id | 9.53 | 73.7 | 0.9418 | 367 | 101 |
| LightGBM | Yes | physical_point_id | 9.59 | 77.1 | 0.9301 | 367 | 101 |
| RandomForest | Yes | physical_point_id | 9.89 | 80.6 | 0.9222 | 367 | 101 |

## Table 3. Model performance on white paper + table surfaces (mixed)

| Model | Log-space | Split | MAPE (%) | MAE (lux) | R² | n_train | n_test |
|---|---|---|---|---|---|---|---|
| XGBoost | Yes | session | **4.87** | 28.3 | 0.9866 | 912 | 229 |
| ExtraTrees | Yes | session | 4.95 | 30.3 | 0.9843 | 912 | 229 |
| ExtraTrees | No | session | 5.20 | 30.8 | 0.9843 | 912 | 229 |
| XGBoost | No | session | 5.23 | 30.8 | 0.9859 | 912 | 229 |
| LightGBM | No | session | 5.45 | 31.3 | 0.9825 | 912 | 229 |
| LightGBM | Yes | session | 5.47 | 32.9 | 0.9809 | 912 | 229 |
| RandomForest | Yes | session | 6.08 | 36.2 | 0.9786 | 912 | 229 |
| RandomForest | No | session | 6.31 | 35.4 | 0.9819 | 912 | 229 |
| ExtraTrees | Yes | physical_point_id | **8.72** | 59.4 | 0.9550 | 835 | 306 |
| XGBoost | Yes | physical_point_id | 9.15 | 61.3 | 0.9523 | 835 | 306 |
| LightGBM | No | physical_point_id | 9.66 | 61.2 | 0.9529 | 835 | 306 |
| RandomForest | Yes | physical_point_id | 9.82 | 67.3 | 0.9446 | 835 | 306 |
| XGBoost | No | physical_point_id | 9.83 | 62.5 | 0.9545 | 835 | 306 |
| ExtraTrees | No | physical_point_id | 9.85 | 63.5 | 0.9524 | 835 | 306 |
| LightGBM | Yes | physical_point_id | 9.85 | 64.1 | 0.9493 | 835 | 306 |
| RandomForest | No | physical_point_id | 11.05 | 68.8 | 0.9469 | 835 | 306 |

## Table 4. Threshold classification results

| Threshold (lux) | Subset | Split | Model | AUC | Accuracy | Balanced Acc. | n_test |
|---|---|---|---|---|---|---|---|
| 200 | white_plus_tables | physical_point_id | RandomForest | 0.996 | 0.970 | 0.930 | 422 |
| 500 | white_paper | session | ExtraTrees | 1.000 | 0.995 | 0.994 | 220 |
| 200 | white_paper (5pt mean) | session | RandomForest | ~1.000 | 1.000 | 1.000 | 31 |

## Top 10 Features (LightGBM, table-only, session split)

| Rank | Feature | Importance (gain) | Physical interpretation |
|---|---|---|---|
| 1 | square_right_minus_left_luma | 1026 | Horizontal lighting gradient across surface |
| 2 | UL_minus_squarecell_0_0_luma | 1000 | Corner ROI vs. grid cell — spatial consistency |
| 3 | C_minus_UL_luma | 735 | Center vs. upper-left corner brightness difference |
| 4 | C_minus_squarecell_2_2_luma | 722 | Center ROI vs. center grid cell |
| 5 | square_bottom_minus_top_luma | 592 | Vertical lighting gradient across surface |
| 6 | UR_minus_squarecell_0_4_luma | 458 | Upper-right corner vs. grid cell |
| 7 | C_minus_LL_luma | 425 | Center vs. lower-left corner |
| 8 | C_minus_LR_luma | 422 | Center vs. lower-right corner |
| 9 | sq_r0_c0_std_luma | 391 | Texture in upper-left grid cell |
| 10 | UL_grad_mean | 370 | Gradient magnitude at upper-left ROI |

**Key finding:** Spatial gradient and difference features dominate over absolute
brightness values. This suggests the model learns to estimate illuminance from
the spatial distribution of brightness across the surface — partially decoupling
from surface reflectance, which confounds absolute brightness.
