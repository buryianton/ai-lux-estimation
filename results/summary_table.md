# Results Summary (v4 Dataset)

All results use the corrected v4 dataset (5,814 images).
Reproduced by running `models/train_extratrees.py --all` and `models/train_xgboost.py --all`
with `feature_master_latest.csv`.

## Table 1. Regression — White paper surfaces

| Model | Split | MAPE (%) | MAE (lux) | RMSE | R² | n_train | n_test |
|---|---|---|---|---|---|---|---|
| ExtraTrees (5pt) | session | **4.13** | 28.3 | 43.2 | 0.9863 | 709 | 177 |
| ExtraTrees (5pt) | physical_point_id | **3.81** | 24.5 | 35.2 | 0.9891 | 792 | 94 |
| ExtraTrees (all labels) | session | **3.61** | 24.5 | 37.7 | 0.9899 | 1,251 | 312 |
| ExtraTrees (all labels) | physical_point_id | 5.20 | 46.7 | 85.4 | 0.9682 | 1,300 | 263 |

## Table 2. Regression — Table surfaces (5-point labels only)

| Model | Split | MAPE (%) | MAE (lux) | RMSE | R² | n_train | n_test |
|---|---|---|---|---|---|---|---|
| ExtraTrees | session | 16.42 | 46.5 | 76.8 | 0.9569 | 662 | 166 |
| ExtraTrees | physical_point_id | 16.15 | 75.7 | 100.0 | 0.9128 | 733 | 95 |

## Table 3. Regression — White paper + table surfaces (5-point labels)

| Model | Split | MAPE (%) | MAE (lux) | RMSE | R² | n_train | n_test |
|---|---|---|---|---|---|---|---|
| ExtraTrees | session | **6.01** | 33.2 | 48.2 | 0.9804 | 1,373 | 341 |
| ExtraTrees | physical_point_id | 10.08 | 51.5 | 78.4 | 0.9461 | 1,525 | 189 |

## Table 4. Regression — Per table type (5-point labels, session split)

| Table type | MAPE (%) | MAE (lux) | R² | n_train | n_test | Note |
|---|---|---|---|---|---|---|
| brown_wood_like | 31.81* | 32.6 | 0.978 | 261 | 67 | *MAPE artifact at low lux |
| laminate | 6.60 | 43.5 | 0.975 | 309 | 78 | |
| white_table | 3.99 | 17.9 | 0.994 | 64 | 17 | |

## Table 5. Cross-surface generalization

| Train → Test | MAPE (%) | MAE (lux) | R² | n_train | n_test |
|---|---|---|---|---|---|
| white_paper → white_table | **3.52** | 22.2 | 0.993 | 886 | 81 |
| white_table → white_paper | 10.28 | 45.2 | 0.968 | 81 | 886 |
| white_paper → all_tables | 58.44 | 405.1 | -0.921 | 886 | 828 |
| all_tables → white_paper | 23.53 | 120.2 | 0.745 | 828 | 886 |
| brown → laminate | 45.18 | 225.6 | 0.575 | 328 | 387 |
| laminate → brown | 55.28 | 292.6 | -0.232 | 387 | 328 |
| laminate → white_table | 124.85 | 544.2 | -1.802 | 387 | 81 |

## Table 6. Threshold classification results

| Threshold (lux) | Subset | Split | AUC | Accuracy | Balanced Acc. | Correct | n_test |
|---|---|---|---|---|---|---|---|
| 500 | white_paper (5pt) | session | 0.999 | 0.977 | 0.974 | 173/177 | 177 |
| 500 | white_paper (5pt) | physical_point_id | 0.996 | 0.957 | 0.957 | 90/94 | 94 |
| 200 | white_paper (5pt) | session | 1.000 | 0.994 | 0.969 | 176/177 | 177 |
| 200 | table (5pt) | session | 0.986 | 0.988 | 0.917 | 164/166 | 166 |
| 200 | white+table (5pt) | physical_point_id | 1.000 | 0.984 | 0.992 | 186/189 | 189 |
| 200 | brown_wood_like | session | 0.997 | 0.985 | 0.900 | 66/67 | 67 |
| 200 | laminate | session | 1.000 | 0.974 | 0.875 | 76/78 | 78 |
| 200 | white_table | session | 1.000 | 1.000 | 1.000 | 17/17 | 17 |

## Table 7. Color classification accuracy by illuminance level

Classifier: ExtraTrees, 22 colored paper classes, session split, n=1,428 five-point images.
Overall accuracy: 69.8% (vs. 4.5% random chance for 22 classes).

| Lux bin | Accuracy | n images |
|---|---|---|
| 100–200 | 9.1% | 22 |
| 200–300 | 43.2% | 44 |
| 300–500 | 87.3% | 110 |
| 500–750 | **93.9%** | 66 |
| 750–1,000 | 77.3% | 44 |
| >1,000 | 9.1% | 22 |

Peak accuracy at 500–750 lux aligns with ISO 3664 (500 lux for practical color appraisal)
and EN 12464-1 (300–750 lux for color-critical work).

## Top 10 Features (ExtraTrees, table-only, session split)

| Rank | Feature | Physical interpretation |
|---|---|---|
| 1 | LL_mean_luma | Mean brightness at lower-left ROI |
| 2 | LL_mean_r | Red channel at lower-left ROI |
| 3 | square_cell_r4_c1_mean_luma | Grid cell brightness (bottom region) |
| 4 | sq_r4_c1_mean_r | Red channel in bottom grid row |
| 5 | sq_r4_c1_mean_luma | Luma in bottom grid row |
| 6 | sq_r4_c2_mean_r | Red channel center-bottom |
| 7 | square_cell_r4_c0_mean_luma | Bottom-left grid cell |
| 8 | sq_r4_c0_mean_luma | Bottom-left luma |
| 9 | sq_r4_c0_mean_r | Bottom-left red channel |
| 10 | square_cell_r4_c2_mean_luma | Bottom-center grid cell |

**Key finding:** Spatial gradient and lower-region features dominate over absolute
brightness values. The model learns to estimate illuminance from the spatial
distribution of brightness across the surface.
