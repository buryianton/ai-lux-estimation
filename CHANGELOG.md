# Changelog

## v2.0.0 — Dataset Correction Release

### Dataset corrections (annotation_master_latest.csv, feature_master_latest.csv)

The annotation and feature files have been substantially corrected. All lux
measurements, image features, and image files are unchanged. Only metadata
columns were affected.

**surface_group corrections (494 rows)**
- 468 rows previously classified as `other` (malformed `ul__ur__lr__ll__table`
  surface names) reclassified to `table`
- 24 rows with `table_lc###` surface names reclassified to `table`
- 2 rows (`yellow_paper (2)`) reclassified from `other` to `colored_paper`

**label_kind corrections (935 rows)**
- 935 rows marked `1point` were found to contain all four corner lux values
  (UL, UR, LR, LL) in their filenames; reclassified to `5point` and corner
  lux columns populated from filename

**surface column corrections (935 rows)**
- Malformed surface values (`ul536__ur408__lr536__ll684__table`) corrected to
  clean values (`table`, `white_paper`) parsed from filename

**white_paper group correction (467 rows)**
- 467 images with surface=`white_paper` photographed during table sessions
  were miscategorized as `colored_paper`; reclassified to `white_paper`

**surface_group_norm sync**
- `surface_group_norm` column was stale (961 conflicts with `surface_group`);
  synced to match `surface_group` exactly

**surface name normalization (24 rows)**
- `table_lc###` surface names normalized to `table`

### Updated dataset counts

| Surface category    | 1-point | 5-point | Total |
|---------------------|---------|---------|-------|
| Colored paper       | 946     | 1,430   | 2,376 |
| White paper         | 677     | 886     | 1,563 |
| Real table surfaces | 677     | 828     | 1,505 |
| Other surfaces      | 0       | 370     | 370   |
| **Total**           | **2,300** | **3,514** | **5,814** |

Previous (v1) counts were:
colored_paper=2,841 / white_paper=1,096 / table=1,013 / other=864 /
1point=3,235 / 5point=2,579

### Updated illuminance range
- Central measurements: 2–2,280 lux (previously reported as 48–1,833 lux)
- All measured points including corners: 1–2,280 lux

### Code changes
- Updated docstrings in `models/train_extratrees.py`,
  `models/train_xgboost.py`, and `models/threshold_classification.py`
  to reflect v4 dataset results and corrected lux range
- Fixed stale filename reference (`optionB_master_enriched37...` →
  `feature_master_latest.csv`) in `train_extratrees.py`
- No changes to code logic; all results recomputed from corrected dataset

### Updated key results (v4 dataset)

**Regression — ExtraTrees:**

| Subset               | Split              | MAPE   | R²     |
|----------------------|--------------------|--------|--------|
| White paper (5pt)    | session            | 4.13%  | 0.986  |
| White paper (5pt)    | physical_point_id  | 3.81%  | 0.989  |
| White paper (all)    | session            | 3.61%  | 0.990  |
| Table (5pt)          | session            | 16.42% | 0.957  |
| White+Table (5pt)    | session            | 6.01%  | 0.980  |
| White+Table (5pt)    | physical_point_id  | 10.08% | 0.946  |

**Classification:**

| Subset            | Threshold | Split             | AUC   | Accuracy |
|-------------------|-----------|-------------------|-------|----------|
| White paper (5pt) | 500 lux   | session           | 0.999 | 0.977    |
| White paper (5pt) | 200 lux   | session           | 1.000 | 0.994    |
| Table (5pt)       | 200 lux   | session           | 0.986 | 0.988    |
| White+Table (5pt) | 200 lux   | physical_point_id | 1.000 | 0.984    |

---

## v1.0.0 — Initial release
