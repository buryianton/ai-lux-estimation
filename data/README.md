# Dataset Description

## Overview

The dataset consists of **5,814 RGB photographs** collected under controlled conditions
to train machine-learning models for indoor illuminance estimation.

**Camera:** Samsung Galaxy A16  
**Fixed settings:** ISO 50, 1/50 s exposure, white balance 4000 K, 45° tilt, 31 cm height above surface  
**Lux meter:** Lutron LX-101A  
**Total images:** 5,814 (2,300 with central measurement only; 3,514 with full 5-point lux readings)  
**Illuminance range:** 2–2,280 lux (central measurements); 1–2,280 lux (all measured points)

## Access

The full dataset (images + annotations + feature table) is hosted on Zenodo:

**[→ Download from Zenodo](https://doi.org/10.5281/zenodo.20499312)**

The master annotation CSV:
```
annotation_master_latest.csv  — 5,814 rows × 23 columns
```

The master feature CSV used for all experiments:
```
feature_master_latest.csv  — 5,814 rows × 133 columns
```

## Surface Conditions

| Surface group     | 1-point | 5-point | Total | Description |
|---|---|---|---|---|
| colored_paper     | 946     | 1,430   | 2,376 | 22 colored paper sheets (blush, burgundy, cool red, dark indigo, deep blue, forest green, grass green, lavender, light green, magenta, maroon, midnight blue, navy blue, orange, purple, red, rose, royal blue, steel blue, teal, yellow, and others) |
| white_paper       | 677     | 886     | 1,563 | Standard white reference sheet |
| table             | 677     | 828     | 1,505 | Bare table surfaces (various materials and colors) |
| other             | 0       | 370     | 370   | Printed materials and additional surface types |
| **Total**         | **2,300** | **3,514** | **5,814** | |

## Lighting Conditions

Two adjustable LED lamps were used in separate rooms:
- **Lamp 1:** 3 color temperatures (warm ~2700 K, neutral ~4000 K, cool ~6500 K)
- **Lamp 2:** 4 color temperatures

Each lamp's dimmer was used to generate multiple illuminance levels per color temperature,
spanning from dim (~2 lux) to bright (~2280 lux) indoor conditions.

## Measurement Grid

Each 5-point photograph documents illuminance at **5 locations** using a fixed grid:

```
  UL -------- UR
  |           |
  |     C     |
  |           |
  LL -------- LR
```

- **C:** center of the image
- **UL, UR, LR, LL:** four corner intersections of a 3×3 grid overlay

1-point images record only the center (C) measurement.

The camera grid overlay (displayed on-screen during shooting) ensured that all
measurement points were photographically consistent across sessions.

## Annotation Format

Lux values are stored in `annotation_master_latest.csv` with one row per image.
Key columns: `lux_C`, `lux_UL`, `lux_UR`, `lux_LR`, `lux_LL`, `label_kind` (1point/5point),
`surface_group`, `session`, `physical_point_id`.

The column `target_lux` holds the lux reading for the center point (C), which is the
primary prediction target throughout this study.

## Data Collection Protocol

1. Set lamp to target color temperature and dimmer level
2. Record lux at all 5 grid points with Lutron LX-101A
3. Immediately capture photograph (camera settings locked)
4. Repeat for all surface conditions in the session
5. Complete the full procedure twice with a slight camera position perturbation

Sessions were tracked with a unique `session` identifier; unique spatial positions
with `physical_point_id`.

## Key Columns in the Feature CSV

| Column | Description |
|---|---|
| `target_lux` | Center-point lux (primary prediction target) |
| `surface_group` | `white_paper` / `table` / `colored_paper` / `other` |
| `label_kind` | `1point` (center only) or `5point` (center + 4 corners) |
| `session` | Unique session ID (used for grouped train-test split) |
| `physical_point_id` | Unique spatial position ID (used for strict grouped split) |
| `C_mean_luma` … `LL_grad_std` | Per-ROI image features (5 ROIs × basic stats) |
| `square_*` | Square context region features |
| `square_cell_r{i}_c{j}_*` | 5×5 grid cell features (25 cells) |
| `*_minus_*` | Spatial difference features |

Total: 100 numeric feature columns (columns 24–133).

## Version History

See [CHANGELOG.md](CHANGELOG.md) for details on dataset corrections between versions.

