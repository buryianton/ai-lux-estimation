# Dataset Description

## Overview

The dataset consists of **5,814 RGB photographs** collected under controlled conditions
to train machine-learning models for indoor illuminance estimation.

**Camera:** Samsung Galaxy A16  
**Fixed settings:** ISO 50, 1/20 s exposure, white balance 4000 K, 45° tilt, 31 cm height above surface  
**Lux meter:** TES-1330A Digital Lux Meter (ISO 17025-grade)  
**Total annotated images:** 3,157 (with complete 5-point lux readings)  
**Target lux range:** 48–1,833 lux (mean = 608, SD = 295)

## Access

The full dataset (photos + annotations) is hosted on Google Drive due to size:

**[→ Download from Google Drive](https://drive.google.com/drive/folders/18vgAnvUA8uLOCT_fZ1HK85zBR4VqMzUk)**

The master feature CSV used for all experiments is:
```
AI_Lux_Project/Experiments_4/optionB_master_enriched37_with_pointid_grid5.csv
```
Shape: 3,157 rows × 325 columns (294 numeric features + metadata + lux labels).

## Surface Conditions

| Surface group     | n images | Description |
|---|---|---|
| colored_paper     | 1,465    | 24 colored paper sheets (blush, burgundy, cool red, dark indigo, deep blue, forest green, grass green, lavender, light green, magenta, maroon, midnight blue, navy blue, orange, purple, red, rose, royal blue, steel blue, teal, yellow, and others) |
| white_paper       | 1,096    | Standard white reference sheet (≈95% reflectance) |
| table             | 1,013    | 10 bare table surfaces (various materials and colors) |
| other             | 549      | Additional surface types |

## Lighting Conditions

Two adjustable LED lamps were used in separate rooms:
- **Lamp 1:** 3 color temperatures (warm ~2700 K, neutral ~4000 K, cool ~6500 K)
- **Lamp 2:** 4 color temperatures

Each lamp's dimmer was used to generate multiple illuminance levels per color temperature,
spanning from dim (≈50 lux) to bright (≈1800 lux) indoor conditions.

## Measurement Grid

Each photograph documents illuminance at **5 locations** using a fixed grid:

```
  UL -------- UR
  |           |
  |     C     |
  |           |
  LL -------- LR
```

- **C:** center of the image
- **UL, UR, LR, LL:** four corner intersections of a 3×3 grid overlay

The camera grid overlay (displayed on-screen during shooting) ensured that all
measurement points were photographically consistent across sessions.

## Annotation Format

Lux values are stored in `annotations_long_updated.csv` with one row per measurement point
and in the master feature CSV as columns: `lux_C`, `lux_UL`, `lux_UR`, `lux_LR`, `lux_LL`.

The column `target_lux` holds the lux reading for the center point (C), which is the
primary prediction target throughout this study.

## Data Collection Protocol

1. Set lamp to target color temperature and dimmer level
2. Record lux at all 5 grid points with TES-1330A
3. Immediately capture photograph (camera settings locked)
4. Repeat for all surface conditions
5. Complete the full procedure twice with a slight camera position perturbation

Sessions were tracked with a unique `session` identifier; unique spatial positions
with `physical_point_id` (25 unique positions across all sessions).

## Key Files in the Feature CSV

| Column | Description |
|---|---|
| `target_lux` | Center-point lux (primary prediction target) |
| `surface_group` | `white_paper` / `table` / `colored_paper` / `other` |
| `session` | Unique session ID (used for grouped train-test split) |
| `physical_point_id` | Unique spatial position ID (used for strict grouped split) |
| `C_mean_luma` … `LL_mean_sat_proxy` | Per-ROI image features (5 ROIs × 7 features = 35) |
| `square_*` | Square context region features |
| `sq_r{i}_c{j}_*` | 5×5 grid cell features (25 cells × 2 features = 50) |
| `square_cell_r{i}_c{j}_*` | Alternative prefix for grid features |
| `*_minus_*` | Spatial difference features (top-ranked by feature importance) |
