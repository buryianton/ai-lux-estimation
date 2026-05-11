"""
Feature extraction pipeline for AI-based indoor illuminance estimation.

Extracts 294 numeric features from a fixed-geometry smartphone photograph:
  - 5 ROI patches (C, UL, UR, LR, LL): mean RGB, luma, std_luma, grad_mean, sat_proxy
  - Square context region (whole-image crop): same statistics + top/bottom/left/right means
  - 5×5 grid cells within the square context: mean_luma, std_luma per cell
  - Spatial gradient / difference features: right−left, bottom−top, center−corner, diagonals
  - Corner-ROI vs grid-cell differences

Usage:
    from features.extract_features import FeatureExtractor

    extractor = FeatureExtractor(roi_radius=32, square_fraction=0.85, grid_size=5)
    features = extractor.extract_from_image(image_path, roi_coords)
"""

import numpy as np
from PIL import Image
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional


# ── ROI and image parameters ──────────────────────────────────────────────────

ROI_RADIUS   = 32     # px — radius of circular ROI around each measurement point
SQUARE_FRAC  = 0.85   # fraction of shorter image dimension for square context crop
GRID_SIZE    = 5      # n×n grid cells within the square context region


# ── Luma formula (ITU-R BT.601) ──────────────────────────────────────────────

def rgb_to_luma(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Convert R, G, B arrays (0–255 float) to perceptual luminance."""
    return 0.299 * r + 0.587 * g + 0.114 * b


def saturation_proxy(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> float:
    """Simple saturation proxy: max(RGB) - min(RGB), averaged over pixels."""
    stacked = np.stack([r, g, b], axis=-1)
    return float(np.mean(stacked.max(axis=-1) - stacked.min(axis=-1)))


def gradient_mean(luma: np.ndarray) -> float:
    """Mean gradient magnitude via Sobel-like finite differences."""
    if luma.size == 0:
        return 0.0
    gx = np.diff(luma, axis=1) if luma.shape[1] > 1 else np.zeros_like(luma)
    gy = np.diff(luma, axis=0) if luma.shape[0] > 1 else np.zeros_like(luma)
    # Pad to same shape for magnitude
    gx_pad = np.pad(gx, ((0,0),(0,1)), mode='edge')
    gy_pad = np.pad(gy, ((0,1),(0,0)), mode='edge')
    return float(np.mean(np.sqrt(gx_pad**2 + gy_pad**2)))


# ── Patch statistics ──────────────────────────────────────────────────────────

def patch_stats(patch: np.ndarray, prefix: str) -> Dict[str, float]:
    """
    Compute 7 statistics from an RGB patch (H×W×3, values 0–255).

    Returns dict with keys:
        {prefix}_mean_r, _mean_g, _mean_b, _mean_luma, _std_luma,
        _grad_mean, _mean_sat_proxy
    """
    if patch.size == 0:
        return {f'{prefix}_{k}': 0.0
                for k in ['mean_r','mean_g','mean_b','mean_luma',
                          'std_luma','grad_mean','mean_sat_proxy']}
    r = patch[:,:,0].astype(float)
    g = patch[:,:,1].astype(float)
    b = patch[:,:,2].astype(float)
    luma = rgb_to_luma(r, g, b)
    return {
        f'{prefix}_mean_r':         float(np.mean(r)),
        f'{prefix}_mean_g':         float(np.mean(g)),
        f'{prefix}_mean_b':         float(np.mean(b)),
        f'{prefix}_mean_luma':      float(np.mean(luma)),
        f'{prefix}_std_luma':       float(np.std(luma)),
        f'{prefix}_grad_mean':      gradient_mean(luma),
        f'{prefix}_mean_sat_proxy': saturation_proxy(r, g, b),
    }


# ── Circular ROI extraction ───────────────────────────────────────────────────

def extract_circular_roi(img_array: np.ndarray,
                         cx: int, cy: int,
                         radius: int) -> np.ndarray:
    """
    Extract pixels within a circle of given radius around (cx, cy).
    Returns an (N, 3) array of RGB values inside the circle.
    """
    H, W = img_array.shape[:2]
    y0, y1 = max(0, cy - radius), min(H, cy + radius + 1)
    x0, x1 = max(0, cx - radius), min(W, cx + radius + 1)
    patch = img_array[y0:y1, x0:x1]

    # Build circular mask
    ys, xs = np.ogrid[y0:y1, x0:x1]
    mask = (xs - cx)**2 + (ys - cy)**2 <= radius**2
    return patch[mask]  # shape (N, 3)


# ── Square context extraction ─────────────────────────────────────────────────

def extract_square_context(img_array: np.ndarray,
                           roi_coords: Dict[str, Tuple[int,int]],
                           square_fraction: float = SQUARE_FRAC) -> np.ndarray:
    """
    Extract a square region centred on the centroid of all ROI points.
    Side length = square_fraction × min(image_height, image_width).
    """
    H, W = img_array.shape[:2]
    xs = [c[0] for c in roi_coords.values()]
    ys = [c[1] for c in roi_coords.values()]
    cx, cy = int(np.mean(xs)), int(np.mean(ys))
    half = int(min(H, W) * square_fraction / 2)
    x0, x1 = max(0, cx - half), min(W, cx + half)
    y0, y1 = max(0, cy - half), min(H, cy + half)
    return img_array[y0:y1, x0:x1]


# ── 5×5 grid cell features ────────────────────────────────────────────────────

def grid_cell_features(square: np.ndarray,
                       grid_size: int = GRID_SIZE) -> Dict[str, float]:
    """
    Divide a square region into grid_size×grid_size cells and compute
    mean_luma and std_luma for each cell. Returns dict with keys
    sq_r{i}_c{j}_mean_luma and sq_r{i}_c{j}_std_luma.
    """
    features = {}
    H, W = square.shape[:2]
    rh = H / grid_size
    rw = W / grid_size
    for ri in range(grid_size):
        for ci in range(grid_size):
            y0, y1 = int(ri * rh), int((ri + 1) * rh)
            x0, x1 = int(ci * rw), int((ci + 1) * rw)
            cell = square[y0:y1, x0:x1]
            if cell.size == 0:
                features[f'sq_r{ri}_c{ci}_mean_luma'] = 0.0
                features[f'sq_r{ri}_c{ci}_std_luma']  = 0.0
            else:
                r = cell[:,:,0].astype(float)
                g = cell[:,:,1].astype(float)
                b = cell[:,:,2].astype(float)
                luma = rgb_to_luma(r, g, b)
                features[f'sq_r{ri}_c{ci}_mean_luma'] = float(np.mean(luma))
                features[f'sq_r{ri}_c{ci}_std_luma']  = float(np.std(luma))
    return features


# ── Spatial gradient / difference features ────────────────────────────────────

def spatial_gradient_features(roi_stats: Dict[str, float],
                               sq_stats: Dict[str, float],
                               sq_top: float, sq_bottom: float,
                               sq_left: float, sq_right: float,
                               sq_center: float, sq_corners: float,
                               grid_feats: Dict[str, float]) -> Dict[str, float]:
    """
    Compute normalized spatial gradient and difference features.
    These were the most important features in LightGBM feature importance analysis,
    outperforming absolute brightness values for table-surface generalization.
    """
    eps = 1e-6
    sq_mean = sq_stats.get('square_mean_luma', eps)
    c_luma  = roi_stats.get('C_mean_luma',  eps)
    ul_luma = roi_stats.get('UL_mean_luma', eps)
    ur_luma = roi_stats.get('UR_mean_luma', eps)
    lr_luma = roi_stats.get('LR_mean_luma', eps)
    ll_luma = roi_stats.get('LL_mean_luma', eps)

    corner_mean = (ul_luma + ur_luma + lr_luma + ll_luma) / 4

    feats = {}

    # Horizontal and vertical gradients (raw from square context)
    feats['square_bottom_minus_top_luma']   = sq_bottom - sq_top
    feats['square_right_minus_left_luma']   = sq_right  - sq_left
    feats['square_center_minus_corners_luma'] = sq_center - sq_corners
    feats['square_grid_range_mean_luma']    = sq_right - sq_left  # alias kept for compatibility
    feats['square_grid_std_mean_luma']      = sq_stats.get('square_std_luma', 0.0)

    # ROI-level differences
    feats['C_minus_corner_mean_luma'] = c_luma - corner_mean
    feats['C_over_corner_mean_luma']  = c_luma / (corner_mean + eps)
    feats['C_minus_UL_luma']          = c_luma - ul_luma
    feats['C_minus_UR_luma']          = c_luma - ur_luma
    feats['C_minus_LR_luma']          = c_luma - lr_luma
    feats['C_minus_LL_luma']          = c_luma - ll_luma
    feats['corner_mean_luma']         = corner_mean
    feats['corner_std_luma']          = float(np.std([ul_luma, ur_luma, lr_luma, ll_luma]))
    feats['corner_range_luma']        = max(ul_luma, ur_luma, lr_luma, ll_luma) - \
                                        min(ul_luma, ur_luma, lr_luma, ll_luma)

    # ROI vs nearest grid cell (diagonal alignment)
    feats['UL_minus_squarecell_0_0_luma'] = ul_luma - grid_feats.get('sq_r0_c0_mean_luma', 0.0)
    feats['UR_minus_squarecell_0_4_luma'] = ur_luma - grid_feats.get('sq_r0_c4_mean_luma', 0.0)
    feats['C_minus_squarecell_2_2_luma']  = c_luma  - grid_feats.get('sq_r2_c2_mean_luma', 0.0)
    feats['LR_minus_squarecell_4_4_luma'] = lr_luma - grid_feats.get('sq_r4_c4_mean_luma', 0.0)
    feats['LL_minus_squarecell_4_0_luma'] = ll_luma - grid_feats.get('sq_r4_c0_mean_luma', 0.0)

    # Normalized versions (reflectance-invariant)
    feats['feat_horiz_grad_norm']        = (sq_right - sq_left) / (sq_mean + eps)
    feats['feat_vert_grad_norm']         = (sq_bottom - sq_top)  / (sq_mean + eps)
    feats['feat_center_over_corners']    = c_luma / (corner_mean + eps)
    feats['feat_center_over_square']     = c_luma / (sq_mean + eps)
    feats['feat_sq_center_over_corners'] = sq_center / (sq_corners + eps)
    feats['feat_diag1_norm']             = (ul_luma - lr_luma) / (sq_mean + eps)
    feats['feat_diag2_norm']             = (ur_luma - ll_luma) / (sq_mean + eps)

    return feats


# ── Main extractor class ──────────────────────────────────────────────────────

class FeatureExtractor:
    """
    Extracts the full feature set from a single photograph given ROI coordinates.

    Parameters
    ----------
    roi_radius : int
        Radius of circular ROI patches in pixels. Default: 32.
    square_fraction : float
        Fraction of shorter image dimension for square context. Default: 0.85.
    grid_size : int
        Number of cells along each axis of the grid. Default: 5.
    """

    ROI_NAMES = ['C', 'UL', 'UR', 'LR', 'LL']

    def __init__(self, roi_radius: int = ROI_RADIUS,
                 square_fraction: float = SQUARE_FRAC,
                 grid_size: int = GRID_SIZE):
        self.roi_radius      = roi_radius
        self.square_fraction = square_fraction
        self.grid_size       = grid_size

    def extract_from_image(self,
                           image_path: str,
                           roi_coords: Dict[str, Tuple[int, int]]) -> Dict[str, float]:
        """
        Extract all features from one image.

        Parameters
        ----------
        image_path : str
            Path to the RGB image file.
        roi_coords : dict
            Pixel coordinates for each ROI, e.g.::

                {'C':  (640, 480),
                 'UL': (200, 150),
                 'UR': (1080, 150),
                 'LR': (1080, 810),
                 'LL': (200, 810)}

        Returns
        -------
        dict
            Feature name → float value.
        """
        img = Image.open(image_path).convert('RGB')
        arr = np.array(img)

        features = {}

        # 1. Per-ROI circular patch statistics
        roi_stats = {}
        for name in self.ROI_NAMES:
            if name not in roi_coords:
                continue
            cx, cy = roi_coords[name]
            pixels = extract_circular_roi(arr, cx, cy, self.roi_radius)
            if pixels.ndim == 1:
                pixels = pixels.reshape(-1, 3)
            # Build small patch array for patch_stats
            patch = pixels.reshape(1, -1, 3) if pixels.ndim == 2 else pixels
            stats = patch_stats(patch, prefix=name)
            features.update(stats)
            roi_stats.update(stats)

        # 2. Square context region statistics
        square = extract_square_context(arr, roi_coords, self.square_fraction)
        sq_full = patch_stats(square, prefix='square')
        features.update(sq_full)

        # 3. Square top/bottom/left/right mean luma
        H, W = square.shape[:2]
        half_h, half_w = H // 2, W // 2

        def mean_luma_region(region):
            if region.size == 0: return 0.0
            r = region[:,:,0].astype(float)
            g = region[:,:,1].astype(float)
            b = region[:,:,2].astype(float)
            return float(np.mean(rgb_to_luma(r, g, b)))

        sq_top    = mean_luma_region(square[:half_h, :])
        sq_bot    = mean_luma_region(square[half_h:, :])
        sq_left   = mean_luma_region(square[:, :half_w])
        sq_right  = mean_luma_region(square[:, half_w:])
        cH, cW    = H // 3, W // 3
        sq_center = mean_luma_region(square[cH:2*cH, cW:2*cW])
        # corner cells (outer 1/3)
        corner_regions = [
            square[:cH, :cW], square[:cH, 2*cW:],
            square[2*cH:, :cW], square[2*cH:, 2*cW:]
        ]
        sq_corners = float(np.mean([mean_luma_region(r) for r in corner_regions]))

        features['square_top_mean_luma']          = sq_top
        features['square_bottom_mean_luma']       = sq_bot
        features['square_left_mean_luma']         = sq_left
        features['square_right_mean_luma']        = sq_right
        features['square_center_cell_mean_luma']  = sq_center
        features['square_corner_cells_mean_luma'] = sq_corners

        # Gradient std
        sq_luma = rgb_to_luma(
            square[:,:,0].astype(float),
            square[:,:,1].astype(float),
            square[:,:,2].astype(float)
        )
        gx = np.diff(sq_luma, axis=1)
        gy = np.diff(sq_luma, axis=0)
        gx_pad = np.pad(gx, ((0,0),(0,1)), mode='edge')
        gy_pad = np.pad(gy, ((0,1),(0,0)), mode='edge')
        features['square_grad_std'] = float(np.std(np.sqrt(gx_pad**2 + gy_pad**2)))

        # C gradient std
        if 'C' in roi_coords:
            cx, cy = roi_coords['C']
            pixels_c = extract_circular_roi(arr, cx, cy, self.roi_radius)
            if pixels_c.ndim == 2 and pixels_c.shape[0] > 1:
                sq_side = int(np.sqrt(pixels_c.shape[0]))
                patch_c = pixels_c[:sq_side**2].reshape(sq_side, sq_side, 3)
                lc = rgb_to_luma(patch_c[:,:,0].astype(float),
                                 patch_c[:,:,1].astype(float),
                                 patch_c[:,:,2].astype(float))
                features['C_grad_std'] = float(np.std(lc))
            else:
                features['C_grad_std'] = 0.0

        # 4. 5×5 grid cell features
        grid_feats = grid_cell_features(square, self.grid_size)
        features.update(grid_feats)

        # Extra grid cell features (square_cell_ prefix used in training data)
        for ri in range(self.grid_size):
            for ci in range(self.grid_size):
                k_mean = f'sq_r{ri}_c{ci}_mean_luma'
                k_std  = f'sq_r{ri}_c{ci}_std_luma'
                features[f'square_cell_r{ri}_c{ci}_mean_luma'] = grid_feats.get(k_mean, 0.0)
                features[f'square_cell_r{ri}_c{ci}_std_luma']  = grid_feats.get(k_std, 0.0)

        # 5. Spatial gradient / difference features
        grad_feats = spatial_gradient_features(
            roi_stats=roi_stats,
            sq_stats=sq_full,
            sq_top=sq_top, sq_bottom=sq_bot,
            sq_left=sq_left, sq_right=sq_right,
            sq_center=sq_center, sq_corners=sq_corners,
            grid_feats=grid_feats
        )
        features.update(grad_feats)

        return features

    def extract_from_dataframe(self,
                               df: pd.DataFrame,
                               image_col: str = 'image_path',
                               coord_cols: Optional[Dict[str, Tuple[str,str]]] = None
                               ) -> pd.DataFrame:
        """
        Extract features for all rows in a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain image paths and ROI pixel coordinates.
        image_col : str
            Column name with image file paths.
        coord_cols : dict, optional
            Maps ROI name → (x_col, y_col). If None, uses default column names
            (x_C, y_C, x_UL, y_UL, x_UR, y_UR, x_LR, y_LR, x_LL, y_LL).

        Returns
        -------
        pd.DataFrame
            One row per image with all extracted features.
        """
        if coord_cols is None:
            coord_cols = {
                'C':  ('x_C',  'y_C'),
                'UL': ('x_UL', 'y_UL'),
                'UR': ('x_UR', 'y_UR'),
                'LR': ('x_LR', 'y_LR'),
                'LL': ('x_LL', 'y_LL'),
            }

        rows = []
        for _, row in df.iterrows():
            roi_coords = {}
            for roi, (xc, yc) in coord_cols.items():
                if xc in row and yc in row:
                    roi_coords[roi] = (int(row[xc]), int(row[yc]))

            try:
                feats = self.extract_from_image(row[image_col], roi_coords)
            except Exception as e:
                print(f"Warning: failed on {row[image_col]}: {e}")
                feats = {}
            rows.append(feats)

        return pd.DataFrame(rows)


# ── Standalone script ─────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Extract features from annotated lux dataset'
    )
    parser.add_argument('--annotations', required=True,
                        help='Path to annotation CSV (must have image_path, x_C, y_C, etc.)')
    parser.add_argument('--output', required=True,
                        help='Path to save feature CSV')
    parser.add_argument('--roi-radius', type=int, default=ROI_RADIUS)
    parser.add_argument('--grid-size',  type=int, default=GRID_SIZE)
    args = parser.parse_args()

    df = pd.read_csv(args.annotations)
    print(f'Loaded {len(df)} rows from {args.annotations}')

    extractor = FeatureExtractor(
        roi_radius=args.roi_radius,
        grid_size=args.grid_size
    )
    features_df = extractor.extract_from_dataframe(df)

    # Merge with original metadata
    meta_cols = [c for c in df.columns if not c.startswith('feat_')]
    out = pd.concat([df[meta_cols].reset_index(drop=True),
                     features_df.reset_index(drop=True)], axis=1)
    out.to_csv(args.output, index=False)
    print(f'Saved {len(out)} rows × {len(out.columns)} columns → {args.output}')
