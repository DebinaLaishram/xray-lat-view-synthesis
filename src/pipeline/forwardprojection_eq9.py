"""
Eq.(9) forward projection with Beer–Lambert law.

Supports:
- PARALLEL-BEAM  (paper-faithful Eq.9)
- CONE-BEAM      (perspective Eq.9, still Beer–Lambert)

Physics saved to .npy (metrics-correct transmission):
    A(u,v) = ∫ μ(x(u,v,s)) ds
    X(u,v) = exp(-A)

Display saved to .png (visualization ONLY):
    -log(X) + percentile windowing

Input:
    runs/ct_refine/test_predictions_nifti/*_CT_pred_hu.nii.gz

Output:
    runs/ct_refine/lat_eq9_cone/*_LAT_eq9.npy
    runs/ct_refine/lat_eq9_cone/*_LAT_eq9.png
"""

import os
import glob
import numpy as np
import nibabel as nib
from tqdm import tqdm
import matplotlib.pyplot as plt

# ============================================================
# USER SWITCH
# ============================================================
GEOMETRY = "cone"   # "parallel" or "cone"

# ============================================================
# PATHS
# ============================================================
CT_NII_DIR = "runs/ct_refine/test_predictions_nifti"
OUT_DIR = "runs/ct_refine/lat_eq9_cone" if GEOMETRY == "cone" else "runs/ct_refine/lat_eq9_parallel"
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# DETECTOR
# ============================================================
DET_H = 512
DET_W = 512
PIXEL_SIZE_MM = 1.5  # match DeepDRR generation

# ============================================================
# PHYSICS
# ============================================================
MU_WATER = 0.02   # 1/mm (scaling only)
EPS = 1e-6

# ============================================================
# RAY INTEGRATION
# ============================================================
STEP_MM = 1.5
MARGIN_MM = 200.0

# ============================================================
# CONE-BEAM GEOMETRY (MATCH DeepDRR)
# DeepDRR used: SDD=1500, source_to_point_fraction=0.5 -> SID=750
# ============================================================
SDD_MM = 1500.0
SID_MM = 0.5 * SDD_MM  # 750.0

# ============================================================
# DISPLAY (PNG ONLY)
# ============================================================
DISPLAY_MODE = "log"     # "raw" or "log"
DISPLAY_INVERT = False
PLOW, PHIGH = 1.0, 99.0

# ============================================================
# ORIENTATION FIX (IMPORTANT)
# Apply the same transform to ALL Eq9 outputs to match DeepDRR view convention.
# If you already empirically found best transform, set it here.
#
# Options:
#   "none"
#   "T"
#   "flip_lr"
#   "flip_ud"
#   "T_flip_lr"
#   "T_flip_ud"
#   "flip_lr_ud"
#   "T_flip_lr_ud"
# ============================================================
EQ9_VIEW_FIX = "none"   # <-- change to e.g. "T_flip_lr" if that matched DeepDRR best

# ============================================================
# DEBUG
# ============================================================
DEBUG_ONE = False  # print stats for first case only


# ============================================================
# UTILS
# ============================================================
def hu_to_mu(hu: np.ndarray) -> np.ndarray:
    return MU_WATER * (hu / 1000.0 + 1.0)


def normalize01_window(x: np.ndarray, plow=1.0, phigh=99.0) -> np.ndarray:
    lo = np.percentile(x, plow)
    hi = np.percentile(x, phigh)
    x = np.clip(x, lo, hi)
    return (x - lo) / (hi - lo + EPS)


def xray_to_display_png(xray: np.ndarray) -> np.ndarray:
    if DISPLAY_MODE == "raw":
        vis = xray
    elif DISPLAY_MODE == "log":
        vis = -np.log(xray + EPS)
    else:
        raise ValueError(f"Unknown DISPLAY_MODE: {DISPLAY_MODE}")

    vis = normalize01_window(vis, PLOW, PHIGH)
    if DISPLAY_INVERT:
        vis = 1.0 - vis
    return vis.astype(np.float32)


def save_png(xray: np.ndarray, path: str) -> None:
    plt.imsave(path, xray_to_display_png(xray), cmap="gray")


def apply_view_fix(img2d: np.ndarray) -> np.ndarray:
    """Apply detector-space transform to match DeepDRR image convention."""
    if EQ9_VIEW_FIX == "none":
        return img2d
    if EQ9_VIEW_FIX == "T":
        return img2d.T
    if EQ9_VIEW_FIX == "flip_lr":
        return np.fliplr(img2d)
    if EQ9_VIEW_FIX == "flip_ud":
        return np.flipud(img2d)
    if EQ9_VIEW_FIX == "flip_lr_ud":
        return np.flipud(np.fliplr(img2d))
    if EQ9_VIEW_FIX == "T_flip_lr":
        return np.fliplr(img2d.T)
    if EQ9_VIEW_FIX == "T_flip_ud":
        return np.flipud(img2d.T)
    if EQ9_VIEW_FIX == "T_flip_lr_ud":
        return np.flipud(np.fliplr(img2d.T))
    raise ValueError(f"Unknown EQ9_VIEW_FIX: {EQ9_VIEW_FIX}")


def trilinear_sample(vol: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    vol is (D,H,W) = (z,y,x). x,y,z are float voxel coords (in that same order).
    """
    D, H, W = vol.shape

    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    z0 = np.floor(z).astype(np.int32)

    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1

    inside = (x >= 0) & (x <= W - 1) & (y >= 0) & (y <= H - 1) & (z >= 0) & (z <= D - 1)

    x0c = np.clip(x0, 0, W - 1); x1c = np.clip(x1, 0, W - 1)
    y0c = np.clip(y0, 0, H - 1); y1c = np.clip(y1, 0, H - 1)
    z0c = np.clip(z0, 0, D - 1); z1c = np.clip(z1, 0, D - 1)

    xd = (x - x0).astype(np.float32)
    yd = (y - y0).astype(np.float32)
    zd = (z - z0).astype(np.float32)

    c000 = vol[z0c, y0c, x0c]
    c100 = vol[z0c, y0c, x1c]
    c010 = vol[z0c, y1c, x0c]
    c110 = vol[z0c, y1c, x1c]
    c001 = vol[z1c, y0c, x0c]
    c101 = vol[z1c, y0c, x1c]
    c011 = vol[z1c, y1c, x0c]
    c111 = vol[z1c, y1c, x1c]

    c00 = c000 * (1 - xd) + c100 * xd
    c10 = c010 * (1 - xd) + c110 * xd
    c01 = c001 * (1 - xd) + c101 * xd
    c11 = c011 * (1 - xd) + c111 * xd

    c0 = c00 * (1 - yd) + c10 * yd
    c1 = c01 * (1 - yd) + c11 * yd

    c = c0 * (1 - zd) + c1 * zd

    out = np.zeros_like(c, dtype=np.float32)
    out[inside] = c[inside].astype(np.float32)
    return out


def world_to_voxel(inv_aff: np.ndarray, pts: np.ndarray) -> np.ndarray:
    ones = np.ones((*pts.shape[:-1], 1), dtype=np.float32)
    hom = np.concatenate([pts.astype(np.float32), ones], axis=-1)
    vox = hom @ inv_aff.T
    return vox[..., :3]


def make_detector_grid(center: np.ndarray, right: np.ndarray, up: np.ndarray) -> np.ndarray:
    u = (np.arange(DET_W, dtype=np.float32) - (DET_W - 1) / 2.0) * PIXEL_SIZE_MM
    v = (np.arange(DET_H, dtype=np.float32) - (DET_H - 1) / 2.0) * PIXEL_SIZE_MM
    uu, vv = np.meshgrid(u, v)
    return (center[None, None, :]
            + uu[..., None] * right[None, None, :]
            + vv[..., None] * up[None, None, :]).astype(np.float32)


def integration_length(vol_shape, affine) -> float:
    D, H, W = vol_shape
    vx = float(np.linalg.norm(affine[:3, 0]))
    vy = float(np.linalg.norm(affine[:3, 1]))
    vz = float(np.linalg.norm(affine[:3, 2]))
    diag = np.sqrt((W * vx) ** 2 + (H * vy) ** 2 + (D * vz) ** 2)
    return float(diag + 2.0 * MARGIN_MM)


# ============================================================
# PROJECTORS
# ============================================================
def project_parallel(mu_zyx, affine, det_center, det_right, det_up):
    inv_aff = np.linalg.inv(affine).astype(np.float32)
    det = make_detector_grid(det_center, det_right, det_up)

    ray = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # +X
    L = integration_length(mu_zyx.shape, affine)
    n = int(np.ceil(L / STEP_MM))
    start = det - (L / 2.0) * ray[None, None, :]

    acc = np.zeros((DET_H, DET_W), dtype=np.float32)
    for i in range(n):
        pts = start + (i * STEP_MM) * ray[None, None, :]
        vox = world_to_voxel(inv_aff, pts)
        vals = trilinear_sample(mu_zyx, vox[..., 0].ravel(), vox[..., 1].ravel(), vox[..., 2].ravel())
        acc += vals.reshape(DET_H, DET_W) * STEP_MM

    return np.exp(-acc).astype(np.float32)


def project_cone(mu_zyx, affine, source, det_center, det_right, det_up):
    inv_aff = np.linalg.inv(affine).astype(np.float32)

    # Detector grid (H, W, 3)
    det = make_detector_grid(det_center, det_right, det_up)

    # Ray directions: source -> detector pixel
    dirs = det - source[None, None, :]
    ray_len = np.linalg.norm(dirs, axis=-1)            # (H, W)
    dirs = dirs / (ray_len[..., None] + EPS)           # unit vectors

    # Integration parameters
    n_steps = int(np.ceil(ray_len.max() / STEP_MM))

    acc = np.zeros((DET_H, DET_W), dtype=np.float32)

    for i in range(n_steps):
        s = i * STEP_MM

        # Only integrate while ray has not reached detector
        mask = s <= ray_len
        if not np.any(mask):
            break

        pts = source[None, None, :] + s * dirs
        vox = world_to_voxel(inv_aff, pts)

        vals = trilinear_sample(
            mu_zyx,
            vox[..., 0].ravel(),
            vox[..., 1].ravel(),
            vox[..., 2].ravel()
        ).reshape(DET_H, DET_W)

        acc[mask] += vals[mask] * STEP_MM

    return np.exp(-acc).astype(np.float32)


# ============================================================
# MAIN
# ============================================================
def main():
    nii_paths = sorted(glob.glob(os.path.join(CT_NII_DIR, "*.nii.gz")))
    if not nii_paths:
        raise RuntimeError(f"No CT NIfTIs found in {CT_NII_DIR}")

    print(f"Eq.(9) forward projection | geometry = {GEOMETRY}")
    print(f"Detector: {DET_W}x{DET_H}, pixel={PIXEL_SIZE_MM} mm")
    if GEOMETRY == "cone":
        print(f"Cone: SID={SID_MM} mm, SDD={SDD_MM} mm (matched to DeepDRR)")
    print(f"EQ9_VIEW_FIX: {EQ9_VIEW_FIX}")
    print(f"Saving to: {OUT_DIR}")

    printed_debug = False

    for ct_path in tqdm(nii_paths, desc="Eq9 forwardprojection"):
        base = os.path.basename(ct_path).replace(".nii.gz", "")
        out_npy = os.path.join(OUT_DIR, f"{base}_LAT_eq9.npy")
        out_png = os.path.join(OUT_DIR, f"{base}_LAT_eq9.png")

        img = nib.load(ct_path)
        ct = img.get_fdata(dtype=np.float32)
        if ct.ndim != 3:
            raise RuntimeError(f"Unexpected CT dims {ct.shape} for {base}")

        mu = hu_to_mu(ct)  # HU -> mu
        mu_zyx = np.transpose(mu, (2, 1, 0)).astype(np.float32)  # (z,y,x)

        affine = img.affine.astype(np.float32)

        # volume center in world coords (isocenter)
        center_vox = np.array([(ct.shape[0] - 1) / 2.0,
                               (ct.shape[1] - 1) / 2.0,
                               (ct.shape[2] - 1) / 2.0], dtype=np.float32)
        center_world = nib.affines.apply_affine(affine, center_vox).astype(np.float32)

        # LAT detector axes (world)
        det_right = np.array([0.0, 1.0, 0.0], dtype=np.float32)  # +Y
        det_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)     # +Z

        if GEOMETRY == "parallel":
            det_center = center_world
            xray = project_parallel(mu_zyx, affine, det_center, det_right, det_up)

        elif GEOMETRY == "cone":
            # match DeepDRR: SDD=1500, SID=750 (source_to_point_fraction=0.5)
            source = center_world - np.array([SID_MM, 0.0, 0.0], dtype=np.float32)  # along -X
            det_center = center_world + np.array([SDD_MM - SID_MM, 0.0, 0.0], dtype=np.float32)  # along +X
            xray = project_cone(mu_zyx, affine, source, det_center, det_right, det_up)

        else:
            raise ValueError("GEOMETRY must be 'parallel' or 'cone'")

        # Apply view fix to match DeepDRR convention (both .npy and .png)
        xray = apply_view_fix(xray)

        # Optional debug stats (first case only)
        if DEBUG_ONE and not printed_debug:
            printed_debug = True
            print("\n[DEBUG] xray stats for first case:")
            print("  min:", float(xray.min()))
            print("  max:", float(xray.max()))
            print("  mean:", float(xray.mean()))
            print("  std:", float(xray.std()), "\n")

        np.save(out_npy, xray.astype(np.float32))
        save_png(xray, out_png)

    print("\nDone. Saved to:", OUT_DIR)


if __name__ == "__main__":
    main()
