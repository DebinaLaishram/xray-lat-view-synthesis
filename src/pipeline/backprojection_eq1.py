"""
backprojection_eq1.py

Implements Eq. (1): Back-projection of AP (0°) X-ray images
into a rough 3D volume (Vbp).

Pipeline:
AP (512 x 512, detector space)
→ resample to (160 x 160, CT in-plane grid)
→ Vbp (160 x 160 x 128)

Equation:
V_bp(x, y, z) = I_AP(x, y) / |L|

Also saves ONE PNG per volume containing 3 slices
for visual sanity check.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.ndimage import zoom

# --------------------------------------------------
# Paths & constants
# --------------------------------------------------

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../")
)

AP_DIR  = os.path.join(PROJECT_ROOT, "data/projections/AP")
VBP_DIR = os.path.join(PROJECT_ROOT, "data/vbp")
PNG_DIR = os.path.join(PROJECT_ROOT, "data/vbp_png")

AP_TARGET_SHAPE = (160, 160)   # CT in-plane grid
DEPTH = 128                    # Number of slices

os.makedirs(VBP_DIR, exist_ok=True)
os.makedirs(PNG_DIR, exist_ok=True)


# --------------------------------------------------
# Resample detector AP → CT grid
# --------------------------------------------------

def resample_ap_to_ct_grid(ap_2d: np.ndarray,
                           target_shape=(160, 160)) -> np.ndarray:
    """
    Resamples AP from detector space (e.g., 512x512)
    to CT in-plane grid (160x160).

    This is a coordinate alignment step, NOT Eq. (1).
    """

    h, w = ap_2d.shape
    th, tw = target_shape

    zoom_factors = (th / h, tw / w)

    ap_resampled = zoom(
        ap_2d,
        zoom_factors,
        order=1  # linear interpolation
    )

    assert ap_resampled.shape == target_shape, (
        f"Resampling failed: got {ap_resampled.shape}"
    )

    return ap_resampled.astype(np.float32)


# --------------------------------------------------
# Eq. (1): Back-projection
# --------------------------------------------------

def backproject_ap_to_vbp(ap_2d: np.ndarray, depth: int) -> np.ndarray:
    """
    Implements Eq. (1):

        V_bp(x, y, z) = I_AP(x, y) / |L|
    """

    ap_2d = ap_2d.astype(np.float32)

    # Normalize by ray length |L| ≈ depth
    ap_2d = ap_2d / depth

    # Replicate along depth (z)
    vbp = np.repeat(ap_2d[:, :, None], depth, axis=2)

    return vbp.astype(np.float32)


# --------------------------------------------------
# Save PNG montage for QC
# --------------------------------------------------

def save_vbp_slices_png(vbp: np.ndarray, out_path: str):
    """
    Saves ONE PNG per volume containing 3 slices:
    [center-16 | center | center+16]
    """

    z_center = DEPTH // 2
    slice_ids = [z_center - 16, z_center, z_center + 16]

    slices = []
    for z in slice_ids:
        img = vbp[:, :, z]
        img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)
        slices.append(img_norm)

    # Concatenate horizontally
    montage = np.concatenate(slices, axis=1)

    plt.imsave(out_path, montage, cmap="gray")


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    ap_files = sorted(
        f for f in os.listdir(AP_DIR) if f.endswith(".npy")
    )

    print("AP → VBP back-projection (Eq. 1)")
    print(f"AP dir : {AP_DIR}")
    print(f"VBP dir: {VBP_DIR}")
    print(f"PNG dir: {PNG_DIR}")
    print(f"Files  : {len(ap_files)}\n")

    for fname in tqdm(ap_files, desc="Backprojecting"):
        ap_path = os.path.join(AP_DIR, fname)

        base = fname.replace("_AP.npy", "")
        vbp_path = os.path.join(VBP_DIR, f"{base}_VBP.npy")
        png_path = os.path.join(PNG_DIR, f"{base}_vbp_slices.png")

        # Load detector-space AP
        ap = np.load(ap_path)

        # Resample detector → CT grid
        ap_resampled = resample_ap_to_ct_grid(
            ap, AP_TARGET_SHAPE
        )

        # Eq. (1) back-projection
        vbp = backproject_ap_to_vbp(ap_resampled, DEPTH)

        # Safety checks
        assert vbp.shape == (160, 160, 128)
        assert vbp.dtype == np.float32
        assert not np.isnan(vbp).any()

        np.save(vbp_path, vbp)
        save_vbp_slices_png(vbp, png_path)

    print("\nVBP generation + PNG montage completed.")


if __name__ == "__main__":
    main()
