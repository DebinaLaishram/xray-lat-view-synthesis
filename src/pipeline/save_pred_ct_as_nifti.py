import os
import numpy as np
import nibabel as nib
from tqdm import tqdm

# --------------------
# PATHS
# --------------------
CT_HU_DIR = "runs/ct_refine/test_predictions_hu"
OUT_NIFTI_DIR = "runs/ct_refine/test_predictions_nifti"

os.makedirs(OUT_NIFTI_DIR, exist_ok=True)

# --------------------
# AFFINE (must match preprocessing)
# --------------------
def make_centered_affine(shape_xyz, spacing_xyz):
    X, Y, Z = shape_xyz
    sx, sy, sz = spacing_xyz

    aff = np.eye(4, dtype=np.float32)
    aff[0, 0] = sx
    aff[1, 1] = sy
    aff[2, 2] = sz

    aff[0, 3] = -((X - 1) * sx) / 2.0
    aff[1, 3] = -((Y - 1) * sy) / 2.0
    aff[2, 3] = -((Z - 1) * sz) / 2.0

    return aff


AFFINE = make_centered_affine(
    shape_xyz=(160, 160, 128),
    spacing_xyz=(1.5, 1.5, 1.5),
)

# --------------------
# RUN
# --------------------
files = sorted(f for f in os.listdir(CT_HU_DIR) if f.endswith("_CT_pred_hu.npy"))
print(f"Found {len(files)} CT HU predictions")

for f in tqdm(files, desc="Saving NIfTI"):
    case_id = f.replace("_CT_pred_hu.npy", "")
    ct = np.load(os.path.join(CT_HU_DIR, f)).astype(np.float32)

    # --------------------------------------------------
    # IMPORTANT FIX: remove channel dimension if present
    # --------------------------------------------------
    if ct.ndim == 4 and ct.shape[0] == 1:
        ct = ct[0]   # (Z, Y, X)

    # Safety check
    assert ct.ndim == 3, f"Expected 3D CT, got shape {ct.shape}"

    nii = nib.Nifti1Image(ct, AFFINE)
    nib.save(
        nii,
        os.path.join(OUT_NIFTI_DIR, f"{case_id}_CT_pred_hu.nii.gz")
    )

print("Done.")
print("Saved to:", OUT_NIFTI_DIR)
