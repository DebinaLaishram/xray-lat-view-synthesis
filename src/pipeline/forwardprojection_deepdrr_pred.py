import os
import glob
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import killeengeo as kg
from deepdrr.vol import Volume
from deepdrr import SimpleDevice, Projector

# --------------------
# PATHS
# --------------------
CT_NIFTI_DIR = "runs/ct_refine/test_predictions_nifti"
OUT_LAT_DIR  = "runs/ct_refine/lat_deepdrr_predct"

os.makedirs(OUT_LAT_DIR, exist_ok=True)

SOURCE_TO_POINT_FRACTION = 0.5
SAVE_PNG = True

# --------------------
# UTILS
# --------------------
def save_png(img, path):
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    plt.imsave(path, img, cmap="gray")

# --------------------
# MAIN
# --------------------
def main():
    nii_paths = sorted(glob.glob(os.path.join(CT_NIFTI_DIR, "*.nii.gz")))
    if not nii_paths:
        raise RuntimeError(f"No predicted CT NIfTI files found in {CT_NIFTI_DIR}")

    print(f"Found {len(nii_paths)} predicted CT volumes")

    for idx, ct_path in enumerate(nii_paths, start=1):
        case_id = os.path.basename(ct_path).replace(".nii.gz", "")
        out_npy = os.path.join(OUT_LAT_DIR, f"{case_id}_LAT_pred_deepdrr.npy")
        out_png = os.path.join(OUT_LAT_DIR, f"{case_id}_LAT_pred_deepdrr.png")

        print(f"[{idx:04d}/{len(nii_paths)}] Projecting {case_id}")

        # Load CT
        ct = Volume.from_nifti(ct_path)

        # Fresh device per case (safe)
        device = SimpleDevice(
            sensor_height=512,
            sensor_width=512,
            pixel_size=1.5,  # mm
            source_to_detector_distance=1500.0,
        )

        # World directions
        right = ct.world_from_anatomical @ kg.vector(1, 0, 0)
        superior = ct.world_from_anatomical @ kg.vector(0, 0, 1)

        with Projector(ct, device=device) as projector:
            device.set_view(
                point=ct.center_in_world,
                direction=right,     # LAT
                up=superior,
                source_to_point_fraction=SOURCE_TO_POINT_FRACTION,
            )
            lat = projector()        # DeepDRR Eq. (9)

        np.save(out_npy, lat.astype(np.float32))
        if SAVE_PNG:
            save_png(lat, out_png)

    print("\nDeepDRR forward projection complete.")
    print("Saved to:", OUT_LAT_DIR)

if __name__ == "__main__":
    main()
