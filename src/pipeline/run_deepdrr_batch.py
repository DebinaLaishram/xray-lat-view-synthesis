import os
import glob
import numpy as np
import matplotlib.pyplot as plt

import killeengeo as kg
from deepdrr import Volume, SimpleDevice, Projector


# -----------------------------
# Config
# -----------------------------
NIFTI_DIR = "data/nifti_resampled_hu"
OUT_AP_DIR = "data/projections/AP"
OUT_LAT_DIR = "data/projections/LAT"

SENSOR_H = 512
SENSOR_W = 512
PIXEL_SIZE_MM = 1.5
SDD_MM = 1500.0
SOURCE_TO_POINT_FRACTION = 0.5

SAVE_PNG = True
SAVE_NPY = True

SKIP_IF_EXISTS = True  # set False if you want to overwrite


def save_png(img: np.ndarray, path: str) -> None:
    # Normalize for visualization only
    imgn = (img - img.min()) / (img.max() - img.min() + 1e-8)
    plt.imsave(path, imgn, cmap="gray")


def main():
    os.makedirs(OUT_AP_DIR, exist_ok=True)
    os.makedirs(OUT_LAT_DIR, exist_ok=True)

    # Collect all nii.gz
    nii_paths = sorted(glob.glob(os.path.join(NIFTI_DIR, "*.nii.gz")))
    if not nii_paths:
        raise RuntimeError(f"No .nii.gz files found in: {NIFTI_DIR}")

    # Device (cone-beam)
    device = SimpleDevice(
        sensor_height=SENSOR_H,
        sensor_width=SENSOR_W,
        pixel_size=PIXEL_SIZE_MM,
        source_to_detector_distance=SDD_MM,
    )

    total = len(nii_paths)
    print(f"Found {total} CTs in {NIFTI_DIR}")

    for idx, ct_path in enumerate(nii_paths, start=1):
        case_id = os.path.basename(ct_path).replace(".nii.gz", "")
        ap_npy = os.path.join(OUT_AP_DIR, f"{case_id}_AP.npy")
        lat_npy = os.path.join(OUT_LAT_DIR, f"{case_id}_LAT.npy")
        ap_png = os.path.join(OUT_AP_DIR, f"{case_id}_AP.png")
        lat_png = os.path.join(OUT_LAT_DIR, f"{case_id}_LAT.png")

        if SKIP_IF_EXISTS:
            need_ap = not (os.path.exists(ap_npy) if SAVE_NPY else os.path.exists(ap_png))
            need_lat = not (os.path.exists(lat_npy) if SAVE_NPY else os.path.exists(lat_png))
            if not need_ap and not need_lat:
                print(f"[{idx:04d}/{total}] Skipping {case_id} (exists)")
                continue

        print(f"[{idx:04d}/{total}] Processing {case_id} ...")

        try:
            ct = Volume.from_nifti(ct_path)

            # Anatomical directions in WORLD coords
            right = ct.world_from_anatomical @ kg.vector(1, 0, 0)
            anterior = ct.world_from_anatomical @ kg.vector(0, 1, 0)
            superior = ct.world_from_anatomical @ kg.vector(0, 0, 1)

            # One projector per CT (safe, predictable)
            with Projector(ct, device=device) as projector:

                # AP
                device.set_view(
                    point=ct.center_in_world,
                    direction=anterior,
                    up=superior,
                    source_to_point_fraction=SOURCE_TO_POINT_FRACTION,
                )
                ap_img = projector()

                # LAT
                device.set_view(
                    point=ct.center_in_world,
                    direction=right,
                    up=superior,
                    source_to_point_fraction=SOURCE_TO_POINT_FRACTION,
                )
                lat_img = projector()

            # Save
            if SAVE_NPY:
                np.save(ap_npy, ap_img.astype(np.float32))
                np.save(lat_npy, lat_img.astype(np.float32))
            if SAVE_PNG:
                save_png(ap_img, ap_png)
                save_png(lat_img, lat_png)

            print(f"      saved -> {os.path.basename(ap_png)} , {os.path.basename(lat_png)}")

        except Exception as e:
            print(f"[ERROR] {case_id}: {repr(e)}")
            continue

    print("Done.")


if __name__ == "__main__":
    main()
