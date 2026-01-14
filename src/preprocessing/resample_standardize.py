import os
import csv
import numpy as np
import nibabel as nib
import scipy.ndimage


# ==================================================
# PATHS (WINDOWS)
# ==================================================
PROJECT_ROOT = r"C:\Users\nithi\OneDrive\Documents\xrayct_project"

input_dir = os.path.join(PROJECT_ROOT, "data", "nifti_raw")
out_hu_dir = os.path.join(PROJECT_ROOT, "data", "nifti_resampled_hu")
out_norm_dir = os.path.join(PROJECT_ROOT, "data", "nifti_resampled_norm")

os.makedirs(out_hu_dir, exist_ok=True)
os.makedirs(out_norm_dir, exist_ok=True)

csv_file = os.path.join(out_hu_dir, "resample_summary.csv")


# ==================================================
# PARAMETERS
# ==================================================
target_spacing = (1.5, 1.5, 1.5)     # mm (sx, sy, sz)
target_shape = (160, 160, 128)       # (X, Y, Z)

hu_window = (-1000, 400)
eps = 1e-8


# ==================================================
# HELPERS
# ==================================================
def get_spacing_from_affine(aff):
    return tuple(float(np.linalg.norm(aff[:3, i])) for i in range(3))


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


def resample_volume(vol, orig_spacing, new_spacing):
    zoom_factors = [o / n for o, n in zip(orig_spacing, new_spacing)]
    return scipy.ndimage.zoom(vol, zoom=zoom_factors, order=1)


def pad_crop_volume(vol, target_shape_xyz):
    padded = np.zeros(target_shape_xyz, dtype=vol.dtype)
    src_slices = []
    dst_slices = []
    info = ""

    for i in range(3):
        if vol.shape[i] < target_shape_xyz[i]:
            pad_before = (target_shape_xyz[i] - vol.shape[i]) // 2
            pad_after = pad_before + vol.shape[i]
            src_slices.append(slice(0, vol.shape[i]))
            dst_slices.append(slice(pad_before, pad_after))
            info += f"{i}:pad "
        else:
            crop_start = (vol.shape[i] - target_shape_xyz[i]) // 2
            crop_end = crop_start + target_shape_xyz[i]
            src_slices.append(slice(crop_start, crop_end))
            dst_slices.append(slice(0, target_shape_xyz[i]))
            info += f"{i}:crop "

    padded[
        dst_slices[0],
        dst_slices[1],
        dst_slices[2]
    ] = vol[
        src_slices[0],
        src_slices[1],
        src_slices[2]
    ]

    return padded, info.strip()


# ==================================================
# RUN
# ==================================================
# ==================================================
# RUN
# ==================================================
if os.path.exists(csv_file):
    os.remove(csv_file)

with open(csv_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "filename",
        "orig_shape",
        "orig_spacing",
        "final_shape",
        "final_spacing",
        "pad_crop_info",
    ])


existing = set(os.listdir(out_hu_dir))

for fname in sorted(os.listdir(input_dir)):
    if not fname.endswith(".nii.gz"):
        continue

    if fname in existing:
        print(f"Skipping {fname}, already resampled.")
        continue

    try:
        in_path = os.path.join(input_dir, fname)
        nii = nib.load(in_path)
        vol = nii.get_fdata().astype(np.float32)

        orig_shape = vol.shape
        orig_spacing = get_spacing_from_affine(nii.affine)

        resampled = resample_volume(vol, orig_spacing, target_spacing)
        final_vol, pad_crop_info = pad_crop_volume(resampled, target_shape)
        final_vol = np.clip(final_vol, *hu_window).astype(np.float32)

        new_affine = make_centered_affine(target_shape, target_spacing)

        nib.save(
            nib.Nifti1Image(final_vol, new_affine),
            os.path.join(out_hu_dir, fname)
        )

        norm_vol = (final_vol - hu_window[0]) / (hu_window[1] - hu_window[0] + eps)
        nib.save(
            nib.Nifti1Image(norm_vol.astype(np.float32), new_affine),
            os.path.join(out_norm_dir, fname)
        )

        with open(csv_file, "a", newline="") as f:
            csv.writer(f).writerow([
                fname, orig_shape, orig_spacing,
                final_vol.shape, target_spacing, pad_crop_info
            ])

        print(f"Processed {fname}")

    except Exception as e:
        print(f"[ERROR] {fname}: {e}")
        continue

print("\nDONE.")
print("HU volumes   :", out_hu_dir)
print("Norm volumes :", out_norm_dir)
print("CSV summary  :", csv_file)
