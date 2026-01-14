import os
import nibabel as nib
import numpy as np
import pandas as pd

nifti_root = r"C:\Users\nithi\OneDrive\Documents\xrayct_project\data\nifti_raw"

files = sorted(f for f in os.listdir(nifti_root) if f.endswith(".nii.gz"))

summary = []

for f in files:
    file_path = os.path.join(nifti_root, f)

    nii = nib.load(file_path)
    dataobj = nii.dataobj  # memory-mapped, NOT loaded fully
    zooms = nii.header.get_zooms()

    # Compute stats safely
    data_min = np.min(dataobj)
    data_max = np.max(dataobj)
    data_mean = np.mean(dataobj)
    data_std = np.std(dataobj)

    info = {
        "file": f,
        "shape": nii.shape,
        "slices": nii.shape[2],
        "voxel_spacing_x": zooms[0],
        "voxel_spacing_y": zooms[1],
        "voxel_spacing_z": zooms[2],
        "dtype": nii.get_data_dtype(),
        "min": float(data_min),
        "max": float(data_max),
        "mean": float(data_mean),
        "std": float(data_std),
    }

    summary.append(info)

df = pd.DataFrame(summary)

# Print overall statistics
print("Total files:", len(df))
print("Unique shapes:", df["shape"].unique())
print(
    "Unique voxel spacings:",
    df[["voxel_spacing_x", "voxel_spacing_y", "voxel_spacing_z"]]
    .drop_duplicates()
    .values,
)
print("Data types:", df["dtype"].unique())
print("Slices: min/max:", df["slices"].min(), "/", df["slices"].max())
print("Intensity ranges: min/max:", df["min"].min(), "/", df["max"].max())
print("Mean intensity range:", df["mean"].min(), "/", df["mean"].max())
print("Std dev range:", df["std"].min(), "/", df["std"].max())

# Save summary
out_csv = os.path.join(nifti_root, "nifti_summary.csv")
df.to_csv(out_csv, index=False)

print(f"Summary saved to: {out_csv}")
