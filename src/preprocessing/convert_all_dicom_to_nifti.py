import os
import subprocess

PROJECT_ROOT = r"C:\Users\nithi\OneDrive\Documents\xrayct_project"
DICOM_ROOT = os.path.join(PROJECT_ROOT, "data", "dicom_raw")
NIFTI_ROOT = os.path.join(PROJECT_ROOT, "data", "nifti_raw")

# create output directory once
os.makedirs(NIFTI_ROOT, exist_ok=True)

patients = sorted(
    p for p in os.listdir(DICOM_ROOT)
    if p.startswith("LIDC-IDRI")
)

for patient in patients:
    dicom_dir = os.path.join(DICOM_ROOT, patient)
    out_file = os.path.join(NIFTI_ROOT, patient + ".nii.gz")

    if os.path.exists(out_file):
        print(f"Skipping {patient}, already converted.")
        continue

    print(f"Converting {patient}...")

    cmd = [
        "python",
        "src\\preprocessing\\convert_dicom_to_nifti.py",
        "--input", dicom_dir,
        "--output", out_file
    ]

    try:
        subprocess.run(cmd, check=True)
    except Exception as e:
        print(f"[ERROR] {patient}: {e}")
        continue