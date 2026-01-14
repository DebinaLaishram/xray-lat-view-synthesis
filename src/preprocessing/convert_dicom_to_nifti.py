import os
import numpy as np
import pydicom
import nibabel as nib

def convert_to_nifti(dicom_root, output_path):
    dcm_files = []

    # collect CT DICOMs only
    for root, _, files in os.walk(dicom_root):
        for f in files:
            if f.lower().endswith(".dcm"):
                path = os.path.join(root, f)
                try:
                    ds = pydicom.dcmread(path, stop_before_pixels=True)
                    if ds.Modality == "CT":
                        dcm_files.append(path)
                except Exception:
                    pass

    if len(dcm_files) == 0:
        raise RuntimeError(f"No CT DICOMs found in {dicom_root}")

    slices = []
    for f in dcm_files:
        ds = pydicom.dcmread(f)
        slices.append(ds)

    # sort slices by z-position
    slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))

    volume = np.stack(
        [(s.pixel_array * float(s.RescaleSlope) + float(s.RescaleIntercept))
         for s in slices],
        axis=-1
    ).astype(np.float32)

    # spacing
    px, py = map(float, slices[0].PixelSpacing)
    pz = abs(
        float(slices[1].ImagePositionPatient[2]) -
        float(slices[0].ImagePositionPatient[2])
    )
    spacing = (px, py, pz)

    affine = np.diag([px, py, pz, 1.0])
    nii = nib.Nifti1Image(volume, affine)
    nib.save(nii, output_path)

    print(f"Saved NIfTI → {output_path}")
    print(f"Shape: {volume.shape}, Spacing: {spacing}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    convert_to_nifti(args.input, args.output)
