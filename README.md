# New X-ray View Synthesis from a Single AP Projection

This repository implements **Section III-D (New X-ray View Synthesis)** from the paper:

**DVG-Diffusion: Dual-View Guided Diffusion Model for CT Reconstruction from X-Rays**  
Xing Xie *et al.*, arXiv:2503.17804

The goal is to synthesize a **LAT (90°) X-ray view** from a **single AP (0°) X-ray**, using analytical back-/forward-projection and a 3D convolutional neural network, **without using diffusion models**.

---

## Project Scope

**Implemented components (Section III-D):**
- Generation of ground-truth AP (0°) and LAT (90°) X-ray projections using **DeepDRR**
- Analytical back-projection from a single AP X-ray to a rough 3D volume (**Eq. 1**)
- A **3D U-Net (Gz)** to refine the back-projected volume and predict a CT latent representation
- Analytical forward projection from the predicted CT to synthesize a LAT view (**Eq. 9**)
- Quantitative evaluation using **PSNR** and **SSIM** against ground-truth LAT projections

**Out of scope (not implemented):**
- Diffusion models
- VPGE / VQGAN components
- Multi-view or biplanar diffusion refinement

This repository focuses **only on the image-processing and CNN-based components** explicitly described in Section III-D of the paper, as required by the task description.

---

## Method Overview (Section III-D)

Given a single **AP (0°) X-ray projection**, the pipeline proceeds as follows:

```text
AP X-ray
↓
Back-projection to 3D volume (Eq. 1)
↓
3D U-Net (Gz)
↓
Predicted CT volume
↓
Forward projection (Eq. 9)
↓
Synthesized LAT (90°) X-ray
```


- **Eq. (1)** produces a rough 3D volume by back-projecting the AP X-ray under a parallel-beam assumption.
- A **canonical 3D U-Net (Gz)** refines this volume into a latent CT representation using voxel-wise regression.
- **Eq. (9)** forward-projects the predicted CT to generate a synthesized LAT view.
- For comparison, a physics-based forward projection using **DeepDRR** is also performed.

---

## Dataset

- **Dataset:** LIDC-IDRI
- A subset of approximately **200 CT volumes** is used for training.
- Test cases are evaluated using held-out CT volumes.

---

## Data and File Policy

This repository **does not distribute medical imaging volumes**.

- No DICOM files (`.dcm`)
- No CT volumes (`.nii`, `.nii.gz`)
- Only **PNG images** and **CSV results** are included

All AP, LAT (ground truth), and predicted X-ray images are stored as `.png` files.  
These PNGs are generated from the LIDC-IDRI dataset following the preprocessing and projection steps described in the paper.

This design ensures:
- Compliance with data-sharing policies
- Lightweight repository size
- Reproducibility using publicly available datasets

---

## Repository Structure

Only representative files are shown below.  
Directories may contain many PNG images generated during preprocessing, inference, and evaluation.

```text
src/
├── preprocessing/
│   ├── convert_dicom_to_nifti.py         # Convert single DICOM series to NIfTI
│   ├── convert_all_dicom_to_nifti.py     # Batch DICOM → NIfTI conversion
│   ├── resample_standardize.py           # Resampling, HU clipping, normalization
│   └── dataset_split.py                  # Train / val / test split
│
├── pipeline/
│   ├── run_deepdrr_batch.py              # Generate AP and GT LAT using DeepDRR
│   ├── backprojection_eq1.py             # Implements Eq. (1)
│   ├── forward_projection_eq9.py         # Implements Eq. (9)
│   ├── forward_projection_deepdrr_pred.py# LAT projection from predicted CT (DeepDRR)
│   ├── unet3d.py                         # 3D U-Net (Gz)
│   ├── train_ct.py                       # CT refinement training
│   ├── infer_ct.py                       # CT inference
│   └── evaluation.py                    # PSNR / SSIM computation
│
├── visualization/
│   └── examples.py                       # Qualitative comparison figures
│
data/
├── projections/
│   ├── AP/                               # Input AP X-rays (.png)
│   └── LAT/                              # Ground-truth LAT (.png)
│
runs/
├── ct_refine/
│   ├── debug_slices/                     # Training sanity-check visualizations (.png)
│   ├── lat_eq9_cone/                     # Synthesized LAT via Eq. (9) (.png)
│   └── lat_deepdrr_predct/               # Synthesized LAT via DeepDRR (.png)
│
results/
├── metrics/
│   └── metrics_all.csv
└── summary.txt
```

## Evaluation Protocol

- **Eq. (9) Evaluation:**  
  Synthesized LAT views from analytical forward projection are compared against ground-truth LAT images.

- **DeepDRR Evaluation:**  
  Synthesized LAT views generated using DeepDRR are compared against ground-truth LAT images.

- **Metrics:**  
  - Peak Signal-to-Noise Ratio (**PSNR**)  
  - Structural Similarity Index (**SSIM**, computed in the log-attenuation domain)

On the held-out test set, the proposed pipeline achieves:

- Eq.(9) vs GT: PSNR ≈ 12–13, SSIM ≈ 0.60
- DeepDRR vs GT (orientation-corrected): PSNR ≈ 14.4, SSIM ≈ 0.76

---

## How to Run (High-Level)

1. **Preprocessing**
   - Convert CT volumes to a standardized format
   - Generate AP and LAT ground-truth projections using DeepDRR

2. **Back-projection**
   - Apply Eq. (1) to generate rough 3D volumes from AP X-rays

3. **3D U-Net Training**
   - Train the U-Net to map back-projected volumes to CT latent representations

4. **Inference**
   - Predict CT volumes from unseen AP X-rays

5. **Forward Projection**
   - Apply Eq. (9) to synthesize LAT views
   - Generate DeepDRR-based LAT projections for comparison

6. **Evaluation & Visualization**
   - Compute PSNR and SSIM
   - Generate qualitative comparison figures

Refer to individual scripts under `src/` for detailed execution steps.

---

## References

If you use this code, please cite the original paper:

```bibtex
@article{xie2025dvgdiffusion,
  title   = {DVG-Diffusion: Dual-View Guided Diffusion Model for CT Reconstruction from X-Rays},
  author  = {Xie, Xing and Liu, Jiawei and Fan, Huijie and Han, Zhi and Tang, Yandong and Qu, Liangqiong},
  journal = {arXiv preprint arXiv:2503.17804},
  year    = {2025}
}

If you use the physics-based projections, please also cite DeepDRR:

@inproceedings{unberath2018deepdrr,
  title     = {DeepDRR: A Catalyst for Machine Learning in Fluoroscopy-guided Procedures},
  author    = {Unberath, Mathias and Zaech, Jan-Nico and Gao, Cong and Taylor, Russell H.},
  booktitle = {Medical Image Computing and Computer Assisted Intervention (MICCAI)},
  year      = {2018}
}





