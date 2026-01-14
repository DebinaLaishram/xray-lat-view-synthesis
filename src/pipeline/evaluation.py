"""
Evaluation script for LAT view synthesis (0° to 90°).

Evaluates:
1) Eq.(9) LAT (predicted CT) vs GT LAT
2) DeepDRR LAT (predicted CT) vs GT LAT

IMPORTANT:
- Full detector images (no cropping, no resizing)
- Eq.(9) orientation is FIXED (transpose + LR flip)
- DeepDRR orientation is FIXED (transpose + UD flip)
- GT LAT: DO NOT TOUCH

Outputs:
- results/metrics/metrics_all.csv
- results/summary.txt
"""

import os
import csv
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# -----------------------
# PATHS
# -----------------------
TEST_SPLIT_FILE = "data/splits/test.txt"

LAT_GT_DIR = "data/projections/LAT"
LAT_EQ9_DIR = "runs/ct_refine/lat_eq9_cone"
LAT_DEEPDRR_DIR = "runs/ct_refine/lat_deepdrr_predct"

RESULTS_DIR = "results"
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")
os.makedirs(METRICS_DIR, exist_ok=True)

EPS = 1e-6

# -----------------------
# GLOBAL CALIBRATION (Eq.9 only)
# -----------------------
A_CAL = -0.2987
B_CAL =  1.5195

# -----------------------
# UTILS
# -----------------------
def normalize01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    x = x - x.min()
    return x / (x.max() + EPS)

def to_log_domain(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, EPS, 1.0)
    return -np.log(x)

def to_intensity(a: np.ndarray) -> np.ndarray:
    return np.exp(-a).astype(np.float32)

def extract_case_id(fname: str) -> str:
    return fname.split("_")[0]

def fix_eq9_orientation(x: np.ndarray) -> np.ndarray:
    # empirically validated orientation
    return np.fliplr(x.T)

def apply_global_calibration(eq9_I: np.ndarray) -> np.ndarray:
    eq9_A = to_log_domain(eq9_I)
    eq9_A_cal = A_CAL * eq9_A + B_CAL
    eq9_I_cal = to_intensity(eq9_A_cal)
    return np.clip(eq9_I_cal, 0.0, 1.0)

def fix_deepdrr_orientation(x: np.ndarray) -> np.ndarray:
    # empirically validated DeepDRR LAT orientation
    return np.flipud(x.T)

# -----------------------
# MAIN
# -----------------------
def main():
    # Load test split
    with open(TEST_SPLIT_FILE) as f:
        test_ids = [line.strip() for line in f if line.strip()]

    if not test_ids:
        raise RuntimeError("Test split file is empty.")

    # Index GT files
    gt_files = {
        extract_case_id(f): os.path.join(LAT_GT_DIR, f)
        for f in os.listdir(LAT_GT_DIR)
        if f.endswith("_LAT.npy")
    }

    # Index Eq.(9) files
    eq9_files = {
        f.split("_CT_")[0]: os.path.join(LAT_EQ9_DIR, f)
        for f in os.listdir(LAT_EQ9_DIR)
        if f.endswith(".npy") and "_LAT_eq9" in f
    }

    # Index DeepDRR files
    deepdrr_files = {
        extract_case_id(f): os.path.join(LAT_DEEPDRR_DIR, f)
        for f in os.listdir(LAT_DEEPDRR_DIR)
        if f.endswith("_LAT_pred_deepdrr.npy")
    }

    rows = []
    psnr_eq9_list, ssim_eq9_list = [], []
    psnr_ddr_list, ssim_ddr_list = [], []

    print(f"Evaluating {len(test_ids)} test cases...\n")

    for case_id in test_ids:
        if case_id not in gt_files or case_id not in eq9_files or case_id not in deepdrr_files:
            print(f"[WARN] Missing data for {case_id}")
            continue

        # -----------------------
        # Load GT
        # -----------------------
        gt_I = np.load(gt_files[case_id]).astype(np.float32)

        # -----------------------
        # Eq.(9): orientation + calibration (UNCHANGED)
        # -----------------------
        eq9_I = np.load(eq9_files[case_id]).astype(np.float32)
        eq9_I = fix_eq9_orientation(eq9_I)
        eq9_I = apply_global_calibration(eq9_I)

        # -----------------------
        # DeepDRR: FIX ORIENTATION
        # -----------------------
        ddr_I = np.load(deepdrr_files[case_id]).astype(np.float32)
        ddr_I = fix_deepdrr_orientation(ddr_I)

        # -----------------------
        # Shape check
        # -----------------------
        if gt_I.shape != eq9_I.shape or gt_I.shape != ddr_I.shape:
            raise RuntimeError(f"Shape mismatch for {case_id}")

        # -----------------------
        # PSNR (intensity domain)
        # -----------------------
        psnr_eq9 = peak_signal_noise_ratio(gt_I, eq9_I, data_range=1.0)
        psnr_ddr = peak_signal_noise_ratio(gt_I, ddr_I, data_range=1.0)

        # -----------------------
        # SSIM (log domain)
        # -----------------------
        gt_log = normalize01(to_log_domain(gt_I))
        eq9_log = normalize01(to_log_domain(eq9_I))
        ddr_log = normalize01(to_log_domain(ddr_I))

        ssim_eq9 = structural_similarity(gt_log, eq9_log, data_range=1.0)
        ssim_ddr = structural_similarity(gt_log, ddr_log, data_range=1.0)

        rows.append([
            case_id,
            psnr_eq9, ssim_eq9,
            psnr_ddr, ssim_ddr
        ])

        psnr_eq9_list.append(psnr_eq9)
        ssim_eq9_list.append(ssim_eq9)
        psnr_ddr_list.append(psnr_ddr)
        ssim_ddr_list.append(ssim_ddr)

        print(
            f"{case_id} | "
            f"Eq.(9): PSNR={psnr_eq9:.2f}, SSIM={ssim_eq9:.3f} | "
            f"DeepDRR (fixed): PSNR={psnr_ddr:.2f}, SSIM={ssim_ddr:.3f}"
        )

    # -----------------------
    # SUMMARY
    # -----------------------
    mean_psnr_eq9 = float(np.mean(psnr_eq9_list))
    mean_ssim_eq9 = float(np.mean(ssim_eq9_list))
    mean_psnr_ddr = float(np.mean(psnr_ddr_list))
    mean_ssim_ddr = float(np.mean(ssim_ddr_list))

    csv_path = os.path.join(METRICS_DIR, "metrics_all.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "case_id",
            "PSNR_eq9", "SSIM_eq9",
            "PSNR_deepdrr", "SSIM_deepdrr"
        ])
        writer.writerows(rows)

    summary_path = os.path.join(RESULTS_DIR, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("LAT View Synthesis Evaluation (0° to 90°)\n")
        f.write("========================================\n\n")
        f.write(f"Evaluated test cases: {len(rows)}\n\n")

        f.write("Eq.(9) vs GT:\n")
        f.write(f"  Mean PSNR: {mean_psnr_eq9:.2f}\n")
        f.write(f"  Mean SSIM: {mean_ssim_eq9:.3f}\n\n")

        f.write("DeepDRR vs GT:\n")
        f.write(f"  Mean PSNR: {mean_psnr_ddr:.2f}\n")
        f.write(f"  Mean SSIM: {mean_ssim_ddr:.3f}\n")

    print("\n===== FINAL RESULTS =====")
    print(f"Eq.(9) vs GT:        PSNR={mean_psnr_eq9:.2f}, SSIM={mean_ssim_eq9:.3f}")
    print(f"DeepDRR s GT:     PSNR={mean_psnr_ddr:.2f}, SSIM={mean_ssim_ddr:.3f}")
    print(f"Saved metrics CSV: {csv_path}")
    print(f"Saved summary TXT: {summary_path}")

if __name__ == "__main__":
    main()
