"""
Evaluation script for LAT view synthesis (0° to 90°).

Evaluates:
1) Eq.(9) parallel-beam LAT (Beer–Lambert, predicted CT) vs GT LAT
2) DeepDRR LAT (predicted CT) vs GT LAT

IMPORTANT:
- Metrics computed on FULL detector images (512×512)
- No cropping, no resizing
- Matches Table VI evaluation protocol

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
LAT_EQ9_DIR = "runs/ct_refine/lat_eq9_pred"
LAT_DEEPDRR_DIR = "runs/ct_refine/lat_deepdrr_predct"

RESULTS_DIR = "results"
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")
os.makedirs(METRICS_DIR, exist_ok=True)


# -----------------------
# UTILS
# -----------------------
def normalize(img: np.ndarray) -> np.ndarray:
    """Normalize to [0,1] for PSNR/SSIM."""
    img = img.astype(np.float32, copy=False)
    return img / (img.max() + 1e-6)


def extract_case_id(fname: str) -> str:
    """Extract canonical case ID: LIDC-IDRI-XXXX"""
    return fname.split("_")[0]


# -----------------------
# MAIN
# -----------------------
def main():
    # Load test split
    with open(TEST_SPLIT_FILE) as f:
        test_ids = [line.strip() for line in f if line.strip()]

    if not test_ids:
        raise RuntimeError("Test split file is empty.")

    # -----------------------
    # Index files
    # -----------------------
    gt_files = {
        extract_case_id(f): os.path.join(LAT_GT_DIR, f)
        for f in os.listdir(LAT_GT_DIR)
        if f.endswith("_LAT.npy")
    }

    eq9_files = {}
    for f in os.listdir(LAT_EQ9_DIR):
        if "_LAT_eq9" in f and f.endswith(".npy"):
            case_id = f.split("_CT_")[0]
            eq9_files[case_id] = os.path.join(LAT_EQ9_DIR, f)

    deepdrr_files = {
        extract_case_id(f): os.path.join(LAT_DEEPDRR_DIR, f)
        for f in os.listdir(LAT_DEEPDRR_DIR)
        if f.endswith("_LAT_pred_deepdrr.npy")
    }

    rows = []
    psnr_eq9_list, ssim_eq9_list = [], []
    psnr_deepdrr_list, ssim_deepdrr_list = [], []

    print(f"Evaluating {len(test_ids)} test cases...\n")

    # -----------------------
    # Evaluation loop
    # -----------------------
    for case_id in test_ids:
        if case_id not in gt_files:
            print(f"[WARN] Missing GT LAT for {case_id}")
            continue
        if case_id not in eq9_files:
            print(f"[WARN] Missing Eq.9 LAT for {case_id}")
            continue
        if case_id not in deepdrr_files:
            print(f"[WARN] Missing DeepDRR LAT for {case_id}")
            continue

        lat_gt = normalize(np.load(gt_files[case_id]))
        lat_eq9 = normalize(np.load(eq9_files[case_id]))
        lat_deepdrr = normalize(np.load(deepdrr_files[case_id]))

        if lat_gt.shape != lat_eq9.shape:
            raise RuntimeError(
                f"Shape mismatch GT vs Eq9 for {case_id}: "
                f"{lat_gt.shape} vs {lat_eq9.shape}"
            )

        if lat_gt.shape != lat_deepdrr.shape:
            raise RuntimeError(
                f"Shape mismatch GT vs DeepDRR for {case_id}: "
                f"{lat_gt.shape} vs {lat_deepdrr.shape}"
            )

        psnr_eq9 = peak_signal_noise_ratio(lat_gt, lat_eq9, data_range=1.0)
        ssim_eq9 = structural_similarity(lat_gt, lat_eq9, data_range=1.0)

        psnr_deepdrr = peak_signal_noise_ratio(lat_gt, lat_deepdrr, data_range=1.0)
        ssim_deepdrr = structural_similarity(lat_gt, lat_deepdrr, data_range=1.0)

        rows.append([
            case_id,
            psnr_eq9, ssim_eq9,
            psnr_deepdrr, ssim_deepdrr,
        ])

        psnr_eq9_list.append(psnr_eq9)
        ssim_eq9_list.append(ssim_eq9)
        psnr_deepdrr_list.append(psnr_deepdrr)
        ssim_deepdrr_list.append(ssim_deepdrr)

        print(
            f"{case_id} | "
            f"Eq9: PSNR={psnr_eq9:.2f}, SSIM={ssim_eq9:.3f} | "
            f"DeepDRR: PSNR={psnr_deepdrr:.2f}, SSIM={ssim_deepdrr:.3f}"
        )

    if not rows:
        raise RuntimeError("No valid test cases evaluated.")

    # -----------------------
    # Aggregate stats
    # -----------------------
    mean_psnr_eq9 = float(np.mean(psnr_eq9_list))
    mean_ssim_eq9 = float(np.mean(ssim_eq9_list))
    mean_psnr_deepdrr = float(np.mean(psnr_deepdrr_list))
    mean_ssim_deepdrr = float(np.mean(ssim_deepdrr_list))

    # -----------------------
    # Save CSV
    # -----------------------
    csv_path = os.path.join(METRICS_DIR, "metrics_all.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "case_id",
            "PSNR_eq9", "SSIM_eq9",
            "PSNR_deepdrr", "SSIM_deepdrr",
        ])
        writer.writerows(rows)

    # -----------------------
    # Save summary
    # -----------------------
    summary_path = os.path.join(RESULTS_DIR, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("LAT View Synthesis Evaluation (0° to 90°)\n")
        f.write("======================================\n\n")
        f.write(f"Evaluated test cases: {len(rows)}\n\n")
        f.write("Eq.(9) Parallel-beam vs GT LAT:\n")
        f.write(f"  Mean PSNR: {mean_psnr_eq9:.2f}\n")
        f.write(f"  Mean SSIM: {mean_ssim_eq9:.3f}\n\n")
        f.write("DeepDRR vs GT LAT:\n")
        f.write(f"  Mean PSNR: {mean_psnr_deepdrr:.2f}\n")
        f.write(f"  Mean SSIM: {mean_ssim_deepdrr:.3f}\n")

    print("\n===== FINAL RESULTS =====")
    print(f"Evaluated cases: {len(rows)}\n")
    print("Eq.9 vs GT:")
    print(f"  Mean PSNR: {mean_psnr_eq9:.2f}")
    print(f"  Mean SSIM: {mean_ssim_eq9:.3f}\n")
    print("DeepDRR vs GT:")
    print(f"  Mean PSNR: {mean_psnr_deepdrr:.2f}")
    print(f"  Mean SSIM: {mean_ssim_deepdrr:.3f}\n")
    print(f"Saved metrics CSV: {csv_path}")
    print(f"Saved summary TXT: {summary_path}")


if __name__ == "__main__":
    main()
