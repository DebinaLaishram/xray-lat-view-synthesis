"""
Generate qualitative examples for LAT view synthesis (VISUALIZATION ONLY).

For each test case:
AP → GT LAT → Eq.(9) → DeepDRR LAT   (all visually aligned to GT)

IMPORTANT:
- GT LAT is the visual reference (NOT rotated)
- Eq.(9) orientation is fixed (transpose + LR flip) + global calibration
- DeepDRR is oriented for DISPLAY ONLY by choosing the best of 8 transforms vs GT
- Quantitative metrics are loaded from precomputed CSV (no evaluation recomputed)
- Visualization is min–max normalized
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

# -------------------------------------------------
# PATHS
# -------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

AP_DIR       = os.path.join(PROJECT_ROOT, "data/projections/AP")
LAT_GT_DIR   = os.path.join(PROJECT_ROOT, "data/projections/LAT")
LAT_EQ9_DIR  = os.path.join(PROJECT_ROOT, "runs/ct_refine/lat_eq9_cone")
LAT_DDR_DIR  = os.path.join(PROJECT_ROOT, "runs/ct_refine/lat_deepdrr_predct")

TEST_SPLIT   = os.path.join(PROJECT_ROOT, "data/splits/test.txt")
METRICS_CSV  = os.path.join(PROJECT_ROOT, "results/metrics/metrics_all.csv")

RESULTS_DIR  = os.path.join(PROJECT_ROOT, "results/examples")
os.makedirs(RESULTS_DIR, exist_ok=True)

EPS = 1e-6

# -------------------------------------------------
# GLOBAL CALIBRATION (Eq.9)
# -------------------------------------------------
A_CAL = -0.2987
B_CAL =  1.5195

# -------------------------------------------------
# UTILS
# -------------------------------------------------
def normalize_vis(x):
    x = x.astype(np.float32, copy=False)
    return (x - x.min()) / (x.max() - x.min() + EPS)

def to_log(x):
    return -np.log(np.clip(x, EPS, 1.0))

def fix_eq9_orientation(x):
    # empirically validated Eq.(9) -> GT detector alignment
    return np.fliplr(x.T)

def apply_global_calibration(eq9_I):
    A = to_log(eq9_I)
    A_cal = A_CAL * A + B_CAL
    return np.clip(np.exp(-A_cal), 0.0, 1.0)

def load_metrics(csv_path):
    metrics = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            metrics[row["case_id"]] = {
                "eq9_psnr": float(row["PSNR_eq9"]),
                "eq9_ssim": float(row["SSIM_eq9"]),
                "ddr_psnr": float(row["PSNR_deepdrr"]),
                "ddr_ssim": float(row["SSIM_deepdrr"]),
            }
    return metrics

# --- 8 orientation candidates (for DISPLAY ONLY) ---
def apply_orient(x, name):
    if name == "none": return x
    if name == "flip_lr": return np.fliplr(x)
    if name == "flip_ud": return np.flipud(x)
    if name == "flip_lr_ud": return np.flipud(np.fliplr(x))
    if name == "T": return x.T
    if name == "T_flip_lr": return np.fliplr(x.T)
    if name == "T_flip_ud": return np.flipud(x.T)
    if name == "T_flip_lr_ud": return np.flipud(np.fliplr(x.T))
    raise ValueError(name)

def best_display_orient(pred_I, gt_I):
    """
    Pick the transform that best matches GT (visual-only).
    Uses SSIM in log-domain to decide orientation.
    """
    # log + normalize to stabilize contrast for SSIM
    gt = normalize_vis(to_log(gt_I))

    candidates = ["none", "T_flip_ud"]
    best_name, best_score, best_img = None, -1e9, None

    for name in candidates:
        p = apply_orient(pred_I, name)
        if p.shape != gt_I.shape:
            continue
        p_log = normalize_vis(to_log(p))
        score = ssim(gt, p_log, data_range=1.0)
        if score > best_score:
            best_score, best_name, best_img = score, name, p

    return best_img, best_name, best_score

def save_panel(out_path, ap, gt, eq9, ddr, ddr_orient_name):
    plt.figure(figsize=(18, 4))

    panels = [
        (ap,  "AP input"),
        (gt,  "GT LAT"),
        (eq9, "Eq.(9) "),
        (ddr, f"DeepDRR LAT "),
    ]

    for i, (img, title) in enumerate(panels):
        plt.subplot(1, 4, i + 1)
        plt.imshow(normalize_vis(img), cmap="gray")
        plt.title(title)
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

# -------------------------------------------------
# MAIN
# -------------------------------------------------
def main():
    with open(TEST_SPLIT) as f:
        case_ids = [l.strip() for l in f if l.strip()]

    metrics = load_metrics(METRICS_CSV)

    print(f"Generating qualitative examples for {len(case_ids)} cases...\n")

    for case_id in case_ids:
        if case_id not in metrics:
            print(f"[WARN] Metrics missing for {case_id}, skipping.")
            continue

        try:
            ap = np.load(os.path.join(AP_DIR, f"{case_id}_AP.npy")).astype(np.float32)
            gt = np.load(os.path.join(LAT_GT_DIR, f"{case_id}_LAT.npy")).astype(np.float32)

            eq9 = np.load(os.path.join(LAT_EQ9_DIR, f"{case_id}_CT_pred_hu_LAT_eq9.npy")).astype(np.float32)
            ddr = np.load(os.path.join(LAT_DDR_DIR, f"{case_id}_CT_pred_hu_LAT_pred_deepdrr.npy")).astype(np.float32)

        except FileNotFoundError:
            print(f"[WARN] Missing files for {case_id}, skipping.")
            continue

        # Eq.(9): fixed alignment + calibration
        eq9 = fix_eq9_orientation(eq9)
        eq9 = apply_global_calibration(eq9)

        # DeepDRR: choose best DISPLAY orientation vs GT
        ddr_vis, ddr_orient, ddr_score = best_display_orient(ddr, gt)

        if ddr_vis is None:
            print(f"[WARN] Could not align DeepDRR for {case_id} (shape mismatch), skipping.")
            continue

        # Save
        case_dir = os.path.join(RESULTS_DIR, case_id)
        os.makedirs(case_dir, exist_ok=True)

        save_panel(
            os.path.join(case_dir, "comparison.png"),
            ap, gt, eq9, ddr_vis, ddr_orient
        )

        m = metrics[case_id]
        with open(os.path.join(case_dir, "metrics.txt"), "w") as f:
            f.write("Metrics are loaded from results/metrics/metrics_all.csv\n\n")
            f.write("Eq 9-FP:\n")
            f.write(f"  PSNR: {m['eq9_psnr']:.2f}\n")
            f.write(f"  SSIM: {m['eq9_ssim']:.3f}\n\n")

            f.write("DeepDRR-FP:\n")
            f.write(f"  PSNR: {m['ddr_psnr']:.2f}\n")
            f.write(f"  SSIM: {m['ddr_ssim']:.3f}\n\n")

        print(f"Saved example for {case_id}")

    print("\n Qualitative examples generated.")
    print(f"Results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()

