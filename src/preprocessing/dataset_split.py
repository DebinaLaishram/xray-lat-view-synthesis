import os
import random
import shutil

# -------------------------------
# Configuration
# -------------------------------
AP_DIR = "/Users/debinalaishram/xrayct_project/data/projections/AP"
LAT_DIR = "/Users/debinalaishram/xrayct_project/data/projections/LAT"
OUTPUT_DIR = "/Users/debinalaishram/xrayct_project/data"
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SEED = 42

# -------------------------------
# Helper functions
# -------------------------------
def make_dirs(base_dir):
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(base_dir, split, "AP"), exist_ok=True)
        os.makedirs(os.path.join(base_dir, split, "LAT"), exist_ok=True)

def copy_subject_files(subject_list, src_ap, src_lat, dst_dir):
    for subj in subject_list:
        ap_file = os.path.join(src_ap, f"{subj}_AP.npy")
        lat_file = os.path.join(src_lat, f"{subj}_LAT.npy")

        if os.path.exists(ap_file) and os.path.exists(lat_file):
            shutil.copy(ap_file, os.path.join(dst_dir, "AP", f"{subj}_AP.npy"))
            shutil.copy(lat_file, os.path.join(dst_dir, "LAT", f"{subj}_LAT.npy"))
        else:
            print(f"WARNING: Missing .npy files for subject {subj}, skipping.")

# -------------------------------
# Main
# -------------------------------
def main():
    # Get all AP .npy files
    ap_files = [f for f in os.listdir(AP_DIR) if f.endswith(".npy")]
    subjects = [f.replace("_AP.npy", "") for f in ap_files]
    subjects.sort()

    random.seed(SEED)
    random.shuffle(subjects)

    n_total = len(subjects)
    n_train = int(TRAIN_RATIO * n_total)
    n_val = int(VAL_RATIO * n_total)
    n_test = n_total - n_train - n_val

    train_subjects = subjects[:n_train]
    val_subjects = subjects[n_train:n_train + n_val]
    test_subjects = subjects[n_train + n_val:]

    print(f"Total subjects (with .npy AP): {n_total}")
    print(f"Train: {len(train_subjects)}, Val: {len(val_subjects)}, Test: {len(test_subjects)}")

    make_dirs(OUTPUT_DIR)

    copy_subject_files(train_subjects, AP_DIR, LAT_DIR, os.path.join(OUTPUT_DIR, "train"))
    copy_subject_files(val_subjects, AP_DIR, LAT_DIR, os.path.join(OUTPUT_DIR, "val"))
    copy_subject_files(test_subjects, AP_DIR, LAT_DIR, os.path.join(OUTPUT_DIR, "test"))

    print("Dataset split complete!")

if __name__ == "__main__":
    main()
