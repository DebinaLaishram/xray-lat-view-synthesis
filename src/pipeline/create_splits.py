"""
create_splits.py

Creates train / val / test splits based on patient IDs.
Splits are deterministic and applied consistently across
VBP, CT, and LAT data.
"""

import os
import random

# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../")
)

VBP_DIR = os.path.join(PROJECT_ROOT, "data/vbp")
SPLIT_DIR = os.path.join(PROJECT_ROOT, "data/splits")

os.makedirs(SPLIT_DIR, exist_ok=True)

# --------------------------------------------------
# Split configuration
# --------------------------------------------------

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

RANDOM_SEED = 42  # reproducibility

# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    # Extract patient IDs from VBP filenames
    ids = sorted([
        f.replace("_VBP.npy", "")
        for f in os.listdir(VBP_DIR)
        if f.endswith("_VBP.npy")
    ])

    print(f"Total cases found: {len(ids)}")

    random.seed(RANDOM_SEED)
    random.shuffle(ids)

    n_total = len(ids)
    n_train = int(n_total * TRAIN_RATIO)
    n_val   = int(n_total * VAL_RATIO)

    train_ids = ids[:n_train]
    val_ids   = ids[n_train:n_train + n_val]
    test_ids  = ids[n_train + n_val:]

    # Save splits
    def save_split(name, split_ids):
        path = os.path.join(SPLIT_DIR, f"{name}.txt")
        with open(path, "w") as f:
            for pid in split_ids:
                f.write(pid + "\n")
        print(f"{name}: {len(split_ids)}")

    save_split("train", train_ids)
    save_split("val", val_ids)
    save_split("test", test_ids)

    print("\nSplits written to:", SPLIT_DIR)


if __name__ == "__main__":
    main()
