#!/usr/bin/env python3
import os
import csv
import argparse
from pathlib import Path

import numpy as np
import nibabel as nib


def fmt_tuple(x):
    return "(" + ",".join(f"{v:.4g}" for v in x) + ")"


def get_spacing_from_affine(aff):
    # voxel sizes = norms of affine columns (first 3 rows, first 3 cols)
    return tuple(float(np.linalg.norm(aff[:3, i])) for i in range(3))


def check_centering(aff, shape, tol_mm=5.0):
    """
    Heuristic: if affine is roughly centered, the world coord of the volume center
    should be near (0,0,0) (or at least not absurdly far).
    """
    ijk_center = np.array([(shape[0]-1)/2, (shape[1]-1)/2, (shape[2]-1)/2, 1.0], dtype=float)
    xyz_center = aff @ ijk_center
    dist = float(np.linalg.norm(xyz_center[:3]))
    return dist <= tol_mm, xyz_center[:3], dist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hu_dir", required=True, help="Folder: data/nifti_resampled_hu")
    ap.add_argument("--norm_dir", required=True, help="Folder: data/nifti_resampled_norm")
    ap.add_argument("--expected_shape", default="160,160,128")
    ap.add_argument("--expected_spacing", default="1.5,1.5,1.5")
    ap.add_argument("--max_files", type=int, default=0, help="0 = all")
    ap.add_argument("--csv_out", default="", help="Optional CSV path to write a validation report")
    args = ap.parse_args()

    expected_shape = tuple(int(x) for x in args.expected_shape.split(","))
    expected_spacing = tuple(float(x) for x in args.expected_spacing.split(","))

    hu_dir = Path(args.hu_dir)
    norm_dir = Path(args.norm_dir)

    hu_files = sorted([p for p in hu_dir.glob("*.nii*") if p.is_file()])
    if args.max_files and args.max_files > 0:
        hu_files = hu_files[: args.max_files]

    rows = []
    issues = 0

    for hu_path in hu_files:
        name = hu_path.name
        norm_path = norm_dir / name

        row = {
            "filename": name,
            "hu_exists": True,
            "norm_exists": norm_path.exists(),
            "hu_shape": "",
            "norm_shape": "",
            "hu_spacing": "",
            "norm_spacing": "",
            "hu_axcodes": "",
            "norm_axcodes": "",
            "hu_center_dist_mm": "",
            "norm_center_dist_mm": "",
            "shape_ok": "",
            "spacing_ok": "",
            "axcodes_ok": "",
            "centered_ok": "",
            "notes": "",
        }

        try:
            hu_img = nib.load(str(hu_path))
            hu_data = hu_img.get_fdata(dtype=np.float32)
            hu_shape = tuple(int(x) for x in hu_data.shape)
            hu_spacing = get_spacing_from_affine(hu_img.affine)
            hu_ax = nib.aff2axcodes(hu_img.affine)

            row["hu_shape"] = str(hu_shape)
            row["hu_spacing"] = fmt_tuple(hu_spacing)
            row["hu_axcodes"] = str(hu_ax)

            # center check
            hu_center_ok, hu_center_xyz, hu_center_dist = check_centering(
                hu_img.affine, hu_shape, tol_mm=25.0
            )
            row["hu_center_dist_mm"] = f"{hu_center_dist:.3f}"

            # norm
            if norm_path.exists():
                norm_img = nib.load(str(norm_path))
                norm_data = norm_img.get_fdata(dtype=np.float32)
                norm_shape = tuple(int(x) for x in norm_data.shape)
                norm_spacing = get_spacing_from_affine(norm_img.affine)
                norm_ax = nib.aff2axcodes(norm_img.affine)

                row["norm_shape"] = str(norm_shape)
                row["norm_spacing"] = fmt_tuple(norm_spacing)
                row["norm_axcodes"] = str(norm_ax)

                norm_center_ok, _, norm_center_dist = check_centering(
                    norm_img.affine, norm_shape, tol_mm=25.0
                )
                row["norm_center_dist_mm"] = f"{norm_center_dist:.3f}"
            else:
                norm_shape = None
                norm_spacing = None
                norm_ax = None
                norm_center_ok = False

            # expected checks (HU is the canonical one)
            shape_ok = (hu_shape == expected_shape) and (norm_shape == expected_shape if norm_shape else False)
            spacing_ok = all(abs(hu_spacing[i] - expected_spacing[i]) < 1e-3 for i in range(3))
            if norm_spacing:
                spacing_ok = spacing_ok and all(abs(norm_spacing[i] - expected_spacing[i]) < 1e-3 for i in range(3))

            ax_ok = (hu_ax == ("R", "A", "S")) and (norm_ax == ("R", "A", "S") if norm_ax else False)

            centered_ok = hu_center_ok and (norm_center_ok if norm_path.exists() else False)

            row["shape_ok"] = str(shape_ok)
            row["spacing_ok"] = str(spacing_ok)
            row["axcodes_ok"] = str(ax_ok)
            row["centered_ok"] = str(centered_ok)

            notes = []
            if not row["norm_exists"]:
                notes.append("missing_norm")
            if not shape_ok:
                notes.append("shape_mismatch")
            if not spacing_ok:
                notes.append("spacing_mismatch")
            if not ax_ok:
                notes.append("axcodes_not_RAS")
            if not centered_ok:
                notes.append("affine_not_centered_enough")

            row["notes"] = "|".join(notes)

            if notes:
                issues += 1

        except Exception as e:
            issues += 1
            row["notes"] = f"ERROR:{type(e).__name__}:{e}"

        rows.append(row)

    # Print summary
    total = len(rows)
    ok = sum(1 for r in rows if r["notes"] == "")
    print(f"\nValidated {total} files")
    print(f"OK: {ok}")
    print(f"With issues: {issues}\n")

    # Show a few problematic rows
    bad = [r for r in rows if r["notes"]]
    for r in bad[:10]:
        print(f"- {r['filename']}: {r['notes']}")
        print(f"  HU shape={r['hu_shape']} spacing={r['hu_spacing']} ax={r['hu_axcodes']} centerDist={r['hu_center_dist_mm']}mm")
        if r["norm_exists"]:
            print(f"  N  shape={r['norm_shape']} spacing={r['norm_spacing']} ax={r['norm_axcodes']} centerDist={r['norm_center_dist_mm']}mm")

    # Optional CSV
    if args.csv_out:
        outp = Path(args.csv_out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with open(outp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote CSV report: {outp}")

    # Fail fast if needed
    if issues > 0:
        print("\n There are issues. Fix preprocessing before projections.")
    else:
        print("\n Preprocessing looks consistent. Safe to proceed to projections.")


if __name__ == "__main__":
    main()
