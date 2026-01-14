import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset


class VbpToCTDataset(Dataset):
    """
    Dataset for:
        Vbp (rough 3D volume from AP back-projection) --> CT_gt (ground-truth CT)

    Splits are defined by text files:
        data/splits/train.txt
        data/splits/val.txt
        data/splits/test.txt

    Vbp files are expected at:
        data/vbp/{CASE_ID}_AP.npy

    CT ground truth is loaded from either:
        data/nifti_resampled_norm/{CASE_ID}.nii.gz   (recommended for training)
        data/nifti_resampled_hu/{CASE_ID}.nii.gz     (for HU-space evaluation/forward projection)

    Notes:
    - Vbp is min-max normalized to [0, 1] per volume for stable training.
    - Returned tensors are (C, D, H, W) for Conv3d.
    """

    def __init__(self, root_dir: str, split: str, ct_space: str = "norm"):
        """
        Args:
            root_dir: Project root directory (xrayct_project)
            split: One of ['train', 'val', 'test']
            ct_space: 'norm' or 'hu'
        """
        super().__init__()
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        assert ct_space in ["norm", "hu"], f"Invalid ct_space: {ct_space}"

        self.root_dir = root_dir
        self.split = split
        self.ct_space = ct_space

        # --- Paths ---
        self.split_file = os.path.join(root_dir, "data", "splits", f"{split}.txt")
        self.vbp_dir = os.path.join(root_dir, "data", "vbp")

        if ct_space == "norm":
            self.ct_dir = os.path.join(root_dir, "data", "nifti_resampled_norm")
        else:
            self.ct_dir = os.path.join(root_dir, "data", "nifti_resampled_hu")

        # --- Checks ---
        assert os.path.isfile(self.split_file), f"Split file not found: {self.split_file}"
        assert os.path.isdir(self.vbp_dir), f"Vbp directory not found: {self.vbp_dir}"
        assert os.path.isdir(self.ct_dir), f"CT directory not found: {self.ct_dir}"

        # --- Read case IDs ---
        with open(self.split_file, "r") as f:
            self.case_ids = [line.strip() for line in f if line.strip()]

        assert len(self.case_ids) > 0, f"No case IDs found in {self.split_file}"

        print(f"[Dataset] Split={split} | ct_space={ct_space} | Samples={len(self.case_ids)}")

    def __len__(self):
        return len(self.case_ids)

    @staticmethod
    def _minmax01(vol: np.ndarray) -> np.ndarray:
        vmin = float(vol.min())
        vmax = float(vol.max())
        return (vol - vmin) / (vmax - vmin + 1e-6)

    def __getitem__(self, idx):
        case_id = self.case_ids[idx]

        # -----------------------------
        # Load Vbp (AP back-projection)
        # -----------------------------
        vbp_path = os.path.join(self.vbp_dir, f"{case_id}_VBP.npy")

        if not os.path.isfile(vbp_path):
            raise FileNotFoundError(f"Vbp file not found: {vbp_path}")

        vbp = np.load(vbp_path).astype(np.float32)
        if vbp.ndim != 3:
            raise ValueError(f"Vbp must be 3D, got shape {vbp.shape} for {case_id}")

        # Normalize Vbp to [0,1] for stable training
        vbp = self._minmax01(vbp)

        # -----------------------------
        # Load GT CT
        # -----------------------------
        ct_path = os.path.join(self.ct_dir, f"{case_id}.nii.gz")
        if not os.path.isfile(ct_path):
            raise FileNotFoundError(f"CT file not found: {ct_path}")

        ct = nib.load(ct_path).get_fdata().astype(np.float32)
        if ct.shape != vbp.shape:
            raise ValueError(
                f"Shape mismatch for {case_id}: Vbp {vbp.shape} vs CT {ct.shape} "
                f"(ct_space={self.ct_space})"
            )

        # -----------------------------
        # Convert to torch tensors
        # vbp/ct are assumed (H, W, D) based on your earlier pipeline
        # Convert to (C, D, H, W) for Conv3d
        # -----------------------------
        vbp_t = torch.from_numpy(vbp).unsqueeze(0)  # (1, H, W, D)
        ct_t = torch.from_numpy(ct).unsqueeze(0)

        vbp_t = vbp_t.permute(0, 3, 1, 2).contiguous()  # (1, D, H, W)
        ct_t = ct_t.permute(0, 3, 1, 2).contiguous()

        return {
            "vbp": vbp_t,
            "ct": ct_t,
            "id": case_id,
        }
