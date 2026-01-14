import os
import sys
import time
import argparse

# ------------------------------------------------------------------
# Ensure project root is on PYTHONPATH so `src.*` imports work
# ------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.pipeline.dataset import VbpToCTDataset
from src.pipeline.unet3d import UNet3D

# ------------------------------------------------------------------
# Stability fixes (Windows / OpenMP)
# ------------------------------------------------------------------
torch.set_num_threads(1)

# ------------------------------------------------------------------
# Device selection
# ------------------------------------------------------------------
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# ------------------------------------------------------------------
# Checkpoint utilities
# ------------------------------------------------------------------
def save_checkpoint(path, model, optimizer, epoch, best_val_loss):
    ckpt = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
    }
    torch.save(ckpt, path)

# ------------------------------------------------------------------
# Debug visualization
# ------------------------------------------------------------------
@torch.no_grad()
def save_debug_slices(out_dir, case_id, vbp, ct_gt, ct_pred, epoch):
    """
    Save a 3-slice comparison:
      Row 1: Vbp
      Row 2: CT_gt
      Row 3: CT_pred
    """
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)

    v = vbp.squeeze(0).cpu().numpy()   # (D,H,W)
    g = ct_gt.squeeze(0).cpu().numpy()
    p = ct_pred.squeeze(0).cpu().numpy()

    D = v.shape[0]
    zs = [0, D // 2, D - 1]

    fig = plt.figure(figsize=(12, 6))
    for i, z in enumerate(zs):
        ax = fig.add_subplot(3, 3, 1 + i)
        ax.imshow(v[z], cmap="gray")
        ax.set_title(f"Vbp z={z}")
        ax.axis("off")

        ax = fig.add_subplot(3, 3, 4 + i)
        ax.imshow(g[z], cmap="gray")
        ax.set_title(f"CT_gt z={z}")
        ax.axis("off")

        ax = fig.add_subplot(3, 3, 7 + i)
        ax.imshow(p[z], cmap="gray")
        ax.set_title(f"CT_pred z={z}")
        ax.axis("off")

    fig.suptitle(f"{case_id} | epoch {epoch}", y=0.98)
    fig.tight_layout()
    out_path = os.path.join(out_dir, f"epoch_{epoch:03d}_{case_id}.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

# ------------------------------------------------------------------
# Training / validation loops
# ------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    running, n = 0.0, 0

    for batch in loader:
        vbp = batch["vbp"].to(device)
        ct = batch["ct"].to(device)

        optimizer.zero_grad(set_to_none=True)
        pred = model(vbp)
        loss = loss_fn(pred, ct)
        loss.backward()
        optimizer.step()

        running += loss.item() * vbp.size(0)
        n += vbp.size(0)

    return running / max(n, 1)

@torch.no_grad()
def validate(model, loader, loss_fn, device):
    model.eval()
    running, n = 0.0, 0

    for batch in loader:
        vbp = batch["vbp"].to(device)
        ct = batch["ct"].to(device)
        pred = model(vbp)
        loss = loss_fn(pred, ct)

        running += loss.item() * vbp.size(0)
        n += vbp.size(0)

    return running / max(n, 1)

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=".")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--save_dir", type=str, default="runs/ct_refine")
    parser.add_argument("--save_debug_every", type=int, default=1)
    args = parser.parse_args()

    device = get_device()
    print("Device:", device)

    # --------------------------------------------------------------
    # Track TOTAL training time
    # --------------------------------------------------------------
    train_start_time = time.time()

    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_dir = os.path.join(args.save_dir, "checkpoints")
    dbg_dir = os.path.join(args.save_dir, "debug_slices")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(dbg_dir, exist_ok=True)

    # --------------------------------------------------------------
    # Datasets (normalized CT targets)
    # --------------------------------------------------------------
    train_ds = VbpToCTDataset(args.root, "train", ct_space="norm")
    val_ds = VbpToCTDataset(args.root, "val", ct_space="norm")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # --------------------------------------------------------------
    # Model / loss / optimizer
    # --------------------------------------------------------------
    model = UNet3D(base_channels=args.base_channels).to(device)
    loss_fn = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # --------------------------------------------------------------
    # Early stopping parameters
    # --------------------------------------------------------------
    best_val = float("inf")
    patience = 8
    min_delta = 1e-4
    epochs_no_improve = 0

    print("\nStarting training (CT regression)")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss = validate(model, val_loader, loss_fn, device)

        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} | "
            f"train {train_loss:.6f} | val {val_loss:.6f} | "
            f"{time.time() - t0:.1f}s"
        )

        # ----------------------------------------------------------
        # Check improvement
        # ----------------------------------------------------------
        if val_loss < best_val - min_delta:
            best_val = val_loss
            epochs_no_improve = 0
            print(f"  ✓ New best val loss: {best_val:.6f}")

            save_checkpoint(
                os.path.join(ckpt_dir, "best.pt"),
                model, optimizer, epoch, best_val
            )
        else:
            epochs_no_improve += 1
            print(f"  ✗ No improvement for {epochs_no_improve}/{patience} epochs")

        # Always save last checkpoint
        save_checkpoint(
            os.path.join(ckpt_dir, "last.pt"),
            model, optimizer, epoch, best_val
        )

        # ----------------------------------------------------------
        # Debug slices
        # ----------------------------------------------------------
        if args.save_debug_every > 0 and epoch % args.save_debug_every == 0:
            print(f"[DEBUG] Saving debug slices for epoch {epoch}")

            batch = next(iter(val_loader))
            with torch.no_grad():
                pred = model(batch["vbp"].to(device))

            save_debug_slices(
                dbg_dir,
                batch["id"][0],
                batch["vbp"][0],
                batch["ct"][0],
                pred[0],
                epoch,
            )

        # ----------------------------------------------------------
        # Early stopping condition
        # ----------------------------------------------------------
        if epochs_no_improve >= patience:
            print(
                f"\nEarly stopping triggered at epoch {epoch} "
                f"(no improvement for {patience} epochs)"
            )
            break

    # --------------------------------------------------------------
    # Print TOTAL training time
    # --------------------------------------------------------------
    total_time = time.time() - train_start_time
    print("\nTraining finished.")
    print("Best val loss:", best_val)
    print(f"Total training time: {total_time/60:.2f} minutes ({total_time:.1f} seconds)")

if __name__ == "__main__":
    main()
