"""
Train, validate, test and save a HipMRI segmentation model.

Both models use the same patient-level split, pre-processing, augmentation, loss and
optimiser, so the only difference between runs is 3D vs 2D spatial context:
  * unet3d: ImprovedUNet3D trained on whole (resampled) volumes
  * unet2d: UNet2D trained on the axial slices of those same volumes

Each run writes to --out:
  best.pt            weights with the best validation prostate Dice
  history.json       per-epoch losses and validation Dice
  curves.png         loss and validation Dice plots
  test_metrics.json  test Dice/IoU per class, parameter count, peak VRAM, inference time
  test_per_case.csv  per-case test Dice/IoU, for picking failure cases in predict.py

Usage:
  python train.py --model unet3d --synthetic --epochs 2          # CPU smoke test
  python train.py --model unet3d --out runs/unet3d               # real data on Rangpur
  python train.py --model unet2d --out runs/unet2d
"""
import argparse
import json
import random
import time
from pathlib import Path

import torch

from dataset import CLASS_NAMES, DEFAULT_SHAPE, NUM_CLASSES, PROSTATE, get_loaders
from modules import DiceCELoss, ImprovedUNet3D, UNet2D, count_parameters
from utils import evaluate


def parse_args():
    p = argparse.ArgumentParser(description="Train a HipMRI segmentation model")
    p.add_argument("--model", choices=["unet3d", "unet2d"], required=True)
    p.add_argument("--root", default="/home/groups/comp3710/HipMRI_Study_open")
    p.add_argument("--synthetic", action="store_true", help="use synthetic phantoms (CPU smoke test)")
    p.add_argument("--shape", type=int, nargs=3, default=DEFAULT_SHAPE, metavar=("D", "H", "W"))
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=2, help="volumes per step")
    p.add_argument("--slice-batch", type=int, default=32, help="unet2d only: slices per step")
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--lr-decay", type=float, default=0.985, help="per-epoch exponential decay (Isensee et al.)")
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--amp", action="store_true", help="bfloat16 autocast (CUDA)")
    p.add_argument("--max-hours", type=float, default=None, help="stop training early to leave time for testing")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None, help="output directory (default runs/<model>)")
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(name):
    return ImprovedUNet3D(num_classes=NUM_CLASSES) if name == "unet3d" else UNet2D(num_classes=NUM_CLASSES)


def train_one_epoch(model, loader, optimiser, loss_fn, device, is_2d, slice_batch, amp):
    """One pass over the training volumes. Returns the mean training loss per optimiser step."""
    model.train()
    losses = []

    def step(x, y):
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            logits = model(x)
        loss = loss_fn(logits.float(), y)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        losses.append(loss.item())

    for image, label, _ in loader:
        image, label = image.to(device), label.to(device)
        if not is_2d:
            step(image, label)
            continue
        # 2D baseline: cut the augmented volumes into axial slices and shuffle them
        b, c, d, h, w = image.shape
        slices = image.permute(0, 2, 1, 3, 4).reshape(b * d, c, h, w)
        targets = label.reshape(b * d, h, w)
        order = torch.randperm(b * d, device=device)
        for i in range(0, b * d, slice_batch):
            idx = order[i:i + slice_batch]
            step(slices[idx], targets[idx])

    return sum(losses) / len(losses)


def plot_curves(history, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    fig, (ax_loss, ax_dice) = plt.subplots(1, 2, figsize=(11, 4))
    ax_loss.plot(epochs, [h["train_loss"] for h in history], label="train")
    ax_loss.plot(epochs, [h["val_loss"] for h in history], label="validation")
    ax_loss.set(xlabel="epoch", ylabel="Dice + CE loss", title="Loss")
    ax_loss.legend()
    for c in range(1, NUM_CLASSES):  # skip background
        ax_dice.plot(epochs, [h["val_dice"][c] for h in history], label=CLASS_NAMES[c])
    ax_dice.set(xlabel="epoch", ylabel="Dice", title="Validation Dice per class", ylim=(0, 1))
    ax_dice.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    args = parse_args()
    shape = tuple(args.shape)
    out = Path(args.out or f"runs/{args.model}")
    out.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = args.amp and device.type == "cuda"
    is_2d = args.model == "unet2d"
    print(f"device {device}, amp {amp}, args {vars(args)}", flush=True)

    t0 = time.time()
    train_loader, val_loader, test_loader = get_loaders(
        args.root, args.synthetic, shape, args.batch_size, args.num_workers, args.seed)
    print(f"data: {len(train_loader.dataset)} train / {len(val_loader.dataset)} val / "
          f"{len(test_loader.dataset)} test volumes, loaded in {time.time() - t0:.0f}s", flush=True)

    model = build_model(args.model).to(device)
    n_params = count_parameters(model)
    print(f"{args.model}: {n_params:,} trainable parameters", flush=True)

    loss_fn = DiceCELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimiser, gamma=args.lr_decay)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    history, best_score, best_dice, best_epoch = [], (-1.0, -1.0), -1.0, 0
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimiser, loss_fn, device, is_2d, args.slice_batch, amp)
        scheduler.step()
        val = evaluate(model, val_loader, device, NUM_CLASSES, is_2d, loss_fn=loss_fn, amp=amp)
        val_dice = val["dice"].nanmean(0)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val["loss"],
                        "val_dice": val_dice.tolist(), "lr": scheduler.get_last_lr()[0],
                        "elapsed_s": time.time() - start})

        # Select on prostate Dice; break ties (e.g. prostate still 0 early on) by mean foreground Dice
        prostate = val_dice[PROSTATE].item()
        score = (prostate, val_dice[1:].nanmean().item())
        if score > best_score:
            best_score, best_dice, best_epoch = score, prostate, epoch
            torch.save({"model": model.state_dict(), "args": vars(args), "epoch": epoch}, out / "best.pt")
        print(f"epoch {epoch:3d} | train {train_loss:.4f} | val {val['loss']:.4f} | "
              f"val Dice " + " ".join(f"{CLASS_NAMES[c][:4]} {val_dice[c]:.3f}" for c in range(1, NUM_CLASSES))
              + f" | best prostate {best_dice:.3f} @ {best_epoch} | {time.time() - start:.0f}s", flush=True)

        (out / "history.json").write_text(json.dumps(history, indent=1))
        plot_curves(history, out / "curves.png")
        if args.max_hours and time.time() - start > args.max_hours * 3600:
            print(f"stopping early: --max-hours {args.max_hours} reached", flush=True)
            break

    train_peak_gb = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else None
    train_hours = (time.time() - start) / 3600

    # Test the best checkpoint, recording inference memory and time per case
    model.load_state_dict(torch.load(out / "best.pt", map_location=device)["model"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    test = evaluate(model, test_loader, device, NUM_CLASSES, is_2d, amp=amp, timing=True)
    infer_peak_gb = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else None

    dice_mean, iou_mean = test["dice"].nanmean(0), test["iou"].nanmean(0)
    seconds = test["seconds_per_case"]
    metrics = {
        "model": args.model, "best_epoch": best_epoch, "best_val_prostate_dice": best_dice,
        "test_dice": dict(zip(CLASS_NAMES, dice_mean.tolist())),
        "test_iou": dict(zip(CLASS_NAMES, iou_mean.tolist())),
        "test_dice_foreground_mean": dice_mean[1:].nanmean().item(),
        "n_test_cases": len(test["case_ids"]), "parameters": n_params,
        "train_peak_vram_gb": train_peak_gb, "inference_peak_vram_gb": infer_peak_gb,
        # first case includes CUDA warm-up, so report the median
        "inference_s_per_case_median": sorted(seconds)[len(seconds) // 2],
        "train_hours": train_hours, "epochs_run": len(history), "shape": list(shape),
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
    }
    (out / "test_metrics.json").write_text(json.dumps(metrics, indent=1))

    rows = ["case_id," + ",".join(f"dice_{n}" for n in CLASS_NAMES) + "," + ",".join(f"iou_{n}" for n in CLASS_NAMES)]
    for k, case_id in enumerate(test["case_ids"]):
        rows.append(",".join([case_id] + [f"{v:.4f}" for v in test["dice"][k].tolist() + test["iou"][k].tolist()]))
    (out / "test_per_case.csv").write_text("\n".join(rows) + "\n")

    print("\nTEST (best checkpoint, epoch %d)" % best_epoch)
    for c, name in enumerate(CLASS_NAMES):
        print(f"  {name:10s} Dice {dice_mean[c]:.3f}  IoU {iou_mean[c]:.3f}")
    print(f"  params {n_params:,} | train peak VRAM {train_peak_gb} GB | inference peak VRAM {infer_peak_gb} GB | "
          f"median {metrics['inference_s_per_case_median']:.3f} s/case")
    print(f"outputs in {out}")


if __name__ == "__main__":
    main()
