"""
Run trained models on the held-out test patients and analyse where they fail.

Loads one or both checkpoints written by train.py, rebuilds the same patient-level test
split from the seed stored in the checkpoint, and reports:
  * per-class Dice / IoU on whole test volumes
  * prostate Dice slice by slice, from apex to base (the open research question: does 3D
    context fix the end slices, or only the easy central ones?)
  * leakage between prostate and the rectum / bladder
  * disconnected prostate fragments (the true prostate is a single component)
and saves figures:
  * prostate_profile.png   mean slice Dice along the apex-base axis, per model
  * case_<id>.png          input / ground truth / prediction / error overlays for the
                           worst test cases (failure autopsy) and the best case (example output)
  * predict_results.json   all numbers above, per model and per case

Usage:
  python predict.py --unet3d runs/unet3d_<job>/best.pt --unet2d runs/unet2d_<job>/best.pt
  python predict.py --unet3d runs/smoke/best.pt --synthetic     # pipeline check
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap

from dataset import CLASS_NAMES, NUM_CLASSES, PROSTATE, HipMRI3D, SyntheticHipMRI, find_cases, split_by_patient
from modules import ImprovedUNet3D, UNet2D
from utils import count_components, dice_iou, predict_logits

BLADDER, RECTUM = CLASS_NAMES.index("bladder"), CLASS_NAMES.index("rectum")
N_BINS = 10  # bins along the prostate's apex-base extent
LABEL_CMAP = ListedColormap(["black", "#555555", "#e8e8a0", "#3a7bd5", "#a0522d", "#e03030"])
MODEL_NAMES = {"unet3d": "3D Improved U-Net", "unet2d": "2D U-Net"}


def parse_args():
    p = argparse.ArgumentParser(description="Test-set inference and failure analysis")
    p.add_argument("--unet3d", help="checkpoint from train.py --model unet3d")
    p.add_argument("--unet2d", help="checkpoint from train.py --model unet2d")
    p.add_argument("--root", default="/home/groups/comp3710/HipMRI_Study_open")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--n-worst", type=int, default=4, help="number of failure cases to plot")
    p.add_argument("--out", default="figures")
    args = p.parse_args()
    if not (args.unet3d or args.unet2d):
        p.error("give at least one of --unet3d / --unet2d")
    return args


def load_model(path, device):
    ckpt = torch.load(path, map_location=device)
    name = ckpt["args"]["model"]
    model = ImprovedUNet3D(num_classes=NUM_CLASSES) if name == "unet3d" else UNet2D(num_classes=NUM_CLASSES)
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), ckpt["args"]


def load_test_set(args, train_args):
    """Rebuild the exact test split used in training (same seed and shape)."""
    shape, seed = tuple(train_args["shape"]), train_args["seed"]
    if args.synthetic:
        return SyntheticHipMRI(2, shape, seed=seed + 2)
    _, _, test_cases = split_by_patient(find_cases(args.root), seed=seed)
    return HipMRI3D(test_cases, shape)


# ---------------------------------------------------------------------------
# Analysis of one predicted volume
# ---------------------------------------------------------------------------

def slice_dice(pred_mask, gt_mask):
    """Dice per axial slice (D,), NaN where both are empty."""
    inter = (pred_mask & gt_mask).sum(dim=(1, 2)).float()
    total = (pred_mask.sum(dim=(1, 2)) + gt_mask.sum(dim=(1, 2))).float()
    return torch.where(total > 0, 2 * inter / total.clamp(min=1), torch.full_like(total, float("nan")))


def apex_base_positions(label):
    """
    Map each slice inside the ground-truth prostate to a position in [0, 1], 0 = apex, 1 = base.
    The base is the end nearer the bladder (it sits superior to the prostate), so the orientation
    comes from the anatomy instead of assuming a Nifti axis direction.
    Returns (z indices, positions) or None if the case has no prostate.
    """
    z = torch.nonzero((label == PROSTATE).any(dim=(1, 2))).flatten()
    if z.numel() == 0:
        return None
    z0, z1 = z.min().item(), z.max().item()
    bladder_z = torch.nonzero(label == BLADDER)[:, 0].float()
    base_is_high = bladder_z.numel() == 0 or bladder_z.mean().item() >= (z0 + z1) / 2
    zs = torch.arange(z0, z1 + 1)
    t = (zs - z0).float() / max(z1 - z0, 1)
    return zs, (t if base_is_high else 1 - t)


def analyse_case(pred, label):
    """Per-case numbers for one model's prediction."""
    dice, iou = dice_iou(pred, label, NUM_CLASSES)
    gt_p, pr_p = label == PROSTATE, pred == PROSTATE
    result = {"dice": dice.tolist(), "iou": iou.tolist(),
              "prostate_components": count_components(pr_p),
              # leakage: fraction of each organ's true voxels labelled as the other
              "rectum_as_prostate": ((label == RECTUM) & pr_p).sum().item() / max((label == RECTUM).sum().item(), 1),
              "bladder_as_prostate": ((label == BLADDER) & pr_p).sum().item() / max((label == BLADDER).sum().item(), 1),
              "prostate_as_rectum_or_bladder": (gt_p & ((pred == RECTUM) | (pred == BLADDER))).sum().item()
                                               / max(gt_p.sum().item(), 1)}

    sd = slice_dice(pr_p, gt_p)
    positions = apex_base_positions(label)
    if positions is not None:
        zs, t = positions
        result["slice_positions"] = t.tolist()
        result["slice_dice"] = sd[zs].tolist()
        inside = torch.zeros(label.shape[0], dtype=torch.bool)
        inside[zs] = True
        # predicted prostate on slices where there is none: overshooting past apex or base
        result["false_prostate_slices"] = int((pr_p.any(dim=(1, 2)) & ~inside).sum())
    return result


def profile(cases):
    """Mean slice Dice in N_BINS bins from apex to base, plus thirds, over all cases."""
    bins = [[] for _ in range(N_BINS)]
    for c in cases:
        for t, d in zip(c.get("slice_positions", []), c.get("slice_dice", [])):
            if not np.isnan(d):
                bins[min(int(t * N_BINS), N_BINS - 1)].append(d)
    means = [float(np.mean(b)) if b else float("nan") for b in bins]
    stds = [float(np.std(b)) if b else float("nan") for b in bins]
    third = N_BINS / 3
    thirds = {name: float(np.mean([d for k, b in enumerate(bins) if lo <= k < hi for d in b]))
              for name, lo, hi in (("apex", 0, third), ("mid", third, 2 * third), ("base", 2 * third, N_BINS))}
    return means, stds, thirds


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def error_overlay(image_slice, pred_slice, gt_slice):
    """RGB image: greyscale MR with prostate true positives green, false positives red, false negatives blue."""
    img = (image_slice - image_slice.min()) / (np.ptp(image_slice) + 1e-8)
    rgb = np.stack([img] * 3, axis=-1) * 0.7
    pred_p, gt_p = pred_slice == PROSTATE, gt_slice == PROSTATE
    rgb[pred_p & gt_p] = [0.1, 0.8, 0.1]
    rgb[pred_p & ~gt_p] = [0.95, 0.1, 0.1]
    rgb[~pred_p & gt_p] = [0.1, 0.4, 1.0]
    return rgb


def choose_slices(label, preds):
    """Three slices to show: worst apex-third slice, largest-prostate slice, worst base-third slice."""
    positions = apex_base_positions(label)
    if positions is None:
        return [label.shape[0] // 2]
    zs, t = positions
    first = next(iter(preds.values())) == PROSTATE
    sd = slice_dice(first, label == PROSTATE)[zs].nan_to_num(1.0)
    area = (label[zs] == PROSTATE).sum(dim=(1, 2))
    chosen = []
    for lo, hi in ((0, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 1.01)):
        idx = torch.nonzero((t >= lo) & (t < hi)).flatten()
        if idx.numel() == 0:
            continue
        pick = idx[area[idx].argmax()] if lo == 1 / 3 else idx[sd[idx].argmin()]
        chosen.append((zs[pick].item(), ["apex", "mid", "base"][[0, 1 / 3, 2 / 3].index(lo)]))
    return chosen


def plot_case(case_id, image, label, preds, results, path):
    """Rows: apex / mid / base slices. Columns: MR, ground truth, then prediction + error overlay per model."""
    slices = choose_slices(label, preds)
    slices = [s if isinstance(s, tuple) else (s, "middle") for s in slices]
    n_cols = 2 + 2 * len(preds)
    fig, axes = plt.subplots(len(slices), n_cols, figsize=(2.6 * n_cols, 2.6 * len(slices)), squeeze=False)
    img, lab = image.numpy(), label.numpy()
    for r, (z, where) in enumerate(slices):
        row = [("MR", img[z], "gray"), ("ground truth", lab[z], "label")]
        for name, pred in preds.items():
            p = pred.numpy()
            d = results[name]["dice"][PROSTATE]
            row.append((f"{MODEL_NAMES[name]}\nprostate Dice {d:.2f}", p[z], "label"))
            row.append((f"{MODEL_NAMES[name]} errors", error_overlay(img[z], p[z], lab[z]), "rgb"))
        for c, (title, data, kind) in enumerate(row):
            ax = axes[r, c]
            if kind == "label":
                ax.imshow(data, cmap=LABEL_CMAP, vmin=0, vmax=NUM_CLASSES - 1, interpolation="nearest")
            else:
                ax.imshow(data, cmap="gray" if kind == "gray" else None)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(title, fontsize=8)
            if c == 0:
                ax.set_ylabel(f"{where} (z={z})", fontsize=8)
    fig.suptitle(f"{case_id}   errors: green = correct prostate, red = false positive, blue = missed", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_profile(profiles, path):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    centres = (np.arange(N_BINS) + 0.5) / N_BINS
    for name, (means, stds, _) in profiles.items():
        means, stds = np.array(means), np.array(stds)
        ax.plot(centres, means, marker="o", label=MODEL_NAMES[name])
        ax.fill_between(centres, means - stds, means + stds, alpha=0.15)
    ax.set(xlabel="position along prostate (0 = apex, 1 = base)", ylabel="slice Dice (prostate)",
           ylim=(0, 1), title="Prostate Dice by slice position, test set")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models, train_args = {}, None
    for name in ("unet3d", "unet2d"):
        path = getattr(args, name)
        if path:
            models[name], a = load_model(path, device)
            if train_args and (a["seed"], a["shape"]) != (train_args["seed"], train_args["shape"]):
                raise ValueError("checkpoints were trained with different seeds or shapes; test sets would differ")
            train_args = a
            print(f"loaded {name} from {path} (epoch {torch.load(path, map_location='cpu')['epoch']})")

    test_set = load_test_set(args, train_args)
    print(f"test set: {len(test_set)} volumes")

    images, labels, preds, case_ids = [], [], {n: [] for n in models}, []
    with torch.no_grad():
        for i in range(len(test_set)):
            image, label, case_id = test_set[i]
            for name, model in models.items():
                logits = predict_logits(model, image[None].to(device), is_2d=(name == "unet2d"))
                preds[name].append(logits.argmax(1)[0].to(torch.uint8).cpu())
            images.append(image[0])
            labels.append(label.to(torch.uint8))
            case_ids.append(case_id)

    per_case = {n: [analyse_case(p.long(), l.long()) for p, l in zip(preds[n], labels)] for n in models}
    summary, profiles = {}, {}
    for name, cases in per_case.items():
        dice = np.array([c["dice"] for c in cases])
        iou = np.array([c["iou"] for c in cases])
        profiles[name] = profile(cases)
        summary[name] = {
            "dice_mean": dict(zip(CLASS_NAMES, np.nanmean(dice, 0).tolist())),
            "dice_std": dict(zip(CLASS_NAMES, np.nanstd(dice, 0).tolist())),
            "iou_mean": dict(zip(CLASS_NAMES, np.nanmean(iou, 0).tolist())),
            "prostate_dice_by_third": profiles[name][2],
            "prostate_dice_profile": profiles[name][0],
            "mean_prostate_components": float(np.mean([c["prostate_components"] for c in cases])),
            "cases_with_fragmented_prostate": sum(c["prostate_components"] > 1 for c in cases),
            "mean_false_prostate_slices": float(np.mean([c.get("false_prostate_slices", 0) for c in cases])),
            "mean_rectum_as_prostate": float(np.mean([c["rectum_as_prostate"] for c in cases])),
            "mean_bladder_as_prostate": float(np.mean([c["bladder_as_prostate"] for c in cases])),
            "mean_prostate_as_rectum_or_bladder": float(np.mean([c["prostate_as_rectum_or_bladder"] for c in cases])),
        }

    # Print a compact report
    for name, s in summary.items():
        print(f"\n=== {MODEL_NAMES[name]} ({len(case_ids)} test volumes) ===")
        for c in CLASS_NAMES:
            print(f"  {c:10s} Dice {s['dice_mean'][c]:.3f} +/- {s['dice_std'][c]:.3f}   IoU {s['iou_mean'][c]:.3f}")
        t = s["prostate_dice_by_third"]
        print(f"  prostate slice Dice  apex {t['apex']:.3f} | mid {t['mid']:.3f} | base {t['base']:.3f}")
        print(f"  prostate fragments: mean {s['mean_prostate_components']:.2f} components, "
              f"{s['cases_with_fragmented_prostate']} cases with >1; "
              f"{s['mean_false_prostate_slices']:.2f} slices of prostate beyond apex/base per case")
        print(f"  leakage: rectum->prostate {100 * s['mean_rectum_as_prostate']:.2f}%, "
              f"bladder->prostate {100 * s['mean_bladder_as_prostate']:.2f}%, "
              f"prostate->rectum/bladder {100 * s['mean_prostate_as_rectum_or_bladder']:.2f}%")

    plot_profile(profiles, out / "prostate_profile.png")

    # Failure autopsy: worst cases by prostate Dice of the first model (3D if given), plus the best as an example
    ranking_model = next(iter(models))
    prostate = np.nan_to_num([c["dice"][PROSTATE] for c in per_case[ranking_model]], nan=0.0)
    order = np.argsort(prostate)
    chosen = list(order[:args.n_worst]) + ([order[-1]] if len(order) > args.n_worst else [])
    for k in chosen:
        case_preds = {n: preds[n][k] for n in models}
        case_results = {n: per_case[n][k] for n in models}
        path = out / f"case_{case_ids[k]}.png"
        plot_case(case_ids[k], images[k], labels[k], case_preds, case_results, path)
        print(f"saved {path} (prostate Dice " +
              ", ".join(f"{n} {per_case[n][k]['dice'][PROSTATE]:.3f}" for n in models) + ")")

    results = {"summary": summary, "train_args": train_args,
               "per_case": {n: {cid: c for cid, c in zip(case_ids, cases)} for n, cases in per_case.items()}}
    (out / "predict_results.json").write_text(json.dumps(results, indent=1))
    print(f"\nsaved {out / 'prostate_profile.png'} and {out / 'predict_results.json'}")


if __name__ == "__main__":
    main()
