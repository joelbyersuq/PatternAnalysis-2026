"""
Shared evaluation helpers for train.py and predict.py.

All metrics work on hard label maps (argmax of the logits), per volume, so the 3D model
and the 2D baseline (after segment_volume_2d) are scored identically.
"""
import time

import torch

from modules import segment_volume_2d


def dice_iou(pred, target, num_classes):
    """
    Per-class Dice and IoU for one volume.

    pred, target: integer label maps of the same shape.
    Returns two float tensors of length num_classes. A class absent from both the prediction
    and the ground truth gets NaN, so it is skipped by nanmean instead of counting as perfect.
    """
    dice = torch.full((num_classes,), float("nan"))
    iou = torch.full((num_classes,), float("nan"))
    for c in range(num_classes):
        p, t = pred == c, target == c
        inter = (p & t).sum().item()
        p_sum, t_sum = p.sum().item(), t.sum().item()
        if p_sum + t_sum == 0:
            continue
        dice[c] = 2 * inter / (p_sum + t_sum)
        iou[c] = inter / (p_sum + t_sum - inter)
    return dice, iou


def predict_logits(model, image, is_2d):
    """Logits (B, C, D, H, W) for a (B, 1, D, H, W) volume from either model type."""
    return segment_volume_2d(model, image) if is_2d else model(image)


@torch.no_grad()
def evaluate(model, loader, device, num_classes, is_2d, loss_fn=None, amp=False, timing=False):
    """
    Score a model over a loader of whole volumes.

    Returns a dict with the mean loss (if loss_fn is given), per-case Dice/IoU tensors of shape
    (num_cases, num_classes), the case IDs, and optionally per-case inference times in seconds.
    """
    model.eval()
    dices, ious, case_ids, times, losses = [], [], [], [], []
    for image, label, ids in loader:
        image, label = image.to(device), label.to(device)
        if timing and device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            logits = predict_logits(model, image, is_2d)
        if timing and device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        logits = logits.float()
        if loss_fn is not None:
            losses.append(loss_fn(logits, label).item())
        pred = logits.argmax(1)
        for b in range(image.shape[0]):
            d, i = dice_iou(pred[b], label[b], num_classes)
            dices.append(d)
            ious.append(i)
            case_ids.append(ids[b])
            times.append(elapsed / image.shape[0])

    result = {"dice": torch.stack(dices), "iou": torch.stack(ious), "case_ids": case_ids}
    if loss_fn is not None:
        result["loss"] = sum(losses) / len(losses)
    if timing:
        result["seconds_per_case"] = times
    return result
