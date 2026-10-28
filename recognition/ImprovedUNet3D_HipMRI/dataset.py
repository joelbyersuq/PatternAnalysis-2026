"""
Data loading for 3D multi-class segmentation of the HipMRI prostate study.

Two data sources share the same interface:
  * HipMRI3D:        real Nifti volumes (MR + semantic labels) from Rangpur
  * SyntheticHipMRI: pelvis-like ellipsoid phantoms for CPU smoke tests (--synthetic)

Every sample is (image, label, case_id):
  image   float32 tensor (1, D, H, W), z-score normalised
  label   int64 tensor   (D, H, W), values in [0, NUM_CLASSES)
  case_id string, used later to trace failure cases back to a scan

Splits are made per patient, so every scan (week) of a patient lands in exactly one split.

Usage:
  python dataset.py --synthetic
  python dataset.py --root /home/groups/comp3710/HipMRI_Study_open --preview preview.png
"""
import argparse
import random
import re
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Label convention assumed for HipMRI semantic labels. TODO: verify against the real data.
CLASS_NAMES = ["background", "body", "bone", "bladder", "rectum", "prostate"]
NUM_CLASSES = len(CLASS_NAMES)
PROSTATE = CLASS_NAMES.index("prostate")

DEFAULT_SHAPE = (64, 128, 128)  # (D, H, W) every volume is resampled to


# ---------------------------------------------------------------------------
# File discovery and patient-level splitting
# ---------------------------------------------------------------------------

def find_cases(root):
    """
    Pair every MR volume with its label volume and tag it with a patient ID.

    This is the ONLY function that knows the on-disk layout and file naming.
    Assumed layout (check on Rangpur, change here if it differs):
        root/semantic_MRs/<PATIENT>_Week<N>_LFOV.nii.gz
        root/semantic_labels_only/<PATIENT>_Week<N>_SEMANTIC.nii.gz
    The scan key is "<PATIENT>_Week<N>" and the patient ID is "<PATIENT>".

    Returns a sorted list of (image_path, label_path, patient_id, case_id) tuples.
    """
    root = Path(root)
    image_dir = root / "semantic_MRs"
    label_dir = root / "semantic_labels_only"
    pattern = re.compile(r"^(?P<patient>.+?)_(?P<week>Week\d+)", re.IGNORECASE)

    def parse(name):
        m = pattern.match(name)
        if m is None:
            raise ValueError(f"Unexpected file name (update find_cases): {name}")
        return m.group("patient"), f"{m.group('patient')}_{m.group('week')}"

    labels = {parse(p.name)[1]: p for p in label_dir.glob("*.nii*")}
    cases = []
    for image_path in sorted(image_dir.glob("*.nii*")):
        patient_id, case_id = parse(image_path.name)
        if case_id not in labels:
            print(f"warning: no label for {image_path.name}, skipping")
            continue
        cases.append((image_path, labels[case_id], patient_id, case_id))

    if not cases:
        raise FileNotFoundError(f"No image/label pairs found under {root}")
    return cases


def split_by_patient(cases, val_frac=0.15, test_frac=0.15, seed=42):
    """
    Split cases into train/val/test so that no patient appears in more than one split.

    Patients (not scans) are shuffled with a fixed seed and allocated to splits, which
    prevents leakage between weekly scans of the same patient.
    """
    by_patient = defaultdict(list)
    for case in cases:
        by_patient[case[2]].append(case)

    patients = sorted(by_patient)
    random.Random(seed).shuffle(patients)
    n_test = max(1, round(len(patients) * test_frac))
    n_val = max(1, round(len(patients) * val_frac))
    groups = {
        "test": patients[:n_test],
        "val": patients[n_test:n_test + n_val],
        "train": patients[n_test + n_val:],
    }
    assert not (set(groups["train"]) & set(groups["val"]) | set(groups["train"]) & set(groups["test"])
                | set(groups["val"]) & set(groups["test"])), "patient leakage between splits"

    return tuple([c for p in groups[name] for c in by_patient[p]] for name in ("train", "val", "test"))


# ---------------------------------------------------------------------------
# Pre-processing and augmentation
# ---------------------------------------------------------------------------

def load_nifti(path, is_label):
    """Load a Nifti volume as a (D, H, W) tensor. Nifti stores (X, Y, Z); Z becomes depth."""
    data = nib.load(str(path)).get_fdata(caching="unchanged")
    if data.ndim == 4:  # some HipMRI files carry an extra trailing dimension
        data = data[..., 0]
    data = np.transpose(data, (2, 0, 1))
    if is_label:
        return torch.from_numpy(np.rint(data).astype(np.int64))
    return torch.from_numpy(np.ascontiguousarray(data, dtype=np.float32))


def normalise(image):
    """Per-volume z-score intensity normalisation."""
    return (image - image.mean()) / (image.std() + 1e-8)


def resample(image, label, shape):
    """Resize to a fixed (D, H, W): trilinear for the image, nearest for labels."""
    image = F.interpolate(image[None, None], size=shape, mode="trilinear", align_corners=False)[0, 0]
    label = F.interpolate(label[None, None].float(), size=shape, mode="nearest")[0, 0].long()
    return image, label


def augment(image, label, generator=None):
    """Random left-right flip and intensity scale/shift, applied identically to image and label."""
    if torch.rand(1, generator=generator) < 0.5:
        image, label = image.flip(-1), label.flip(-1)
    scale = 1 + (torch.rand(1, generator=generator) - 0.5) * 0.2
    shift = (torch.rand(1, generator=generator) - 0.5) * 0.2
    return image * scale + shift, label


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class HipMRI3D(Dataset):
    """
    Real HipMRI 3D volumes. Volumes are pre-processed once and kept in memory
    (about 5 MB per case at the default shape), so epochs do not re-read Nifti files.
    """

    def __init__(self, cases, shape=DEFAULT_SHAPE, train=False):
        self.case_ids = [c[3] for c in cases]
        self.train = train
        self.images, self.labels = [], []
        for image_path, label_path, _, case_id in cases:
            image = normalise(load_nifti(image_path, is_label=False))
            label = load_nifti(label_path, is_label=True)
            if image.shape != label.shape:
                raise ValueError(f"{case_id}: image {tuple(image.shape)} vs label {tuple(label.shape)}")
            image, label = resample(image, label, shape)
            self.images.append(image)
            self.labels.append(label.to(torch.uint8))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        image, label = self.images[i], self.labels[i].long()
        if self.train:
            image, label = augment(image, label)
        return image[None], label, self.case_ids[i]


class SyntheticHipMRI(Dataset):
    """
    Pelvis-like phantoms for testing the pipeline without real data.

    Each class is an ellipsoid with a jittered centre and size, roughly in its anatomical
    place (bladder anterior, rectum posterior and elongated, small prostate between them,
    bones lateral). Intensities differ per class, with noise and a linear bias field.
    Sample i is deterministic for a given seed.
    """

    def __init__(self, n, shape=DEFAULT_SHAPE, seed=0, train=False):
        self.n, self.shape, self.seed, self.train = n, shape, seed, train
        axes = [torch.linspace(-1, 1, s) for s in shape]
        self.grid = torch.stack(torch.meshgrid(*axes, indexing="ij"))  # (3, D, H, W) as (z, y, x)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        g = torch.Generator().manual_seed(self.seed * 100_000 + i)

        def ellipsoid(centre, radii):
            centre = torch.tensor(centre) + (torch.rand(3, generator=g) - 0.5) * 0.1
            radii = torch.tensor(radii) * (1 + (torch.rand(3, generator=g) - 0.5) * 0.2)
            d = ((self.grid - centre[:, None, None, None]) / radii[:, None, None, None]) ** 2
            return d.sum(0) <= 1

        # (class, centre (z, y, x), radii (z, y, x)); later entries overwrite earlier ones
        parts = [
            (1, (0, 0, 0), (0.95, 0.8, 0.9)),          # body
            (2, (0, 0.1, -0.6), (0.6, 0.2, 0.15)),     # bone (left)
            (2, (0, 0.1, 0.6), (0.6, 0.2, 0.15)),      # bone (right)
            (3, (0.2, -0.35, 0), (0.3, 0.25, 0.3)),    # bladder, anterior
            (4, (0, 0.4, 0), (0.6, 0.12, 0.12)),       # rectum, posterior
            (5, (-0.1, 0.1, 0), (0.15, 0.12, 0.13)),   # prostate, small
        ]
        label = torch.zeros(self.shape, dtype=torch.int64)
        for cls, centre, radii in parts:
            label[ellipsoid(centre, radii)] = cls

        intensity = torch.tensor([0.0, 0.4, 0.9, 0.2, 0.55, 0.7])
        bias = 0.2 * (self.grid * (torch.rand(3, 1, 1, 1, generator=g) - 0.5)).sum(0)
        image = intensity[label] + bias + 0.1 * torch.randn(self.shape, generator=g)
        image = normalise(image)

        if self.train:
            image, label = augment(image, label, generator=g)
        return image[None], label, f"synthetic_{self.seed}_{i:03d}"


def get_datasets(root=None, synthetic=False, shape=DEFAULT_SHAPE, seed=42, n_synthetic=(8, 2, 2)):
    """Build (train, val, test) datasets from the real data or the synthetic generator."""
    if synthetic:
        return tuple(SyntheticHipMRI(n, shape, seed=seed + k, train=(k == 0))
                     for k, n in enumerate(n_synthetic))
    train, val, test = split_by_patient(find_cases(root), seed=seed)
    return (HipMRI3D(train, shape, train=True), HipMRI3D(val, shape), HipMRI3D(test, shape))


def get_loaders(root=None, synthetic=False, shape=DEFAULT_SHAPE, batch_size=2, num_workers=0, seed=42):
    """Wrap get_datasets in DataLoaders. Only the training loader is shuffled."""
    train, val, test = get_datasets(root, synthetic, shape, seed)
    return (DataLoader(train, batch_size, shuffle=True, num_workers=num_workers),
            DataLoader(val, batch_size, num_workers=num_workers),
            DataLoader(test, batch_size, num_workers=num_workers))


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Check the HipMRI 3D data pipeline")
    parser.add_argument("--root", default="/home/groups/comp3710/HipMRI_Study_open")
    parser.add_argument("--synthetic", action="store_true", help="use synthetic phantoms instead of real data")
    parser.add_argument("--shape", type=int, nargs=3, default=DEFAULT_SHAPE, metavar=("D", "H", "W"))
    parser.add_argument("--preview", help="save a mid-slice image/label figure to this path")
    args = parser.parse_args()
    shape = tuple(args.shape)

    if not args.synthetic:
        cases = find_cases(args.root)
        splits = split_by_patient(cases)
        for name, split in zip(("train", "val", "test"), splits):
            print(f"{name:5s}: {len(split):3d} scans from {len({c[2] for c in split}):2d} patients")

    train_loader, val_loader, test_loader = get_loaders(args.root, args.synthetic, shape)
    image, label, case_ids = next(iter(train_loader))
    print(f"batch image {tuple(image.shape)} {image.dtype}, label {tuple(label.shape)} {label.dtype}")
    print(f"cases: {list(case_ids)}")

    counts = torch.bincount(label.flatten(), minlength=NUM_CLASSES).float()
    for name, frac in zip(CLASS_NAMES, counts / counts.sum()):
        print(f"  {name:10s} {100 * frac:6.2f}% of voxels")

    if args.preview:
        import matplotlib.pyplot as plt

        # Show the axial slice with the most prostate, falling back to the middle slice
        per_slice = (label[0] == PROSTATE).sum(dim=(1, 2))
        z = int(per_slice.argmax()) if per_slice.max() > 0 else shape[0] // 2
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(image[0, 0, z], cmap="gray")
        axes[0].set_title(f"{case_ids[0]} image, slice {z}")
        axes[1].imshow(label[0, z], cmap="tab10", vmin=0, vmax=9, interpolation="nearest")
        axes[1].set_title("label")
        for ax in axes:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(args.preview, dpi=100)
        print(f"saved preview to {args.preview}")


if __name__ == "__main__":
    main()
