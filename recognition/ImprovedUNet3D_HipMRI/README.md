# 3D Improved U-Net for Prostate Segmentation on HipMRI

> Work in progress. Sections marked TODO are filled in as the project develops.

## Problem and algorithm

TODO: one paragraph on the problem (prostate radiotherapy contouring on 3D pelvic MRI,
HipMRI study) and one paragraph on how the 3D Improved U-Net works (Isensee et al., 2018),
plus a figure.

## Feasibility review

TODO: one-page feasibility review presented at the midpoint check-off (task sheet Section 3):
user need and acceptance criteria, model choice, preliminary evidence, risks and fallback.

## Dependencies and reproducibility

See `requirements.txt`. TODO: Rangpur environment versions, random seeds, exact commands.

## Data, pre-processing and splits

TODO: data layout, intensity normalisation, resampling, augmentation, and justification of
the patient-level train/validation/test split.

## Usage

```bash
# Smoke test the data pipeline on CPU without the real data
python dataset.py --synthetic
```

TODO: training and prediction commands.

## Results

### Baseline (3D U-Net) vs 3D Improved U-Net

TODO: per-class Dice and IoU on the held-out test split.

### Resource profiling

TODO: peak GPU VRAM, parameter count, per-case inference time.

### Failure case autopsy

TODO: 3–5 boundary failure cases (prostate base/apex slices, disconnected artefacts).

### Recommendation

TODO: does 3D context justify its cost for an operational contouring pipeline?

## Artificial Intelligence Usage Disclosure

| Date | Tool | What it helped with | How I verified it |
|------|------|---------------------|-------------------|
| 2026-10-28 | Claude Code (Claude Opus 5.5) | Repository setup, project folder scaffold, README skeleton, first version of `dataset.py` (synthetic generator, file pairing, patient-level split) | TODO |

## References

1. F. Isensee et al., "Brain tumor segmentation and radiomics survival prediction: Contribution
   to the BRATS 2017 challenge," *Brainlesion*, 2018.
2. Ö. Çiçek et al., "3D U-Net: Learning dense volumetric segmentation from sparse annotation,"
   *MICCAI*, 2016.
