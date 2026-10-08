# 08_preprocessing: preparing hospital scans for R*

R* expects BraTS-style input: the four sequences (T1c, T1n, T2-FLAIR, T2w) aligned to each other, skull-stripped, on a
1 mm grid. This module will hold the steps that turn a hospital scan into that. **Part 1, done: sequence alignment.**
Skull stripping, DICOM conversion and resampling wait for the first real hospital scans, so they are built against real
data.

## Alignment

Each of T1n, T2-FLAIR and T2w is rigidly registered to the same patient's T1c (rotation + translation, mutual
information) and resampled onto the T1c grid. A sequence the registration would move by less than 1 mm on average is
left untouched (resampling an already-aligned scan only blurs it).

```
python -m prep.align --t1c t1c.nii.gz --t1n t1n.nii.gz --t2f flair.nii.gz --t2w t2.nii.gz --out-dir aligned/
```

Writes `aligned/aligned-{t1c,t1n,t2f,t2w}.nii.gz` and `aligned/alignment.json` (per sequence: rotation, translation,
displacement, whether it was skipped). `--skip-below-mm 0` always resamples. Needs SimpleITK (installed in
`C:\Users\ahmed\nnunet_env`).

## How well it works (see RESULTS_alignment.md)

- Undoes 2-3 mm / 2-5 degree misalignments to a consistent position (median 0.12 mm between repeated recoveries).
- Disagrees with BraTS's own alignment by 0.5-1.5 mm on some FLAIR/T2w scans, so it does not meet its original 0.5 mm
  targets; which of the two is closer to the truth is not settled.
- On already-aligned BraTS scans, 8 of 9 sequences are left byte-identical by the skip rule.
- End to end, realigning a deliberately misaligned scan flipped one tumour-free patient (00077) to a false enhancing
  spot: R*'s enhancing-tumour decision is fragile to small image changes.

Tests: `python -m pytest 08_preprocessing/tests` (13 tests, synthetic volumes, about 15 s).
