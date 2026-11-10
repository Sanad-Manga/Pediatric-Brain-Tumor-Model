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
  targets. Which of the two is closer to the truth is decided per scan by `tools/visual_check.py` (below); on the
  BraTS cases in issue #65 it has not been run yet for want of the data.
- On already-aligned BraTS scans, 8 of 9 sequences are left byte-identical by the skip rule.
- End to end, realigning a deliberately misaligned scan flipped one tumour-free patient (00077) to a false enhancing
  spot: R*'s enhancing-tumour decision is fragile to small image changes.

Tests: `python -m pytest 08_preprocessing/tests` (28 tests, synthetic volumes, about 90 s).

## Is the step right, or was the scan already right? (issue #65)

On BraTS scans the step disagrees with BraTS's own inter-sequence alignment by 0.5-1.6 mm. `tools/visual_check.py`
decides, per patient and sequence, which of the two positions lines the sequence up with T1c better, and draws the
difference:

```
python tools/visual_check.py --t1c <p>-t1c.nii.gz --t1n <p>-t1n.nii.gz --t2f <p>-t2f.nii.gz \
    --t2w <p>-t2w.nii.gz --out-dir out/00127 --label BraTS-PED-00127-000
```

Writes `out/00127/<label>-{t2f,t2w}.png` - T1c in grey with each candidate's edges drawn over it (red = where the scan
has the sequence, cyan = where our step would put it) in three planes plus a zoom on the largest disagreement - and
`verdicts.json` with the verdict and every score behind it. `--ants` adds a slower second re-registration; `--seg`
centres the slices on the tumour.

The verdict comes from `prep/judge.py`, not from the picture. Both positions are resampled once onto a common
half-voxel-shifted grid (otherwise the untouched scan is the sharper one for free and every edge metric prefers it
whatever its position) and scored inside an eroded brain mask by four criteria the step does not optimise. It answers
"aligned better", "original better", "no disagreement" or "can't tell". See RESULTS_alignment.md, Addendum 2, and
`tools/check_judge.py` for the phantom check that it gets a known 1.5 mm answer right in both directions.
