# rstar: R* inference for pediatric brain tumour MRI

`rstar` segments one co-registered scan with the recipe measured on 2026-09-26: the shipped 2D ensemble and the
four-member 3D dropout family, averaged 50/50 in probability space, with the background probability halved before the
argmax and a small enhancing-tumour clean-up rule.

> **Validated on US BraTS-PEDs only. Every outline needs radiologist review. This is research code, not a medical device.**
> The measured numbers (0.805 mean Dice on 30 patients never used to choose anything, 0.810 on the 81 held-out patients) come
> from the public US BraTS-PEDs data. Nothing here says how it behaves on other scanners, protocols or patient populations.

## Input contract

* A float array `(4, 240, 240, 155)`, channels in the order `t1c, t1n, t2f, t2w`, in BraTS space: 1 mm isotropic,
  skull-stripped (background exactly 0), all sequences co-registered to each other. Misaligned sequences are the largest
  measured risk: shifting T2w and FLAIR by 1, 2 and 3 mm costs about 0.02, 0.08 and 0.13 mean Dice.
* The module does **not** register, skull-strip, resample or read DICOM. Do that first, with the same pipeline BraTS used.
* A sequence that was not acquired is declared absent (`present=(True, True, False, True)`, or an omitted CLI flag).
* Broken input raises `ContractError` with the problem and the fix: wrong shape, NaN, negative values, a declared-present
  sequence that is all zero, or a sequence with under 15% exactly-zero voxels (it still has a skull).

## Modes and statuses

| Sequences present | Mode | What runs |
|---|---|---|
| all four | `R*` | 2D ensemble + 3D family, fused, small-ET rule |
| one to three | `3D-only` | the 3D family only (the 2D models collapse without every sequence: 0.14 without FLAIR), plain argmax, small-ET rule |

`status` is `ok` or `review`. `review` means: two or more sequences absent (not validated), or the 2D and 3D outlines
disagree (whole-tumour agreement below 0.70; below 0.10 the warning names a possible 2D collapse or a frame/orientation
error). The agreement flag is a weak signal (flagged patients scored 0.15-0.18 lower on average, but it catches few failures),
and labels are **never** withheld because of it.

## Use

```python
from rstar import RStarSegmenter, RStarConfig

seg = RStarSegmenter(RStarConfig(models_root="D:/models"))   # hash-verifies the checkpoints, runs a geometry self-check
result = seg.segment(volume, present=(True, True, True, True))
result.labels        # uint8 (240, 240, 155): 0 background, 1 enhancing, 2 non-enhancing core, 3 cystic, 4 edema
result.status, result.warnings, result.diagnostics
```

```bash
python -m rstar --t1c t1c.nii.gz --t1n t1n.nii.gz --t2f flair.nii.gz --t2w t2.nii.gz --out seg.nii.gz --json report.json
```

Exit code 0 = written, 2 = the input broke the contract.

## Review flags (optional)

With `RStarConfig(review_flags=True)` (CLI: `--review-mask flags.nii.gz`), R* also returns the enhancing-tumour spots it
is unsure about, so a reader can check them instead of their being silently kept or deleted. Rule, chosen in the
pre-registered review-flag study of 2026-10-02: every connected enhancing-tumour spot of at least 50 voxels, taken
**before** the 500 mm³ rule, whose mean enhancing-tumour probability is below 0.7 (`review_min_voxels`,
`review_prob_cut`). On the 30-patient clean test this removed every silent false spot (3 -> 0) at about one spot to
review per two patients.

`result.review_mask` (uint8: 0 = not flagged, k = spot k) and `result.review_spots` (a list of `spot_id`, `voxels`,
`mean_et_prob`, `models_agree` (always `null` here), `reason`) are `None` when flags are off. Flags are added
information only: the labels, mode, status and warnings are identical with flags on or off.

## Fragment cleanup (optional, off by default)

With `RStarConfig(fragment_cleanup=True)` (CLI: `--fragment-cleanup`), every connected piece of predicted tumour
smaller than 200 voxels (`cleanup_min_voxels`) is removed, and every enhancing-tumour piece smaller than that becomes
non-enhancing, before the 500 mm³ rule. Review flags are still computed from the uncleaned prediction. In the
pre-registered tests of 2026-10-04 (`PREREGISTERED_fragments_hybrid`, `PREREGISTERED_combo`) this left the usual Dice
unchanged on the 30-patient clean test and raised the official BraTS-PED lesion-wise Dice by about 0.07, because the
lesion-wise metric counts every stray fragment as a false lesion. The size of 200 was chosen on the held-out patients,
so the option stays off by default until cross-validation confirms it.

## Checkpoints

Checkpoints are not in git. `config/models.default.json` pins each file by path (relative to the models root: argument, else
`RSTAR_MODELS_ROOT`, else the repository root) and SHA-256; a mismatch raises `ModelIntegrityError` before the file is opened.
Use exactly these files: retraining a 3D member changes the fusion, and the recipe was validated with this family only.

## Tests

```bash
python -m pytest 06_rstar_inference/tests          # CPU only, stub models, no data
RSTAR_MODELS_ROOT=... RSTAR_TEST_DATA=... python -m pytest 06_rstar_inference/tests/test_regression.py   # real checkpoints + 3 patients
```
