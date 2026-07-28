# 03 — 2D Slice Augmentation + Ablation (BraTS-PEDs)

One entry point: `python run.py <command>`. Anything not reachable through it is
not part of this section.

---

## Run these, in this order

### 0. Install

```bash
pip install -r requirements.txt
```

### 1. Verify everything with no data and no model

```bash
python run.py test
```

Expected — the whole suite is CPU-only and never touches the real cache:

```
.........................................................................
85 passed in 11s

wrote: nothing (test run only; suite at .../03_augmentation_eval/tests)
```

This is the command to run first on a clean checkout. If it passes, the
pipeline is sound; only real data is missing.

### 2. Export 2D slices from the volumes

```bash
python run.py slices --out cache_2d
```

**Prerequisite:** the raw BraTS-PEDs NIfTI dataset, which lives on Ahmed's
laptop only. Point `paths.nifti_root` in `config.yaml` at it first.

Expected, with the data present:

```
planes exported: axial, coronal (sagittal dropped)
subjects written: 227 of 227
files written: 454
wrote: D:\Medical AI Workshop\cache_2d
```

Expected, without it — the command refuses rather than writing a partial cache:

```
error: raw NIfTI dataset not found: D:\Medical AI Workshop\BraTS-PEDs\train
  set paths.nifti_root in config.yaml to the BraTS-PEDs folder
```

Useful flags: `--limit 5` (first N subjects), `--manifest hospitalA`,
`--planes axial`.

### 3. Build the per-hospital balanced slice plan

```bash
python run.py plan --report
```

Expected:

```
  hospital   subjects   entries     tumor       ET     ET%     empty    empty%   augment     cap   max/pt   med/pt
------------------------------------------------------------------------------------------------------------------
 hospitalA         53      8000      6524     2000    25.0      1476      18.4         0     302      151      151
 hospitalB         92      8000      5600     2000    25.0      2400      30.0         0     174       89       87

wrote: .../plans/hospitalA_plan.json
wrote: .../plans/hospitalB_plan.json
```

Read it as: both hospitals get the same 8000-slice budget, never pooled; no
patient exceeds `cap`; empty (tumour-free) slices stay under 30%; at least 25%
of slices contain enhancing tumour.

**Two ratios, pulling opposite ways** — worth not confusing them:

| Key | Direction | Meaning |
|---|---|---|
| `empty_slice_ratio: 0.30` | **ceiling** | at most 30% tumour-free → ≥70% has tumour |
| `et_slice_ratio: 0.25` | **floor** | at least 25% contains enhancing tumour |

The empty ceiling fixes the raw imbalance (~70% of slices off the scanner have no
tumour, so an unweighted model learns to predict nothing). The ET floor fixes a
second, subtler one: balancing tumour-vs-empty is **subregion-blind**, and ET —
the smallest of the three scored regions, absent entirely in 34 of 82 held-out
subjects — lands at 2.8–4.9% of the plan without it. Set `et_slice_ratio: 0.0`
to turn it off and see the difference.

### 4. Sanity-check augmentation on a few slices

```bash
python run.py preview --n 8 --out preview.png
```

Expected:

```
previewed 8 slices (top row unaugmented, bottom row augmented)
wrote: .../preview.png
```

Top row is `use_augmentation: false`, bottom row is `true`, with the
segmentation overlaid. Add `--dummy-data` to run it with no cache at all.

### 5. Score a checkpoint on the held-out set

```bash
python run.py eval --checkpoint <path> --experiment-name baseline
```

To prove the path works before any model exists:

```bash
python run.py eval --experiment-name baseline --dummy-checkpoint --dummy-data
```

Expected:

```
baseline: ET=0.0024  NC=0.0340  WT=0.0591
  mean_dice=0.0318
  scored 4 subjects per patient from restacked volumes
  eval plane policy: axial
  subjects with an empty ground-truth region: ET=1, NC=0, WT=0
  wrote: .../results/ablation_results.csv
```

Those numbers are near zero because the weights are random. That is the point —
the command validates the pipeline, not a model.

---

## What this section decided

**2D, not 3D.** The team reversed the "full 3D volumes only" line in
`00_shared/CONTRACTS.md` deliberately. Slicing the native 1mm data instead of
resampling to 96³ removes the resampling ceiling entirely: round-tripping ground
truth through 96³ capped achievable Dice at ~0.93 WT and ~0.74–0.87 ET.

**Sagittal is dropped.** Axial (155 slices of 240×240) and coronal (240 slices
of 240×155) only.

**Padding, never resizing.** The two planes are brought to 256×256 by centred
zero-padding. Stretching coronal slices to square would distort aspect and
change tumour morphology. Padding is also exactly invertible, which is what lets
predictions be cropped back to native shape and restacked.

**Normalisation is per volume, not per slice.** Z-score over non-zero (brain)
voxels, computed once per volume at export time and stored in the cache as
`norm_mean` / `norm_std`. Per-slice normalisation would change what an intensity
means from slice to slice.

**Dice is scored per patient, not per slice.** Every slice of a subject is
predicted, un-padded, stacked back into a 240×240×155 volume, and scored once.
Per-slice averaging is not comparable to any BraTS number and badly overstates
results — ~70% of slices contain no tumour, and an empty slice predicted empty
scores 1.0. `tests/test_evaluate.py` measures the gap: a volume whose tumour sits
on one slice, predicted entirely empty, scores 0.0 per patient and >0.9 per
slice.

**Eval plane policy is explicit.** `eval.plane` defaults to `axial` — it has the
higher tumour-bearing fraction (~43% vs ~25%) and needs no probability
accumulator. `coronal` and `both` are also implemented; `both` averages per-class
softmax probabilities in the shared (C, 240, 240, 155) voxel grid, which is legal
because axial and coronal slices index the same volume, then argmaxes once.

**ET is largely presence/absence.** 34 of the 82 held-out subjects have no
enhancing tumour at all, so every run logs how many subjects had an empty
ground-truth region per region. Empty in both prediction and truth scores 1.0;
empty in exactly one scores 0.0.

---

## Leakage rules

The single most common failure mode in 2D medical imaging is a slice-level
split: with ~57,000 slices, adjacent near-identical slices of the same patient
land in both train and validation and you get a meaningless Dice around 0.95.

- **Patients are split before slicing** — the manifests define the split.
- A / B / held-out disjointness is asserted at subject level against the real
  manifests, and the assertion names the offending IDs
  (`tests/test_leakage.py`).
- Held-out is never augmented: `EvalSubjectDataset` ignores `use_augmentation`
  entirely, and a test asserts its output is identical with the flag on and off.
- No subject appears in more than one hospital's plan.

---

## The ablation flags

```yaml
use_augmentation: true    # masters the MONAI stack AND mixup
use_mixup: true           # nested inside it — no effect when the master is off
use_federation: false     # recorded in the CSV; implemented in section 01
use_domain_adaptation: false  # recorded in the CSV; implemented in section 02
```

All four augmentation conditions are reachable from `config.yaml` alone, with no
code change. `use_augmentation: false` disables mixup even when
`use_mixup: true`.

Results append to `results/ablation_results.csv` with the header fixed by
`00_shared/CONTRACTS.md`:

```
experiment_name,use_augmentation,use_federation,use_domain_adaptation,dice_ET,dice_NC,dice_WT,mean_dice
```

One row per run. A pre-existing file with a different header is a hard error
rather than a silent misaligned append.

---

## What this section does not own

- The 2D model architecture and the federated training loop — **section 01**.
  There is deliberately no `run.py train`.
- CORAL, PCA, LDA — **section 02**. This pipeline only makes them possible:
  `PatientBalancedBatchSampler` draws few slices from many patients (slices from
  one patient are near-duplicates and give a rank-deficient covariance) and keeps
  every batch plane-homogeneous, so CORAL can pair axial-A with axial-B. Plane is
  its own domain — axial vs coronal differ more than hospital A vs B.
- Clinical narrative — **section 04**. Streamlit demo — **section 05**.

---

## Layout

```
run.py                  the entry point: test | slices | plan | preview | eval
config.yaml             every knob; ablation flags at the top
SPEC.md                 what was built, as 53 testable requirements
src/
  config.py             YAML -> Config; the mixup gating rule
  slices.py             NIfTI -> 2D cache; volume-level norm stats; pad/unpad
  augment.py            MONAI 2D stack + segmentation mixup
  plan.py               per-hospital balanced slice plan + provenance
  dataset.py            SliceDataset, EvalSubjectDataset, batch sampler
  dummy.py              DummySegNet2D + synthetic cache (the only fake data)
  metrics.py            ET/NC/WT regions and Dice conventions (reused from 3D)
  evaluate.py           restack -> per-patient Dice -> one CSV row
tests/                  the suite behind `run.py test`
```

`run.py` is a thin dispatcher with no pipeline logic, so the CLI and the library
cannot drift apart.
