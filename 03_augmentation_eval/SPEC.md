# Spec: Section 03 — 2D Slice Augmentation + Ablation (BraTS-PEDs)

Supersedes the 3D implementation currently in `03_augmentation_eval/`. Source brief:
`BRIEF_2D.md`. Shared contract: `00_shared/CONTRACTS.md` (its "full 3D volumes only,
no 2D slices" line is deliberately overridden by the team — 2D is settled).

---

## 1. Goal

A single-entry-point 2D pipeline that turns BraTS-PEDs volumes into axial + coronal
slices, applies a config-gated MONAI augmentation stack and segmentation mixup, builds a
leakage-free per-hospital balanced slice plan, and scores a checkpoint **per patient**
from restacked volumes into one contract-exact row of `results/ablation_results.csv`.

---

## 2. In Scope

- **One CLI entry point** — `python run.py <command>` with exactly five commands:
  `test`, `slices`, `plan`, `preview`, `eval`. Nothing in this section is reachable any
  other way.
- **Slice extraction** (`slices`) — read NIfTI volumes, drop sagittal, emit axial +
  coronal slices per subject to a `.npz` cache, carrying per-volume normalization
  statistics and full provenance.
- **2D augmentation stack** (MONAI) — random flip, rotate, zoom, intensity scale/shift,
  Gaussian noise; spatial transforms applied jointly to image and mask with
  nearest-neighbour interpolation on the mask.
- **2D segmentation mixup** — same λ ~ Beta(α, α) applied to the image and to the
  **one-hot mask**, same-plane pairs only.
- **Balanced slice plan** (`plan`) — per-hospital slice budget with a per-patient cap,
  augmentation-based up-sampling of under-represented patients, hospitals never pooled,
  full provenance on every entry.
- **Config gating** — `use_augmentation` masters the entire augmentation branch with
  `use_mixup` nested inside it; all four combinations reachable from config alone.
- **Augmentation preview** (`preview`) — PNG grid of original vs augmented slices.
- **Ablation runner** (`eval`) — checkpoint → held-out inference → restack to volume →
  per-patient ET/NC/WT Dice → one appended CSV row matching the contract byte-for-byte.
- **Patient-diverse, plane-homogeneous batch sampler** — so section 02's CORAL is usable.
- **Dummy test suite** — CPU-only, random tensors, dummy 2D network, no real cache and no
  trained model required.
- **README.md** opening with copy-pasteable commands in run order, each with its expected
  output.

## 3. Out of Scope

- The 2D model architecture and the federated training loop — section 01. **There is no
  `run.py train` command.**
  **Assumption:** `BRIEF_2D.md` §0 lists `train` among example commands, but §7 puts the
  training loop out of scope and the §8 required README shape omits it. §7 wins; a `train`
  command would duplicate section 01's ownership.
- CORAL, PCA, LDA — section 02. This spec only guarantees the sampler and model interface
  that make CORAL possible.
- Clinical narrative (section 04) and the Streamlit demo (section 05).
- Any real trained model, real training, or GPU code paths. Everything here runs on CPU.
- Brain-box cropping — the brain fills ~90–96% of the frame, so it buys nothing.
- Sagittal slices, and any 3D volumetric augmentation.
- Modifying `00_shared/CONTRACTS.md` or any other section's files.
- Re-running or replacing the existing 3D modules on other branches; the 3D code in this
  directory is replaced in place on this branch only.

---

## 4. Requirements

### Entry point and packaging

1. `python run.py` with no arguments, or `python run.py --help`, prints the five command
   names (`test`, `slices`, `plan`, `preview`, `eval`) and exits 0.
2. Every command prints, on success, a line naming each file it wrote and its absolute
   path. A command that writes nothing says so explicitly.
3. `requirements.txt` exists and pins the top-level dependencies (`torch`, `monai`,
   `numpy`, `pyyaml`, `nibabel`, `matplotlib`, `pytest`).
4. `python run.py test` runs the full suite on CPU with no real cache, no NIfTI data and
   no trained checkpoint, and exits 0. It must not read `paths.cache_2d`.

> **Amended after the team confirmed the data contract.** The 2D slices are produced from
> the original volumes by the team and are the dataset from now on — this section consumes
> them rather than exporting them. Requirements 5–9 below are superseded by §A′; the
> `slices` command is replaced by `index`. Everything else stands unchanged.

### A. Slice extraction (`run.py slices`) — SUPERSEDED, see §A′

5. Given a source volume of shape `(240, 240, 155)` the extractor emits **axial** slices
   along the last axis (155 slices of `240×240`) and **coronal** slices along the middle
   axis (240 slices of `240×155`). Sagittal (first axis) is never emitted.
   **Assumption:** axis→plane mapping is `axis 0 = sagittal, axis 1 = coronal,
   axis 2 = axial`, which is the only assignment consistent with the slice counts and
   shapes measured in `BRIEF_2D.md` §2.
6. Output is one `.npz` per subject per plane at
   `<cache_2d>/<subject_id>_<plane>.npz`, `plane ∈ {axial, coronal}`, containing:
   - `t1c`, `t1n`, `t2f`, `t2w` — `(N, H, W)` `float16`, **raw un-normalized**
   - `seg` — `(N, H, W)` `uint8`, values ⊆ `{0,1,2,3,4}`
   - `norm_mean`, `norm_std` — `(4,)` `float32`, per-modality z-score statistics computed
     over **non-zero voxels of the whole volume** (not per slice), in the modality order
     `["t1c","t1n","t2f","t2w"]`
   - `has_tumor` — `(N,)` `bool`, true where `seg[i]` has any non-zero voxel
   - `has_et` — `(N,)` `bool`, true where `seg[i]` contains label `1`
   - `subject_id`, `plane`, `hospital` — scalar strings
7. Normalization is applied at load time as `(x - norm_mean[m]) / norm_std[m]` using the
   **volume-level** statistics stored in the cache. A test asserts that two different
   slices of the same volume are normalized with identical constants.
8. `norm_std` is clamped to a minimum of `1e-6`; a modality whose volume is entirely zero
   yields an all-zero normalized slice rather than NaN or a divide-by-zero.
9. `run.py slices` accepts `--out <dir>`, `--limit <n>`, `--manifest <name|all>` and
   `--planes axial,coronal`. With no readable NIfTI source it exits non-zero with a
   message naming the missing path and the config key that sets it — it never writes a
   partial cache silently.
   **Assumption:** section 03 owns this exporter. `BRIEF_2D.md` §7 leaves ownership open
   ("confirm who owns it"), but §0 requires everything be reachable through `run.py`, so
   it ships here; it is a thin `nibabel` reader and costs little.

### A′. Slice cache and index (`run.py index`) — replaces Reqs 5–9

55. The cache layout is fixed by the team's exporter, one file per slice:
    `<cache_2d>/<subject_id>/<plane>/slice_<NNN>.npz`, `plane ∈ {axial, coronal}`, with
    - `image` — `(4, H, W)` `float32`, channels **`t1c, t1n, t2f, t2w`** in that order,
      already z-scored per volume over non-zero (brain) voxels
    - `mask` — `(1, H, W)` `uint8`, values ⊆ `{0,1,2,3,4}`

    The four modalities are **not** stored as separate arrays; `image` is already the
    stacked form the model consumes. Nothing in this section re-normalises or re-stacks.
56. `run.py index` scans every mask once and writes `<cache_2d>/index.json` recording, per
    subject and plane: `indices` (every slice present), `tumor` (mask has any non-zero
    label) and `et` (mask contains label `1`). `plan` reads that index, never the masks.
    Rationale: tumour presence is needed *before* slice selection, and deriving it means
    opening ~90,000 files.
57. **Every list in the index holds actual slice indices, never positions.** Subjects whose
    blank edge slices were dropped run e.g. `slice_008..slice_134`; the number in the
    filename is the volume position. A missing index file raises, naming the command that
    builds it.
58. `restack_volume` places each predicted slice at its **true volume position** into the
    full `data.volume_shape`, leaving dropped slices as background. This keeps axial and
    coronal restacks the same shape — which `eval.plane: both` requires in order to average
    them — and keeps a prediction aligned with its ground truth. A dropped slice is
    background in prediction and truth alike, so Dice is unaffected.
59. `run.py index` warns if the cache looks **per-slice** normalised (brain mean/std pinned
    to 0/1 on every slice) rather than per-volume, since that is invisible later and
    changes what an intensity means from slice to slice.
60. The cache is verified **internally coherent**: restacking a subject's masks from axial
    and from coronal yields bitwise-identical volumes.

### Shape harmonization

10. Axial (`240×240`) and coronal (`240×155`) slices are brought to one common network
    size of `256×256` by **zero-padding only** (`data.resize_mode: "pad"`). No resize, no
    stretch, no aspect change. Padding is centered; odd remainders put the extra row/column
    at the end.
11. The padding applied is recorded per sample as `orig_shape` and `pad` = `((top, bottom),
    (left, right))`, and `unpad(pad(x)) == x` exactly, for both plane shapes. A test
    asserts the round-trip is bitwise-identical.
12. A source slice larger than the common size is a hard error naming both shapes — it is
    never centre-cropped silently.

### B. 2D augmentation stack

13. The augmentation stack is MONAI dictionary transforms over keys `{"image", "label"}`
    and includes, in order: random flip per spatial axis, random rotate, random zoom,
    random intensity scale, random intensity shift, random Gaussian noise.
14. Every **spatial** transform (flip, rotate, zoom) applies to image and label together;
    the label always uses nearest-neighbour interpolation. Intensity transforms apply to
    the image only.
15. After augmentation the label value set is ⊆ `{0,1,2,3,4}` — asserted in the transform
    pipeline itself (raising on violation), not only in tests. No fractional labels.
16. Augmented image shape equals input image shape `(4, 256, 256)`; augmented label shape
    equals `(1, 256, 256)`.
17. With `use_augmentation: false`, loading the same plan entry twice returns
    **bitwise-identical** arrays (`np.array_equal` on both image and label).
18. With `use_augmentation: true`, loading the same plan entry twice with different
    `aug_seed` produces at least one differing pixel, while shapes and the label value set
    are unchanged.
19. Given the same `aug_seed`, augmentation is reproducible: two loads produce
    bitwise-identical output.

### C. Segmentation mixup

20. `mixup(img_a, oh_a, img_b, oh_b, alpha)` draws one `λ ~ Beta(alpha, alpha)` and
    returns `λ·img_a + (1-λ)·img_b` and `λ·oh_a + (1-λ)·oh_b`. Scalar labels are never
    mixed.
21. The mixed one-hot sums to `1.0 ± 1e-5` across the channel axis at every pixel, and has
    shape `(5, 256, 256)`.
22. Mixup raises `ValueError` when the two samples come from different planes; the batch
    mixup helper only ever pairs samples within the same plane.
23. Mixup runs only when `use_augmentation and use_mixup` are both true.
    `use_augmentation: false` with `use_mixup: true` performs **no** mixup — asserted by a
    test on `Config.mixup_enabled` and on the dataset output.

### D. Balanced expansion (`run.py plan`)

24. The plan is built per hospital from `hospitalA.json` and `hospitalB.json` only.
    Hospital A and Hospital B entries are never pooled, and no held-out subject ever
    appears in a plan.
25. Each hospital gets `expansion.target_per_hospital` slice entries (default **8000**);
    if a hospital cannot reach the target even after up-sampling it emits what it can and
    logs the shortfall rather than borrowing from the other hospital.
26. Slice selection keeps **all** tumor-bearing slices (subject to the per-patient cap) and
    subsamples empty slices so that empty slices are at most
    `expansion.empty_slice_ratio` (default **0.30**) of that hospital's entries.
27. No patient dominates: each patient contributes at most
    `ceil(expansion.cap_multiplier * target_per_hospital / n_patients_in_hospital)` entries
    (`cap_multiplier` default **2.0**). A test asserts every patient's share is under the
    cap and that the maximum patient share is ≤ `cap_multiplier / n_patients`.
28. Patients with fewer available slices than their fair share are up-sampled by repeating
    slices with `is_augmented: true` and distinct `aug_seed` values; original (unaugmented)
    entries are emitted before any augmented copy of the same slice.
29. Each hospital keeps its own patient-level composition: **every** subject in a
    hospital's manifest that has at least one cached slice appears at least once in that
    hospital's plan. A and B are not forced to match each other.
30. Every plan entry carries exactly these provenance fields: `hospital`,
    `source_subject_id`, `plane`, `slice_index`, `is_augmented`, `aug_seed`. A test asserts
    no entry is missing any field and that `aug_seed` is `null` iff `is_augmented` is false.
31. `run.py plan --report` prints a per-hospital table with: subjects, entries, tumor
    entries, empty entries, empty fraction, augmented entries, per-patient cap, and
    max/median entries per patient — and writes the plan to `plans/<hospital>_plan.json`.
32. Plan construction is deterministic for a fixed `expansion.seed`: two runs produce
    byte-identical plan JSON.

### Leakage rules

33. A test asserts the three manifests are **disjoint at subject level** (A ∩ B, A ∩
    held-out, B ∩ held-out all empty) and fails loudly naming the offending IDs.
34. Splitting is by patient, never by slice: no subject ID appears in more than one plan,
    and the eval subject list is exactly the held-out manifest.
35. Held-out and validation slices are **never** augmented — the eval dataset ignores
    `use_augmentation` entirely and a test asserts that eval output is identical with the
    flag on and off.

### CORAL-enabling sampler

36. `PatientBalancedBatchSampler` yields batches drawn from **distinct patients**, taking
    at most `sampler.max_slices_per_patient_per_batch` (default **1**) slices per patient
    per batch.
37. Every batch it yields is **plane-homogeneous** (all axial or all coronal), so section
    02 can form plane-matched CORAL pairs.
38. A test asserts that for a batch size of 8 over a 20-patient plan, every yielded batch
    has 8 distinct `source_subject_id` values and one distinct `plane`.

### F. Ablation runner (`run.py eval`)

39. Dice is computed **per patient from restacked volumes**, never per slice: all slices of
    a subject are predicted, un-padded to their native shape, stacked back into a
    `(240, 240, 155)` label volume, and one ET/NC/WT Dice is computed per subject, then
    averaged across subjects.
40. **Plane policy is explicit:** `eval.plane` ∈ `{axial, coronal, both}`, default
    **`axial`**. `axial`/`coronal` restack argmax class labels along that plane's axis.
    `both` averages per-class **softmax probabilities** in the shared `(C, 240, 240, 155)`
    voxel grid — legal because axial and coronal slices index the same volume — and takes
    the argmax once at the end. The chosen policy is printed on every run.
    **Assumption:** `axial` is the default because it has the higher tumor-bearing slice
    fraction (~43% vs ~25%) and needs no probability accumulator; `both` is implemented and
    tested but not the default.
41. Regions follow `metrics.py` unchanged: `ET = {1}`, `NC = {1,2,3}`, `WT = {1,2,3,4}`.
42. Empty-region conventions, asserted by tests: empty in **both** prediction and ground
    truth → Dice `1.0`; empty in **exactly one** → Dice `0.0`.
43. Every run logs how many subjects had an empty **ground-truth** region, per region, so
    the mean stays interpretable (34 of 82 held-out subjects have no ET).
44. Exactly one row is appended to `results/ablation_results.csv` per run. The header is
    byte-for-byte:
    `experiment_name,use_augmentation,use_federation,use_domain_adaptation,dice_ET,dice_NC,dice_WT,mean_dice`
45. `mean_dice` equals the mean of the three rounded region Dice values, to 6 decimals.
46. Appending to a CSV whose existing header differs from the contract raises an error and
    writes nothing.
47. `run.py eval --dummy-checkpoint --dummy-data` completes with no real cache and no
    trained model and appends a valid row — this is the proof the path works.
48. Without `--checkpoint` and without `--dummy-checkpoint`, `eval` exits non-zero with a
    message naming both options.

### E. Config flag

49. `use_augmentation` gates the MONAI stack **and** mixup; `use_mixup` is nested inside
    it. All four conditions — (aug off), (aug on, mixup off), (aug on, mixup on), and
    (aug off, mixup on ≡ aug off) — are reachable by editing `config.yaml` alone, with no
    code change, and each is covered by a test.
50. `use_federation` and `use_domain_adaptation` are read from config and written to the
    CSV row unchanged; this section does not implement either.

### Preview

51. `run.py preview --n 8 --out preview.png` writes a PNG grid of `n` slices showing
    original and augmented image with the label overlaid, and prints the output path. It
    works against dummy data via `--dummy-data`.

### README

52. `README.md`'s first section is a copy-pasteable command block in run order — install,
    `test`, `slices`, `plan --report`, `preview`, `eval` — each with the expected output
    stated beneath it.
53. Every command in that block runs successfully on a clean checkout, except `slices`,
    which is documented as requiring the raw NIfTI dataset and states the exact error shown
    when it is absent.

### ET floor (added after the initial build, at Ahmed's request)

The tumour-vs-empty balance in Req 26 is **subregion-blind**: a slice containing only
edema counts the same as one containing the enhancing core. ET is the smallest of the
three scored regions and is absent entirely in 34 of 82 held-out subjects, so a
tumour-only balance leaves it badly under-represented — measured at **2.8% (hospital A)
and 4.9% (hospital B)** of the plan before this was added.

54. `expansion.et_slice_ratio` (default **0.25**) is a **floor** on the fraction of a
    hospital's plan whose slices contain ET (label `1`), honoured whenever cache supply
    allows. Specifically:
    - ET slices take first claim on the tumour budget, before non-ET tumour slices.
    - Once the floor is met, selection returns to general tumour slices, so a hospital
      with plentiful ET does not over-weight it.
    - A hospital whose cache holds fewer ET slices than the floor emits what exists,
      logs the shortfall, and still meets `target_per_hospital` — it never invents ET.
    - Augmented up-sampling follows the same preference (ET → other tumour → empty) but
      stops drawing ET once the floor is met, so a handful of ET slices are not
      duplicated dozens of times.
    - `et_slice_ratio: 0.0` disables the reservation entirely.
    - The empty ceiling (Req 26) and the per-patient cap (Req 27) both still hold.
    - `plan --report` gains `ET` and `ET%` columns; `stats` gains `et_entries`,
      `et_fraction`, `et_available`, `et_floor`.

---

## 5. Structure

```
03_augmentation_eval/
├── run.py                  # THE entry point: test | slices | plan | preview | eval
├── config.yaml             # every knob; the ablation flags live at the top level
├── requirements.txt
├── README.md               # opens with the copy-pasteable run block
├── SPEC.md
├── BRIEF_2D.md             # (existing) source brief
├── src/
│   ├── __init__.py
│   ├── config.py           # YAML → Config dataclass; `mixup_enabled` gating logic
│   ├── slices.py           # NIfTI → 2D .npz cache; volume-level norm stats; pad/unpad
│   ├── augment.py          # MONAI 2D dict-transform stack + segmentation mixup
│   ├── plan.py             # per-hospital balanced slice plan + provenance
│   ├── dataset.py          # SliceDataset, EvalSubjectDataset, PatientBalancedBatchSampler
│   ├── dummy.py            # DummySegNet2D + synthetic cache/plan generators
│   ├── metrics.py          # REUSED AS-IS from the 3D version (dimension-agnostic)
│   └── evaluate.py         # restack → per-patient Dice → one CSV row (CSV writer reused)
├── plans/                  # written by `run.py plan`
├── results/
│   └── ablation_results.csv  # appended by `run.py eval`
└── tests/
    ├── conftest.py         # tmp synthetic cache fixtures
    ├── test_slices.py      # planes, shapes, norm stats, pad round-trip
    ├── test_augment.py     # determinism, label set, mixup one-hot, same-plane rule
    ├── test_plan.py        # balance, per-patient cap, provenance, determinism
    ├── test_leakage.py     # subject-level disjointness, held-out never augmented
    └── test_evaluate.py    # restacking, empty-region conventions, CSV schema
```

Roles worth stating:

- **`run.py`** is a thin argparse dispatcher. It contains no pipeline logic — every command
  is a call into one `src/` function, so the CLI and the library cannot drift.
- **`src/metrics.py`** is copied unchanged from the current 3D implementation. Its region
  logic and empty-region conventions are dimension-agnostic and already tested.
- **`src/evaluate.py`** reuses the `append_result_row` CSV writer from the 3D `runner.py`
  verbatim, including its "existing header must match" guard.
- **`src/dummy.py`** exists so the whole suite runs with no data; it is the only place that
  fabricates tensors.
- Files removed on this branch: `src/data.py`, `src/strata.py`, `src/expansion.py`,
  `src/augmentation.py`, `src/mixup.py`, `src/runner.py`, `src/dummy_model.py`,
  `tests_dummy.py` — all volume-shaped, rewritten rather than patched.

### Config keys added or changed vs the 3D `config.yaml`

```yaml
data:
  planes: ["axial", "coronal"]     # sagittal deliberately absent
  common_size: [256, 256]
  resize_mode: "pad"               # pad only — never "resize"
  volume_shape: [240, 240, 155]
  plane_axis: {sagittal: 0, coronal: 1, axial: 2}

expansion:
  target_per_hospital: 8000
  empty_slice_ratio: 0.30          # CEILING on tumour-free slices
  et_slice_ratio: 0.25             # FLOOR on slices containing ET (label 1)
  cap_multiplier: 2.0
  seed: 1337

sampler:
  max_slices_per_patient_per_batch: 1
  plane_homogeneous_batches: true

eval:
  plane: "axial"                   # axial | coronal | both
```

`strata:` (the 3D tumor-type proxy) is deleted — the 2D balance rule is patient-level, not
tumor-type-level.

---

## 6. Edge Cases

| Scenario | Expected Behaviour |
|---|---|
| `paths.cache_2d` does not exist during `plan`/`preview`/`eval` | Exit non-zero: `2D slice cache not found: <abs path> (set paths.cache_2d in config.yaml, or run `python run.py slices --out <dir>`)`. Never falls back to random data unless `--dummy-data` is passed. |
| Raw NIfTI source missing during `slices` | Exit non-zero naming the missing path and the config key. No partial cache written. |
| `nibabel` not installed and `slices` invoked | Exit non-zero: `slices requires nibabel (pip install -r requirements.txt)`. Other commands still work. |
| Subject listed in a manifest but absent from the cache | Warn once per subject, skip it, and print the skipped count in the summary. If **all** subjects of a hospital are missing, exit non-zero. |
| Subject volume is not `(240,240,155)` | Slice it anyway using the configured axis map, record the true `orig_shape`, and warn. Only a slice exceeding `common_size` is fatal. |
| A modality volume is entirely zero | `norm_std` clamped to `1e-6`; normalized slice is all zeros. No NaN, no warning spam (one warning per subject/modality). |
| Subject has zero tumor-bearing slices in a plane | Its empty slices remain eligible; the subject still appears in the plan (Req 29) but contributes only empty entries. |
| `empty_slice_ratio` unreachable — a hospital has too few tumor slices | Emit what exists, log the achieved ratio, do not pad with duplicates just to hit the ratio. |
| Requested `target_per_hospital` exceeds cap × n_patients | Emit `cap × n_patients` entries and log the shortfall explicitly. |
| Mixup with `alpha <= 0` | Raise `ValueError("mixup.alpha must be > 0")`. |
| Mixup pairing an axial with a coronal slice | Raise `ValueError` naming both planes. |
| Augmentation produces a label outside `{0,1,2,3,4}` | Raise `AssertionError` inside the pipeline naming the offending values — never silently clipped. |
| A slice larger than `common_size` | Raise `ValueError` naming source shape and common size. |
| Region empty in both prediction and ground truth | Dice `1.0`. |
| Region empty in exactly one of the two | Dice `0.0`. |
| All 82 held-out subjects lack ET | `dice_ET` is the mean over the presence/absence outcomes; the empty-GT count line reports `ET=82`. |
| Existing `ablation_results.csv` has a different header | Raise `ValueError` showing expected vs found; nothing appended. |
| `results/` does not exist | Created automatically before the first append. |
| `eval` given neither `--checkpoint` nor `--dummy-checkpoint` | Exit non-zero naming both options. |
| `eval --checkpoint` pointing at a missing file | `FileNotFoundError` with the path. |
| Subject with slices in only one plane while `eval.plane: both` | Use whatever planes exist, log which subjects were single-plane, and count it in the summary. |
| Empty evaluation set (no subjects loadable) | Raise `ValueError("evaluation set is empty; nothing to score")` — never writes a row of `1.0`s. |
| `run.py` invoked with an unknown command | argparse error listing the five valid commands, exit non-zero. |

---

## 7. Done Checklist

**Entry point and packaging**
- [ ] Req 1: `run.py` / `run.py --help` lists the five commands, exits 0
- [ ] Req 2: every command prints what it wrote and where
- [ ] Req 3: `requirements.txt` pins torch, monai, numpy, pyyaml, nibabel, matplotlib, pytest
- [ ] Req 4: `python run.py test` passes on CPU with no data, no model, no cache access

**Slice extraction**
- [ ] Req 5: axial along axis 2 (155×240×240), coronal along axis 1 (240×240×155); sagittal never emitted
- [ ] Req 6: `<subject>_<plane>.npz` holds 4 modalities (float16), `seg` (uint8), `norm_mean`/`norm_std`, `has_tumor`, and string provenance
- [ ] Req 7: normalization uses volume-level stats; two slices of a volume share identical constants
- [ ] Req 8: `norm_std` clamped to 1e-6; all-zero modality → all-zero slice, no NaN
- [ ] Req 9: `slices` supports `--out/--limit/--manifest/--planes`; missing source exits non-zero naming path and config key

**Shape harmonization**
- [ ] Req 10: 240×240 and 240×155 padded (never resized) to 256×256, centered
- [ ] Req 11: `orig_shape` and `pad` recorded; `unpad(pad(x)) == x` bitwise for both shapes
- [ ] Req 12: slice larger than common size is a hard error naming both shapes

**Augmentation**
- [ ] Req 13: MONAI dict stack — flip, rotate, zoom, intensity scale, intensity shift, Gaussian noise
- [ ] Req 14: spatial transforms joint on image+label, label nearest-neighbour; intensity image-only
- [ ] Req 15: label set ⊆ {0,1,2,3,4} asserted inside the pipeline, not only in tests
- [ ] Req 16: shapes hold — image `(4,256,256)`, label `(1,256,256)`
- [ ] Req 17: `use_augmentation: false` → bitwise-identical repeated loads
- [ ] Req 18: `use_augmentation: true` → output varies, shapes and label set unchanged
- [ ] Req 19: same `aug_seed` → bitwise-identical output

**Mixup**
- [ ] Req 20: one shared λ ~ Beta(α,α) on image and one-hot mask; no scalar-label mixing
- [ ] Req 21: mixed one-hot sums to 1.0 ± 1e-5 everywhere; shape `(5,256,256)`
- [ ] Req 22: cross-plane mixup raises `ValueError`; batch helper pairs same-plane only
- [ ] Req 23: `use_augmentation: false` disables mixup even when `use_mixup: true`

**Balanced expansion**
- [ ] Req 24: per-hospital only; A and B never pooled; no held-out subject in any plan
- [ ] Req 25: `target_per_hospital` = 8000 per hospital, shortfall logged not borrowed
- [ ] Req 26: all tumor slices kept (under cap); empty slices ≤ 30% of entries
- [ ] Req 27: per-patient cap enforced; no patient dominates
- [ ] Req 28: under-represented patients up-sampled via augmented copies; originals first
- [ ] Req 29: every cached subject of a hospital appears ≥ once; A and B not forced to match
- [ ] Req 30: full provenance on every entry; `aug_seed` null iff `is_augmented` false
- [ ] Req 31: `plan --report` prints the per-hospital table and writes `plans/<hospital>_plan.json`
- [ ] Req 32: plan is byte-identical across runs for a fixed seed

**Leakage**
- [ ] Req 33: A / B / held-out disjoint at subject level, asserted by a test that fails loudly
- [ ] Req 34: split is by patient, never by slice; eval set is exactly the held-out manifest
- [ ] Req 35: held-out never augmented — eval output identical with the flag on and off

**Sampler (CORAL enablement)**
- [ ] Req 36: batches draw ≤ 1 slice per patient (configurable), from distinct patients
- [ ] Req 37: every batch is plane-homogeneous
- [ ] Req 38: batch of 8 over 20 patients → 8 distinct subject IDs, 1 distinct plane

**Ablation runner**
- [ ] Req 39: Dice per patient from restacked `(240,240,155)` volumes, never per slice
- [ ] Req 40: `eval.plane` ∈ {axial, coronal, both}, default `axial`; `both` averages softmax probabilities; policy printed
- [ ] Req 41: ET/NC/WT regions unchanged from `metrics.py`
- [ ] Req 42: empty in both → 1.0; empty in exactly one → 0.0
- [ ] Req 43: empty-ground-truth subject counts logged per region on every run
- [ ] Req 44: exactly one row appended; header byte-for-byte matches the contract
- [ ] Req 45: `mean_dice` = mean of the three rounded region Dice values, 6 decimals
- [ ] Req 46: mismatched existing header raises and writes nothing
- [ ] Req 47: `eval --dummy-checkpoint --dummy-data` appends a valid row with no data or model
- [ ] Req 48: missing checkpoint options exits non-zero naming both

**Config flag**
- [ ] Req 49: all four aug/mixup conditions reachable from config alone, each tested
- [ ] Req 50: `use_federation` / `use_domain_adaptation` pass through to the CSV unchanged

**Preview**
- [ ] Req 51: `preview --n 8 --out preview.png` writes the grid, prints the path, works with `--dummy-data`

**README**
- [ ] Req 52: first section is the copy-pasteable run block with expected output per command
- [ ] Req 53: every block command works on a clean checkout; `slices` documents its data prerequisite and exact error

**ET floor**
- [ ] Req 54: `et_slice_ratio` floor met when supply allows; ET takes first claim on the tumour budget; shortage degrades gracefully without missing the target; `0.0` disables it; empty ceiling and per-patient cap still hold; `ET`/`ET%` in the report

**Slice cache and index (replaces Reqs 5–9)**
- [ ] Req 55: cache layout `<subject>/<plane>/slice_NNN.npz` with `image` (4,H,W) t1c/t1n/t2f/t2w and `mask` (1,H,W); no re-normalising, no re-stacking
- [ ] Req 56: `run.py index` writes `index.json` with `indices`/`tumor`/`et`; `plan` reads it, never the masks
- [ ] Req 57: index lists hold actual slice indices, not positions; missing index names the command that builds it
- [ ] Req 58: `restack_volume` places slices at true volume positions into the full shape; axial and coronal restack to a common shape
- [ ] Req 59: `index` warns on a per-slice-normalised cache
- [ ] Req 60: cache verified internally coherent — axial and coronal restack to identical volumes
