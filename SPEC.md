# Spec: BraTS-PEDs-v1 Augmentation + Ablation Pipeline (`03_augmentation_eval`)

## 1. Goal

A toggleable, hospital-separated, ratio-preserving 3D segmentation augmentation pipeline for BraTS-PEDs-v1 (Mixup + MONAI transforms, each hospital expanded to exactly 150 training subjects) plus an ablation runner that scores a checkpoint on the held-out institution and appends one ET/NC/WT/mean-Dice row to the results CSV.

## 2. In Scope

- **Data access layer** over the existing cache: `<cache>/<subject_id>.npz` with keys `t1c`, `t1n`, `t2f`, `t2w` (96³ float16, raw un-normalized) and `seg` (96³ uint8, labels 0–4). Loads a manifest (`hospitalA.json` / `hospitalB.json` / `heldout.json` — each a flat JSON list of subject IDs), stacks the 4 modalities into a `(4, 96, 96, 96)` tensor, and z-score normalizes each modality over non-zero (brain) voxels only.
- **Tumor-type stratification via imaging proxy.** No tumor-type label exists in the manifests or the cache; a per-subject stratum is derived from the `seg` mask geometry (midline-ness + inferior/brainstem position + enhancing fraction) and written to a CSV with an explicit `source` column. A real label file, when supplied, overrides the proxy wholesale.
- **Ratio-preserving expansion** to exactly 150 subjects per hospital, computed independently per hospital, preserving that hospital's own stratum ratio via largest-remainder allocation. Produces a JSON expansion plan carrying full provenance (`hospital`, `source_subject_id`, `stratum`, `is_augmented`, `aug_seed`).
- **MONAI 3D augmentation stack** (`RandFlipd`, `RandAffined` / rotation, `RandZoomd`, `RandScaleIntensityd`, `RandShiftIntensityd`, `RandGaussianNoised`) applied jointly to image and label, with nearest-neighbour interpolation on the label for every spatial transform.
- **3D segmentation Mixup**: one-hot the 5-class mask, sample λ~Beta(α,α), mix image and one-hot mask together. Works at `batch_size=1` via a one-sample carry-over buffer.
- **Single config flag** `use_augmentation: true|false` gating the whole augmentation branch, with a sub-flag `use_mixup` inside it, so all four conditions (none / MONAI / mixup / both) are reachable from config alone.
- **Metrics module**: converts class masks (0–4) into BraTS-PEDs regions — ET = `{1}`, NC = `{1,2,3}`, WT = `{1,2,3,4}` — and computes per-region Dice.
- **Ablation runner**: loads a checkpoint, runs inference over `heldout.json`, computes mean per-subject ET/NC/WT Dice, appends one row to `results/ablation_results.csv` with the exact contract schema.
- **Dummy-data test suite** that exercises every module (transforms, mixup, metrics, expansion, CSV writing, runner end-to-end) on random tensors and a dummy segmentation network, with no dependency on a real checkpoint or the real cache.

## 3. Out of Scope

- The 3D U-Net architecture and the FedAvg training loop — owned by `01_model_federated`. This section consumes checkpoints, it does not produce them.
- CORAL / domain adaptation and PCA/LDA visualization — owned by `02_domain_adaptation`. The `use_domain_adaptation` column is recorded, not implemented here.
- The actual training script / optimizer loop. This section provides the `Dataset` + transform pipeline that a trainer consumes, not the trainer.
- Streamlit demo, clinical narrative, plots of the results table.
- Re-deriving the hospital A/B/held-out split. The manifests are fixed inputs.
- Regenerating the 96³ cache from raw NIfTI.
- Test-time augmentation, ensembling, post-processing of predictions.
- Any real histopathology/radiology tumor-type ground truth. The stratum is an imaging proxy and is labelled as such everywhere it appears.

## 4. Requirements

**Assumption:** Python + PyTorch 2.13 (CPU-capable) + MONAI 1.6 + NumPy + pandas, all already present in the environment. Config is YAML.

**Assumption:** Augmented subjects are **virtual by default** — the expansion plan stores `(source_subject_id, aug_seed)` and the `Dataset` materializes the augmented volume on-the-fly at `__getitem__`. Writing 300 augmented 96³×5 volumes to disk is opt-in via `--materialize`. Rationale: identical semantics, ~2 GB of disk saved, and the plan JSON is the durable provenance record either way.

**Assumption:** The proxy stratum uses two classes, `dmg_like` and `astrocytoma_like`, since the brief names exactly two pediatric HGG types.

1. `load_subject(subject_id)` returns `image` of shape `(4, 96, 96, 96)` float32 and `label` of shape `(1, 96, 96, 96)` with integer values in `{0,1,2,3,4}`, for every one of the 227 subject IDs across the three manifests, with zero failures.
2. Each modality channel is z-score normalized over non-zero voxels only: for every loaded subject and every channel, `|mean(ch[ch!=0])| < 1e-3` and `|std(ch[ch!=0]) - 1| < 1e-3`. Background voxels remain exactly 0.
3. `build_strata()` assigns exactly one stratum to each of the 227 subjects and writes `strata.csv` with columns `subject_id, hospital, stratum, source, midline_offset, inferior_frac, et_frac`. Every row's `source` is `proxy_v1` unless an override label file was supplied, in which case it is `external`.
4. Supplying `--tumor-type-csv <path>` (columns `subject_id,tumor_type`) makes those labels the sole source of truth for every subject present in the file; subjects absent from it fall back to the proxy, and their `source` column reflects which path was taken.
5. `plan_expansion(hospital)` returns a plan whose total subject count is **exactly 150** for Hospital A and **exactly 150** for Hospital B.
6. Per-stratum proportions in each hospital's plan are within **±1/150 (0.67 pp)** of that hospital's original proportion, computed independently per hospital — A's plan is unaffected by B's ratios and vice versa.
7. Every plan entry retains provenance: `hospital`, `source_subject_id`, `stratum`, `is_augmented`, `aug_seed`. All 53 Hospital A originals appear in A's plan with `is_augmented=false`; all 92 Hospital B originals appear in B's plan with `is_augmented=false`.
8. No subject ID appears in more than one hospital's plan, and **no** held-out subject ID appears in any hospital's plan. A unit test asserts three-way set disjointness between A, B and held-out at the source-subject level.
9. The held-out set is never augmented: the eval `Dataset` is constructed with augmentation force-disabled regardless of the config flag, and a test asserts two loads of the same held-out subject return bitwise-identical tensors.
10. With `use_augmentation: false`, the transform pipeline applies only deterministic loading + normalization: two consecutive loads of the same training subject return bitwise-identical tensors.
11. With `use_augmentation: true`, two loads of the same training subject with different seeds return different tensors, while output shapes stay exactly `(4,96,96,96)` and `(1,96,96,96)`, and the label still contains only values in `{0,1,2,3,4}` (i.e. no interpolation-induced fractional labels).
12. `mixup_3d(batch, alpha)` accepts image `(B,4,96,96,96)` and one-hot label `(B,5,96,96,96)` and returns the same shapes, with the mixed one-hot label summing to 1.0 across the channel axis at every voxel (tolerance 1e-4).
13. Mixup works at `batch_size=1` via a carry-over buffer: the first call returns the sample unmixed, and every subsequent call returns a mix of the current and the previously buffered sample. A test asserts no exception and correct shapes over a 10-step `batch_size=1` loop.
14. Setting `use_augmentation: false` disables mixup too, regardless of the `use_mixup` sub-flag value — verified by a test that sets `use_mixup: true, use_augmentation: false` and asserts the returned batch is unmixed.
15. `dice_regions(pred_classes, true_classes)` maps to ET=`{1}`, NC=`{1,2,3}`, WT=`{1,2,3,4}` and returns a dict with keys `dice_ET`, `dice_NC`, `dice_WT`. Identical input masks give Dice `1.0` for all three regions; disjoint non-empty masks give `0.0`.
16. When a region is empty in **both** prediction and ground truth, its Dice is `1.0` for that subject; when empty in ground truth but non-empty in prediction (or vice versa), it is `0.0`. Both conventions are asserted by tests and stated in the module docstring.
17. `verify_label_ids()` loads one real sample and asserts `set(unique(seg)) ⊆ {0,1,2,3,4}`, failing loudly with the offending subject ID otherwise.
18. The ablation runner writes a CSV whose header is exactly `experiment_name,use_augmentation,use_federation,use_domain_adaptation,dice_ET,dice_NC,dice_WT,mean_dice` — byte-for-byte, in that order.
19. `mean_dice` in every written row equals `(dice_ET + dice_NC + dice_WT) / 3` to within 1e-6.
20. The runner appends exactly **one** row per invocation. Running it twice against an existing CSV yields a 2-row file with a single header line.
21. The runner completes successfully against a randomly-initialized dummy checkpoint and dummy predictions, with no trained model present — proving it does not depend on `01_model_federated` being finished.
22. The runner accepts `--dummy-data` and runs end-to-end on random tensors without reading `cache_96cube` at all.
23. `python tests_dummy.py` (or `pytest tests_dummy.py`) exits 0 with every test passing, on CPU, in under 120 seconds, without touching the real cache.
24. `python -m src.expansion --report` prints a per-hospital table of original vs expanded counts and proportions per stratum, and the printed totals read 150 and 150.

## 5. Structure

```
03_augmentation_eval/
├── BRIEF.md                 # existing, unchanged
├── config.yaml              # use_augmentation / use_mixup / paths / aug hyperparams
├── src/
│   ├── __init__.py
│   ├── config.py            # YAML load + dataclass, CLI overrides
│   ├── data.py              # manifest reading, npz cache access, per-channel z-score, verify_label_ids
│   ├── strata.py            # imaging-proxy tumor-type derivation + external CSV override -> strata.csv
│   ├── expansion.py         # per-hospital largest-remainder expansion to exactly 150 + plan JSON + --report
│   ├── augmentation.py      # MONAI 3D transform stack, gated by use_augmentation
│   ├── mixup.py             # 3D segmentation mixup, one-hot, batch_size=1 carry-over buffer
│   ├── dataset.py           # torch Dataset over an expansion plan; train vs eval mode
│   ├── metrics.py           # class-mask -> ET/NC/WT regions, per-region Dice
│   ├── dummy_model.py       # tiny random seg net + dummy checkpoint writer, for testing without 01_
│   └── runner.py            # ablation entry point: checkpoint -> heldout inference -> one CSV row
├── tests_dummy.py           # full dummy-tensor test suite
├── plans/                   # generated: hospitalA_plan.json, hospitalB_plan.json
├── strata.csv               # generated
└── results/
    └── ablation_results.csv # generated, appended one row per run
```

Non-obvious roles:

- **`strata.py`** — the compensating layer for the missing tumor-type metadata. It computes three geometric features from `seg`: `midline_offset` (|centroid_x − 48| / 48), `inferior_frac` (fraction of tumor voxels below the axial midpoint), and `et_frac` (label-1 voxels / all tumor voxels). `dmg_like` requires low midline offset **and** inferior position **and** low enhancing fraction; everything else is `astrocytoma_like`. Thresholds live in `config.yaml`, not in the code. Every output carries `source=proxy_v1` so no downstream consumer can mistake it for histology.
- **`expansion.py`** — allocates the 150-subject budget per hospital: `target_s = 150 × n_s / N` rounded by largest remainder so the targets sum to exactly 150. Originals are always kept (real data is never dropped); the shortfall `target_s − n_s` is filled by augmented copies of randomly-chosen source subjects from within that same stratum and that same hospital.
- **`dataset.py`** — resolves a plan entry to tensors. `is_augmented=true` entries run the MONAI stack under the entry's fixed `aug_seed`; `is_augmented=false` entries in a training plan still get live random augmentation when `use_augmentation: true`. Eval mode hard-disables both paths.
- **`dummy_model.py`** — exists purely so the runner is testable today. Emits a `(B,5,96,96,96)` logit tensor and a bottleneck feature vector, matching the `model(x) -> (seg_logits, features)` contract, so swapping in the real network later is a one-line change.

## 6. Edge Cases

| Scenario | Expected Behaviour |
|---|---|
| A stratum has 0 subjects in a hospital | Allocated 0 of the 150; no divide-by-zero; the other stratum absorbs the full budget |
| Rounding makes stratum targets sum to 149 or 151 | Largest-remainder allocation corrects it; a post-condition `assert sum == 150` fails loudly if not |
| `target_s < n_originals` for a stratum (over-represented stratum, shrinking) | Keep all originals — never drop real data — and reduce another stratum's augmented quota to hold the total at 150; log a warning naming the stratum. Cannot occur at 53→150 or 92→150 but is guarded |
| Requested subject `.npz` missing from the cache | `FileNotFoundError` naming the full path and the subject ID |
| `.npz` present but missing a modality key | `KeyError` naming the subject ID and the missing key, not a bare KeyError from NumPy |
| `seg` contains a label outside 0–4 | `ValueError` naming the subject ID and the offending values |
| A modality channel is entirely zero (all background) | Skip normalization for that channel, leave it as zeros, emit a warning with the subject ID — no NaN from division by zero std |
| Spatial transform introduces fractional label values | Label uses `mode="nearest"` on every spatial transform; a test asserts the output label set stays ⊆ {0,1,2,3,4} |
| Mixup called with `batch_size=1` on the very first step (empty buffer) | Return the sample unmixed, fill the buffer, no exception |
| `alpha <= 0` in mixup config | `ValueError: mixup alpha must be > 0` |
| Region absent from both prediction and ground truth | Dice = 1.0 for that region on that subject (documented convention) |
| Region absent from ground truth but predicted, or present but not predicted | Dice = 0.0 |
| ET region empty in ground truth for a DMG/DIPG subject (clinically expected) | Handled by the two rules above; per-subject ET values are averaged as-is, and the row count of ET-empty subjects is logged so the mean is interpretable |
| `results/ablation_results.csv` does not exist | Create it, write the header, then the row |
| `results/ablation_results.csv` exists with a different header | `ValueError` reporting expected vs found header; do not silently append misaligned columns |
| `results/` directory does not exist | Created automatically |
| Checkpoint file missing or unloadable | `FileNotFoundError` / clear load error naming the path; suggest `--dummy-checkpoint` |
| Checkpoint state dict does not match the model | Raise with the mismatched key names; do not silently `strict=False` |
| Held-out manifest requested with `use_augmentation: true` | Augmentation is force-disabled for eval and a warning is logged; results are unaffected |
| Same `aug_seed` requested twice for the same source subject | Produces bitwise-identical output — augmentation is reproducible |
| No tumor-type CSV supplied | Proxy path taken; a prominent warning states the strata are an imaging proxy, not histology |

## 7. Done Checklist

- [ ] Req 1: `load_subject` returns `(4,96,96,96)` image + `(1,96,96,96)` label for all 227 subjects
- [ ] Req 2: Per-channel z-score over non-zero voxels; mean≈0, std≈1, background stays 0
- [ ] Req 3: `build_strata()` covers all 227 subjects and writes `strata.csv` with the specified columns
- [ ] Req 4: `--tumor-type-csv` overrides the proxy; `source` column reflects the path taken
- [ ] Req 5: Hospital A plan totals exactly 150; Hospital B plan totals exactly 150
- [ ] Req 6: Per-hospital stratum proportions within ±0.67 pp of original, computed independently per hospital
- [ ] Req 7: Every plan entry carries full provenance; all originals present with `is_augmented=false`
- [ ] Req 8: A / B / held-out source subjects are three-way disjoint (asserted)
- [ ] Req 9: Held-out is never augmented; repeated loads are bitwise-identical
- [ ] Req 10: `use_augmentation: false` → deterministic, bitwise-identical repeated loads
- [ ] Req 11: `use_augmentation: true` → varied output, correct shapes, label values still ⊆ {0,1,2,3,4}
- [ ] Req 12: `mixup_3d` preserves shapes; mixed one-hot sums to 1.0 across channels
- [ ] Req 13: Mixup works over a 10-step `batch_size=1` loop via carry-over buffer
- [ ] Req 14: `use_augmentation: false` disables mixup even when `use_mixup: true`
- [ ] Req 15: `dice_regions` maps ET/NC/WT correctly; identical→1.0, disjoint→0.0
- [ ] Req 16: Empty-region conventions implemented, tested, and documented
- [ ] Req 17: `verify_label_ids()` checks a real sample's label IDs and fails loudly
- [ ] Req 18: CSV header byte-for-byte matches the contract schema
- [ ] Req 19: `mean_dice == (ET+NC+WT)/3` within 1e-6
- [ ] Req 20: Exactly one row appended per run; two runs → 2 rows, 1 header
- [ ] Req 21: Runner succeeds against a dummy checkpoint with no trained model present
- [ ] Req 22: `--dummy-data` runs end-to-end without reading the real cache
- [ ] Req 23: `tests_dummy.py` exits 0, CPU-only, under 120 s, no real cache access
- [ ] Req 24: `expansion --report` prints per-hospital stratum tables with totals of 150 / 150
