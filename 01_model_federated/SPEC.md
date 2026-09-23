# Spec: Federated 3D U-Net for Pediatric Brain Tumor Segmentation (Section 01: model_federated)

> **Addendum (2026-09-23):** §2, §3, §4 (Req 17-24), §5 and §7 were updated to bring the real 3D data-augmentation stack into this section's scope. The original spec (below, everything else unchanged) said augmentation logic belonged to section 03 and only the hook would live here — that was true while the 2D pipeline was the active model. The project has since started exploring a 3D pipeline (2D progress plateaued; a supervising doctor recommended revisiting 3D), and section 03's augmentation code is slice-based and does not apply to 3D volumes. A 3D augmentation stack was built directly in this section instead, superseding the original out-of-scope line. This addendum documents what was already built and verified, not a forward plan.

## 1. Goal
A MONAI 3D U-Net trained from scratch that exposes `(seg_logits, features)`, runnable both as a single-client sanity-check loop and as a FedAvg loop across simulated Hospital A / Hospital B clients, with per-epoch checkpoint/resume, **and a real 3D augmentation stack behind `use_augmentation`** — built and testable now against dummy tensors, ready to swap to the real resampled-cache data the moment it lands.

## 2. In Scope
- A 3D U-Net model (MONAI `UNet` or equivalent) built for 96³ single-channel-per-modality (4-channel: t1c, t1n, t2f, t2w) input, trained from scratch (no pretrained weights).
- Model forward signature `model(x) -> (seg_logits, features)` where `seg_logits` has 5 channels (background + ET, NET, CC, ED per `CONTRACTS.md`) and `features` is a fixed-size bottleneck embedding vector (pooled), exposed even though this section doesn't consume it.
- A `Dataset`/loader abstraction that:
  - Reads subject IDs from a manifest JSON (`hospitalA.json`, `hospitalB.json`, `heldout.json`).
  - In "dummy mode" (default until the real cache exists), generates random tensors of the correct shape (4×96×96×96 input, 96×96×96 integer label in {0,1,2,3,4}) keyed by the manifest's subject IDs — same subject count as the real manifest.
  - In "real mode", loads `.nii.gz` volumes from a configurable shared-cache path, filtered by the manifest, stacking the 4 modalities into one input tensor.
- A single-client training loop: trains on one manifest/subject set, FP16 mixed precision, batch size 1, logs loss per epoch.
- A FedAvg loop: independently trains local copies of the model on Hospital A and Hospital B subsets for a configurable number of local epochs, then averages weights (parameter-count-weighted by client subject count) into a global model, repeated for a configurable number of rounds.
- Config flags respected everywhere per `CONTRACTS.md`: `use_augmentation`, `use_federation`, `use_domain_adaptation`.
  - `use_augmentation`: when true, builds and applies a real 3D augmentation transform to each training sample (image + mask together); when false, no-op. **This is the addendum: the transform's actual logic now lives here (see below), not just the hook.**
  - `use_federation`: when true, runs the FedAvg loop across hospitalA + hospitalB; when false, runs the single-client loop on whichever manifest is passed in.
  - `use_domain_adaptation`: accepted and threaded through config/CLI for interface compatibility; this section does not implement CORAL. When true, the loop still runs identically (no-op) — flag is exposed so the config schema is stable for section 02 to plug into later.
- **A 3D augmentation stack (`src/augment3d.py`)**, applied only when `use_augmentation=true`:
  - Spatial transforms applied to the image and the mask together, mask always via nearest-neighbour interpolation so no fractional label values can appear: random flip (each of the 3 spatial axes independently), random rotation (about all 3 axes), random zoom.
  - Intensity transforms applied to the image only: random intensity scale, random intensity shift, random Gaussian noise.
  - A guard that raises immediately if any transform ever produces a label value outside `{0,1,2,3,4}`.
  - No mixup/sample-blending of any kind (see Out of Scope).
  - Exposed as an `Augment3D` callable matching the existing `transform(x, y) -> (x, y)` hook contract in `train_single.py`, operating on batched `(B, 4, D, H, W)` / `(B, D, H, W)` tensors.
- `run.py` builds a real `Augment3D()` and passes it through to both the single-client and federated training paths when `--use-augmentation` is passed (previously a complete no-op end to end: the flag and the hook existed, but nothing ever constructed a transform to pass through it).
- Per-epoch checkpointing to `checkpoints/<run_id>/epoch_<N>.pt` containing model state, optimizer state, epoch number, and (for federation) round number.
- Resume-from-checkpoint: given a `run_id` (and, for federation, which client), training resumes from the latest saved epoch rather than epoch 0.
- A minimal CLI/entry point to launch: single-client run, federated run, and resume of either.
- Unit tests (dummy-tensor mode only — no real data required) covering model I/O shape, single-client loop, FedAvg aggregation correctness, checkpoint save/resume, augmentation correctness, and flag behavior.
- A `requirements.txt` / dependency list scoped to this section (torch, monai, numpy — nothing else).

## 3. Out of Scope
- CORAL / any domain adaptation math (section 02).
- ~~The augmentation transform's internal logic — Mixup or any other technique (section 03).~~ **Superseded by this addendum** — the spatial/intensity augmentation logic now lives here (see §2). Mixup specifically remains out of scope (next bullet), but no longer because it belongs to another section.
- **Mixup / any cross-sample blending for the 3D pipeline.** The 2D pipeline's mixup blends two 2D slices; a 3D-volume equivalent would blend anatomy at every voxel of a full segmentation target and has never been validated as sane for this task. Left out deliberately, not ported speculatively.
- Final ablation-matrix result generation and the results CSV (section 03).
- Any real `.nii.gz` I/O testing (no real data cache exists yet) — real-mode code path is written but only exercised once the cache lands; this build validates it structurally, not against real files.
- Streamlit demo / any UI (section 05).
- Clinical narrative content (section 04).
- Multi-GPU / distributed-data-parallel training, secure aggregation, differential privacy, or any FedAvg variant beyond plain weighted averaging.
- Hyperparameter tuning or achieving any particular Dice score — this section delivers a correct, runnable pipeline, not a tuned model. **This applies to the augmentation stack too: no training run on real data with augmentation enabled has been done yet (see Req 24) — this addendum covers building and verifying the mechanism, not proving it improves held-out Dice.**

## 4. Requirements

1. `model(x)` where `x` has shape `(batch, 4, 96, 96, 96)` returns a tuple `(seg_logits, features)`; `seg_logits` has shape `(batch, 5, 96, 96, 96)`; `features` has shape `(batch, D)` for some fixed `D`.
2. Model is constructed with randomly-initialized weights only — no pretrained-weight loading path exists.
3. Running the single-client loop for N epochs on a dummy dataset of size ≥ 2 completes without error and produces a monotonically-tracked loss log of length N.
4. Running the single-client loop with `use_augmentation=true` and an injected transform calls that transform on every sample; with `use_augmentation=false`, the transform (if any) is never called.
5. Running the FedAvg loop for R rounds × E local epochs against dummy Hospital A (53 dummy subjects) and Hospital B (92 dummy subjects) manifests completes without error.
6. FedAvg aggregation is subject-count-weighted: given two clients with known subject counts and manually-set distinct local weights, the aggregated global weight equals the analytically-computed weighted average (verified in a unit test with tolerance ≤ 1e-5).
7. `use_federation=false` runs only the single-client path (no aggregation step ever invoked); `use_federation=true` runs only the federated path.
8. A checkpoint file is written after every epoch (single-client) or every round (federated) to `checkpoints/<run_id>/`, containing at minimum model state dict, optimizer state dict, and the current epoch/round number.
9. Given a `run_id` whose checkpoint directory has N saved epochs, resuming training continues from epoch N+1 (not epoch 0), and the restored model/optimizer state matches what was saved (verified by unit test comparing state dicts before checkpoint and after resume-load).
10. Interrupting and resuming produces the same next-epoch loss (within FP16 tolerance) as an uninterrupted run trained to the same point, given a fixed random seed — verified in a unit test on dummy data.
11. All batch sizes used in training are 1; a config value other than 1 is rejected with a clear error (`Assumption`: batch size is hard-fixed by contract, not user-configurable).
12. Training runs under `torch.autocast`/AMP FP16 on CUDA when available; on CPU-only environments (e.g., CI), the loop still runs correctly with autocast disabled or no-op (`Assumption`: FP16 is a CUDA-only optimization, tests must still pass on CPU).
13. `use_domain_adaptation` is accepted as a config flag and stored/logged, but toggling it produces no behavioral difference in this section's code (no-op), and no error is raised for either value.
14. Manifests are loaded from `00_shared/manifests/{hospitalA,hospitalB,heldout}.json` by path, and the dummy dataset for a given manifest produces exactly as many samples as subject IDs in that manifest.
15. A `--dummy`/`--real` (or equivalent config) switch selects data source; default is dummy mode; real mode requires an explicit cache-path argument and is not exercised by the test suite.
16. All unit tests pass using only dummy tensors, run on CPU, in under 2 minutes total.
17. `Augment3D()(x, y)` called on `(B, 4, D, H, W)` / `(B, D, H, W)` tensors returns tensors of the same shape and dtype as the input.
18. With every transform's probability forced to 1.0, the returned label tensor's value set is a subset of `{0, 1, 2, 3, 4}` (nearest-neighbour interpolation on the mask never introduces a fractional or out-of-range label).
19. With every transform's probability forced to 1.0, the returned image tensor is not bitwise-identical to the input (the stack is not a silent no-op).
20. With every transform's probability set to 0.0, the returned image and label tensors are bitwise-identical to the input (a true, verifiable no-op, distinct from the `use_augmentation=false` case covered by Req 4).
21. Constructing two `Augment3D` instances with the same seed and calling each once on the same input produces bitwise-identical output (reproducibility).
22. `AssertLabelValuesd` raises `AssertionError` when given a tensor containing a value outside the configured valid label set.
23. A batch with `B > 1` is accepted and each sample is transformed independently (the callable does not hardcode the batch_size=1 contract even though every other call site in this section does).
24. `run.py ... --use-augmentation` (single-client and federated paths) constructs a real `Augment3D` and passes it through to `train_single_client`, verified by an end-to-end test that runs the real transform (not a mock) through the actual training loop on dummy data for at least 1 epoch and asserts the resulting loss is finite. **No claim is made or tested here about the effect of augmentation on held-out Dice on real data** — that requires an actual training run, which is out of scope for this addendum (see §3).

**Assumption:** FedAvg round/local-epoch counts, learning rate, and optimizer are exposed as config parameters with reasonable defaults (Adam, lr=1e-3, 1 local epoch/round, 2 rounds for smoke tests) rather than fixed — no spec constraint dictates specific values.
**Assumption:** `features` is produced by global-average-pooling the U-Net bottleneck activation map to a 1D vector; exact dimensionality is an implementation default, not contractually fixed.
**Assumption:** Checkpoint resume identifies "latest" by highest epoch/round number found in the run's checkpoint directory, not by a separately tracked "latest" pointer file.
**Assumption:** Loss function is Dice+CrossEntropy over the 5 label classes (standard MONAI choice for multi-class 3D segmentation); not contractually specified. **Superseded in part:** `run.py --loss` also accepts `dice_focal` (a class-imbalance-aware variant); `dice_ce` remains the default.
**Assumption:** Test framework is `pytest`, consistent with a Python/MONAI/PyTorch stack.
**Assumption (addendum):** Augmentation transform probabilities and magnitudes (flip 0.5, rotate 0.3 @ ±10°, zoom 0.3 @ 0.9-1.1×, intensity scale/shift 0.3 @ ±0.1, Gaussian noise 0.2 @ std 0.05) mirror section 03's 2D defaults for consistency across the project; not contractually specified, and untuned against real 3D held-out Dice.

## 5. Structure

```
01_model_federated/
├── BRIEF.md
├── SPEC.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── model.py            # 3D U-Net wrapper: model(x) -> (seg_logits, features)
│   ├── data.py             # Manifest-driven Dataset: dummy-tensor mode + real .nii.gz mode
│   ├── augment3d.py        # 3D MONAI augmentation stack + Augment3D callable (addendum)
│   ├── train_single.py     # Single-client training loop
│   ├── federated.py        # FedAvg orchestration: local training + weighted aggregation
│   ├── checkpoint.py        # save/load/resume helpers
│   └── config.py           # Config dataclass/CLI: use_augmentation, use_federation, use_domain_adaptation, etc.
├── run.py                  # CLI entry point (single-client / federated / resume); builds Augment3D() when --use-augmentation is set
├── checkpoints/            # Created at runtime, gitignored
└── tests/
    ├── test_model.py
    ├── test_data.py
    ├── test_single_client.py
    ├── test_federated.py
    ├── test_checkpoint.py
    └── test_augment3d.py   # addendum
```

## 6. Edge Cases

| Scenario | Expected Behaviour |
|---|---|
| Manifest file missing/unreadable | Raise `FileNotFoundError` naming the missing manifest path |
| Manifest has 0 subject IDs | Raise a clear `ValueError` ("empty manifest") before training starts |
| `batch_size != 1` passed in config | Raise `ValueError` immediately, no training attempted |
| Resume requested but no checkpoint exists for `run_id` | Start fresh from epoch/round 0 and log a warning, do not error |
| Resume requested, checkpoint dir exists but is corrupted/unreadable | Raise a clear error naming the bad file, do not silently start over |
| `use_augmentation=true` but no transform provided | No-op (identity), do not error |
| `use_federation=true` with only one client manifest configured | Raise `ValueError` requiring at least 2 clients for FedAvg |
| Running on CPU-only machine (no CUDA) | Training still runs (FP16 autocast disabled/no-op), no crash |
| `use_domain_adaptation=true` or `false` | No behavioral difference; both accepted without error |
| Real-mode selected but cache path missing/unset | Raise a clear `ValueError`/`FileNotFoundError` naming the missing path, before any training |
| A transform in the augmentation stack produces an out-of-range label value | `AssertLabelValuesd` raises `AssertionError` naming the bad value(s) immediately, no silent pass-through |
| All augmentation transform probabilities set to 0.0 | Output is bitwise-identical to input (true no-op, not just "usually unchanged") |
| Augmentation called on a batch with `B > 1` | Each sample transformed independently; no cross-sample interaction |

## 7. Done Checklist
- [x] Req 1: `model(x)` returns `(seg_logits, features)` with correct shapes for `(batch, 4, 96, 96, 96)` input
- [x] Req 2: Model weights are randomly initialized only, no pretrained-loading path
- [x] Req 3: Single-client loop runs N epochs on dummy data without error, logs loss per epoch
- [x] Req 4: `use_augmentation` flag correctly gates whether injected transform is called
- [x] Req 5: FedAvg loop runs R rounds × E local epochs on dummy Hospital A/B without error
- [x] Req 6: FedAvg aggregation is subject-count-weighted, verified against analytic weighted average (tol ≤ 1e-5)
- [x] Req 7: `use_federation` flag correctly selects single-client vs. federated path exclusively
- [x] Req 8: Checkpoint written every epoch/round to `checkpoints/<run_id>/` with model+optimizer+epoch/round state
- [x] Req 9: Resume continues from N+1, restored state matches saved state
- [x] Req 10: Interrupt+resume yields same next-epoch loss as uninterrupted run (fixed seed, FP16 tolerance)
- [x] Req 11: Non-1 batch size rejected with clear error
- [x] Req 12: FP16 autocast used on CUDA; CPU-only environments still run correctly
- [x] Req 13: `use_domain_adaptation` accepted, logged, no-op, no error either value
- [x] Req 14: Dummy dataset sample count exactly matches manifest subject-ID count
- [x] Req 15: Dummy/real mode switch exists, defaults to dummy, real mode requires explicit cache path
- [x] Req 16: Full test suite passes on CPU, dummy data only, under 2 minutes
- [x] Req 17: `Augment3D` preserves shape and dtype
- [x] Req 18: Forced-on augmentation keeps labels within `{0,1,2,3,4}`
- [x] Req 19: Forced-on augmentation is not a silent no-op (image actually changes)
- [x] Req 20: All-probabilities-zero is a bitwise-identical true no-op
- [x] Req 21: Same-seed reproducibility
- [x] Req 22: `AssertLabelValuesd` raises on an out-of-range label
- [x] Req 23: `B > 1` batches processed correctly, independently per sample
- [x] Req 24: `run.py --use-augmentation` wires a real `Augment3D` end-to-end through `train_single_client` (dummy data, finite loss) — no held-out-Dice effectiveness claim made
