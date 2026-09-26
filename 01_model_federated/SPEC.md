# Spec: Federated 3D U-Net for Pediatric Brain Tumor Segmentation (Section 01: model_federated)

> **Addendum 4 (2026-09-26) — sequence-shift augmentation (misregistration robustness):** §2, §3, §4 (Req 50-60), §5, §6 and §7 were extended. This is a *forward* plan. A stress test measured 2026-09-26 on 30 patients that no model or setting had ever been chosen on shifted the T2w and FLAIR sequences against T1c/T1n by 1, 2 and 3 mm: the 2D+3D ensemble lost 0.023, 0.078 and 0.127 mean Dice, and the 3D family alone lost 0.023, 0.029 and 0.103. Contrast, shading, noise and blur changes cost the 3D family at most 0.015, so imperfect co-registration between sequences is the largest measured deployment risk, and no current augmentation simulates it. This addendum adds a random per-sequence translation to the 3D augmentation stack. It does not change any existing behaviour when the new flag is off.
>
> **Addendum 3 (2026-09-26) — patch-based full-resolution training and sliding-window inference:** §2, §3, §4 (Req 35-49), §5, §6 and §7 were extended. This is a *forward* plan. The 3D pipeline trains on ONE 96³-downsampled volume per patient per epoch (145 samples, no sampling balance, 2.5 mm in-plane voxels), while the shipped 2D pipeline trains on ~244 full-detail slices per patient (~35k samples, deliberately balanced: at most 30% tumour-free slices, at least 25% containing enhancing tumour). Measured 2026-09-25: the 3D ensemble matches 2D on NC/WT but plateaus at ~0.70 mean Dice without the ET cleanup rule no matter which loss, augmentation or dropout setting is used, and the number of distinct training views per patient is the largest untested difference between the two pipelines. This addendum adds full-resolution cropping ("patches") with the same style of tumour/ET-balanced sampling, plus the sliding-window inference needed to score such a model on whole volumes. It does not change any existing 96³ behaviour.
>
> **Addendum 2 (2026-09-25) — input-sequence dropout:** §2, §3, §4 (Req 25-34), §6 and §7 were extended to add random input-sequence ("modality") dropout to the augmentation stack. This is a *forward* plan (unlike Addendum 1, which documented already-built work). Motivation, measured 2026-09-25 on the 81 scoreable held-out subjects with the best 3D checkpoint (mean Dice 0.6814 with all 4 sequences): removing T2-FLAIR drops it to 0.25, T2w to 0.49, T1c to 0.54 (ET 0.39 -> 0.09), T1n to 0.55 — while realistic thick-slice/gap acquisition geometry costs only ~0.005 (routine axial 5 mm/1 mm). A clinic missing one sequence is therefore a far bigger deployment risk than slice geometry, and the model has never seen an absent sequence in training.
>
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
- **Input-sequence dropout (Addendum 2), in `src/augment3d.py`:** a new image-only MONAI transform `RandModalityDropoutd` (a `MapTransform` + `RandomizableTransform`), enabled by a new `modality_dropout_prob` kwarg on `build_transforms3d` / `Augment3D` (default `0.0` = the transform is not added at all, behaviour bitwise-identical to today).
  - Each of the input's channels is dropped independently with probability `p`; a dropped channel is set to exactly `0.0` (the z-score mean — the hook receives already-normalized tensors, so "sequence missing" == an all-zero channel). If a draw would drop every channel, one channel chosen uniformly at random is kept (never an all-zero input; no rejection loop).
  - Appended at the END of the chain — after the intensity/noise transforms, before `AssertLabelValuesd` — so nothing can re-introduce non-zero values into a dropped channel. The label is never touched.
  - Draws are independent per sample in a batch and participate in `Compose.set_random_state`, so a fixed seed is reproducible.
  - This zero-channel encoding is also the convention deployment must use for a sequence a clinic does not have.
- `run.py` gains `--modality-dropout FLOAT` (default `0.0`), passed to `Augment3D(modality_dropout_prob=...)` for both the single-client and federated paths; a positive value without `--use-augmentation` (which would otherwise silently do nothing) is a hard CLI error.
- **Patch-based full-resolution training (Addendum 3):**
  - `tools/build_fullres_cache.py`: raw BraTS-PEDs NIfTI (`<raw_dir>/<sid>/<sid>-{t1c,t1n,t2f,t2w,seg}.nii.gz`, 240×240×155 at 1 mm; **no resampling and no reorientation**) → per patient `<out>/<sid>.img.npy` (float16, `(4,X,Y,Z)`, channel order t1c,t1n,t2f,t2w, RAW intensities), `<sid>.seg.npy` (uint8 `(X,Y,Z)`, labels 0-4) and `<sid>.stats.npy` (float32 `(4,2)` = mean, std over voxels > 0 of each raw modality; `(0,1)` for a modality with no voxel > 0). Atomic writes (temp file handle, then rename — `np.save` on a string path silently appends `.npy`), skips patients already present, reports per-patient errors without aborting the batch, uses a process pool, writes `_build_summary.json`, and rejects a patient whose five files disagree on shape or whose labels leave `{0..4}`.
  - `src/patch_data.py` — `PatchDataset(manifest_path, cache_path, patch_size=(128,128,128), patches_per_epoch, fractions=(0.35,0.45,0.20), seed=42)` over the categories `('et','tumor','random')`. `len` is `patches_per_epoch`; every `__getitem__` draws its OWN patch from an internal numpy `Generator` (the index is ignored, so each epoch sees new patches). Category-first sampling: choose the category by `fractions`; choose a patient uniformly among those ELIGIBLE (et: at least one ET voxel; tumor: at least one voxel with label > 0; random: any); choose a centre voxel (et: a random ET voxel; tumor: a random label > 0 voxel; random: uniform in the volume); patch start = centre − patch//2 + jitter, jitter a uniform integer in `[−patch//4, +patch//4]` per axis, then clipped to `[0, vol − patch]` so the patch always lies fully inside the volume and an et/tumor centre voxel is always inside its patch. Per-patient ET/tumour voxel coordinate lists are computed once at construction (subsampled to at most 4000 each). Crops are read with `np.load(mmap_mode='r')`. Normalisation follows `src/data.py::_zscore_normalize` semantics but with the stored full-volume stats (voxels > 0 → `(v−mean)/std`, or `v−mean` when `std < 1e-8`; every other voxel exactly `0`). Returns `x` float32 `(4,*patch)` and `y` int64 `(*patch)`.
  - `TrainConfig.data_mode` accepts `'patch'` (requires `cache_path`); new fields `patch_size` (default `(128,128,128)`), `patches_per_epoch` (default 580), `patch_fractions` (default `(0.35,0.45,0.20)`). `data.py::build_dataset` dispatches `'patch'` to `PatchDataset`. `run.py` gains `--data-mode patch`, `--patch-size D H W`, `--patches-per-epoch N`, `--patch-fractions ET TUMOR RANDOM`; patch mode is single-client only. `train_single.py` needs no functional change: an "epoch" is simply `patches_per_epoch` samples through the existing loop, and `Augment3D` + modality dropout apply to patches unchanged.
  - `src/patch_infer.py` — `predict_volume(model, image, roi_size, overlap=0.5, flips=((),), sw_batch_size=2, device=...)` for a normalised float tensor `(4,X,Y,Z)`: `monai.inferers.sliding_window_inference` with gaussian blending on the segmentation logits `model(x)[0]`, softmax per window, averaged over the flip list after un-flipping; returns float32 probabilities `(5,X,Y,Z)` on the CPU; autocast on CUDA only.
  - `tools/eval_patch3d.py` — `apply_et_min_voxels(pred, T)` (relabel predicted ET, label 1, to non-enhancing, label 2, when the total predicted ET voxel count is below `T`; `T=0` is the identity; NC = {1,2,3} and WT = {1,2,3,4} are unchanged by 1→2) and `evaluate(checkpoints, cache_path, manifest, roi_size, flips, et_min_voxels_list, save_probs_dir=None)`: full-resolution Dice for ET/NC/WT over the manifest using the full-res cache, one row per threshold, `RESULT_JSON:` line from a CLI wrapper, optional float16 probability maps per patient for later ensembling. Dice comes from `03_augmentation_eval/src/metrics.py`, loaded by file path exactly as `tools/eval_heldout_3d.py` does (both sections have a top-level package named `src`).
  - Tests use a TINY synthetic cache written in the real file format (a handful of patients around 48×48×40 voxels, patch sizes 16-32), so nothing needs the real data or a GPU.
- **Sequence-shift augmentation (Addendum 4), in `src/augment3d.py`:** a new image-only MONAI transform `RandSequenceShiftd(keys, prob, max_shift_voxels)` (a `MapTransform` + `RandomizableTransform`). With probability `prob` (one draw per call) it picks one **reference** channel uniformly at random that is never shifted; every other channel is selected independently with probability 0.5 (if none is selected, one of the others is chosen uniformly, so an applied call always shifts at least one channel); each selected channel gets its own translation vector, uniform in `[-max_shift_voxels, +max_shift_voxels]` on each axis (float, sub-voxel allowed). The translation uses `torch.nn.functional.grid_sample` (bilinear, `padding_mode='zeros'`, `align_corners=True`), so vacated voxels are exactly 0 (the hook receives z-scored tensors whose background is 0); semantics `out[x] = in[x - s]`, i.e. an integer shift equals `np.roll` on the interior. The label is never touched. Input `(C,D,H,W)` tensor or ndarray; output has the same type, shape and dtype; the input is never modified in place.
  - `build_transforms3d` / `Augment3D` gain `sequence_shift_prob` (default `0.0`) and `sequence_shift_max_voxels` (default `1.2`). With `prob > 0` the transform is inserted after the intensity transforms and before `RandModalityDropoutd` (dropout must stay last so dropped channels stay exactly 0); with the default `0.0` the chain is unchanged.
  - `run.py` gains `--sequence-shift P` (float in `[0,1)`, default `0.0`) and `--sequence-shift-max-voxels V` (float `> 0`, default `1.2`). `--sequence-shift > 0` requires `--use-augmentation`, exactly like `--modality-dropout`. `V` is in voxels of whatever grid is being trained on: 1.2 voxels is about 3 mm in-plane (and 1.9 mm through-plane) on the 96³ grid, whose voxels are 2.5×2.5×1.6 mm, but about 1.2 mm in `--data-mode patch`, where voxels are 1 mm. Both data modes receive the value.
- Per-epoch checkpointing to `checkpoints/<run_id>/epoch_<N>.pt` containing model state, optimizer state, epoch number, and (for federation) round number.
- Resume-from-checkpoint: given a `run_id` (and, for federation, which client), training resumes from the latest saved epoch rather than epoch 0.
- A minimal CLI/entry point to launch: single-client run, federated run, and resume of either.
- Unit tests (dummy-tensor mode only — no real data required) covering model I/O shape, single-client loop, FedAvg aggregation correctness, checkpoint save/resume, augmentation correctness, and flag behavior.
- A `requirements.txt` / dependency list scoped to this section (torch, monai, numpy — nothing else).

## 3. Out of Scope
- CORAL / any domain adaptation math (section 02).
- ~~The augmentation transform's internal logic — Mixup or any other technique (section 03).~~ **Superseded by this addendum** — the spatial/intensity augmentation logic now lives here (see §2). Mixup specifically remains out of scope (next bullet), but no longer because it belongs to another section.
- **Mixup / any cross-sample blending for the 3D pipeline.** The 2D pipeline's mixup blends two 2D slices; a 3D-volume equivalent would blend anatomy at every voxel of a full segmentation target and has never been validated as sane for this task. Left out deliberately, not ported speculatively.
- **(Addendum 2)** Any change to the model, loss, data loading, federated/CORAL logic, or the default behaviour of any existing flag/transform; a missing-sequence evaluation tool (the 2026-09-25 measurements used a throwaway script outside the repo); sequence-specific or learned dropout probabilities (one scalar `p` for all channels only); dropping anything other than whole input channels; and **any claim that dropout improves held-out or missing-sequence Dice** — verifying that needs a real training run plus evaluation, not this spec's tests.
- **(Addendum 3)** The federated/CORAL path in patch mode; any change to the model, loss, `Augment3D` logic, or the existing 96³ `dummy`/`real` behaviour (including Req 10's interrupt-and-resume determinism, which continues to apply to those modes only); multi-process `DataLoader` workers; a missing-sequence evaluation tool; 2D+3D ensembling scripts (they stay outside the repo); resampling, registration or skull-stripping of any kind; and **any claim that patch training improves held-out Dice** — that needs a real training run plus an evaluation, not this spec's tests.
- **(Addendum 4)** Any change to the model, loss, data loading or evaluation code; shifts specific to one sub-region; rotations or scale differences between sequences (translations only); the federated/CORAL path; the missing-sequence tools; a registration or skull-stripping step of any kind; and **any claim that shift augmentation improves misregistration robustness or held-out Dice** — that needs a real training run plus the shifted-input stress test, not this spec's tests.
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

25. `build_transforms3d(...)` and `Augment3D(...)` accept a `modality_dropout_prob` kwarg, default `0.0`. With the default, the returned `Compose` contains no `RandModalityDropoutd`, and `Augment3D(seed=s)` and `Augment3D(seed=s, modality_dropout_prob=0.0)` produce bitwise-identical output on the same input (existing behaviour unchanged; Req 17-24 keep passing unmodified).
26. A `modality_dropout_prob` that is negative, `>= 1.0`, or NaN raises `ValueError` at construction time, naming the offending value.
27. With `modality_dropout_prob = p` (`0 < p < 1`) and every other transform's probability set to 0.0, every output image channel is either bitwise-identical to the same input channel or exactly all-zero (`== 0.0` everywhere) — no channel is partially modified — and the label tensor is bitwise-identical to the input label.
28. With every OTHER transform's probability forced to 1.0 and `p = 0.99`, every dropped output channel is exactly all-zero (dropout runs after the intensity/noise transforms, so scale/shift/noise cannot re-introduce non-zero values into a dropped channel), and the label value set is still a subset of `{0, 1, 2, 3, 4}`.
29. Over at least 400 independent draws on a `(4, D, H, W)` input that is non-zero everywhere, with `p = 0.99`: no output is ever all-zero across all 4 channels, and each of the 4 channel indices is kept (non-zero) in at least one draw (the keep-one fallback is not biased toward a fixed channel).
30. Over at least 400 independent seeded draws with `p = 0.3` (all other probabilities 0.0), the observed drop rate of each of the 4 channels lies in `[0.2, 0.4]`.
31. Two `Augment3D(seed=s, modality_dropout_prob=p)` instances called once on the same input produce bitwise-identical output (reproducibility).
32. A `B = 16` batch of identical samples with `p = 0.5` produces drop patterns that are not all identical across the samples (independent per-sample draws).
33. `run.py --modality-dropout FLOAT` exists with default `0.0`. A value `> 0` without `--use-augmentation`, or any value outside `[0.0, 1.0)`, exits through an argparse error (`SystemExit` code 2) whose message names the offending flag(s), before any data loading or training. With `--use-augmentation --modality-dropout 0.2` the transform handed to training has dropout enabled at 0.2 (inspectable on its `Compose`) at both the single-client and federated call sites; with `--use-augmentation` alone it has no dropout transform (unchanged behaviour).
34. `train_single_client` runs at least 1 epoch on dummy data with a real (not mocked) `Augment3D(modality_dropout_prob=0.3)` and finishes with a finite loss. **No claim is made or tested about held-out Dice or missing-sequence robustness** — that requires a real training run and evaluation, outside this spec's tests.
35. `tools/build_fullres_cache.py`, given a synthetic NIfTI patient with known small volumes, writes `<sid>.img.npy` (float16 `(4,X,Y,Z)`, channel order t1c,t1n,t2f,t2w, equal to the raw data within float16 rounding), `<sid>.seg.npy` (uint8 `(X,Y,Z)`, exactly equal to the raw labels) and `<sid>.stats.npy` (float32 `(4,2)`, equal to the mean/std of the voxels > 0 of each RAW modality with rtol 1e-4; a modality with no voxel > 0 gets `(0,1)`).
36. The builder is safe and restartable: no `*.tmp` file and no partial `.npy` remains after a run; a second run reports the patients as skipped and leaves their files untouched (modification times unchanged); a patient with a missing file, mismatched shapes among the five files, or labels outside `{0..4}` is reported as an error, leaves NO output files, and does not stop the other patients from being built; `_build_summary.json` lists ok / skipped / errors.
37. `len(PatchDataset)` equals `patches_per_epoch`, and every item has `x` float32 of shape `(4,*patch)` and `y` int64 of shape `patch` with values a subset of `{0,1,2,3,4}`.
38. Normalisation and alignment: for a sampled patch, `x` equals an independent computation from the raw crop and the stored stats (voxels > 0 → `(v−mean)/std`, or `v−mean` when `std < 1e-8`; every other voxel exactly `0.0`), and `x` and `y` come from the SAME crop — verified on a synthetic volume whose image values encode voxel position, by locating the crop in the source volume and matching both `x` and `y` against it exactly.
39. Category sampling: with `fractions=(0.35,0.45,0.20)` over at least 2000 draws on a synthetic cohort, the observed rate of each category is within ±0.05 of its fraction (the chosen category and patient are observable through a documented method so this is testable).
40. Sampling guarantees, checked on every draw of at least 2000: an `et` patch contains at least one ET voxel and only ever comes from a patient that has ET; a `tumor` patch contains at least one voxel with label > 0; every patch lies fully inside the volume (`start >= 0` and `start + patch <= volume shape` on every axis).
41. Reproducibility: two datasets built with the same seed return bitwise-identical first 20 samples; different seeds return different samples; and successive `__getitem__` calls (the same index) return different patches.
42. Construction-time validation, each with a clear error: a patch dimension that is not a multiple of 16 → `ValueError`; a patch larger than the volume on any axis → `ValueError`; a missing cache file → `FileNotFoundError` naming the path; a category with positive fraction and no eligible patient → `ValueError`; fractions that are negative or do not sum to 1 (tolerance 1e-6) → `ValueError`.
43. `predict_volume` stitches correctly: with a pointwise (1×1×1 convolution) stub model and a volume whose shape is not divisible by the window size or the stride, the output equals the direct full-volume softmax (atol 1e-5), has shape `(5,X,Y,Z)`, dtype float32, lives on the CPU, and sums to 1 over the class axis (atol 1e-5).
44. Flip test-time augmentation is aligned: with the same pointwise stub, `flips=[(),(0,),(1,),(2,)]` gives the same probabilities as no flips (atol 1e-5) — a wrong un-flip would misalign positions and fail.
45. `TrainConfig` accepts `data_mode='patch'` with a `cache_path`; `'patch'` without `cache_path`, a `patch_size` that is not three positive integers, `patches_per_epoch < 1`, or invalid `patch_fractions` raises `ValueError`; the defaults and behaviour of the existing `dummy`/`real` modes are unchanged (Req 1-34 keep passing unmodified).
46. `run.py --data-mode patch --patch-size D H W --patches-per-epoch N --patch-fractions ET TUMOR RANDOM` reaches training with those values in the config; combining patch mode with `--use-federation` exits through an argparse error (`SystemExit` code 2) before any data is loaded; existing invocations behave as before.
47. End to end: `train_single_client` runs at least 1 epoch in patch mode with a real `Augment3D(modality_dropout_prob=0.3)` on the tiny synthetic cache, finishes with a finite loss, and writes a checkpoint. No held-out-Dice claim is made or tested.
48. `apply_et_min_voxels(pred, T)`: when the total ET voxel count is below `T`, every label-1 voxel becomes 2 and nothing else changes; at or above `T` the array is unchanged; `T=0` is the identity; the input array is never mutated in place.
49. `evaluate()` on the tiny synthetic cache with a freshly built tiny-run checkpoint returns, for each threshold in the list, mean/ET/NC/WT Dice all in `[0,1]`, with NC and WT identical across thresholds; and the CLI's `RESULT_JSON:` line parses as JSON.

50. Validation: `RandSequenceShiftd(keys, prob, max_shift_voxels)` raises `ValueError` naming the value when `prob` is outside `[0, 1]` or NaN, or when `max_shift_voxels` is negative, NaN or infinite; calling it on an input with fewer than 2 channels raises `ValueError` at call time (even when the draw would not apply the shift). `prob = 0` and `max_shift_voxels = 0` are accepted.
51. Identity when not applied: with `prob = 0` (or a call whose draw is "not applied") the output equals the input bitwise, is a distinct object (no aliasing), and the input is never modified in place (checked for both a `torch.Tensor` and a NumPy array).
52. Reference channel: over at least 400 applied draws (`prob = 1`, 4 channels) the reference channel is bitwise unchanged in the output, and it is uniformly distributed (each of the 4 channels is the reference in 15-35% of the draws). The drawn plan is observable through a `randomize(n_channels)` method that returns the reference index and a `(C, 3)` array of shift vectors (zero rows for the reference and for unselected channels).
53. At least one shift per applied call: in every one of at least 200 applied draws at least one non-reference channel has a non-zero shift vector, and on a non-constant volume the output differs from the input in at least one channel.
54. Translation semantics: with integer shift vectors forced through a subclass or monkeypatch of `randomize`, the output equals the corresponding `np.roll` shift on the interior and every vacated voxel is exactly 0; a sub-voxel shift of a linear ramp equals the analytic shifted ramp on the interior (atol 1e-4) along each of the three axes.
55. Magnitude and independence: over at least 200 applied calls, the recovered shift of every shifted channel (measured with ramps) never exceeds `max_shift_voxels` by more than 1e-3 on any axis; in calls with two or more shifted channels the shift vectors differ between channels in a majority of draws.
56. Type safety: the output keeps shape and dtype for `float32` and `float16` inputs, and tensor/ndarray type; the transform only reads and writes the keys it is given, so a data dict's label entry is the same object before and after; with `keys=[IMAGE_KEY]` the label is never touched.
57. Determinism: the same seed reproduces the output bitwise; a different seed produces a different output.
58. Chain integration: `build_transforms3d(sequence_shift_prob=0.0)` yields exactly the transform-type list it yielded before this addendum; with `sequence_shift_prob > 0` the new transform sits after `RandGaussianNoised` and before `RandModalityDropoutd` (and before `AssertLabelValuesd`); with both shift and dropout enabled at probability 1 the dropped channels are still exactly 0 in the final output and labels stay valid; `Augment3D` forwards both `sequence_shift_*` kwargs; invalid values raise `ValueError` at construction.
59. CLI: `run.py` defaults are `--sequence-shift 0.0` and `--sequence-shift-max-voxels 1.2`; a `--sequence-shift` outside `[0, 1)` or NaN, and a `--sequence-shift-max-voxels` that is not a positive finite number, exit through argparse (code 2); `--sequence-shift > 0` without `--use-augmentation` exits through argparse naming both flags; with valid values both reach the `Augment3D` constructor, in `--data-mode real` and in `--data-mode patch` (asserted by capturing the constructor's kwargs).
60. End to end: `train_single_client` runs at least 1 epoch with a real `Augment3D(sequence_shift_prob=1.0, modality_dropout_prob=0.15)` and finishes with a finite loss. No claim is made or tested about robustness to misregistration.

**Assumption (Addendum 4):** The default maximum of 1.2 voxels is chosen to match the worst stress-test shift (3 mm) on the 96³ grid; each non-reference channel being selected with probability 0.5 and the default application probability used for the first training run (0.5) are judgment calls, not tuned.
**Assumption (Addendum 4):** Labels are defined in the co-registered BraTS frame, so after shifting a sequence the supervision for that sequence is only approximately aligned; the reference channel is drawn per call because the model cannot know which sequence is correctly registered. Whether this helps is decided by the stress-test comparison, not by this spec.
**Assumption (Addendum 4):** `grid_sample` runs on CPU inside the augmentation hook (a few milliseconds for a 96³ channel); half-precision input is computed in float32 and cast back.

**Assumption (Addendum 3):** The sampling defaults (35% ET-centred, 45% tumour-centred, 20% uniformly random patches) mirror the 2D pipeline's slice expansion (a floor on ET-containing slices, a ceiling on tumour-free ones); they are a judgment call, not tuned. `patches_per_epoch = 580` is about four patches per training patient and is likewise only a default.
**Assumption (Addendum 3):** The full-resolution cache lives outside the repository (about 20 GB for all 257 patients) and is built once; `nibabel` is needed by the builder tool only and is not added to the training requirements.
**Assumption (Addendum 3):** The `DataLoader` keeps `num_workers=0`, so the dataset's internal generator is a single stream. Resume does not restore that stream, so patch mode is not bitwise-resumable (Req 10 covers the 96³ modes only); an epoch after resume simply draws fresh patches.
**Assumption (Addendum 3):** Sliding-window inference defaults (50% overlap, gaussian blending) are the standard choices, not tuned; the window size equals the training patch size.

**Assumption:** FedAvg round/local-epoch counts, learning rate, and optimizer are exposed as config parameters with reasonable defaults (Adam, lr=1e-3, 1 local epoch/round, 2 rounds for smoke tests) rather than fixed — no spec constraint dictates specific values.
**Assumption:** `features` is produced by global-average-pooling the U-Net bottleneck activation map to a 1D vector; exact dimensionality is an implementation default, not contractually fixed.
**Assumption:** Checkpoint resume identifies "latest" by highest epoch/round number found in the run's checkpoint directory, not by a separately tracked "latest" pointer file.
**Assumption:** Loss function is Dice+CrossEntropy over the 5 label classes (standard MONAI choice for multi-class 3D segmentation); not contractually specified. **Superseded in part:** `run.py --loss` also accepts `dice_focal` (a class-imbalance-aware variant); `dice_ce` remains the default.
**Assumption:** Test framework is `pytest`, consistent with a Python/MONAI/PyTorch stack.
**Assumption (addendum):** Augmentation transform probabilities and magnitudes (flip 0.5, rotate 0.3 @ ±10°, zoom 0.3 @ 0.9-1.1×, intensity scale/shift 0.3 @ ±0.1, Gaussian noise 0.2 @ std 0.05) mirror section 03's 2D defaults for consistency across the project; not contractually specified, and untuned against real 3D held-out Dice.

**Assumption (Addendum 2):** A missing sequence is encoded as an all-zero channel *after* z-score normalization. Skull-stripped background is not 0 after z-scoring, so a zeroed channel is not identical to a real sequence's background level; this matches the convention the 2026-09-25 missing-sequence measurements used, and the model learns that convention from training, so deployment must apply exactly the same encoding to an absent sequence.
**Assumption (Addendum 2):** The per-channel drop probability used for the first real training run (0.15: ~52% of samples keep all 4 sequences, ~37% lose one, ~10% two, ~1% three) is a judgment call, not tuned; nothing here fixes a value.
**Assumption (Addendum 2):** The federated path needs no separate wiring because `run.py` builds one augmentation transform and passes it to both paths; Req 33 checks both call sites receive it, and Req 34 exercises the single-client loop only.

## 5. Structure

```
01_model_federated/
├── BRIEF.md
├── SPEC.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── model.py            # 3D U-Net wrapper: model(x) -> (seg_logits, features)
│   ├── data.py             # Manifest-driven Dataset: dummy-tensor mode + real .nii.gz mode; dispatches 'patch' (addendum 3)
│   ├── patch_data.py       # PatchDataset: full-res, tumour/ET-balanced random crops (addendum 3)
│   ├── patch_infer.py      # predict_volume: sliding-window (+ flip TTA) inference (addendum 3)
│   ├── augment3d.py        # 3D MONAI augmentation stack + Augment3D callable (addendum); + RandModalityDropoutd (addendum 2)
│   ├── train_single.py     # Single-client training loop
│   ├── federated.py        # FedAvg orchestration: local training + weighted aggregation
│   ├── checkpoint.py        # save/load/resume helpers
│   └── config.py           # Config dataclass/CLI: use_augmentation, use_federation, use_domain_adaptation, etc.
├── run.py                  # CLI entry point (single-client / federated / resume); builds Augment3D() when --use-augmentation is set; --modality-dropout (addendum 2)
├── tools/
│   ├── build_fullres_cache.py  # raw NIfTI -> full-res .npy cache + stats (addendum 3)
│   └── eval_patch3d.py         # full-res sliding-window held-out Dice + ET rule (addendum 3)
├── checkpoints/            # Created at runtime, gitignored
└── tests/
    ├── test_model.py
    ├── test_data.py
    ├── test_single_client.py
    ├── test_federated.py
    ├── test_checkpoint.py
    ├── test_augment3d.py   # addendum; + sequence-dropout tests and CLI validation (addendum 2)
    ├── test_build_fullres_cache.py  # addendum 3
    ├── test_patch_data.py           # addendum 3 (+ config / CLI validation, end-to-end training)
    ├── test_patch_infer.py          # addendum 3
    ├── test_eval_patch3d.py         # addendum 3
    └── test_sequence_shift.py       # addendum 4 (transform, chain order, CLI, end-to-end)
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
| `modality_dropout_prob = 0.0` (default) | `RandModalityDropoutd` is not added to the chain at all; output identical to before this addendum |
| `modality_dropout_prob` negative, `>= 1.0`, or NaN | `ValueError` naming the value at construction (library); argparse error, exit code 2 (CLI) |
| `--modality-dropout 0.2` given without `--use-augmentation` | argparse error naming both flags, before any data loading or training (would otherwise silently do nothing) |
| A draw would drop every input channel | One channel, chosen uniformly at random, is kept; the model never receives an all-zero input |
| Input has fewer than 4 channels (e.g. 1) | Same rule applies: a lone channel is never dropped |
| Patch size not a multiple of 16, or larger than the volume | `ValueError` at construction naming the offending size/axis (the UNet has 4 stride-2 stages) |
| Cache file missing for a manifest patient | `FileNotFoundError` naming the path, at construction, before any training |
| `--data-mode patch` given without `--cache-path` | Same `ValueError` as `real` mode ("requires an explicit cache_path") |
| `--data-mode patch` together with `--use-federation` | argparse error (exit 2), before any data loading |
| No manifest patient has ET but `fractions[0] > 0` (or none has any tumour but `fractions[1] > 0`) | `ValueError` naming the empty category — never a silent renormalisation |
| Builder meets a corrupt/mismatched patient | That patient is reported as an error with no output files; the rest still build |
| Builder killed mid-write | No partial `.npy` and no `.tmp` is left as a valid-looking file; a rerun rebuilds only the missing patients |
| A volume is smaller than the sliding window on an axis | `predict_volume` still returns a full `(5,X,Y,Z)` result (the inferer pads); no crash |
| `apply_et_min_voxels` receives an array with no ET | Returned unchanged |
| `sequence_shift_prob = 0.0` (default) | `RandSequenceShiftd` is not added to the chain at all; output identical to before this addendum |
| `sequence_shift_prob` negative, `> 1`, or NaN; `max_shift_voxels` negative, NaN or infinite | `ValueError` naming the value at construction (library); argparse error, exit code 2 (CLI) |
| `--sequence-shift 0.5` given without `--use-augmentation` | argparse error naming both flags, before any data loading or training |
| Input has a single channel | `ValueError` at call time: there is no other sequence to be misregistered against |
| A shift moves a channel's content past the volume edge | The vacated voxels are exactly 0 and content shifted out is lost; the label and the reference channel are untouched |
| Shift and modality dropout both enabled | Dropout runs last; dropped channels are exactly 0 regardless of the shift |

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
- [x] Req 25: `modality_dropout_prob` kwarg, default 0.0 adds no dropout transform and output is bitwise-identical to before
- [x] Req 26: negative / >= 1.0 / NaN `modality_dropout_prob` raises `ValueError`
- [x] Req 27: with other probs 0, every output channel is bitwise-unchanged or exactly all-zero; label bitwise-unchanged
- [x] Req 28: with other probs 1.0 and `p=0.99`, dropped channels are still exactly all-zero (dropout runs last); labels stay valid
- [x] Req 29: over >= 400 draws at `p=0.99`, never all channels dropped, and each of the 4 channels is kept at least once
- [x] Req 30: over >= 400 seeded draws at `p=0.3`, each channel's drop rate is in [0.2, 0.4]
- [x] Req 31: same-seed reproducibility with dropout enabled
- [x] Req 32: `B=16` identical samples at `p=0.5` do not all get the same drop pattern
- [x] Req 33: `run.py --modality-dropout` default 0.0; `>0` without `--use-augmentation` or out-of-range exits with argparse error (code 2); with `--use-augmentation` the transform has dropout at the given value (both call sites); without the flag, none
- [x] Req 34: end-to-end `train_single_client` 1 epoch with real `Augment3D(modality_dropout_prob=0.3)` gives a finite loss — no held-out-Dice claim
- [x] Req 35: full-res cache builder writes correct `.img.npy` / `.seg.npy` / `.stats.npy` for a synthetic NIfTI patient
- [x] Req 36: builder is atomic, skips existing patients, isolates bad patients (no output files), writes `_build_summary.json`
- [x] Req 37: `PatchDataset` length, `x` float32 `(4,*patch)`, `y` int64 `(*patch)` with labels in {0..4}
- [x] Req 38: `x` equals independent normalisation of the raw crop (background exactly 0); `x` and `y` come from the same crop
- [x] Req 39: category rates within ±0.05 of the fractions over >= 2000 draws
- [x] Req 40: et patches always contain ET (from ET patients only), tumor patches contain tumour, every patch fully inside the volume
- [x] Req 41: same-seed bitwise reproducibility, different seeds differ, successive calls differ
- [x] Req 42: all construction-time validation errors (patch size, too large, missing file, empty category, bad fractions)
- [x] Req 43: sliding-window with a pointwise stub equals the direct full-volume softmax; shape/dtype/CPU/sums-to-1
- [x] Req 44: flip TTA with the pointwise stub equals no-TTA (un-flip alignment)
- [x] Req 45: `TrainConfig` patch fields validated; existing modes and defaults unchanged
- [x] Req 46: `run.py` patch flags reach the config; patch + federation is an argparse error (exit 2)
- [x] Req 47: end-to-end `train_single_client` in patch mode with real `Augment3D` + dropout: finite loss, checkpoint written
- [x] Req 48: `apply_et_min_voxels` semantics (1→2 below threshold, identity at/above and at T=0, no in-place mutation)
- [x] Req 49: `evaluate()` returns Dice in [0,1] per threshold, NC/WT identical across thresholds; `RESULT_JSON:` parses
- [x] Req 50: constructor validation (`prob`, `max_shift_voxels`) and the fewer-than-2-channels error at call time
- [x] Req 51: identity when not applied, no aliasing, no in-place modification (tensor and ndarray)
- [x] Req 52: reference channel bitwise unchanged and uniformly distributed over channels; `randomize` exposes the plan
- [x] Req 53: at least one non-reference shift per applied call, output differs from input
- [x] Req 54: integer shifts equal `np.roll` on the interior with exact-0 vacated voxels; sub-voxel ramp shift matches the analytic ramp
- [x] Req 55: shift magnitude within `max_shift_voxels`; shifted channels get independent vectors
- [x] Req 56: shape/dtype/type preserved for float32 and float16; label entry untouched
- [x] Req 57: same-seed bitwise reproducibility, different seeds differ
- [x] Req 58: chain order (after noise, before dropout/assert), default chain unchanged, dropped channels still exactly 0, `Augment3D` forwards kwargs
- [x] Req 59: `run.py` flag defaults, ranges, requires `--use-augmentation`, forwarded to `Augment3D` in real and patch modes
- [x] Req 60: end-to-end `train_single_client` 1 epoch with real `Augment3D(sequence_shift_prob=1, modality_dropout_prob=0.15)`: finite loss, no robustness claim
