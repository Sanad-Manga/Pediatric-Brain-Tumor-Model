# Spec: R* inference module (Section 06: rstar_inference)

## 1. Goal
A standalone, guard-railed Python package (`rstar`) that segments one co-registered BraTS-space pediatric brain MRI with the 2D+3D fusion recipe measured on 2026-09-26 (0.805 mean Dice on 30 patients that were never used to train or choose anything, against 0.711 for the shipped 2D ensemble), and fails loudly, not silently, when its inputs or its own geometry are wrong.

## 2. In Scope
- **The recipe (R\*):** the shipped 2D ensemble (`03_augmentation_eval/checkpoints/overnight_run/best.pt`, an `ensemble` payload with two members, verified bit-identical to the two member files) run on axial and coronal slices, and the original four-member 3D dropout family (`01_model_federated/checkpoints_best/session2_A_focal_seqdrop_final_ep262.pt`, `session2_C_ce_seqdrop_final_ep125.pt`, `session3_D_focal_sd25_final_ep224.pt`, `session3_E_ce_sd25_final_ep317.pt`) run on a 96³ resample with flip test-time augmentation. Fusion: `probs = w3d·p3d + (1−w3d)·p2d` with `w3d = 0.5`; `probs[background] *= 0.5`; argmax; then the small-ET rule (relabel enhancing tumour, label 1, as non-enhancing, label 2, when the total predicted ET volume is below 500 mm³).
- **Input contract:** a float array `(4, 240, 240, 155)` in channel order `t1c, t1n, t2f, t2w` in BraTS-PEDs space (1 mm isotropic, skull-stripped so the background is exactly 0, all sequences co-registered). Absent sequences are declared with a `present` tuple of four bools (through the path API: an omitted file). The module does not register, skull-strip, resample or read DICOM.
- **Output:** `RStarResult(labels, mode, status, warnings, diagnostics)`: `labels` is `uint8` with the input's spatial shape and values in `0..4`; `mode` is `'R*'` or `'3D-only'`; `status` is `'ok'` or `'review'`; `warnings` is a `list[str]`; `diagnostics` holds `agreement` (whole-tumour Dice between the 2D-alone and the 3D-alone argmax; `None` in 3D-only mode), `et_voxels_before_rule`, `et_relabelled` (bool), `present`, `checkpoints` (name and SHA-256 of every model file used), `elapsed_s` and `mode`.
- **Package layout** (`06_rstar_inference/`): `rstar/{__init__,config,contract,preprocess,sections,models,fusion,guards,pipeline,cli}.py`, `config/models.default.json`, `tests/`, `README.md`, `SPEC.md`. Run as `python -m rstar --t1c a.nii.gz --t1n b.nii.gz --t2f c.nii.gz --t2w d.nii.gz --out seg.nii.gz [--json report.json] [--models-root DIR]`; a sequence whose flag is omitted is absent. `nibabel` is used only by the CLI and the path API.
- **Section isolation:** sections 01 and 03 both own a top-level package named `src`. `sections.py` imports each under a unique alias (`rstar_sec01_src`, `rstar_sec03_src`) with `importlib` (`spec_from_file_location` with `submodule_search_locations`), registered in `sys.modules`. Neither directory is put on `sys.path`, so `import src` elsewhere is unaffected. Both packages use relative imports only (verified).
- **Checkpoint pinning:** checkpoints are not in git. `config/models.default.json` lists each member as a path relative to `models_root` (argument, else the environment variable `RSTAR_MODELS_ROOT`, else the repository root) plus a SHA-256. Loading verifies every hash and raises `ModelIntegrityError` naming the file on a mismatch (`verify_hashes=False` overrides). The build computes the hashes from the real files and fails loudly if a file is missing.
- **Preprocessing exactly as measured.** *3D:* per-channel trilinear resample of the raw volume to 96³ (`torch.nn.functional.interpolate`, `align_corners=False`), cast to float16 and back to float32, per-channel z-score over voxels `> 0` (`std < 1e-8` → subtract the mean only; background stays 0); absent channels are all-zero after z-scoring; inference averages, for each of the four members, the softmax over the four flips `((), (0,), (1,), (2,))` (each un-flipped), then averages the members, then upsamples trilinearly to the input grid. *2D:* per-channel brain-only z-score at full resolution with float64 statistics (voxels `> 0`, background exactly 0); slices are float16 round-tripped; axial slice `k` = `norm[:, :, :, k][:, ::-1, ::-1]` for every `k` with any brain voxel; coronal slice `k` = `norm[:, :, j, :][:, ::-1, :]` with `j = Y−1−k` for every slice with any brain voxel; slices are padded (never resized) to `cfg.common_size`; each member's softmax is restacked into the volume by the module's own restack (axial along axis 2, coronal along axis 1, slices without brain stay 0), then averaged over members and planes, then flipped along axes 0 and 1 to reach the NIfTI frame (constant `FRAME_FLIPS = (0, 1)`, verified on 111 patients on 2026-09-26).
- **Modes:** all four sequences present → `'R*'`. Any sequence absent → `'3D-only'` (the shipped 2D model collapses when a sequence is missing: 0.14 without FLAIR): only the 3D family runs, absent channels are zeroed, the ET rule still applies; one absent → status `'ok'` plus a warning naming it, two or more absent → status `'review'`; none present → `ContractError`.
- **Guards:** input contract errors (below); a soft agreement check (`agreement < 0.70` → status `'review'` plus a warning; `agreement < 0.10` → status `'review'` plus a stronger warning naming a possible 2D collapse or a frame/orientation error); and `self_check()`, which runs the whole 2D path and the 3D path with identity stub models on a synthetic asymmetric blob and asserts the output blob lands where the input blob was (2D path exactly; 3D path within 3 voxels on the 240-voxel grid), so a silent frame or orientation bug fails loudly.
- **Dependency injection:** `RStarSegmenter(config, models_2d=None, models_3d=None, run_self_check=None)`. Stub models can be supplied for tests. When models are loaded from checkpoints, `self_check` runs once per process unless `run_self_check=False`.
- **Defaults in `RStarConfig`:** `w3d=0.5`, `background_scale=0.5`, `et_min_mm3=500.0`, `voxel_mm3=1.0`, `agreement_review=0.70`, `agreement_strong=0.10`, `expected_shape=(240, 240, 155)` (overridable so tests can use tiny volumes), `verify_hashes=True`, `device='auto'` (CUDA when available).
- **README.md** stating the input contract, the modes and statuses, and: *validated on US BraTS-PEDs only; needs radiologist review; not a medical device*.

## 3. Out of Scope
- Registration, skull-stripping, resampling to BraTS space, DICOM I/O, NIfTI header repair.
- Any change to sections 01, 03 or 05 (`05_frontend_demo` is not touched; a later integration step is the maintainers' call).
- Training, evaluation scripts, batching several patients, multi-GPU, memory tuning for small machines.
- Uncertainty calibration or a learned confidence score. The agreement flag is a weak soft signal only; the module never withholds labels because of low agreement (one genuine 2D collapse on a correctly aligned held-out patient had agreement 0.000).
- **Any claim of clinical validity or of performance on non-US data.** Reported numbers are US BraTS-PEDs only.
- Bundling checkpoints into the repository.

## 4. Requirements

1. `RStarConfig` defaults are exactly `w3d=0.5`, `background_scale=0.5`, `et_min_mm3=500.0`, `voxel_mm3=1.0`, `agreement_review=0.70`, `agreement_strong=0.10`, `expected_shape=(240,240,155)`, `verify_hashes=True`, `device='auto'`. `w3d` outside `[0,1]`, `background_scale` that is non-positive or non-finite, negative `et_min_mm3`, non-positive `voxel_mm3`, agreement thresholds outside `[0,1]`, or `agreement_strong > agreement_review` raise `ValueError` naming the field.
2. Input contract errors: a volume with the wrong ndim, channel count or spatial shape, any non-finite value, more than 5% negative voxels in a declared-present sequence, a declared-present sequence that is entirely zero, or a declared-present sequence whose exactly-zero voxel fraction is below 0.15 raises `ContractError` whose message names the problem and the fix. A zero fraction in `[0.15, 0.25)` produces a warning in the result and no error. Negative voxels are background to the models (only voxels `> 0` count as brain, exactly as in the measured pipeline): up to 1% negative voxels are accepted silently (real BraTS-PEDs scans contain up to about 0.5%, from resampling ringing, with values down to about −340), between 1% and 5% produce a warning, and more than 5% raise `ContractError`. A valid synthetic BraTS-like volume passes with no warning. `present` must be a length-4 sequence of bools (else `ContractError`); no sequence present raises `ContractError`. The content of an absent sequence is ignored (zeros, `None` or garbage are all accepted), and the input array is never modified.
3. Small-ET rule in mm³: ET is relabelled 1→2 exactly when `0 < ET_voxels × voxel_mm3 < et_min_mm3`; a total exactly at the threshold keeps ET; a volume with no ET is unchanged; only label 1 changes (every voxel that was 1 becomes 2, nothing else moves, so the whole-tumour and tumour-core voxel sets are unchanged); `voxel_mm3 ≠ 1` scales the count; the input array is never modified in place.
4. Fusion arithmetic equals an independent NumPy computation (`w3d`, background scale, argmax) on random probability volumes; ties resolve to the lowest class index; `w3d=1` with `background_scale=1` equals the 3D-only argmax and `w3d=0` equals the 2D-only argmax; output shape equals the input's spatial shape.
5. `agreement(p2, p3)` equals an independent whole-tumour Dice between the argmaxes; it is 1.0 for identical outputs, 0.0 for disjoint tumours, 1.0 when both are empty and 0.0 when exactly one is empty, and it is symmetric.
6. 3D preprocessing: the resampled volume has shape `(4,96,96,96)` and dtype float32; every value is float16-representable (`x.astype(float16).astype(float32) == x`); the resample equals an independent `torch.nn.functional.interpolate` call; the z-score leaves background exactly 0 and gives brain voxels (`> 0`) mean 0 and std 1 (atol 1e-4); a constant positive channel becomes all zeros (`std < 1e-8` branch) and an all-zero channel stays all zero, with no NaN; absent channels are exactly all-zero in the 3D input tensor after z-scoring while present channels equal the independent computation.
7. 2D preprocessing: on an asymmetric synthetic volume the extracted axial and coronal slices equal independent indexing (including the reversals and `j = Y−1−k`), only slices containing brain are extracted, values are float16-representable, and restacking the slice stack is the exact inverse of extraction with brainless slices left at 0.
8. Frame: `FRAME_FLIPS == (0, 1)`, and through the identity-stub 2D path an asymmetric blob lands at exactly the voxel coordinates it had in the input (both planes).
9. 3D path geometry: through the identity-stub 3D path an asymmetric blob's centre of mass lands within 3 voxels (scaled to the volume size on tiny test volumes) of its input position. With a pointwise stub model, the flip test-time augmentation output equals the direct softmax (atol 1e-5), proving the un-flip alignment.
10. Averaging: with stub members returning known logits, the ensemble probabilities equal the mean of the members' softmax outputs (over members, then over flips or planes), and each output sums to 1 over classes (atol 1e-4).
11. Section isolation: `load_section_package` returns the alias package for section 01 and for section 03; both alias packages import their `model` submodule; loading changes neither `sys.path` nor creates or replaces `sys.modules['src']`; a second call returns the same cached module; a path that does not exist raises `FileNotFoundError` naming it.
12. Hash verification with temp files: a matching hash passes; a mismatch raises `ModelIntegrityError` naming the file and the first 12 hex characters of both hashes; a missing file raises `FileNotFoundError` naming the path; `verify_hashes=False` skips the check. `config/models.default.json` lists exactly one 2D ensemble file and four 3D files, each with a relative path (no drive letter, not absolute) and a 64-hex-digit SHA-256.
13. Modes: with all four sequences present the mode is `'R*'` and both the 2D and 3D stubs are called; with one absent the mode is `'3D-only'`, the 2D stub is never called, the absent channel reaches the 3D stub as exactly zeros, the status is `'ok'` and a warning names the missing sequence; with two or more absent the status is `'review'`; with none present a `ContractError` is raised.
14. Output: `labels` is `uint8` with the input's spatial shape and values in `0..4`; the input volume is not modified; two runs give identical results; `diagnostics` contains the keys `agreement`, `et_voxels_before_rule`, `et_relabelled`, `present`, `checkpoints`, `elapsed_s`, `mode`.
15. `diagnostics['et_relabelled']` is true exactly when `0 < ET × voxel_mm3 < et_min_mm3`, and then the final labels contain no label 1.
16. Agreement statuses, with controllable stubs: `agreement ≥ 0.70` → `'ok'` (given no other warnings); `0.10 ≤ agreement < 0.70` → `'review'` with a warning; `agreement < 0.10` → `'review'` with a warning containing both the words "collapse" and "orientation"; in every case `labels` is returned (never `None`).
17. `self_check()` passes on the correct code and raises `SelfCheckError` when `FRAME_FLIPS` is monkeypatched to a wrong value and when the 3D path is displaced by monkeypatch; with injected models `self_check` does not run automatically unless `run_self_check=True`, and with models loaded from checkpoints it runs once per process (counter asserted with a patched loader).
18. Path API and CLI: `segment_paths` reads NIfTI with `nibabel`, takes the voxel size from the header zooms, and raises `ContractError` when it is not within 0.9–1.1 mm isotropic; the CLI treats an omitted sequence flag as absent, exits 2 with the message on stderr for a contract error, and on success writes a `uint8` NIfTI with the input affine plus a JSON report (status, mode, warnings, diagnostics) and exits 0.
19. `import rstar` does not modify `sys.path` and does not import section 01 or 03 code until models are loaded.
20. `README.md` contains the phrases "US BraTS-PEDs only", "radiologist review" and "not a medical device", and documents the input contract and the two modes.
21. Slow regression test (skipped unless `RSTAR_MODELS_ROOT` and `RSTAR_TEST_DATA` are set): three fresh-30 patients run end to end reproduce the reference per-patient NC, WT and ET (rule applied) Dice stored in `tests/fixtures/rstar_reference_scores.json`, taken from the 2026-09-26 post-hoc run, within 0.002; the shipped checkpoint hashes in `models.default.json` match the real files.

**Assumption:** The negative-voxel limits (1% warn, 5% error) were set after the regression run showed that real BraTS-PEDs scans do contain negative voxels (0.03-0.47% in the four sequences of one fresh patient); the first draft of this spec wrongly rejected any negative value.
**Assumption:** The thresholds 0.15 and 0.25 (background fraction) come from BraTS volumes being 30–60% background; a non-stripped scan has almost no exactly-zero voxels. They are judgment calls, not tuned.
**Assumption:** The agreement thresholds (0.70 review, 0.10 strong) are judgment calls. On 111 evaluation patients the agreement had median 0.84–0.90, 5th percentile 0.34–0.59 and minimum 0.000, so 0.70 flags roughly the lowest quarter and 0.10 only near-total disagreement.
**Assumption:** Three-or-fewer-sequence inputs run through the 3D family only because that is the only measured-robust path (3D dropout members were trained with sequences hidden); two-or-more absent is unvalidated, hence `'review'`.
**Assumption:** The module returns a single case per call, on one device, in float32; memory use is a full-resolution five-class probability volume per branch (about 180 MB each at 240×240×155).
**Assumption:** Voxel volume is 1 mm³ for the BraTS grid; the mm³ form of the 500-voxel rule is the same rule expressed in physical units.

## 5. Structure

```
06_rstar_inference/
├── SPEC.md
├── README.md                     # contract, modes, statuses, validity disclaimer
├── config/
│   └── models.default.json       # relative checkpoint paths + SHA-256 (1 x 2D ensemble, 4 x 3D)
├── rstar/
│   ├── __init__.py               # public API: RStarSegmenter, RStarConfig, RStarResult, errors
│   ├── config.py                 # RStarConfig (+ validation), models manifest loading
│   ├── contract.py               # input validation, ContractError
│   ├── preprocess.py             # 3D resample + z-score, 2D slices, restack, FRAME_FLIPS
│   ├── sections.py               # alias import of section 01 / 03 `src` packages
│   ├── models.py                 # hash-verified loading of the 2D ensemble and the 3D family
│   ├── fusion.py                 # fuse, small-ET rule (mm^3), agreement
│   ├── guards.py                 # agreement statuses, self_check, SelfCheckError
│   ├── pipeline.py               # RStarSegmenter.segment / segment_paths
│   └── cli.py                    # python -m rstar
└── tests/
    ├── conftest.py               # tiny synthetic volumes, controllable stub models
    ├── test_config_contract.py   # Req 1-2, 19
    ├── test_fusion.py            # Req 3-5, 10
    ├── test_preprocess.py        # Req 6-9
    ├── test_sections_models.py   # Req 11-12
    ├── test_pipeline.py          # Req 13-17
    ├── test_cli_readme.py        # Req 18, 20
    ├── test_regression.py        # Req 21 (slow, skipped without data)
    └── fixtures/rstar_reference_scores.json
```

## 6. Edge Cases

| Scenario | Expected Behaviour |
|---|---|
| Volume shape `(4,240,240,150)` or three channels | `ContractError` naming the expected shape `(4,240,240,155)` |
| A NaN or infinite intensity in a present sequence | `ContractError` naming the problem |
| A few negative voxels (real BraTS scans: up to ~0.5% of the volume) | Accepted silently; treated as background like the measured pipeline |
| 1-5% negative voxels | Runs; result carries a warning |
| More than 5% negative voxels | `ContractError` (not a raw-intensity scan) |
| A present sequence with under 15% exactly-zero voxels (skull still on) | `ContractError` telling the user to skull-strip |
| A present sequence that is all zeros | `ContractError` naming the sequence; declare it absent instead |
| Zero fraction between 15% and 25% | Runs; result carries a warning |
| `present=(False,False,False,False)` | `ContractError` (no sequence) |
| One sequence absent | `'3D-only'`, status `'ok'`, warning naming it |
| Two or more absent | `'3D-only'`, status `'review'` |
| 2D and 3D disagree strongly or 2D predicts nothing (agreement 0.000) | Labels still returned; status `'review'` with the stronger warning |
| ET total exactly 500 mm³ | ET kept (threshold is exclusive: relabel only below) |
| ET total below 500 mm³ | All ET relabelled to non-enhancing; `et_relabelled` true |
| Checkpoint file missing | `FileNotFoundError` naming the path, at load time |
| Checkpoint hash mismatch | `ModelIntegrityError` naming file and hashes |
| `verify_hashes=False` | Hash check skipped; nothing else changes |
| NIfTI voxel size 2 mm or anisotropic | `ContractError` naming the voxel size (models trained at 1 mm) |
| A frame/orientation bug introduced in a later edit | `self_check()` raises `SelfCheckError` before any patient is segmented |
| `import src` elsewhere in the process | Unaffected by loading the section packages |
| Input array is a read-only view | Accepted; never written to |

## 7. Done Checklist

- [x] Req 1: `RStarConfig` defaults and validation errors
- [x] Req 2: input-contract errors and warnings, `present` validation, input never modified
- [x] Req 3: small-ET rule in mm³ (threshold exclusive, only label 1 changes, scaling, no in-place edit)
- [x] Req 4: fusion arithmetic against an independent computation, ties, `w3d` extremes
- [x] Req 5: `agreement` equals an independent whole-tumour Dice, edge cases, symmetric
- [x] Req 6: 3D preprocessing (shape, dtype, float16 round trip, resample, z-score properties, absent channels zero)
- [x] Req 7: 2D slice extraction and restack against independent indexing
- [x] Req 8: `FRAME_FLIPS` and 2D-path blob position exact
- [x] Req 9: 3D-path blob position within tolerance; flip TTA un-flip alignment
- [x] Req 10: ensemble averaging and probability sums
- [x] Req 11: section alias import isolation
- [x] Req 12: hash verification and the default manifest format
- [x] Req 13: R* and 3D-only modes, absent channels zero, statuses and warnings
- [x] Req 14: output shape/dtype/values, no input mutation, determinism, diagnostics keys
- [x] Req 15: `et_relabelled` exact semantics
- [x] Req 16: agreement statuses and warnings, labels always returned
- [x] Req 17: `self_check` behaviour and one-time run
- [x] Req 18: path API and CLI behaviour
- [x] Req 19: import hygiene
- [x] Req 20: README content
- [x] Req 21: regression test reproduces the reference scores (when data and checkpoints are available)
