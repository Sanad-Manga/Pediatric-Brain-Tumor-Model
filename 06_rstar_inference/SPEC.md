# Spec: R* inference module (Section 06: rstar_inference)

> **Addendum 1 (2026-10-02) — spot-level review flags and the error scorecard (team decisions Q1 = C, Q2 = D):** §2, §3, §4 (Req 22-28), §5, §6 and §7 were extended. This is a *forward* plan. The bio-tech doctor said missed and false enhancing tumour (ET) are both unacceptable, and that uncertain areas may be flagged for (biopsy) review. Today the 500 mm³ rule silently relabels all predicted ET as non-enhancing when a patient's total ET is small: on 81 held-out patients it removes 24 real small ET spots to remove 46 false ones, and this week every 30-patient comparison was decided by a few 500-1,000 voxel false ET spots that survived it. This addendum adds a **standalone** module that splits predicted ET into separate spots, describes each (size, confidence, agreement between the model sources) and marks each spot `keep` or `review`; plus a scorecard that counts errors per spot, including **silent** errors (a false spot kept, or a real lesion not covered by any kept or review spot). It is **not wired into `RStarSegmenter`** (the deployed pipeline and its outputs are unchanged); integration is a later, evidence-gated step. The "uncertainty" out-of-scope line of §3 is narrowed accordingly: spot-level review flags from fixed, documented thresholds are in scope; learned or calibrated confidence remains out of scope.
>
> **Addendum 2 (2026-10-04) - review flags wired into `RStarSegmenter`, off by default:** §2, §3, §4 (Req 29-36), §5, §6 and §7 were extended. The pre-registered review-flag study (2026-10-02) chose the rule *flag every enhancing-tumour spot of at least 50 voxels whose mean ET probability is below 0.7* (fresh-30: silent false spots 3 -> 0, silently missed lesions 4 -> 3 of 21, 0.5 spots to review per patient). The team decided to integrate it after Oct 6 as its own reviewed step; this addendum is that step. Flags are **added information only**: with flags on or off the labels, mode, status, warnings and existing diagnostics are identical, so the deployed output does not change. It supersedes Addendum 1's out-of-scope line "Wiring review flags into `RStarSegmenter`, its CLI or its outputs" and the clause of Req 28 that `pipeline.py` does not import `review_flags`.
>
> **Addendum 3 (2026-10-04) - optional fragment cleanup, off by default:** §2, §3, §4 (Req 37-43), §5, §6 and §7 were extended. Under the official BraTS-PED lesion-wise metric every predicted component that touches no real tumour counts as a whole false lesion; R* leaves many tiny stray fragments (held-out tumour core: 110 in 23 of 81 patients, median 4 voxels). A pre-registered test (`PREREGISTERED_fragments_hybrid_2026-10-04`, cleanup size chosen on held-out only) found that removing components under 200 voxels before the 500 mm³ rule leaves legacy Dice unchanged on fresh-30 (0.8050 vs 0.8051) and raises lesion-wise Dice by +0.074 (CI +0.025 to +0.131); combined with review flags (`PREREGISTERED_combo_2026-10-04`) silent false ET spots fell 5 -> 2 (fresh-30) and 22 -> 2 (held-out). The size was chosen after seeing these sets, so the option ships off by default pending cross-validation.
>
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
- **Review flags (Addendum 1), new module `rstar/review_flags.py`, numpy + scipy only:**
  - `find_et_spots(labels, et_prob, source_et_masks=())` -> `(spot_ids, spots)`: 26-connected components of `labels == 1` numbered 1..k in scan order; `spot_ids` is a `uint16` volume (0 = no spot). Each `Spot` (frozen dataclass) has `spot_id`, `voxels`, `mean_et_prob`, `max_et_prob`, and `agreement` = the mean over the given boolean source masks of the fraction of the spot's voxels that source marks ET (`None` when no sources are given).
  - `decide(spots, size_cut, prob_cut, agree_cut, small_cut=500)` -> one `Decision(spot_id, action, reason)` per spot: `action = "review"` if `voxels < size_cut`, or `mean_et_prob < prob_cut`, or (`agreement` is not None and `agreement < agree_cut`); otherwise `"keep"`. `reason` for a review spot is the first that applies in this order: `"small"` (`voxels < small_cut`), `"borderline"` (`small_cut <= voxels < size_cut`), `"low_confidence"`, `"models_disagree"`; `reason` is `None` for kept spots.
  - `review_outputs(spot_ids, spots, decisions)` -> `(review_mask, review_spots)`: `review_mask` uint8 (same shape) holding 1..m for review spots only (renumbered consecutively), 0 elsewhere; `review_spots` a JSON-serialisable list of `{"spot_id", "voxels", "mean_et_prob", "models_agree", "reason"}` using the renumbered ids. This is the file format of issue #44.
  - `score_patient(spot_ids, spots, decisions, gt_labels)` -> dict: GT lesions = 26-connected components of `gt_labels == 1`. Per predicted spot: real if it overlaps GT ET, else false. Per GT lesion: `caught_keep` if it overlaps a kept spot, else `caught_review` if it overlaps a review spot, else `missed`. Returns integer counts `kept_real`, `kept_false`, `review_real`, `review_false`, `lesions`, `lesions_caught_keep`, `lesions_caught_review`, `lesions_missed`, plus `silent_false = kept_false`, `silent_missed = lesions_missed`, and `et_dice_kept` (ET Dice with only kept spots as ET) and `et_dice_with_review` (kept + review as ET), each 1.0 when both prediction and truth are empty.
  - `t500_decisions(spots, total_et_voxels, voxel_mm3=1.0, min_mm3=500.0)` -> today's rule expressed as decisions: every spot `keep` if `total_et_voxels * voxel_mm3 >= min_mm3`, else every spot `"drop"` (relabelled away, i.e. neither kept nor reviewed). `score_patient` accepts `"drop"` and treats a dropped spot as absent.

- **Review flags in the pipeline (Addendum 2):**
  - `RStarConfig` gains `review_flags: bool = False`, `review_prob_cut: float = 0.7` and `review_min_voxels: int = 50`.
  - When `review_flags` is true, `segment` also computes flags from the **pre-rule** labels (the fused labels before `apply_small_et_rule`; in 3D-only mode the 3D argmax) and the enhancing-tumour probability of the same branch (`w3d * p3[1] + (1 - w3d) * p2[1]` in R* mode, `p3[1]` in 3D-only mode, clipped to [0, 1]): `find_et_spots` with no source masks, spots with fewer than `review_min_voxels` voxels discarded (neither kept nor flagged), `decide(big_spots, size_cut=0, prob_cut=review_prob_cut, agree_cut=0.0, small_cut=0)`, then `review_outputs`.
  - `RStarResult` gains `review_mask` (uint8, the input's spatial shape, 0 = not flagged, k = flagged spot k) and `review_spots` (the issue #44 list); both are `None` when flags are off. `diagnostics` gains `review_spot_count` only when flags are on.
  - CLI: `--review-mask PATH` turns flags on, writes the review mask as a uint8 NIfTI with the input affine, and adds `review_spots` to the `--json` report.

- **Fragment cleanup (Addendum 3):**
  - `RStarConfig` gains `fragment_cleanup: bool = False` and `cleanup_min_voxels: int = 200`.
  - New `fusion.remove_fragments(labels, min_voxels)` -> `(new labels, whole-tumour voxels removed, ET voxels relabelled)`, in the style of `apply_small_et_rule`: every 26-connected component of the whole-tumour mask (labels 1-4) with fewer than `min_voxels` voxels becomes 0; then every 26-connected component of the remaining enhancing tumour (label 1) with fewer than `min_voxels` voxels becomes 2. Other labels and the input are unchanged; `min_voxels = 0` returns an equal copy and zero counts.
  - When `fragment_cleanup` is true, `segment` applies it to the fused labels (R* mode) or the 3D argmax (3D-only mode) **after** review flags are computed and **before** the 500 mm³ rule; `diagnostics` gains `fragment_voxels_removed` (whole-tumour voxels set to background) and `et_fragment_voxels_relabelled` only when it is on.
  - CLI: `--fragment-cleanup` turns it on.

## 3. Out of Scope
- Registration, skull-stripping, resampling to BraTS space, DICOM I/O, NIfTI header repair.
- Any change to sections 01, 03 or 05 (`05_frontend_demo` is not touched; a later integration step is the maintainers' call).
- Training, evaluation scripts, batching several patients, multi-GPU, memory tuning for small machines.
- Uncertainty calibration or a learned confidence score. The agreement flag is a weak soft signal only; the module never withholds labels because of low agreement (one genuine 2D collapse on a correctly aligned held-out patient had agreement 0.000).
- **Any claim of clinical validity or of performance on non-US data.** Reported numbers are US BraTS-PEDs only.
- Bundling checkpoints into the repository.
- **(Addendum 3)** Turning cleanup on by default; flagging the removed pieces (tested as COMBO-F: no measurable gain); per-region or per-label sizes; any change to `05_frontend_demo` beyond what Addendum 2 already excludes.
- **(Addendum 2)** Changing the labels, status or warnings because of flags; per-spot model agreement inside the pipeline (it would need four extra 3D passes; the chosen rule does not use it, so `models_agree` is `null`); flags on by default; any change to `05_frontend_demo` or the deployed comparison package; the CLI writing flags without being asked.
- **(Addendum 1)** Wiring review flags into `RStarSegmenter`, its CLI or its outputs; any change to existing `rstar` modules, the deployed models or the Oct-6 comparison package; choosing threshold values in code (evaluation runs pass them explicitly; defaults exist only as documented keyword defaults where stated); learned/calibrated uncertainty; any evaluation script with data paths (those live outside the repository); the Streamlit display (issue #44).

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
22. Spots: on hand-built volumes, `find_et_spots` finds exactly the 26-connected ET components (two blocks touching only at a corner are ONE spot; blocks one voxel apart are two), numbers them 1..k, and reports `voxels`, `mean_et_prob`, `max_et_prob` exactly (1e-6); with two source masks covering 100% and 50% of a spot its `agreement` is 0.75 exactly; with no sources it is `None`; a volume with no ET gives an all-zero `spot_ids` and an empty list; mismatched shapes raise `ValueError` naming both shapes.
23. Decisions: for hand-built spots, `decide` returns `keep`/`review` and the reason exactly as specified, with boundaries pinned (`voxels == size_cut` keeps; `voxels == small_cut` is borderline, not small; `mean_et_prob == prob_cut` keeps; `agreement == agree_cut` keeps), the reason priority order respected when several apply, and `agreement=None` never triggering `models_disagree`. Invalid thresholds (negative size cuts, `small_cut > size_cut`, probabilities or agreement outside [0, 1], NaN) raise `ValueError` naming the value.
24. Outputs: `review_outputs` produces a uint8 mask of the input shape containing exactly the review spots renumbered 1..m in their original order (kept spots are 0), and a list whose entries have exactly the five keys, the renumbered ids, Python `int`/`float`/`str`/`None` values only (`json.dumps` succeeds), and `models_agree` equal to the spot's agreement (or `None`).
25. Scorecard: on a hand-built patient with a known mix (a kept real spot, a kept false spot, a review real spot, a review false spot, a GT lesion covered by nothing, a GT lesion covered only by a review spot), every count returned by `score_patient` is exact, `silent_false == kept_false`, `silent_missed == lesions_missed`, and both Dice values equal hand-computed values (1e-9); an ET-free patient with no spots gives Dice 1.0 for both and all counts 0.
26. Today's rule: `t500_decisions` drops every spot when total ET is below 500 mm³ (respecting `voxel_mm3`) and keeps every spot otherwise (exactly 500 keeps), and `score_patient` with those decisions gives the same `et_dice_kept` as scoring the labels produced by the existing `fusion.apply_small_et_rule` on the same volume (1e-9), for at least a below-threshold, an at-threshold and an above-threshold case.
27. Robustness: integer, boolean or float `labels`/`gt_labels` arrays of the same shape are accepted; `et_prob` outside [0, 1] or with NaN raises `ValueError`; source masks of another shape raise `ValueError`; a spot touching the volume edge is handled; everything is deterministic.
29. Config: `RStarConfig().review_flags is False`, `review_prob_cut == 0.7`, `review_min_voxels == 50`; `review_prob_cut` outside [0, 1] or NaN, a negative or non-integer `review_min_voxels`, or a non-bool `review_flags` raise `ValueError` naming the field.
30. Off by default and unchanged: with stub models on a tiny synthetic volume, `segment` with default config gives `review_mask is None`, `review_spots is None`, no `review_spot_count` key, and the same diagnostics keys as Req 14.
31. Labels never change: for the same inputs, `segment` with `review_flags=True` returns labels, mode, status, warnings and every Req 14 diagnostics value except `elapsed_s` identical to `review_flags=False`, in R* mode and in 3D-only mode, including a case where the 500 mm³ rule relabels all ET.
32. Flag content: with controllable stubs that produce (a) a large confident ET blob, (b) a large low-probability ET blob, (c) a low-probability blob below `review_min_voxels`, the result flags exactly (b): `review_spots` has one entry with `voxels` equal to (b)'s voxel count, `mean_et_prob` equal to the fused ET probability averaged over (b) (1e-6), `reason == "low_confidence"`, `models_agree is None`, and `review_mask` is 1 exactly on (b)'s voxels and 0 elsewhere; `diagnostics["review_spot_count"] == 1`.
33. Pre-rule source: when the 500 mm³ rule relabels a patient's whole ET to non-enhancing, a low-probability spot of at least `review_min_voxels` voxels is still flagged (the flag sees the ET the rule erased).
34. Thresholds respected: raising `review_prob_cut` above (a)'s mean probability flags (a) too; setting `review_min_voxels` to 1 flags (c) too; a spot exactly at `review_prob_cut` is not flagged.
35. CLI: `--review-mask out.nii.gz` with a stub segmenter writes a uint8 NIfTI equal to `review_mask` with the input affine, the `--json` report contains `review_spots` equal to the result's list, and without `--review-mask` no review file is written, the JSON has no `review_spots` key, and the segmenter was built with `review_flags=False`.
36. Real-data check (slow, skipped unless `RSTAR_MODELS_ROOT` and `RSTAR_TEST_DATA` are set): for three of the 14 demo patients, the flagged spots' count and reasons equal, and each spot's voxel count is within 1% of, those in `05_frontend_demo/comparison_cache/<patient>/review_spots.json` (written by `add_review_flags.py` with the same rule), and `mean_et_prob` within 0.01. README documents the option, the rule, its source study and that flags do not change the labels.
37. Config: `RStarConfig().fragment_cleanup is False` and `cleanup_min_voxels == 200`; a non-bool `fragment_cleanup`, or a negative, non-integer or bool `cleanup_min_voxels`, raises `ValueError` naming the field.
38. `remove_fragments` on hand-built volumes: a whole-tumour component of `min_voxels - 1` voxels becomes 0 and one of exactly `min_voxels` is kept; an ET component below `min_voxels` inside a large tumour becomes 2 while the rest of that tumour is unchanged; two blocks touching only at a corner count as one component; the input array is not modified and the output dtype equals the input dtype; `min_voxels = 0` returns an array equal to the input and zero counts; the two counts equal the voxels changed.
39. Off by default: with stub probabilities, `segment` with the default config returns labels identical to the labels computed without any cleanup and has no `fragment_voxels_removed` or `et_fragment_voxels_relabelled` key.
40. On: `segment` with `fragment_cleanup=True` returns labels equal to `apply_small_et_rule(remove_fragments(pre_rule_labels, cleanup_min_voxels)[0])`, in R* and 3D-only modes; the two diagnostics equal the voxel counts actually changed; with `review_flags=True` as well, `review_mask` and `review_spots` are identical to those with cleanup off.
41. Order: on a stub case where removing an ET fragment brings the total ET below 500 mm³, the remaining ET is relabelled by the 500 mm³ rule (cleanup runs first) and `et_relabelled` is true.
42. CLI: `--fragment-cleanup` builds the segmenter with `fragment_cleanup=True` (and an injected segmenter gets it set); without the option it is built with `fragment_cleanup=False`.
43. Real data and docs (slow part skipped unless `RSTAR_MODELS_ROOT` and `RSTAR_TEST_DATA` are set): on three fresh-30 patients, `fragment_cleanup=True` changes the per-patient mean legacy Dice (ET, TC, WT; rule applied) by no more than 0.01 against cleanup off, and never leaves a whole-tumour component smaller than `cleanup_min_voxels`. README documents the option, the size, its source tests, and that it is off by default pending cross-validation.
28. Isolation and regression: the only changed or new files are `rstar/review_flags.py`, `tests/test_review_flags.py` and `SPEC.md`; every other file in `06_rstar_inference/` is byte-identical; `rstar/pipeline.py` does not import `review_flags`; every pre-existing `06_rstar_inference` test passes unedited.

**Assumption:** The negative-voxel limits (1% warn, 5% error) were set after the regression run showed that real BraTS-PEDs scans do contain negative voxels (0.03-0.47% in the four sequences of one fresh patient); the first draft of this spec wrongly rejected any negative value.
**Assumption:** The thresholds 0.15 and 0.25 (background fraction) come from BraTS volumes being 30–60% background; a non-stripped scan has almost no exactly-zero voxels. They are judgment calls, not tuned.
**Assumption:** The agreement thresholds (0.70 review, 0.10 strong) are judgment calls. On 111 evaluation patients the agreement had median 0.84–0.90, 5th percentile 0.34–0.59 and minimum 0.000, so 0.70 flags roughly the lowest quarter and 0.10 only near-total disagreement.
**Assumption:** Three-or-fewer-sequence inputs run through the 3D family only because that is the only measured-robust path (3D dropout members were trained with sequences hidden); two-or-more absent is unvalidated, hence `'review'`.
**Assumption:** The module returns a single case per call, on one device, in float32; memory use is a full-resolution five-class probability volume per branch (about 180 MB each at 240×240×155).
**Assumption:** Voxel volume is 1 mm³ for the BraTS grid; the mm³ form of the 500-voxel rule is the same rule expressed in physical units.

**Assumption (Addendum 3):** Cleanup runs after flags are computed so the flags keep seeing everything the model predicted (the combination tested); it runs before the 500 mm³ rule so removed ET fragments no longer count toward the patient's ET total, exactly as in the test.
**Assumption (Addendum 3):** Removing a whole-tumour component removes all its labels; an ET component is relabelled non-enhancing rather than removed, because it sits inside tumour the cleanup kept.
**Assumption (Addendum 2):** Flags leave `status` alone: a flagged spot is information for the reader, and tying it to `'review'` would change the deployed output and make most patients `'review'` (about 0.5 flagged spots per patient on fresh-30).
**Assumption (Addendum 2):** `models_agree` is `null` in the pipeline: the chosen rule ignores agreement, and computing it needs one extra 3D pass per member. `add_review_flags.py` (demo package) computed it; that is why Req 36 compares everything but agreement.
**Assumption (Addendum 2):** Req 36 compares voxel counts within 1%, not exactly: the demo files were made on a GPU, and CPU and GPU arithmetic differ at a few boundary voxels (first run: 7,067 vs 7,064 voxels on 00004; the same effect as the shipped R* files differing by up to 49 voxels per patient).
**Assumption (Addendum 2):** Specks under 50 voxels are dropped from the flag list, as in the study's amended scorecard; whether the tool should show them is still an open team question.
**Assumption (Addendum 1):** "Agreement" is measured per spot against each source's own argmax-ET mask (the 4 3D members and the 2D model in evaluation runs), so it needs no ground truth and costs one extra argmax per source; the earlier whole-patient agreement (guards.py) was too coarse, catching 19-40% of bad cases, because it measured whole-tumour overlap rather than the ET spots where the errors are.
**Assumption (Addendum 1):** A dropped spot under today's rule is treated as absent (it becomes non-enhancing core, which is how the existing rule scores); review spots count as caught for lesion-level scoring but are NOT counted as ET for `et_dice_kept`. Both Dice views are reported so neither framing hides the other.

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
│   ├── fusion.py                 # fuse, small-ET rule (mm^3), agreement, remove_fragments (addendum 3)
│   ├── review_flags.py           # spot-level review flags + per-spot error scorecard (addendum 1; used by pipeline.py when review_flags is on, addendum 2)
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
    ├── test_review_flags.py      # Req 22-28 (addendum 1)
    ├── test_pipeline_review.py   # Req 29-36 (addendum 2)
    ├── test_fragment_cleanup.py  # Req 37-43 (addendum 3)
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
| Two ET blobs touching only at a corner (Addendum 1) | One spot (26-connectivity), matching how GT lesions are counted |
| No ET predicted at all (Addendum 1) | No spots, empty review outputs; scorecard counts GT lesions as missed |
| Agreement unavailable (no source masks) (Addendum 1) | `agreement = None`; never triggers `models_disagree` |
| A spot exactly at a threshold (Addendum 1) | Kept (all cuts are strict `<` for review) |
| Flags on, no ET predicted (Addendum 2) | `review_mask` all zeros, `review_spots == []`, `review_spot_count == 0` |
| Flags on, 3D-only mode (Addendum 2) | Flags computed from the 3D argmax and `p3[1]` |
| Flags on, the 500 mm³ rule erased all ET (Addendum 2) | Erased spots can still be flagged; labels unchanged |
| `--review-mask` given without `--json` (Addendum 2) | Mask written; no report; exit 0 |
| Cleanup on, no tumour predicted (Addendum 3) | Labels all 0; both cleanup diagnostics 0 |
| Cleanup on, the whole prediction is one small component (Addendum 3) | Removed: labels all 0 (the patient gets no tumour) |
| Cleanup removes an ET fragment and the remaining ET falls below 500 mm³ (Addendum 3) | The rule then relabels the remaining ET; `et_relabelled` true |
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
- [ ] Req 22: spot extraction (26-connectivity), features and agreement exact; empty volume; shape mismatch raises
- [ ] Req 23: keep/review and reasons exact with pinned boundaries and priority; `None` agreement never disagrees; invalid thresholds raise
- [ ] Req 24: review mask and spot list in the issue #44 format, renumbered, JSON-serialisable
- [ ] Req 25: per-spot and per-lesion counts, silent errors and both Dice values exact on a hand-built patient; empty patient
- [ ] Req 26: today's rule as decisions matches `fusion.apply_small_et_rule` scoring below, at and above threshold
- [ ] Req 27: dtype tolerance, invalid probabilities and mismatched masks rejected, edge spots, determinism
- [ ] Req 28: only the new module, its tests and SPEC.md change; pipeline does not import it; existing tests pass unedited
- [x] Req 29: review-flag config defaults and validation
- [x] Req 30: flags off by default; no review fields
- [x] Req 31: labels, mode, status, warnings, diagnostics identical with flags on and off (R*, 3D-only, rule firing)
- [x] Req 32: exactly the low-confidence large spot flagged, values exact, mask exact, count in diagnostics
- [x] Req 33: spots erased by the 500 mm³ rule can still be flagged
- [x] Req 34: thresholds respected, boundary not flagged
- [x] Req 35: CLI --review-mask writes the mask and JSON spots; nothing extra without it
- [x] Req 36: real-data flags match the demo package (slow); README documents the option
- [x] Req 37: fragment-cleanup config defaults and validation
- [x] Req 38: remove_fragments exact on hand-built volumes (boundary, ET inside tumour, corner, no mutation, dtype, 0 = identity)
- [x] Req 39: off by default; labels unchanged; no cleanup diagnostics
- [x] Req 40: on = rule(remove_fragments(pre)) in both modes; diagnostics exact; flags unchanged by cleanup
- [x] Req 41: cleanup runs before the 500 mm³ rule
- [x] Req 42: CLI --fragment-cleanup
- [x] Req 43: real-data Dice change <= 0.01 and no small components left (slow); README documents the option
