# Spec: Review-flag prototype (Section 07: review_prototype)

## 1. Goal
A local, stand-alone Streamlit prototype that lets the bio-tech doctor compare, patient by patient and slice by slice, today's R* enhancing-tumour output with the same output under spot-level review flags (team decision Q2 = D), so she can judge the flag behaviour before it is built into the real app.

## 2. In Scope
- `precompute.py` (run once, locally, needs the saved probability maps and raw NIfTI on Ahmed's machine): for the 14 patients of `05_frontend_demo/comparison_cache/` (all held-out), writes one folder per patient to a cache directory outside the repository:
  - `t1c.npz`, `t2f.npz`: key `img`, uint8 (240, 240, 155), each sequence windowed to its 0.5-99.5 percentile of brain voxels; background 0.
  - `gt.npz`: expert labels (uint8 0-4).
  - `today.npz`: R* labels exactly as deployed: `rstar.fusion.fuse(p2, p3, 0.5, 0.5)` then `rstar.fusion.apply_small_et_rule(..., 500.0, 1.0)`.
  - `flagged.npz`: R* labels from the same `fuse` WITHOUT the small-ET rule (review spots stay visible as ET).
  - `review_mask.npz` / `review_spots.json`: from `rstar.review_flags` (format of issue #44) with the rule chosen in the pre-registered evaluation: a spot (>= 50 voxels) is `review` when its mean fused ET probability < 0.7, else `keep`.
  - `specks.npz` / `specks.json`: ET spots < 50 voxels (ignored by the evaluation; their treatment is an open team question), same format as the review files.
  - `meta.json`: rule parameters, counts (kept / review / specks), and per review spot and speck whether it touches expert ET (`real`), plus its axial slice range.
- `app.py` (Streamlit, reads only the cache directory; no GPU, checkpoints, torch or rstar import): patient selector; axial slice slider; background toggle T1c / FLAIR; two panels side by side, "Today (500 mm³ rule)" and "With review flags"; label colours consistent with the main app (enhancing / non-enhancing core / cyst / edema); review spots drawn with a distinct dashed-looking outline colour in the right panel; toggles for the expert outline, for specks, and for revealing per-spot "real / false" from the expert labels; a table of the patient's review spots (size, confidence, agreement, reason, slices) with a button per spot that jumps the slider to that spot's centre slice; per-patient counts; a visible research-prototype disclaimer.
- A short `README.md` (how to precompute and run).

## 3. Out of Scope
- Any change to `05_frontend_demo` (Tuesday's app and its comparison page), `06_rstar_inference` code, the deployed models, checkpoints, or the comparison package.
- Hosting/deployment (Streamlit Cloud); feedback capture or persistence; more patients than the 14; 3D rendering; non-axial views; changing the flag rule from the UI; any clinical claim.

## 4. Requirements
1. `precompute.py` produces, for each of the 14 patients, every file listed in §2 with the stated keys, dtypes and shape (240, 240, 155).
2. `today.npz` equals, voxel for voxel, `apply_small_et_rule(fuse(p2, p3, 0.5, 0.5), 500.0, 1.0)` computed independently in the check script; `flagged.npz` equals `fuse(p2, p3, 0.5, 0.5)`; the two differ only where the small-ET rule relabelled ET to 2.
3. Every review spot has >= 50 voxels and mean fused ET probability < 0.7; every kept ET spot (>= 50 voxels) has mean probability >= 0.7; every speck has < 50 voxels; review, kept and speck spots together are exactly the ET voxels of `flagged.npz`; ids in each mask match its JSON list (1..m).
4. `meta.json` counts equal the mask/JSON contents; `real` equals "touches expert ET (label 1)" recomputed from `gt.npz`; slice ranges equal the mask's z-extent.
5. `streamlit run 07_review_prototype/app.py` starts with no GPU, and in a browser every one of the 14 patients renders both panels at the initial slice with no exception text on the page.
6. Toggling T1c/FLAIR, expert outline, specks and reveal changes the image or table (each verified once in a browser); a jump button sets the slider to that spot's centre slice and the right panel shows the spot's outline on that slice.
7. The disclaimer "Research prototype - not for clinical use" is visible at the top of the page; the app does not import torch or rstar (checked by a test).
8. Isolation: only files under `07_review_prototype/` are added; nothing else in the repository changes.

**Assumption:** The 14 comparison patients are reused so the doctor sees the same cases as on Tuesday's page; the cache lives outside the repo (it contains derived MRI images).
**Assumption:** Review spots are drawn as an outline on top of the normal ET colour in the right panel, so the doctor sees both "the model thinks ET here" and "but it is unsure".

## 5. Structure
```
07_review_prototype/
├── SPEC.md
├── README.md
├── precompute.py      # saved probabilities + raw NIfTI -> local cache (one folder per patient)
├── app.py             # Streamlit prototype; reads the cache only
└── test_prototype.py  # Req 2-4 checks on the real cache; Req 7 import check
```

## 6. Edge Cases
| Scenario | Expected Behaviour |
|---|---|
| Patient with no review spots | Right panel equals the left except for rule-relabelled ET; table says "No spots flagged for review" |
| Patient with no ET at all | Both panels show no ET; counts are 0 |
| Cache directory missing | The page shows how to run `precompute.py` instead of a traceback |
| Spot spanning many slices | Jump goes to the centre slice of its z-extent |

## 7. Done Checklist
- [ ] Req 1: all files per patient with correct keys/dtypes/shape
- [ ] Req 2: `today` and `flagged` match the deployed fusion and rule exactly
- [ ] Req 3: review/kept/speck partition and thresholds hold; ids match JSON
- [ ] Req 4: meta counts, `real` flags and slice ranges correct
- [ ] Req 5: app starts and all 14 patients render without errors
- [ ] Req 6: toggles and jump-to-slice work
- [ ] Req 7: disclaimer visible; app imports neither torch nor rstar
- [ ] Req 8: only `07_review_prototype/` added
