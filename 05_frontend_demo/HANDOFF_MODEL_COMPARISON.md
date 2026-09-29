# Handoff: model comparison page for the Streamlit app

**Deadline: Tuesday 6 October 2026** (a meeting where these three models get shown side by side on
real test scans). **This task does not need a GPU, a checkpoint, or the training data.** Everything
you need is either in this repo or in the small package described below.

## The task in one sentence

Add a new page to the existing Streamlit app that lets someone pick a patient and see three
segmentations of the same scan — the shipped 2D model, the 3D model, and "R\*" (the combination of
both) — next to each other, with the accuracy numbers for each.

## What already exists — read this before writing any code

- **The app.** `05_frontend_demo/` is a working Streamlit app (`Home.py` + `pages/`). It already has
  a 2D-only inference page (`pages/MRI_Analysis.py`), a metrics dashboard, and a demo patient cache.
  Match its existing look (`components/theme.py`) rather than inventing a new one.
- **The three models are already built, tested and validated.** Nobody needs to train or debug a
  model for this task — see `06_rstar_inference/README.md` for what R\* actually is. The only new
  code needed is a page that *displays* results that are already computed (see below).
- **Why this isn't live inference.** The deployed app (`neuropeds-ai.streamlit.app`) runs on
  Streamlit Community Cloud, which is CPU-only, and the 3D model has never been wired into it. Rather
  than risk building and hardening GPU inference into that hosting in a few days, everything is
  **precomputed once, offline, and saved**. The page you build just reads those saved files. If this
  changes (e.g. you get GPU-backed hosting), that's a separate task — build the display page first.

## The data package: `comparison_cache/`

Generated for **14 patients** (`precompute_comparison.py` in this folder produced it): the 4 original
demo patients that pass the input check, plus 10 picked to show the real spread — a near-total
failure (00188), ET-miss cases (00191, 00108, 00092), a large badly-segmented tumor (00156), a
moderate struggle (00004), clean wins (00190, 00257, 00009) and a correctly-empty-ET case (00212).
The reason each was picked is a comment next to its ID in the script's `PATIENTS` list. Per patient:

```
comparison_cache/<patient_id>/
  labels_2d.npz      # np.load(...)["labels"]: uint8 (240, 240, 155), values 0-4
  labels_3d.npz      # same shape/meaning, 3D model alone
  labels_rstar.npz   # same shape/meaning, R* (2D+3D combined)
  meta.json          # see below
```

`meta.json` shape (real example, patient `BraTS-PED-00021-000`):

```json
{
  "patient_id": "BraTS-PED-00021-000",
  "rstar_mode": "R*",
  "rstar_status": "ok",
  "rstar_warnings": ["sequence t1c has only 19% exactly-zero voxels; check that the scan is skull-stripped"],
  "rstar_agreement": 0.9015639570511284,
  "regions": {
    "2d":    { "ET": {"dice": 0.93, "precision": 0.91, "recall": 0.95}, "NC": {...}, "WT": {...} },
    "3d":    { "ET": {...}, "NC": {...}, "WT": {...} },
    "rstar": { "ET": {...}, "NC": {...}, "WT": {...} }
  }
}
```

Label values: `0` background, `1` enhancing tumor (ET), `2` non-enhancing core, `3` cystic
component, `4` edema. `NC` in the metrics = labels 1+2+3 (tumor core); `WT` = labels 1+2+3+4 (whole
tumor) — same convention the rest of the app already uses.

**No raw scans ship in this package** (only label volumes and numbers), but the demo cache already
in the repo (`05_frontend_demo/demo_cache/<patient_id>/axial/slice_*.npz`) has the actual MRI slices
to render as the background image under each segmentation overlay — reuse it, don't re-fetch data.

> **Gap to decide on (yours):** `demo_cache/` only has MRI slices for the original demo patients
> (00021, 00051, 00093, 00230 among the 14), and only for selected slice indices. The 10 newer
> patients have labels + metrics but **no background image anywhere in the repo**. Measured for
> reference: all 4 sequences of one patient as uint8 (0.5–99.5 percentile windowed) are ~16 MB
> compressed, FLAIR + T1c alone ~8 MB, the ground-truth mask ~0.04 MB. How to handle it (ship a
> display image in the package, show only the demo-cache patients with images, labels-only view…) is
> a UI decision — ask Ahmed for a precompute change if you need one.

## ⚠️ Known issue: patient selection isn't "any patient"

While generating the package, **3 of the original 7 demo patients failed the input safety check** —
their scans don't look properly skull-stripped in one sequence (a real check in `06_rstar_inference`,
not a bug: one had only 7% background voxels in one sequence where BraTS scans are normally
30–60%). Checking a wider sample: **15 of 81 held-out patients (18.5%) trip this same check** on at
least one sequence — this is real heterogeneity in the source dataset, not a fluke of the demo set.

**What this means for you:** when picking which patients go in the comparison page, only use ones
that don't hit this — the precompute script already skips and logs them (`comparison_cache/skipped.json`
lists which and why). Don't build the UI assuming every patient ID "just works"; some legitimately
get rejected before any model runs, and that's a feature of the system worth being able to show, not
a bug to hide. If you want more patients than are already precomputed, ask for the list of the
remaining ~66 held-out patients that pass, or run `precompute_comparison.py` yourself with a longer
`PATIENTS` list (needs the checkpoints and raw data, which live locally — ask first).

## What to build

1. **New page**, e.g. `pages/Model_Comparison.py`, added to the app's existing navigation.
2. **Patient selector** — dropdown or similar, listing the patients that have a `comparison_cache/`
   folder (i.e., actually succeeded).
3. **Three-way view** — for the selected patient, show the 2D, 3D and R\* segmentations. A slice
   slider (axial, matching `demo_cache`'s available slice indices) with the three label volumes
   overlaid on the same underlying MRI slice, side by side or with a toggle, is the simplest version
   that tells the story. An overlay on/off control is worth having.
4. **Metrics table** underneath, pulled straight from `meta.json`: Dice per region per model, so the
   numbers back up what's visible.
5. **R\* status/warnings** — `meta.json`'s `rstar_status` and `rstar_warnings` should be visible
   somewhere (e.g. a small badge) — this is a real guardrail firing, not decoration; hiding it would
   misrepresent what the system does.

## Definition of done

- Runs locally (`streamlit run Home.py`) with no GPU, no checkpoint files, no `06_rstar_inference`
  import at all — the page only reads `comparison_cache/` and `demo_cache/`.
- Every patient in `comparison_cache/` is selectable and renders all three models correctly.
- Metrics shown match the numbers in that patient's `meta.json` exactly (spot-check by hand).
- Matches the existing app's visual style.

## Who to ask

Ping Ahmed if: you need more precomputed patients, the `comparison_cache/` format needs to change to
fit the UI better (reasonable — it's new, adjust it), or the skull-strip finding above raises
questions about the underlying model rather than this page.
