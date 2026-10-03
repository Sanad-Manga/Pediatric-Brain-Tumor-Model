# Review-flag prototype (local)

A stand-alone prototype for the bio-tech doctor: today's R* enhancing-tumour output next to the same output with
**spot-level review flags** (team decision Q2). Separate from the Streamlit app in `05_frontend_demo`; nothing here
changes it. Research prototype, not for clinical use.

## Run
1. Build the local cache once (needs the saved probability maps and raw NIfTI on Ahmed's machine; uses the
   neuropeds environment because it needs torch):
   `C:\Users\ahmed\neuropeds_env\Scripts\python.exe 07_review_prototype/precompute.py --out "D:/NeuroPeds AI/review_prototype"`
2. Start the app (no GPU, no checkpoints):
   `python -m streamlit run 07_review_prototype/app.py`
   (other cache location: set `NEUROPEDS_PROTOTYPE_CACHE`).
3. Checks: `python -m pytest 07_review_prototype/test_prototype.py`

## What the doctor sees
- Left: today's output (all enhancing tumour removed when a patient's total is under 500 mm³).
- Right: the same output with each enhancing spot judged on its own; spots the model is under 70% sure of are
  outlined in cyan "for review" instead of being removed.
- Toggles: T1c / FLAIR background, the expert's enhancing tumour (yellow), specks under 50 voxels (white),
  and "reveal" whether each flagged spot is real according to the expert labels.
- Cases worth opening first: 00188, 00191, 00108, 00230 (today shows no enhancing tumour; the flags show the
  spots the rule removed).
