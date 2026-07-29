# Handoff — pediatric brain tumour segmentation, section 03 + demo

Paste this whole file as the first message of a new chat.

**Working directory (open a chat here — do not paste files, just point at these paths):**
```
D:/Medical AI Workshop/.claude/worktrees/streamlit-cnn-workshop-dcaf1e
```
Sub-directories referenced throughout:
```
D:/Medical AI Workshop/.claude/worktrees/streamlit-cnn-workshop-dcaf1e/03_augmentation_eval          # pipeline, training, evaluation
D:/Medical AI Workshop/.claude/worktrees/streamlit-cnn-workshop-dcaf1e/03_augmentation_eval/src      # train.py, model.py, plan.py, evaluate.py, tumor_type.py
D:/Medical AI Workshop/.claude/worktrees/streamlit-cnn-workshop-dcaf1e/03_augmentation_eval/tools    # history_from_log.py, make_report.py
D:/Medical AI Workshop/.claude/worktrees/streamlit-cnn-workshop-dcaf1e/03_augmentation_eval/checkpoints/overnight_run   # best.pt / last.pt / history.json
D:/Medical AI Workshop/.claude/worktrees/streamlit-cnn-workshop-dcaf1e/05_frontend_demo              # Streamlit demo
D:/Medical AI Workshop/.claude/worktrees/streamlit-cnn-workshop-dcaf1e/05_frontend_demo/utils        # inference.py, metrics.py, loaders.py, build_metrics_cache.py
D:/Medical AI Workshop/.claude/worktrees/streamlit-cnn-workshop-dcaf1e/05_frontend_demo/pages        # Streamlit pages
D:/Medical AI Workshop/.claude/worktrees/streamlit-cnn-workshop-dcaf1e/05_frontend_demo/data         # roc_cache.json (the measured numbers)
D:/Medical AI Workshop/.claude/worktrees/streamlit-cnn-workshop-dcaf1e/00_shared/manifests           # hospitalA.json, hospitalB.json, heldout.json
```
Data (outside the repo):
```
D:/pack_out            # packed 8000-budget cache used for all training so far (5.7 GB)
D:/pack_out/index.json # tumour/ET presence per subject+plane
E:/Processed_2D        # FULL slice cache, 227 subjects, 97,867 slices (external drive)
```

**Repo:** `Sanad-Manga/Pediatric-Brain-Tumor-Model`
**Branch:** `claude/medical-ai-augmentation-eval-7f4098` (1 commit ahead of `main`; earlier work already merged via PR #11)
**Latest commit:** see `git log -1` on the branch

---

## 0. State in one paragraph

A 2D U-Net segments BraTS-PEDs brain tumours. Training ran to epoch 36 on a
Colab T4 before the GPU quota ran out. The shipped model is **epoch 16**, and
every number in the report and the Streamlit demo is measured on 82 held-out
patients that were never trained on. Config has just been changed for the next
run — bigger model, more data, three training fixes — but **that run has not
happened yet**. Nothing has been trained at the new settings.

---

## 1. Current measured results (epoch 16, 82 held-out patients, 492 axial slices)

| Region | Dice = F1 | Sensitivity | Precision | Specificity | HD95 | AUC |
|---|---|---|---|---|---|---|
| ET (enhancing) | 0.599 | 0.496 | 0.757 | 0.9998 | 4.12 mm | 0.982 |
| TC (core) | 0.521 | 0.411 | 0.713 | 0.9997 | 4.82 mm | 0.981 |
| WT (whole) | 0.827 | 0.748 | 0.924 | 0.9992 | 2.24 mm | 0.993 |

Training-time validation best: mean Dice **0.6820** at epoch 16. Tumour-type
auxiliary head peaked at **0.806** accuracy (majority-class baseline 0.634).

**Two things must be stated whenever these are quoted:**

1. **AUC and specificity are inflated.** Tumour is 2–18% of pixels; background
   is trivially easy. Dice and HD95 are the honest figures. Never quote AUC alone.
2. **The tumour-type head is not histology.** No tumour-type ground truth exists
   in this dataset. It is trained against a geometric proxy from the mask
   (midline offset, inferior fraction, enhancing fraction).

**The model under-segments** — high precision, lower sensitivity. It misses
tumour rather than over-calling it. That is the clinically relevant finding.

---

## 2. The result that matters most

Training nominated **epoch 25** as best (val mean Dice 0.6885 > epoch 16's
0.6820). On the held-out patients epoch 25 is **much worse**:

| | epoch 16 | epoch 25 | Δ |
|---|---|---|---|
| ET Dice | 0.599 | 0.434 | **−0.165** |
| TC Dice | 0.521 | 0.375 | **−0.147** |
| WT Dice | 0.827 | 0.849 | +0.023 |
| ET HD95 | 4.12 mm | 9.00 mm | **2.2× worse** |
| Mean Dice | **0.649** | 0.553 | −0.096 |

Cause: checkpoints were selected on the **mean** of three regions, computed on
36 patients drawn from the *training* hospitals. One strong region (WT) masked a
collapse in ET. Fixed — see §3. Epoch 25 is kept at
`checkpoints/overnight_run/best_epoch25.pt` if it is ever needed.

---

## 3. Changes made but NOT yet trained with

All in `03_augmentation_eval/config.yaml`. Leaving any key out reproduces the
old behaviour exactly.

| Key | Value | Why |
|---|---|---|
| `model.width` | **48** (was 16) | 4,333,445 params vs 482,949. The old model was ~1/60th of an nnU-Net; capacity, not data, was the binding constraint. |
| `expansion.target_per_hospital` | **15000** (was 8000) | 24,140 unique slices vs 16,000, at 20% up-sampling. |
| `selection.metric` | **min_region** | Scores the *worst* region. On the real numbers `mean` picks epoch 25, `min_region` picks epoch 16 — it would have caught the mistake above. `best_mean.pt` is still written. |
| `loss.class_weights` | `[0.2, 3.0, 1.0, 1.0, 1.0]` | Uniform CE let the network trade away ET, the smallest and weakest region. Weights the CE term only. |
| `schedule.kind` | **cosine**, min_lr 1e-5 | Val Dice oscillated late at fixed 1e-3 (0.688 → 0.637 → 0.687). Fast-forwards correctly on resume. |
| `eval.plane` | **both** | Trained on axial + coronal but only ever scored on axial. Averages softmax in the shared voxel grid before argmax. |

**Do not raise `target_per_hospital` above 15000.** Measured up-sampling
(duplicated/re-augmented slices, not new data): 15000 → 20%, 20000 → 32%,
26000 → 41%. Beyond 15000 most of the extra budget is re-augmented copies while
costing proportional GPU time.

**Checkpoint geometry:** checkpoints now record `width`/`depth`, and both
loaders recover it from the state_dict when absent. Verified that the old
width-16 checkpoint still loads under the width-48 config. Do not remove this —
without it every pre-existing checkpoint fails with a shape mismatch that reads
like corruption.

---

## 4. Next run

```bash
cd 03_augmentation_eval
python run.py plan --report          # confirm augment column ~20%, ET floor 25%, empty ceiling 30%
python run.py pack --out <dir>       # ~8.5 GB at 15000 (0.20 MB/slice, float16)
# upload to Drive, then on a Colab T4:
python run.py train --device cuda --run-id v2_w48 --epochs 40 \
    --cache "/content/drive/MyDrive/<...>/pack_out"
```

- **`--device cuda` is required.** The default is `cpu` and it will silently use it.
- ~290 s/epoch at width 16 on a T4; expect longer at width 48 with 1.9× the data.
- Checkpoints and `history.json` are written **every epoch** (atomically) to
  `checkpoints/<run_id>/`, and resume automatically. Symlink that directory to
  Drive before starting or a disconnect loses the run.
- **Never copy checkpoints from inside `pack_out`** — that folder holds stale
  originals. Doing so once overwrote a good checkpoint with a 2-epoch one.

After training:
```bash
cd 05_frontend_demo
python -m utils.build_metrics_cache        # ~80 s, rewrites data/roc_cache.json
```
Every page and the report read that cache. Nothing else needs editing.

---

## 5. Repo map

**`03_augmentation_eval/`** — data pipeline, training, evaluation
- `run.py` — single entry point: `test | index | tumor-type | plan | pack | train | preview | eval`
- `src/train.py` — training loop, per-epoch checkpointing, resume, selection
- `src/model.py` — `UNet`, `TumorTypeHead`, `build_model`, `infer_geometry`
- `src/plan.py` — balanced per-hospital slice plan (ET floor, empty ceiling, per-patient cap)
- `src/evaluate.py` — writes `results/ablation_results.csv`
- `src/tumor_type.py` — DMG-like vs astrocytoma-like geometric proxy
- `tools/history_from_log.py` — rebuilds `history.json` from console output
- `checkpoints/overnight_run/` — `best.pt` (epoch 16, shipped), `last.pt` (epoch 36), `best_epoch25.pt`, `history.json` (34 epochs)

**`05_frontend_demo/`** — Streamlit demo
- `utils/inference.py` — checkpoint loading, `predict_slice`, per-region Dice
- `utils/metrics.py` — ROC/AUC, sensitivity/specificity/precision, HD95
- `utils/build_metrics_cache.py` — regenerates `data/roc_cache.json`
- `utils/loaders.py` — **all pages read numbers through here**
- `pages/Model_Performance.py`, `pages/Segmentation_Report.py` — converted to real data
- `HANDOFF_METRICS.md` — brief for whoever works on the UI

**Data:** slice cache `<subject>/<plane>/slice_NNN.npz`, keys `image` (4,H,W)
float32 already z-scored, `mask` (1,H,W) uint8 labels 0–4. 227 subjects,
97,867 slices. Manifests: hospitalA 53, hospitalB 92, heldout 82.

---

## 6. Known-bad / unfinished

- **Ablation table is empty.** `results/ablation_results.csv` holds one stale
  row. A real ablation needs a **separate training run per condition** —
  `run.py eval` only copies the config flags into the CSV, so running it 4× on
  one checkpoint would fabricate the table. A 3-condition CPU run
  (`--epochs 3 --max-steps 250`, same seed) was started and had not finished.
- **Streamlit theme was never visually reviewed.** Dark→light was done by bulk
  colour-token substitution across 9 pages and verified only programmatically
  (pages run clean; `.stApp` computes to white). Expect low-contrast text and
  gradients designed for black grounds. The user reported the ROC/AUC page
  looked wrong; the *report* artifact's ROC was verified correct, so the problem
  is in the Streamlit page.
- **Stale claims still on some pages:** `Dashboard.py` ticks ✓ for *Federated
  Learning across 5 hospital sites*, *3D U-Net*, *CORAL*, *Grad-CAM XAI*,
  *FP16* — none implemented. `Clinical_View.py`, `MRI_Analysis.py`,
  `Domain_Adaptation.py` unaudited.
- **Federated training never ran** (`use_federation: false`); the model is
  trained centrally. CORAL exists only on `origin/codex/domain-adaptation`,
  built for 3D.
- **Epochs 0–2 missing** from `history.json` — an early logging fault, since
  fixed. Affects plots only.
- **`config.yaml`'s `cache_2d` points at a Colab Drive path.** Override with
  `--cache` locally.

---

## 7. Where to improve, in order

1. **Train at the new settings** (width 48 + 15000). Not yet done. Biggest expected gain.
2. **Evaluate with `eval.plane: both`** — already configured, never measured.
3. **Test-time augmentation** (flip, average) — ~20 lines, reliable small gain.
4. **Measure per-hospital scores before attempting domain adaptation.** If
   hospital A and B are within noise, CORAL has nothing to correct, and porting
   it from 3D is days of work. ~10 minutes to check.
5. **Finish the ablation** — it is section 03's actual deliverable.

Do **not** just train more epochs at the old settings: epochs 26–36 produced no
improvement and epoch 25 was worse than 16 where it mattered.

---

## 8. Report for the clinician

Standalone HTML, every value and chart coordinate generated from
`roc_cache.json` / `history.json` rather than transcribed:
<https://claude.ai/code/artifact/c6a087fb-8387-4cdf-8b6b-d6d99cee4492>

Contains: per-region metrics, ROC curves, training curves, the epoch-16 vs
epoch-25 selection finding, methods, limitations, and four specific questions
for clinical feedback. Regenerate with the generator script after re-measuring.
