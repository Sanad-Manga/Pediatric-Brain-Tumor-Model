---
title: NeuroPeds AI
emoji: 🧠
colorFrom: blue
colorTo: blue
sdk: streamlit
sdk_version: 1.45.0
app_file: 05_frontend_demo/Home.py
pinned: false
---

# NeuroPeds AI — Pediatric Brain Tumor Segmentation

A 2D U-Net segmentation system for pediatric brain tumors, trained on the BraTS-PEDs 2024 dataset. Built as part of a medical AI workshop.

**Live demo:** https://neuropeds-ai.streamlit.app

## What it does

- Segments pediatric brain MRI scans across 4 tumor sub-regions: Enhancing Tumor (ET), Non-Enhancing Tumor Core (NETC), Cystic Component (CC), and Peritumoral Edema (ED)
- Runs inference on real held-out patients using the shipped **ensemble** checkpoint (held-out mean Dice 0.754)
- Displays per-region Dice scores, ROC curves, and training history
- Generates clinical PDF reports

## Model

The shipped `best.pt` is an **ensemble of two 2D U-Nets** — the original model (width 16, depth 3) and a wider continued-training model (width 64, depth 3) — whose softmax probabilities are averaged. It scores higher than either member on every region of the headline metric below — but read the ET caveat in the per-patient section before leaning on that.

Held-out evaluation: 82 patients never used for training, per-patient Dice, axial + coronal predictions averaged (`--eval-plane both`).

| Region | Dice |
|--------|------|
| ET (enhancing tumor) | 0.581 |
| NC (labels ET+NETC+CC) | 0.828 |
| WT (whole tumor) | 0.854 |
| **Mean** | **0.754** |

For reference: original model alone 0.710, wide model alone 0.727. The ensemble was chosen by looking at this same held-out set, so treat 0.754 as slightly optimistic; there is no separate untouched test set.

### Per-patient spread (how consistent is it?)

The mean hides a lot. Per-patient ensemble Dice on the same 82 held-out patients (`docs/per_patient_scores.csv` has every patient and all three models):

| Region | Patients scored | Mean | Median | 10th percentile | Worst | Patients < 0.5 |
|--------|-----------------|------|--------|-----------------|-------|----------------|
| ET | 48 (only those with ET) | 0.492 | 0.615 | 0.000 | 0.000 | 21 |
| NC | 82 | 0.828 | 0.903 | 0.488 | 0.001 | 9 |
| WT | 82 | 0.854 | 0.925 | 0.663 | 0.001 | 5 |

Per-patient mean of the three regions: median 0.776, 10th percentile 0.545.

- **The typical patient does better than the mean; a tail of failures drags it down.** Median WT is 0.925 against a mean of 0.854, and five patients fall below 0.5. Patient 188 is a near-total failure on every region.
- **ET is the weakest region, and its headline number overstates it.** 34 of the 82 held-out patients have no enhancing tumor at all. An empty region scores 1.0 if the model predicts none and 0.0 if it predicts any (the BraTS convention), so the 0.581 headline is partly a reward for *not* hallucinating ET: 24 of those 34 empty patients scored 1.0 and contribute 0.293 of the 0.581. On the 48 patients that actually have ET the ensemble averages **0.492**, and it completely misses ET (Dice 0.000) in 10 of them.
- **The ensemble's ET win over its members is that same effect, not better ET segmentation.** On patients with ET the wide model alone is slightly better (0.508 vs 0.492; the original is 0.462). The ensemble's gain comes from predicting ET on fewer ET-free patients (24/34 correct vs 18/34 for the wide model). Its NC and WT gains are real: better than the original on 58/82 (NC) and 61/82 (WT) patients, worse on 11 and 9.

The live demo predicts **axial slices only**, and the Dashboard's ROC/metrics table (`roc_cache.json`) is a different, cheaper measure — axial only, 6 sampled slices per patient, pixels pooled — so its numbers will not match the table above. ET is the weakest region; the model tends to under-segment (precision exceeds sensitivity).

### Pixel-level metrics (Dashboard measure)

Same ensemble, same 82 held-out patients, axial only, 492 sampled slices, pixels pooled (TC = ET + cystic component, labels 1+3):

| Region | Dice / F1 | Precision | Recall (sensitivity) | Specificity | ROC AUC | Median HD95 |
|--------|-----------|-----------|----------------------|-------------|---------|-------------|
| ET | 0.629 | 0.776 | 0.529 | 0.9998 | 0.941 | 4.2 mm |
| TC | 0.552 | 0.728 | 0.445 | 0.9997 | 0.943 | 4.5 mm |
| WT | 0.842 | 0.924 | 0.773 | 0.9992 | 0.992 | 2.0 mm |

- **F1 is the same number as Dice** for pixel-level segmentation (both are 2·TP / (2·TP + FP + FN)), so they are shown together rather than as two separate metrics. Precision is well above recall in every region, which is the under-segmentation noted above.
- **ROC AUC and specificity flatter this model.** Only 2–18 % of pixels are tumor, so an all-background prediction already scores ~0.98 specificity. Use Dice/F1 and recall as the headline numbers. The interactive ROC curves are on the app's Dashboard page.

Architecture: 2D U-Net with optional MixUp augmentation and tumor-type classification head. Trained on axial slices from 4 MRI modalities (T1c, T1n, T2f, T2w). See `HANDOFF.md` for training history, lessons learned and what to do next.

## Project status

**Development stopped here by decision, not because the model converged.** The remaining expected gain (roughly +0.01 to +0.02 held-out mean, an estimate that was never measured) was judged marginal against 10–17 hours of GPU time per attempt on one RTX 3060, so this checkpoint is the final one.

What that means, honestly:

- **The plateau across the last training rounds was mostly bugs and measurement, not a hardware ceiling.** The cosine LR schedule was re-stretched on every resumed session, so the wide model never annealed (epochs 13–20 sit at a flat ~1.1e-4 and bounce between 0.66 and 0.70 validation mean). Axial-only evaluation led to one wrong promotion. A Windows sleep event silently wasted a 13-hour run. All three are fixed and documented in `HANDOFF.md`.
- **Real resource limits did constrain the experiments:** a single 12 GB-class GPU at 20–35 minutes per epoch on the wide model, the width-96 model wedging in CUDA allocator OOM on Windows, and the depth-4 model being numerically unstable early on. They limited which architectures could be tried, but they are not why the numbers stopped improving.
- **The one experiment that would settle whether more training helps was never run:** continue the wide model to a fixed 50-epoch cosine horizon with the fixed schedule, then re-score and re-ensemble. The command is in `HANDOFF.md`, and `03_augmentation_eval/overnight/` has the keep-awake and timeout tooling for running it unattended.
- **Known weaknesses that more training may not fix:** ET (missed entirely in 10 of the 48 patients that have it), a tail of failed patients, and no untouched test set — the ensemble was chosen on the same 82 held-out patients it is reported on. Treat every number here as slightly optimistic.

## Repo structure

| Folder | Contents |
|--------|----------|
| `03_augmentation_eval/` | Training pipeline, evaluation scripts, ablation framework |
| `05_frontend_demo/` | Streamlit app + demo cache (7 representative patients) |
| `00_shared/` | Shared contracts and data specifications |

## Running locally

```bash
cd 05_frontend_demo
pip install -r requirements.txt
streamlit run Home.py
```

The app ships with a demo cache of 7 held-out patients covering all 4 tumor label types. Set `NEUROFED_CACHE_2D` to point at a full 2D slice cache for the complete dataset.
