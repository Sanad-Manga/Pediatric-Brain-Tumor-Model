# Brief: keeping the Streamlit demo's numbers honest

**Working directory (point at these paths; nothing needs pasting):**
```
D:/Medical AI Workshop/.claude/worktrees/streamlit-cnn-workshop-dcaf1e/05_frontend_demo              # run the app from here
D:/Medical AI Workshop/.claude/worktrees/streamlit-cnn-workshop-dcaf1e/05_frontend_demo/utils        # inference.py, metrics.py, loaders.py, build_metrics_cache.py
D:/Medical AI Workshop/.claude/worktrees/streamlit-cnn-workshop-dcaf1e/05_frontend_demo/pages        # the 9 Streamlit pages
D:/Medical AI Workshop/.claude/worktrees/streamlit-cnn-workshop-dcaf1e/05_frontend_demo/components   # theme.py (light palette)
D:/Medical AI Workshop/.claude/worktrees/streamlit-cnn-workshop-dcaf1e/05_frontend_demo/data         # roc_cache.json -- the measured numbers every page reads
D:/Medical AI Workshop/.claude/worktrees/streamlit-cnn-workshop-dcaf1e/03_augmentation_eval/checkpoints/overnight_run   # best.pt + history.json
```
Slice cache: `D:/pack_out` (override with the `NEUROFED_CACHE_2D` env var).

**Repo:** `Sanad-Manga/Pediatric-Brain-Tumor-Model`
**Branch with the latest UI work:** `claude/medical-ai-augmentation-eval-7f4098`
**App entry point:** `05_frontend_demo/Home.py` → `streamlit run 05_frontend_demo/Home.py`

---

## 1. The one rule

**Never put a number on a page that wasn't measured.** This demo shows output
from a medical segmentation model. A plausible-looking invented metric is
indistinguishable from a real one to anyone reading the screen, including the
people evaluating this project.

If a value can't be computed, render "not measured" / "n/a" / an explicit
warning. Do **not** substitute a placeholder. Several pages previously shipped
fabricated figures (92.4% Dice, 4.8mm HD95, 96.2% Dice, "50 FedAvg rounds",
"5 hospital nodes"); those are fixed, and the fix must not regress.

---

## 2. Ground truth — the current real numbers

Source: `05_frontend_demo/data/roc_cache.json`, measured on **82 held-out
patients** (never trained on), 492 tumour-bearing axial slices, using the
**epoch-16** checkpoint (`best.pt`, best validation mean Dice **0.6820**).

| Region | Dice | Sensitivity | Specificity | Precision | HD95 (median) | AUC | Pixel prevalence |
|---|---|---|---|---|---|---|---|
| ET (enhancing, label 1) | 0.5990 | 0.4956 | 0.99978 | 0.7569 | 4.12 mm | 0.9820 | 2.02% |
| TC (core, labels 1+3)   | 0.5213 | 0.4108 | 0.99970 | 0.7129 | 4.82 mm | 0.9805 | 2.58% |
| WT (whole, labels 1-4)  | 0.8267 | 0.7482 | 0.99922 | 0.9237 | 2.24 mm | 0.9930 | 17.85% |

**Training run:** 22 epochs recorded (epochs 3–24; epochs 0–2 were lost to a
logging bug, since fixed). Best epoch = 16, mean Dice 0.682
(ET 0.520 / NC 0.751 / WT 0.775). Tumour-type head peaked at **0.806** accuracy.

**Model:** 2D U-Net, **482,949 parameters** (0.48M), 4 input channels
(t1c/t1n/t2f/t2w), 256×256 padded slices, 5 output classes. Trained **centrally**
— *not* federated.

### Two framing rules that matter

1. **AUC (~0.98–0.99) is the flattering metric; Dice (0.68) is the honest one.**
   Tumour is only 2–18% of pixels and background is trivially easy, which
   inflates per-pixel AUC. Never quote AUC alone. Same for specificity
   (0.999) — it's high because most pixels are background.
2. **The tumour-type head is not a diagnosis.** No histology labels exist
   anywhere in this dataset. The head is trained against a *geometric proxy*
   from the segmentation mask (midline offset, inferior fraction, enhancing
   fraction). Majority-class baseline is **0.634** (92 of 145 subjects are
   astrocytoma-like), so 0.806 is real learning — but it is about tumour
   *geometry*, not tissue type.

---

## 3. Files you need

### Read these first (the data layer — don't duplicate this logic)

| File | What it gives you |
|---|---|
| `05_frontend_demo/utils/loaders.py` | `load_metrics_cache()`, `load_checkpoint_status()`, `load_training_history()`, `list_demo_subjects()`, `cache_available()` |
| `05_frontend_demo/utils/inference.py` | `load_model()`, `predict_slice()`, `dice_per_region()`, `region_breakdown()`, `checkpoint_metadata()`, `LABEL_NAMES` |
| `05_frontend_demo/utils/metrics.py` | ROC/AUC, sensitivity/specificity/precision/Dice from pooled counts, `hd95()` |
| `05_frontend_demo/utils/build_metrics_cache.py` | Recomputes `data/roc_cache.json` (~80 s) |
| `05_frontend_demo/data/roc_cache.json` | The measured numbers every page reads |
| `03_augmentation_eval/checkpoints/overnight_run/history.json` | Per-epoch loss / Dice / type accuracy |

**Always pull numbers through `loaders.py`.** Don't re-read JSON or re-run
inference inline in a page — inference on Drive-backed storage is slow and
Streamlit reruns the whole script on every widget interaction.

### Pages already converted to real data (use these as the pattern)

- `pages/Model_Performance.py` — training curves, ROC, per-region table, type head
- `pages/Segmentation_Report.py` — live inference, GT vs prediction, checkpoint diff
- `pages/Federated_Monitor.py` — states plainly that federated training never ran
- `pages/Dashboard.py` — performance block + case list
- `pages/Model_Intelligence.py` — architecture read off the loaded model

### Files with known remaining problems — **this is your job**

| File | Issue |
|---|---|
| `pages/Dashboard.py` | "Research Contributions" section still ticks ✓ for *Federated Learning across 5 hospital sites*, *3D U-Net*, *CORAL Domain Adaptation*, *Explainable AI (Grad-CAM)*, *Mixed Precision FP16*. **None of these are implemented or run in this repo.** Dataset summary still says "Volume Shape 96 × 96 × 96" and "Precision FP16 Mixed" — real is 240×240 2D slices, float32 training. "Subjects 257" — the manifests cover 227 (53+92+82). |
| `pages/Clinical_View.py` | Not audited for fabricated values. |
| `pages/MRI_Analysis.py` | Not audited. May still use random/mock volumes. |
| `pages/Domain_Adaptation.py` | CORAL exists only on branch `origin/codex/domain-adaptation`, built for 3D. Nothing on this branch runs it — check what the page claims. |
| `pages/About.py`, `Home.py` | Hero text says "Federated deep learning across hospital networks"; federation is not run. |

### Theme

- `05_frontend_demo/components/theme.py` — light palette via CSS variables, `apply_custom_theme()`
- `05_frontend_demo/.streamlit/config.toml` — Streamlit light base

**Known weakness:** the dark→light conversion was done by bulk colour-token
substitution across every page, then verified only programmatically (pages run
without exceptions; `.stApp` computes to `rgb(255,255,255)`). **It was never
visually reviewed page by page.** Expect leftover problems: low-contrast text,
gradient text that was designed to sit on black, borders that vanish on white.
Going through each page visually is worth doing first.

---

## 4. How to regenerate metrics after new training

The model is still training (resuming from epoch 24 toward 40). When a newer
checkpoint lands:

```bash
# 1. drop the new best.pt / last.pt into
#    03_augmentation_eval/checkpoints/overnight_run/

# 2. recompute measured metrics (~80 s)
cd 05_frontend_demo
python -m utils.build_metrics_cache

# 3. nothing else — every page reads the cache and the checkpoint directly
```

`utils/build_metrics_cache.py` stamps the cache with the checkpoint that
produced it, so mismatches are detectable. Point `--cache` at the slice cache
if it isn't at `D:/pack_out` (or set `NEUROFED_CACHE_2D`).

Slice cache layout: `<subject>/<plane>/slice_NNN.npz`, keys `image` (4,H,W)
float32 already z-scored, `mask` (1,H,W) uint8 labels 0–4.

---

## 5. Verify before you push

```bash
cd 05_frontend_demo
python -c "
from streamlit.testing.v1 import AppTest
import glob
for p in ['Home.py'] + sorted(glob.glob('pages/*.py')):
    at = AppTest.from_file(p, default_timeout=250); at.run()
    print(('FAIL' if at.exception else 'ok  '), p,
          (str(at.exception[0].value)[:200] if at.exception else ''))
"
```

All pages must print `ok`. Then actually open the app and look at every page —
that step was skipped last time and it shows.

```bash
streamlit run Home.py
```

---

## 6. Quick sanity checks

- Any percentage on screen should trace to `roc_cache.json`, `history.json`, or a
  live `predict_slice()` call. If you can't trace it, it's fabricated.
- Dice above ~0.85 for ET or TC would be wrong — those regions score 0.52–0.60.
- Anything claiming federated rounds, multiple hospital sites with scanner
  models, or 3D volumes is stale.
- Grep for leftovers:
  ```bash
  grep -rn "9[0-9]\.[0-9]%\|FP16\|96 × 96\|3D U-Net\|FedAvg\|Grad-CAM" --include="*.py" .
  ```
