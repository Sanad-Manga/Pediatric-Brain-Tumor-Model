# NeuroPeds AI — Full Project Handoff

**Date:** 2026-08-02
**Repo:** https://github.com/Sanad-Manga/Pediatric-Brain-Tumor-Model
**Live demo:** https://neuropeds-ai.streamlit.app
**Branch to clone:** `main` (everything is merged)

---

## What This Project Is

A 2D U-Net segmentation system for pediatric brain tumors. Trained on the **BraTS-PEDs 2024** dataset. The model segments MRI scans into 4 tumor sub-regions. The Streamlit frontend connects directly to the trained model and a small embedded demo cache of 5 real patients.

This is a workshop project. Section 03 (`03_augmentation_eval/`) owns the model and training pipeline. Section 05 (`05_frontend_demo/`) is the deployed Streamlit app.

---

## Repository Structure

```
Pediatric-Brain-Tumor-Model/
├── 00_shared/
│   ├── CONTRACTS.md              <- fixed interface rules all sections share
│   └── manifests/                <- train/val/test subject splits (JSON)
│
├── 03_augmentation_eval/
│   ├── run.py                    <- main CLI: train / eval / pack / index
│   ├── config.yaml               <- all hyperparameters (edit this, not source)
│   ├── src/
│   │   ├── model.py              <- 2D U-Net + TumorTypeHead
│   │   ├── train.py              <- training loop
│   │   ├── evaluate.py           <- per-subject Dice / HD95 evaluation
│   │   ├── dataset.py            <- SliceDataset, EvalSubjectDataset
│   │   ├── augment.py            <- MONAI augmentation stack
│   │   ├── slices.py             <- cache I/O, pad_to, unpad, _load_slice
│   │   ├── metrics.py            <- Dice, HD95, sensitivity, specificity
│   │   ├── config.py             <- Config dataclass + load_config()
│   │   ├── pack.py               <- NIfTI -> 2D .npz slice cache builder
│   │   └── tumor_type.py         <- imaging-proxy tumor type classifier
│   ├── checkpoints/
│   │   └── overnight_run/
│   │       ├── best.pt           <- SHIPPED in repo, epoch 16, mean Dice 0.682
│   │       └── history.json      <- 36 epochs of train loss + val Dice
│   └── results/
│       └── ablation_results.csv  <- populated by: run.py eval --out-csv
│
└── 05_frontend_demo/
    ├── Home.py                   <- Streamlit entry point
    ├── requirements.txt          <- streamlit, torch (cpu), numpy, pyyaml, etc.
    ├── pages/
    │   ├── Dashboard.py          <- metrics overview + ROC curves
    │   ├── MRI_Analysis.py       <- live inference on demo patients
    │   ├── Segmentation_Report.py <- per-patient PDF report generation
    │   ├── Clinical_View.py      <- clinical-facing slice viewer
    │   └── About.py
    ├── utils/
    │   ├── inference.py          <- model bridge: loads best.pt, runs predict_slice()
    │   ├── loaders.py            <- data loading (cache, checkpoint, metrics)
    │   ├── metrics.py            <- frontend metric helpers
    │   └── build_metrics_cache.py <- builds data/roc_cache.json from eval output
    ├── components/theme.py
    ├── data/
    │   └── roc_cache.json        <- pre-computed ROC curves + full metrics table
    └── demo_cache/               <- 5 patients x 30 axial slices = 93 MB
        ├── BraTS-PED-00030-000/  <- all 4 label types (ET, NETC, CC, ED)
        ├── BraTS-PED-00021-000/  <- ET + NETC + ED
        ├── BraTS-PED-00093-000/  <- ET + NETC + CC
        ├── BraTS-PED-00028-000/  <- CC + ED only (no ET)
        └── BraTS-PED-00230-000/  <- tiny ET + NETC (near-control)
```

---

## The Model

### Architecture
- **2D U-Net** with `width=48`, `depth=3`
- Input: 4-channel MRI slice `(4, 256, 256)` — modalities t1c, t1n, t2f, t2w
- Output: 5-class segmentation logits + bottleneck feature map
- Auxiliary **TumorTypeHead** (MLP on bottleneck features) predicts imaging-proxy tumor type. Does NOT affect segmentation output.
- Parameters: ~4.3 million
- Trained on 2D slices (axial + coronal), evaluated by restacking into 3D volumes

### Scoring Regions (BraTS convention)
| Region | Labels included | What it captures |
|--------|----------------|-----------------|
| ET — Enhancing Tumor | 1 | Active, blood-brain-barrier-breaching tumor |
| TC — Tumor Core | 1 + 3 | ET + Cystic Component |
| WT — Whole Tumor | 1 + 2 + 3 + 4 | Everything non-background |

### Label Encoding in Masks
- 0 = background
- 1 = ET   (Enhancing Tumor)
- 2 = NETC (Non-Enhancing Tumor Core)
- 3 = CC   (Cystic Component)
- 4 = ED   (Peritumoral Edema)

### Trained Checkpoint — best.pt (epoch 16)
Selected by **min-region** criterion (best worst-region Dice, not best mean). Epoch 25 had higher mean (0.689) but ET collapsed to 0.434 on held-out data (-0.165 vs epoch 16).

| Region | Dice | Sensitivity | Specificity |
|--------|------|-------------|-------------|
| ET     | 0.599 | 0.622 | 0.999 |
| TC     | 0.680 | 0.708 | 0.998 |
| WT     | 0.742 | 0.781 | 0.997 |
| Mean   | 0.682 | | |

**Known weaknesses:** Model under-segments — precision >> sensitivity. Misses tumor rather than over-calls it. AUC/specificity are inflated because only 2-18% of pixels are tumor. ET is the hardest region.

### Training History
36 epochs total (~295 sec/epoch on Colab T4). Loss was still declining at ep36 — the model is under-trained. The RTX 3060 should train at similar or faster speed. More epochs will most benefit ET.

---

## Data

### Full Dataset (NOT in repo — on shared Drive)
- **BraTS-PEDs 2024** — pediatric brain MRI, 4 modalities
- Pre-processed into 2D .npz slices by the team
- Cache path used during training: `/content/drive/MyDrive/Medical AI Workshop/pack_out/pack_out`
- Format: `<cache_2d>/<subject_id>/<plane>/slice_NNN.npz`
  - `image`: `(4, H, W)` float32, already z-scored per volume over brain voxels
  - `mask`: `(1, H, W)` uint8, labels 0-4
- Axial: 240x240, coronal: 240x155. Both padded to 256x256 for the network.

### Manifests (in repo at 00_shared/manifests/)
- `train.json`, `val.json`, `test.json` — subject ID splits
- Do NOT change these. They are the fixed evaluation contract.

### Demo Cache (in repo at 05_frontend_demo/demo_cache/)
5 patients, 30 axial slices each. 93 MB. This is what the live app uses. No external drive needed.

---

## How to Run Locally (RTX 3060)

### Setup
```bash
git clone https://github.com/Sanad-Manga/Pediatric-Brain-Tumor-Model.git
cd Pediatric-Brain-Tumor-Model
```

### Run the Streamlit app (demo cache, no GPU needed)
```bash
cd 05_frontend_demo
pip install -r requirements.txt
streamlit run Home.py
```

### Run the Streamlit app with the full slice cache
```bash
cd 05_frontend_demo
pip install -r requirements.txt
NEUROFED_CACHE_2D=C:/path/to/pack_out streamlit run Home.py
```
On Windows PowerShell: `$env:NEUROFED_CACHE_2D = "C:/path/to/pack_out"; streamlit run Home.py`

### Train on RTX 3060 (CUDA)
```bash
cd 03_augmentation_eval
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install monai pyyaml numpy
```

Edit config.yaml line 22 to set your local cache path:
```yaml
paths:
  cache_2d: "C:/path/to/pack_out"
```

Resume from epoch 16 (recommended — do not start from scratch):
```bash
python run.py train --run-id overnight_run --resume
```

Start a fresh run:
```bash
python run.py train --run-id my_run
```

### Evaluate a checkpoint
```bash
# Print results only — does NOT write to ablation CSV
python run.py eval \
  --experiment-name held_out \
  --checkpoint checkpoints/overnight_run/best.pt \
  --cache C:/path/to/pack_out \
  --eval-plane axial

# Save results to CSV
python run.py eval \
  --experiment-name held_out \
  --checkpoint checkpoints/overnight_run/best.pt \
  --cache C:/path/to/pack_out \
  --eval-plane axial \
  --out-csv results/my_results.csv
```

### All CLI commands (run.py)
```bash
python run.py train --run-id <name> [--resume]
python run.py eval --experiment-name <name> --checkpoint <path> --cache <path> [--out-csv <path>] [--eval-plane axial|coronal|both]
python run.py pack --cache <output_path>        # build 2D slice cache from NIfTI
python run.py index --cache <path>              # build index.json (tumor slice lookup)
```

---

## How the Frontend Connects to the Model

`05_frontend_demo/utils/inference.py` is the bridge. It:
1. Adds `03_augmentation_eval/` to `sys.path` so it can import `src.*`
2. Loads `best.pt` from `03_augmentation_eval/checkpoints/overnight_run/best.pt` (relative path, works from any clone)
3. Imports `_load_slice` from `src.slices` (NOT `src.dataset` — see Gotchas below)
4. `predict_slice()` returns: prediction, ground truth, probabilities, confidence, features

Cache defaults to `05_frontend_demo/demo_cache/`. Override with `NEUROFED_CACHE_2D` env var.

---

## Deployment

### Streamlit Community Cloud (active, use this)
- URL: https://neuropeds-ai.streamlit.app
- Connected to `main` branch, entry point `05_frontend_demo/Home.py`
- Auto-redeploys on every push to main
- No CPU quota — stays up indefinitely as long as visited every 7 days

### HuggingFace Spaces (paused)
- URL: https://huggingface.co/spaces/A-Sanad/NeuroPeds-AI
- Hit free tier CPU quota. Ignore — Streamlit Cloud is the real deployment.

---

## What to Do Next on RTX 3060

### 1. Continue training (highest value)
The model stopped at 36 epochs and loss was still dropping. Resume:
```bash
cd 03_augmentation_eval
python run.py train --run-id overnight_run --resume
```
Watch `dice_ET` in the logs. Stop when it plateaus for 5+ epochs. Current: 0.460 at ep36. Target: >0.60 on training val, >0.65 on held-out.

### 2. Run proper held-out evaluation
Once you have the full cache:
```bash
python run.py eval \
  --experiment-name final_eval \
  --checkpoint checkpoints/overnight_run/best.pt \
  --cache C:/path/to/pack_out \
  --eval-plane both
```
`--eval-plane both` averages axial+coronal softmax before argmax. Usually adds 1-3 Dice points but doubles evaluation time.

### 3. Rebuild roc_cache.json after retraining
This powers the Dashboard ROC curves and metrics table:
```bash
cd 05_frontend_demo
python -m utils.build_metrics_cache
```
Then commit and push — Streamlit Cloud auto-redeploys.

### 4. Secondary: ablation runs
Set `use_augmentation: false` or `use_mixup: false` in config.yaml and run eval with `--out-csv` to compare conditions. Results accumulate in `results/ablation_results.csv`.

### 5. Secondary: try larger model
In config.yaml, change `width: 48, depth: 3` to `width: 64, depth: 4`. Will train slower but capacity was the binding constraint.

---

## Key Gotchas

**`src.dataset` imports MONAI at module level (via augment.py)**
`inference.py` imports `_load_slice` from `src.slices`, not `src.dataset`. Do not change this. If you ever need to use `dataset.py` directly in the frontend, you must either install MONAI or replicate the function elsewhere.

**`config.yaml` cache path**
Line 22 points to a Colab Drive path that will not exist on your PC. Always update `paths.cache_2d` before training.

**`run.py eval` does not write CSV by default**
Pass `--out-csv <path>` explicitly. Without it, results print to stdout only. This was a bug that was fixed — previously it always wrote to the ablation CSV on every eval run.

**Checkpoint epoch field is 0-indexed**
`epoch: 16` in the file means the 17th training epoch.

**Checkpoint selection uses min-region, not mean**
`config.yaml selection.metric: "min_region"` — best.pt is the epoch with the highest worst-region Dice, not the highest mean. `best_mean.pt` is also written each epoch if you want to compare.

**AUC and specificity are inflated**
Only 2-18% of pixels are tumor. A model that predicts all-background gets specificity ~0.98. Do not cite these as headline metrics. Use Dice and sensitivity.

**Slice filenames are zero-padded**
`slice_000.npz`, `slice_001.npz`, etc. The index.json stores integer indices (0, 1, 2...), not filenames.
