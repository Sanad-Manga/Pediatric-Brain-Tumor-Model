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

**Live demo:** https://a-sanad-neuropeds-ai.hf.space *(HuggingFace Space)*

## What it does

- Segments pediatric brain MRI scans across 4 tumor sub-regions: Enhancing Tumor (ET), Non-Enhancing Tumor Core (NETC), Cystic Component (CC), and Peritumoral Edema (ED)
- Runs inference on real held-out patients using a trained checkpoint (epoch 16, mean Dice 0.674)
- Displays per-region Dice scores, ROC curves, and training history
- Generates clinical PDF reports

## Model

| Region | Dice | Sensitivity | Specificity |
|--------|------|-------------|-------------|
| ET | 0.599 | 0.622 | 0.999 |
| TC (ET+NETC) | 0.680 | 0.708 | 0.998 |
| WT (all) | 0.742 | 0.781 | 0.997 |

Architecture: 2D U-Net with optional MixUp augmentation and tumor-type classification head. Trained on axial slices from 4 MRI modalities (T1c, T1n, T2f, T2w).

## Repo structure

| Folder | Contents |
|--------|----------|
| `03_augmentation_eval/` | Training pipeline, evaluation scripts, ablation framework |
| `05_frontend_demo/` | Streamlit app + demo cache (5 representative patients) |
| `00_shared/` | Shared contracts and data specifications |

## Running locally

```bash
cd 05_frontend_demo
pip install -r requirements.txt
streamlit run Home.py
```

The app ships with a demo cache of 5 held-out patients covering all 4 tumor label types. Set `NEUROFED_CACHE_2D` to point at a full 2D slice cache for the complete dataset.
