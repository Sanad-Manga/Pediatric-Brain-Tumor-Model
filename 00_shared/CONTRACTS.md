# Shared contracts — do not change without telling everyone

Project: federated, domain-adaptive 3D segmentation for pediatric brain tumors (BraTS-PEDs).

## Data
- Modalities per subject: t1c, t1n, t2f, t2w (.nii.gz).
- Label (seg file) — verified against actual dataset, 4 tumor subregions, not 3:
  - 1 = Enhancing Tumor (ET)
  - 2 = Non-Enhancing Tumor (NET)
  - 3 = Cystic Component (CC)
  - 4 = Peritumoral Edema (ED)
- Official evaluation regions (per BraTS-PEDs 2023 challenge): ET, NC (enhancing tumor + cystic + necrosis complex, i.e. labels 1+2+3), WT (whole tumor, all non-zero labels).
- Full 3D volumes only — no 2D slices, no multi-planar fusion (hard constraint from supervisor, confirmed by Dr. Youness: no usable 2D representation exists for this data).
- Resample target: 96³, FP16 mixed precision, batch size 1.
- 257 labeled subjects exist total (BraTS-PEDs Training folder). The Validation folder (91 subjects) has NO ground-truth labels — unusable for training/eval.

## Simulated hospitals (no real site labels exist in this dataset — these were derived by clustering scale-invariant intensity/contrast features per subject, since geometry is fully standardized)
- Hospital A: 53 subjects — `00_shared/manifests/hospitalA.json`
- Hospital B: 92 subjects — `00_shared/manifests/hospitalB.json`
- Held-out test institution: 82 subjects — `00_shared/manifests/heldout.json`
- A "client" = a manifest (subject ID list) + the shared cached-data path. Not physically separate storage — federation is simulated at the software/aggregation boundary.

## Model interface
`model(x) -> (seg_logits, features)`
- `features` = bottleneck/embedding vector, used by domain adaptation (CORAL) and by PCA/LDA visualization.

## Config flags every module must respect
- `use_augmentation: true|false`
- `use_federation: true|false`
- `use_domain_adaptation: true|false`

These flags drive the ablation matrix: baseline → +aug → +fed → +DA.

## Results schema (CSV)
Columns: `experiment_name, use_augmentation, use_federation, use_domain_adaptation, dice_ET, dice_NC, dice_WT, mean_dice`
(ET = enhancing tumor, NC = enhancing+cystic+necrosis complex, WT = whole tumor — official BraTS-PEDs evaluation regions.)
Evaluated on the held-out 82-subject set only.

## Checkpointing
Save every epoch to `checkpoints/<run_id>/`. Colab disconnects — resume-from-checkpoint is not optional.

## Team sections (6 people, 5 sections)
- `01_model_federated` — 2 people, model + FedAvg loop
- `02_domain_adaptation` — CORAL + PCA/LDA
- `03_augmentation_eval` — Mixup + ablation results CSV
- `04_clinical_bio` — clinical review + narrative, no code
- `05_frontend_demo` — Streamlit demo, consumes every other section's output; necessarily finishes last
