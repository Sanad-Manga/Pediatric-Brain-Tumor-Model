# Brief: Clinical Framing, Data Sanity-Checking, Results Narrative

Read `../00_shared/CONTRACTS.md` first.

## Your job (no model code)
1. Visually sanity-check the resampled/cached 96³ volumes still look medically correct (tumor regions not destroyed by downsampling, labels still aligned to anatomy).
2. Visually review segmentation outputs from the model on the held-out set (`../00_shared/manifests/heldout.json`) — flag anything clinically implausible.
3. Own the clinical framing of the report/demo: what the 4 tumor subregions (enhancing tumor, non-enhancing tumor, cystic component, peritumoral edema) mean, why pediatric high-grade glioma/DIPG segmentation matters, and how to read the final ablation table and PCA/LDA plots for a non-technical audience.
4. Write the results narrative once the ablation table (baseline → +aug → +fed → +DA) and PCA/LDA plots exist from the other sections.

## Out of scope
- Any model, training, or evaluation code — that's the other three sections.
