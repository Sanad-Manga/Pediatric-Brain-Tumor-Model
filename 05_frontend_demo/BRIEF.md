# Brief: Streamlit Demo / Frontend

Read `../00_shared/CONTRACTS.md` first — it's the fixed interface everyone builds against.

## Context
This is the demo layer — it doesn't do any modeling itself, it just displays what the
other sections produce. Because of that, it's necessarily the last thing to come
together for real, but the UI shell can and should be built early against placeholder/
dummy data, same as every other section.

## Your job
1. Build a Streamlit app that can:
   - Pick a subject from any manifest (`00_shared/manifests/hospitalA.json`,
     `hospitalB.json`, or `heldout.json`) and display its MRI slices (t1c/t1n/t2f/t2w)
     with the predicted segmentation overlaid on top of the ground-truth mask, so a
     viewer can visually compare them.
   - Display the ablation results table produced by `03_augmentation_eval`
     (columns: `experiment_name, use_augmentation, use_federation,
     use_domain_adaptation, dice_ET, dice_NC, dice_WT, mean_dice`).
   - Display the PCA/LDA before/after domain-adaptation plots produced by
     `02_domain_adaptation`.
   - Surface the clinical explanation text written by `04_clinical_bio`, so a
     non-technical viewer (e.g. in the workshop demo) has context for what they're
     looking at, not just raw numbers and images.
2. Build the UI shell first against dummy data (a fake results CSV, a random tensor
   standing in for a segmentation mask, placeholder plot images) — don't wait on the
   other sections to finish before starting.
3. Swap in real outputs from each section as they become available (checkpoint from
   `01_model_federated`, results CSV from `03`, plots from `02`, narrative from `04`).

## Out of scope
- Training, evaluation, or any model code — you only consume outputs, you don't produce them.
- Writing the clinical narrative content — that's `04_clinical_bio`, you just display it.

## Notes
- This section is inherently last-in on integration day (Day 3), since it depends on
  every other section's real output to be fully populated. Get the shell fully working
  against fake data well before then so Day 3 is just swapping data sources, not building UI.
