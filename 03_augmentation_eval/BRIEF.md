# Brief: Augmentation (Mixup) + Evaluation/Ablation Pipeline

Read `../00_shared/CONTRACTS.md` first — it's the fixed interface everyone builds against.

## Context
Dataset: BraTS-PEDs pediatric brain tumor MRI, 4 modalities (t1c, t1n, t2f, t2w), full 3D volumes, resampled to 96³, FP16, batch size 1. Federated clients are simulated via manifests, not physically separate storage:
- Hospital A: 53 subjects (`../00_shared/manifests/hospitalA.json`)
- Hospital B: 92 subjects (`../00_shared/manifests/hospitalB.json`)
- Held-out test institution: 82 subjects (`../00_shared/manifests/heldout.json`)

Per-client sample counts (53/92) are small for training a 3D U-Net from scratch, so augmentation is not optional polish — it needs to be solid and togglable from day one so the ablation table can compare with/without it.

## Your job
1. Build Mixup for 3D volumetric segmentation (mix volume pairs + their labels appropriately for a segmentation task, not classification-style label mixing).
2. Build standard MONAI spatial/intensity augmentations as a complementary transform stack: random flips, rotations, intensity jitter, random crops — appropriate for 96³ full-volume 3D MRI.
3. Wrap all of the above behind a single `use_augmentation: true|false` config flag (per contract) so it can be toggled on/off per ablation run without code changes.
4. Build the ablation runner: given a trained model checkpoint + the held-out manifest, compute Dice for ET (enhancing tumor), NC (enhancing tumor + cystic component + necrosis complex), WT (whole tumor) — the official BraTS-PEDs evaluation regions — write one row to the results CSV per the schema in `CONTRACTS.md`:
   `experiment_name, use_augmentation, use_federation, use_domain_adaptation, dice_ET, dice_NC, dice_WT, mean_dice`
5. Develop and test both the augmentation transforms and the ablation runner against dummy/random tensors and dummy predictions first — the real model from `01_model_federated` won't exist yet. Plug in against the real model once it's ready.

## Out of scope
- The model/U-Net itself and the federated training loop — separate section, you just accept its checkpoints as input.
- CORAL/domain adaptation and PCA/LDA — separate section.
- Clinical narrative/results writeup — separate section (bio).
