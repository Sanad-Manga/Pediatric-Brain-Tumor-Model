# Brief: 3D U-Net + Federated Training Loop

Read `../00_shared/CONTRACTS.md` first — it's the fixed interface everyone builds against. Do not deviate from it without flagging the change to the team.

## Your job
1. 3D U-Net (MONAI), trained from scratch (no pretrained weights), full 3D volumes at 96³, FP16, batch size 1.
2. Model must return `(seg_logits, features)` per the contract — `features` feeds domain adaptation and PCA/LDA later, built by another section. Expose it now even if unused by you.
3. Single-client training loop first (sanity check on a small subset), then extend to FedAvg across Hospital A (`hospitalA.json`, 53 subjects) and Hospital B (`hospitalB.json`, 92 subjects).
4. Accept a pluggable augmentation transform (built by the augmentation section) behind the `use_augmentation` flag — don't build augmentation logic yourself, just leave the hook.
5. Checkpoint every epoch. Support resume-from-checkpoint (Colab sessions disconnect).
6. This is the most blocking piece — domain adaptation and evaluation both depend on your model/features existing. Get a rough end-to-end version working fast, refine after.

## Out of scope
- Domain adaptation logic (CORAL) — separate section.
- Augmentation logic itself — separate section, you just accept the hook.
- Final evaluation/ablation reporting — separate section.
