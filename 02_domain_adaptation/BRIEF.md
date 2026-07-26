# Brief: Domain Adaptation (CORAL) + PCA/LDA Visualization

Read `../00_shared/CONTRACTS.md` first — it's the fixed interface everyone builds against.

## Your job
1. Implement CORAL (Correlation Alignment) loss, operating on the `features` output of the model interface `model(x) -> (seg_logits, features)`. Build and unit-test it standalone against random tensors first — don't wait on the real model to exist.
2. Once the real model (from `01_model_federated`) is available, wire CORAL in behind the `use_domain_adaptation` config flag, aligning feature distributions between Hospital A and Hospital B during federated training.
3. Build a PCA/LDA visualization script: extract `features` for subjects across Hospital A, Hospital B, and the held-out set, project with PCA/LDA, and plot before-adaptation vs after-adaptation to show whether CORAL actually reduced the domain gap. This is a key final deliverable.

## Out of scope
- The model/U-Net itself — separate section.
- Augmentation — separate section.
- Federated averaging logic — separate section (you just plug your loss into their loop via the flag).
