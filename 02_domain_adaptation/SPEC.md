# Spec: CORAL Domain Adaptation and Feature-Space Evaluation

## 1. Goal

Add memory-safe Deep CORAL alignment to federated 3D tumor-segmentation training and provide reproducible PCA/LDA evidence of the feature-domain gap before and after adaptation.

## 2. In Scope

- Deep CORAL loss over the model's two-dimensional bottleneck embeddings.
- Batch-size-1-compatible covariance estimation using bounded feature history.
- Federated-loop integration gated by `use_domain_adaptation`.
- Config and CLI controls for CORAL weight, queue length, and steps per round.
- Verification that the model's exposed embeddings vary between distinct inputs.
- Checkpoint recording of the per-round CORAL loss.
- Feature extraction from Hospital A, Hospital B, and the held-out institution.
- Before/after PCA and LDA projection into two dimensions.
- A saved visualization comparing both checkpoints and all three institutions.
- Dummy-tensor unit/integration tests and a documented real-cache smoke-test procedure.

## 3. Out of Scope

- Changing the U-Net segmentation architecture, decoder, or segmentation loss.
- Implementing or changing FedAvg aggregation mathematics.
- Implementing augmentation or Mixup.
- Training on or adapting to the held-out institution.
- Computing final ET/NC/WT Dice or writing the ablation results CSV.
- Claiming clinical benefit or improved segmentation without full held-out evaluation.
- Secure aggregation, differential privacy, or production multi-hospital deployment.

## 4. Requirements

1. `coral_loss(source, target)` accepts two `(N, D)` feature tensors, rejects non-2D inputs, mismatched feature dimensions, and domains with fewer than two subjects, and equals the normalized squared Frobenius distance between unbiased covariance matrices.
2. CORAL covariance calculations run in float32 under mixed precision, return a finite nonnegative scalar, are symmetric between domains, equal zero for identical inputs, and backpropagate finite gradients to both current-domain inputs.
3. A bounded feature queue supports physical batch size 1, stores previous embeddings detached from their computation graphs, preserves the current embedding's gradient, and rejects a capacity below two.
4. The model feature hook targets the actual deepest residual bottleneck rather than a normalization leaf, returns `(B, D)` features, and produces measurably different embeddings for distinct inputs.
5. `TrainConfig` exposes `coral_weight`, `coral_queue_size`, and optional `coral_steps_per_round`, rejecting negative weights, queue sizes below two, and an explicit step count below two.
6. `use_domain_adaptation=False` leaves the existing FedAvg path unchanged; `True` runs one Hospital A/B CORAL alignment phase after aggregation and before checkpointing in every federated round.
7. The alignment phase reads only the two supplied training manifests, supports both dummy and real cache modes, keeps each image DataLoader batch at one, cycles the shorter domain safely without caching image batches, and performs optimizer updates from finite CORAL losses.
8. Every federated checkpoint records `coral_loss`; it is `None` when adaptation is disabled and a finite nonnegative float when adaptation is enabled.
9. The CLI exposes `--coral-weight`, `--coral-queue-size`, and `--coral-steps-per-round` and passes them into `TrainConfig`.
10. The visualization command strictly loads both model checkpoints, extracts features without gradients for Hospital A, Hospital B, and held-out subjects using the real cache loader, and never performs an optimizer update.
11. PCA and LDA each return exactly two finite coordinates per subject for the three institutions; LDA institution labels are used only for visualization, not training or checkpoint selection.
12. The visualization produces a four-panel image containing PCA-before, PCA-after, LDA-before, and LDA-after plots with visible institution labels and saves it to the requested output path.
13. Automated domain-only tests cover covariance correctness, CORAL invariants and gradients, invalid inputs, queue behavior, flag-to-federated integration, an end-to-end unit-batch alignment phase, embedding non-collapse, and PCA/LDA output validity.
14. A real-cache GPU smoke test using at least two subjects per training hospital completes one federated round, saves a checkpoint, and records finite segmentation losses plus a finite positive CORAL loss.
15. Documentation states how to run domain-only tests and the before/after visualization and explicitly distinguishes technical validation from evidence of held-out segmentation improvement.

**Assumption:** The project has exactly two training domains, Hospital A and Hospital B, in the first two federated client manifests.

**Assumption:** A detached rolling queue is acceptable for covariance estimation under the contractual physical batch size of one.

**Assumption:** Full effectiveness is evaluated later using held-out Dice from section 03; this section supplies feature-space evidence and technical validation.

## 5. Structure

```text
01_model_federated/
â”œâ”€â”€ run.py
â”œâ”€â”€ src/
â”‚   â”œâ”€â”€ config.py                 # CORAL configuration
â”‚   â”œâ”€â”€ domain_adaptation.py      # covariance, CORAL, queue, alignment phase
â”‚   â”œâ”€â”€ federated.py              # gated post-FedAvg integration
â”‚   â””â”€â”€ model.py                  # non-collapsing bottleneck feature hook
â””â”€â”€ tests/
    â””â”€â”€ test_domain_adaptation.py

02_domain_adaptation/
â”œâ”€â”€ BRIEF.md
â”œâ”€â”€ README.md
â”œâ”€â”€ SPEC.md
â”œâ”€â”€ requirements.txt
â”œâ”€â”€ visualize_features.py
â””â”€â”€ tests/
    â””â”€â”€ test_visualization.py
```

## 6. Edge Cases

| Scenario | Expected Behaviour |
|---|---|
| Feature tensor is not `(N, D)` | Raise `ValueError` naming the expected shape |
| Either domain has one subject | Raise `ValueError` requiring at least two subjects |
| Feature dimensions differ | Raise `ValueError` |
| Queue size is less than two | Reject configuration before training |
| CORAL step count is less than two | Reject configuration before training |
| One hospital loader is shorter | Restart only that iterator; do not retain past image batches |
| `use_domain_adaptation` is false | Do not call the alignment phase; save `coral_loss=None` |
| Checkpoint lacks `model_state` wrapper | Visualization accepts a raw state dict |
| Checkpoint state does not match the model | Fail strict loading with the mismatch |
| Cache subject is missing | Propagate a clear `FileNotFoundError` with its path |
| Output directory does not exist | Create its parent directories before saving the plot |
| CUDA is unavailable | Unit tests run on CPU; real 96Â³ smoke training is reported as blocked |
| CORAL loss is zero because embeddings collapse | Fail the embedding non-collapse regression test |

## 7. Done Checklist

- [x] Req 1: `coral_loss(source, target)` accepts two `(N, D)` feature tensors, rejects non-2D inputs, mismatched feature dimensions, and domains with fewer than two subjects, and equals the normalized squared Frobenius distance between unbiased covariance matrices.
- [x] Req 2: CORAL covariance calculations run in float32 under mixed precision, return a finite nonnegative scalar, are symmetric between domains, equal zero for identical inputs, and backpropagate finite gradients to both current-domain inputs.
- [x] Req 3: A bounded feature queue supports physical batch size 1, stores previous embeddings detached from their computation graphs, preserves the current embedding's gradient, and rejects a capacity below two.
- [x] Req 4: The model feature hook targets the actual deepest residual bottleneck rather than a normalization leaf, returns `(B, D)` features, and produces measurably different embeddings for distinct inputs.
- [x] Req 5: `TrainConfig` exposes `coral_weight`, `coral_queue_size`, and optional `coral_steps_per_round`, rejecting negative weights, queue sizes below two, and an explicit step count below two.
- [x] Req 6: `use_domain_adaptation=False` leaves the existing FedAvg path unchanged; `True` runs one Hospital A/B CORAL alignment phase after aggregation and before checkpointing in every federated round.
- [x] Req 7: The alignment phase reads only the two supplied training manifests, supports both dummy and real cache modes, keeps each image DataLoader batch at one, cycles the shorter domain safely without caching image batches, and performs optimizer updates from finite CORAL losses.
- [x] Req 8: Every federated checkpoint records `coral_loss`; it is `None` when adaptation is disabled and a finite nonnegative float when adaptation is enabled.
- [x] Req 9: The CLI exposes `--coral-weight`, `--coral-queue-size`, and `--coral-steps-per-round` and passes them into `TrainConfig`.
- [x] Req 10: The visualization command strictly loads both model checkpoints, extracts features without gradients for Hospital A, Hospital B, and held-out subjects using the real cache loader, and never performs an optimizer update.
- [x] Req 11: PCA and LDA each return exactly two finite coordinates per subject for the three institutions; LDA institution labels are used only for visualization, not training or checkpoint selection.
- [x] Req 12: The visualization produces a four-panel image containing PCA-before, PCA-after, LDA-before, and LDA-after plots with visible institution labels and saves it to the requested output path.
- [x] Req 13: Automated domain-only tests cover covariance correctness, CORAL invariants and gradients, invalid inputs, queue behavior, flag-to-federated integration, an end-to-end unit-batch alignment phase, embedding non-collapse, and PCA/LDA output validity.
- [x] Req 14: A real-cache GPU smoke test using at least two subjects per training hospital completes one federated round, saves a checkpoint, and records finite segmentation losses plus a finite positive CORAL loss.
- [x] Req 15: Documentation states how to run domain-only tests and the before/after visualization and explicitly distinguishes technical validation from evidence of held-out segmentation improvement.

