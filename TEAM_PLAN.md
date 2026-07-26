# Federated, Domain-Adaptive 3D Segmentation — Pediatric Brain Tumors

## The project in 3 lines
We're training a 3D U-Net (from scratch, full 3D MRI volumes) to segment pediatric brain
tumors, simulating 2 hospitals training together via federated learning, then using domain
adaptation (CORAL) to handle the fact that the two hospitals' scans look different. We
compare results with/without augmentation, federation, and domain adaptation in one final
table, tested on a 3rd hospital the model never saw.

## Data — already handled, nothing to set up
- Dataset: BraTS-PEDs (pediatric brain tumor MRI), 4 scan types per patient (t1c, t1n, t2f, t2w).
- 257 usable patients total. Split into 3 simulated "hospitals" (no real hospital labels exist
  in this dataset, so we grouped patients by how their scans look):
  - **Hospital A** — 53 patients (training)
  - **Hospital B** — 92 patients (training)
  - **Held-out hospital** — 82 patients (final test only, never trained on)
- Data lives in one shared Drive folder. Nobody downloads the full dataset individually.
- All shared rules (data format, folder structure, what every piece of code must output)
  are written down in `00_shared/CONTRACTS.md` — read that once, it answers most
  "wait how does X talk to Y" questions.

## Pick a section — 3 available (augmentation is already spoken for)
Each folder below has its own `BRIEF.md`: what to build, what NOT to build (so nobody
duplicates work), and the shared data files you need.

| Folder | What it is | Best fit for |
|---|---|---|
| `01_model_federated` | Build the 3D U-Net + the federated training loop across Hospital A & B. Heaviest engineering, most blocking — everything else depends on this. | 2 people, strongest at ML engineering |
| `02_domain_adaptation` | CORAL (make the model handle the two hospitals' scan differences) + PCA/LDA plots showing the fix worked | 1 person, comfortable with the math/stats side |
| `03_augmentation_eval` | Already started — data augmentation (Mixup) + the final results table | taken |
| `04_clinical_bio` | No coding. Sanity-check the scans/results look medically correct, write the clinical explanation for the report/demo | bio teammate |

## How to start
1. Pick your folder above.
2. Open `00_shared/CONTRACTS.md` and your folder's `BRIEF.md`.
3. Give both files to whatever AI tool you're using (Claude, ChatGPT, Copilot, etc.) as
   context before asking it to start building.
4. Don't touch another section's folder — if you need something from another section
   (e.g. the model's output), it's already defined in `CONTRACTS.md`.

## Timeline — 3 working days
- **Day 1**: Everyone builds their piece independently (against placeholder/dummy data
  where the real thing doesn't exist yet — e.g. domain adaptation tests against fake
  data before the real model exists).
- **Day 2**: Real pieces get plugged together — federated training runs for real, domain
  adaptation and augmentation plug into the real model.
- **Day 3**: Full results table + plots generated on the held-out hospital, report/demo
  assembled. This day has no new building — it's integration, fixing what breaks, and
  writing up results.

## One correction worth knowing
Original plan said tumors split into 3 labeled parts. Checked the actual dataset — it's
**4 parts**: enhancing tumor, non-enhancing tumor, a cystic/fluid component, and swelling
(edema). `CONTRACTS.md` and the briefs are already updated to reflect this.
