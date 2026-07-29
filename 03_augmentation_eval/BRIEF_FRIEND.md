# Brief — inference-side improvements (no GPU needed)

You're working the CPU half while the width-48 model trains on Colab. Everything
here runs against the **existing epoch-16 checkpoint** and transfers unchanged to
the new one when it lands, so nothing you do is blocked on that run finishing.

## Setup

```bash
git pull
cd 03_augmentation_eval
python -m pytest tests/ -q          # expect 161 passed
```

You need the slice cache. Ask for the `Processed_2D` path — it's ~98k `.npz`
files and is not in the repo. Every command below takes `--cache <that path>`;
`config.yaml`'s `cache_2d` points at a Colab Drive path and will not work locally.

**First real task: reproduce the baseline before changing anything.**

```bash
python run.py eval --experiment-name baseline_check --checkpoint checkpoints/overnight_run/best.pt --cache <cache> --eval-plane axial
```

You should get **ET 0.599, NC 0.521, WT 0.827**. If you don't, stop and say so —
something differs in your environment and every number you produce afterwards
would be uncomparable to Ahmed's. This is the only gate; once it passes, go.

Note the naming: the code calls the core region **NC**, the report calls it
**TC**. Same thing.

## What's already built for you

`--tta`, `--et-boost`, `--keep-largest-wt`, `--min-component-voxels` on
`run.py eval`, plus `tools/tune_postproc.py`. All default to off, so the baseline
command above is bit-identical to the pre-existing code path.

## Task 1 — TTA and plane averaging (start here)

Two configured-but-never-measured things and one new one. Measure them separately
so you know which one paid:

```bash
python run.py eval --experiment-name plane_both --checkpoint <ckpt> --cache <cache> --eval-plane both
python run.py eval --experiment-name plane_both_tta --checkpoint <ckpt> --cache <cache> --eval-plane both --tta
```

`--eval-plane both` averages axial and coronal softmax in the shared voxel grid.
The model was trained on both planes but has only ever been scored on axial, so
this is free information already paid for. `--tta` adds a horizontal-flip average.

Expect small gains — 0.01–0.03 Dice is a normal, real TTA result. If you see
+0.15 something is wrong, not brilliant.

## Task 2 — post-processing sweep

The model **under-segments**: ET precision 0.757 against sensitivity 0.496. It
misses tumour rather than over-calling it, so the lever that fits is lowering the
bar for ET (`--et-boost`) and deleting false-positive specks
(`--min-component-voxels`).

```bash
python tools/tune_postproc.py --checkpoint checkpoints/overnight_run/best.pt --cache <cache>
```

This sweeps 8 settings and prints a delta table.

**Read this part carefully — it's the one way to invalidate the whole result.**
The sweep runs on *validation* patients, which it reads out of the checkpoint
itself. Those are training-hospital patients. You then take **one** row — the
single best — and measure it on held-out **once**, via `run.py eval`. If you
sweep on held-out and report the winner, that number is not a generalisation
estimate and the report's headline figures become fiction. The script is built to
make the right thing easy; don't route around it.

If the held-out gain is much smaller than the validation gain, report that
honestly. "TTA gave +0.02, post-processing overfit and gave nothing" is a
perfectly good finding and better than a fake one.

Watch **ET specifically**, not just mean Dice. Mean rising while ET falls is
exactly the failure that made epoch 25 look better than epoch 16 when it was
much worse. `keep_largest_wt` in particular destroys real tumour on genuinely
multifocal subjects — check how many subjects it changes, not just the mean.

## Task 3 — Streamlit audit

`05_frontend_demo/` is what the clinician actually sees, and parts of it are
currently untrue.

- `pages/Dashboard.py` ticks ✓ for *Federated Learning across 5 hospital sites*,
  *3D U-Net*, *CORAL*, *Grad-CAM XAI* and *FP16*. **None of these are
  implemented.** Remove them or mark them as planned.
- `Clinical_View.py`, `MRI_Analysis.py`, `Domain_Adaptation.py` are unaudited —
  assume the same problem.
- The light theme was applied by bulk colour-token substitution across 9 pages
  and never looked at by a human. Expect low-contrast text and gradients built
  for a black background. The ROC/AUC page is known to render wrong, while the
  *report* artifact's ROC is correct — so the bug is in the Streamlit page.

Two framing rules that apply everywhere in the UI:

1. **Never quote AUC or specificity alone.** Tumour is 2–18% of pixels, so
   background is trivially easy and both numbers are inflated. Dice and HD95 are
   the honest figures.
2. **The tumour-type head is not histology.** There is no tumour-type ground
   truth in this dataset; it is trained against a geometric proxy computed from
   the mask. Any page implying diagnosis needs to say so.

## Task 4 — the ablation (needs your GPU, your Colab account)

This is section 03's actual deliverable and the table is currently empty. It runs
on **your** Colab account — Ahmed's quota is used up and the main training run is
on a third person's account, so your GPU is the only one free.

It cannot be produced by running `run.py eval` several times. That command copies
the config flags into the CSV while the Dice columns all come from the same
weights, so you'd get a table that looks like an ablation and measures nothing.
`tools/run_ablation.py` trains each condition separately.

Three conditions, one flag changing at a time, shared seed:

| condition | `use_augmentation` | `use_mixup` |
|---|---|---|
| `no_augmentation` | false | (forced off) |
| `augmentation_no_mixup` | true | false |
| `augmentation_and_mixup` | true | true |

### What you need

**1. The data.** The 9.8 GB `pack_out_15k.zip` — ask Ahmed for a Drive share
link. Put it in your own Drive; don't try to read it out of anyone else's.

**2. A Colab GPU runtime**, then:

```python
from google.colab import drive; drive.mount('/content/drive')
```

```bash
!git clone -b claude/medical-ai-augmentation-eval-7f4098 https://github.com/Sanad-Manga/Pediatric-Brain-Tumor-Model.git /content/repo
!pip -q install monai pyyaml scipy
```

```bash
!mkdir -p /content/data && unzip -q /content/drive/MyDrive/pack_out_15k.zip -d /content/data && find /content/data/pack_out_15k -name "*.npz" | wc -l
```

That count must be ~55,523. Unzip to `/content/data` (local disk), never read the
55k files over the Drive mount — it's slow enough to dominate the run.

**3. Checkpoints on Drive**, or a disconnect loses everything:

```bash
!mkdir -p /content/drive/MyDrive/ablation_ckpt && ln -sfn /content/drive/MyDrive/ablation_ckpt /content/repo/03_augmentation_eval/checkpoints
```

**4. Run it:**

```bash
%cd /content/repo/03_augmentation_eval
```

```bash
!python tools/run_ablation.py --device cuda --epochs 8 --num-workers 2 --cache /content/data/pack_out_15k
```

Safe to interrupt. Each condition resumes from its own checkpoint directory, and
conditions already in the CSV are skipped, so re-running the same command after a
disconnect picks up where it stopped. That will happen — plan for it.

### Cost warning — read before launching

At the config's current `model.width: 48` this is three runs of eight epochs over
24k slices. That is roughly as much GPU time as Ahmed's entire main run, and it
will not fit in a free account's quota.

**Talk to Ahmed before starting.** The ablation asks whether augmentation helps,
and a smaller network answers that just as well — width 16 is about 9× less
compute. Either he adds a `--width` flag or you agree on a reduced `--epochs` /
`--max-steps` budget. Whatever you use has to be stated next to the table,
because it is not the width the shipped model uses.

### Reporting it

The absolute Dice values here will land well below the shipped model's
0.599 / 0.521 / 0.827. They are short comparable runs, not the model's
performance — **the difference between rows is the finding, the absolute numbers
are not**. Quoting an ablation Dice as the model's score understates it badly.

`use_federation` and `use_domain_adaptation` are false in every row. That is a
result, not an omission — see "What NOT to do" below. Write one sentence saying
so rather than leaving the columns looking untested.

Send back `results/ablation_results.csv` plus the three `history.json` files.

## Rules

- **Don't touch `src/train.py` or `config.yaml`.** Ahmed is editing both for AMP
  and the training run. Your knobs are all CLI flags — you never need to.
- `src/inference.py` is shared with the demo. Keep changes additive.
- **Don't regenerate `05_frontend_demo/data/roc_cache.json`.** It gets rebuilt
  exactly once, by Ahmed, after the training run. Two people rebuilding it gives
  two competing sets of "the real numbers".
- Report every result as a **delta against the epoch-16 baseline**, not an
  absolute. When the width-48 checkpoint arrives we re-run your settings on it
  and immediately see whether the gains stack.
- Branch per person; PR to `main`. Don't push to Ahmed's branch mid-run.

## What NOT to do

**Domain adaptation / CORAL is dead — don't start it.** `hospitalA.json` (53),
`hospitalB.json` (92) and `heldout.json` (82) sum to exactly 227, the full
subject count, are disjoint, and their subject IDs interleave. They are a
*partition of one cohort*, not separate institutions — `BRIEF.md` says as much
("federated clients are simulated via manifests"). There is no site shift to
correct, and no site labels in the data with which to measure one. That's worth
one honest paragraph in the report's limitations section, which is a better
deliverable than a port of 3D CORAL. Writing that paragraph is a real task; take
it if you want it.

Also skip: training anything, more epochs, 3D U-Net, federated learning. All
either measured as dead ends or blocked on GPU.

## Runtime

CPU eval over 82 subjects is slow — tens of minutes per configuration, and
`tune_postproc.py` runs 8 of them. Start it early and let it run; don't discover
the runtime at 11pm.
