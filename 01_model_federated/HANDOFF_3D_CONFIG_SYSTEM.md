# Handoff: bring a real config system to 3D training

No deadline pressure — this is backlog work, do it whenever. Fully buildable and testable without
the real dataset (`--data-mode dummy` exists in `run.py` for exactly this).

## The problem, confirmed by reading the code

`03_augmentation_eval` (the 2D section) already has a real hyperparameter config:
`03_augmentation_eval/config.yaml`. Read that file first — it's the template. It covers model
width/depth, loss class weights, LR schedule, checkpoint-selection metric, sampler settings, and a
full augmentation block, all in one YAML file, with `--config path.yaml` to load it.

`01_model_federated` (the 3D section) has **none of that**:
- Network width/depth is hardcoded in `01_model_federated/src/model.py` line ~34:
  `channels=(16, 32, 64, 128, 256)` — not a CLI flag, not configurable at all.
- There's no class-weight flag. `03_augmentation_eval`'s default is
  `class_weights: [0.2, 3.0, 1.0, 1.0, 1.0]` (background, ET, NETC, CC, ED) — ET is up-weighted 15x.
  3D has no equivalent.
- `run.py` has 19 CLI args total (loss choice, LR, epochs, manifest, some augmentation dropout flags)
  — compare to `03_augmentation_eval/run.py`'s ~51, most of which come from its config file.

Every future 3D experiment (loss weights, width, anything) currently means editing source files
directly. That's the gap to close.

## What to build

**Mirror `03_augmentation_eval/config.yaml`'s pattern — don't invent a new one.** Read
`03_augmentation_eval/src/config.py` too (the `Config` dataclass + `load_config()` that reads the
YAML) — that's the loader shape to copy.

1. New file `01_model_federated/config.yaml`. At minimum, port these sections from 03's file,
   adapted to what 3D actually has:
   ```yaml
   model:
     width: 16        # first-level channel width; every level is width * 2**i (currently hardcoded)
     depth: 5          # number of levels -- channels=(16,32,64,128,256) today is 5 levels; keep "depth"
                        # meaning the same thing it means in 03's config.yaml (level count, not downsample count)

   loss:
     kind: dice_ce      # dice_ce | dice_focal (already a CLI choice, now also settable here)
     class_weights: null   # null = uniform (today's behavior); [0.2, 3.0, 1.0, 1.0, 1.0] to up-weight ET like 2D does

   schedule:
     kind: cosine        # 2D already validated this beats a fixed LR late in training
     min_lr: 1.0e-5

   augmentation:           # 01_model_federated/src/augment3d.py's build_transforms3d() already takes
     flip_prob: 0.5        # all of these as keyword args with the same defaults shown here --
     rotate_prob: 0.3       # this section is mostly just exposing what already exists as CLI/config,
     # ... (read augment3d.py's build_transforms3d signature for the full parameter list)
   ```
2. `01_model_federated/src/model.py`: change the hardcoded `channels=(16, 32, 64, 128, 256)` to be
   derived from `width` and `depth` config values — `channels = tuple(width * 2**i for i in range(depth))`,
   matching the comment already in 03's config.yaml ("every level is width * 2**i").
3. `01_model_federated/src/train_single.py`: wire `class_weights` into whatever builds the loss
   function (currently uniform; look at how `03_augmentation_eval` passes `class_weights` into its
   loss construction for the pattern).
4. `run.py`: add `--config path.yaml` (see `03_augmentation_eval/run.py` line ~324 for the flag
   itself); config values become new defaults, and existing CLI flags (`--loss`, `--lr`, etc.) should
   still override the config when passed explicitly — same precedence order 03 already uses, don't
   invent a different one.

## What NOT to do

- Don't remove or rename any existing CLI flag — this has to stay backward compatible with everything
  that already calls `run.py` (the shift-training and sequence-dropout work both use the current flags
  directly, not a config file).
- Don't touch `03_augmentation_eval` — it's the reference, not something to change.
- Don't add augmentation parameters beyond what `build_transforms3d()` in `augment3d.py` already
  accepts — this task exposes existing knobs via config, it doesn't add new ones.

## Definition of done

- `python run.py train --data-mode dummy --config config.yaml --epochs 2` runs successfully with the
  default config (should behave identically to today's hardcoded defaults — width 16, depth 5,
  uniform loss weights).
- Overriding `width: 64` in a config file and re-running produces a model with the expected parameter
  count (spot-check: width 16 → ~4.9M params per the existing checkpoint comments; width 64 should
  scale up accordingly — verify with a quick `sum(p.numel() for p in model.parameters())`).
- A class-weight override actually changes the loss value on a dummy batch (cheapest possible check
  that the wiring reached the loss function, not just the config parser).
- Existing CLI-only usage (no `--config` passed at all) is unaffected — this is the part most likely
  to silently break something if the precedence order is wrong, so check it explicitly.

## One coordination note

`HANDOFF_SEED_AND_CHECKPOINT_SELECTION.md` also touches `run.py`'s argument list. If both PRs are in
flight at once, expect to rebase against whichever lands first — not a real conflict, just a heads-up.

## Who to ask

Ping Ahmed if the precedence rules between CLI flags and config file values need a judgment call, or
if it's unclear which of `augment3d.py`'s existing parameters belong in the config file.
