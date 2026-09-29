# Handoff: reproducible runs + pick the best 3D checkpoint automatically

Two small, independent fixes to `01_model_federated` (the 3D training code). Neither needs the real
dataset to build or test — `run.py` already has a `--data-mode dummy` mode that fabricates synthetic
volumes, made exactly for this kind of change. No deadline pressure on these; do them whenever.

## Fix 1: no run is reproducible

**The problem, confirmed by reading the code, not assumed:** neither `run.py` in this section nor in
`03_augmentation_eval` calls `torch.manual_seed`, `np.random.seed`, or `random.seed` anywhere. Run
the exact same command twice and you get two different models. This makes every experiment result in
this project a little bit luck, and makes bugs impossible to reproduce reliably.

**What to do:**
1. Add `p.add_argument("--seed", type=int, default=None)` to `build_parser()` in `run.py` (see the
   existing arguments around line 17 for the pattern).
2. At the top of `main()`, right after `args = parse_args()`, if `args.seed is not None`:
   ```python
   import random
   random.seed(args.seed)
   np.random.seed(args.seed)
   torch.manual_seed(args.seed)
   torch.cuda.manual_seed_all(args.seed)  # harmless if no CUDA device
   ```
3. Do the same in `03_augmentation_eval/run.py` (it has the same gap) — check `p_train.add_argument`
   calls for the pattern to match, and add it to the `train` subcommand specifically.

**Definition of done:** run the same command twice with `--seed 1337 --data-mode dummy --epochs 2`,
diff the two resulting `history.json` losses — they should be identical, not just close. Without
`--seed` (or with two different seed values), behavior should be unchanged from today.

## Fix 2: 3D never picks its best checkpoint — it just uses whatever epoch it stopped at

**The problem, confirmed by reading the code:** `01_model_federated/src/checkpoint.py` saves every
epoch and prunes old ones by recency (`prune_old_checkpoints`), but there is no concept anywhere of
"best" — no evaluation happens during training, and the deployed checkpoints
(`checkpoints_best/session2_A_focal_seqdrop_final_ep262.pt` etc.) are literally whichever epoch
training happened to be on when the run ended.

**Compare to `03_augmentation_eval`,** which already solves this: `config.yaml` has a `selection:
metric: min_region` option that, after each epoch, scores the checkpoint and keeps the one that's
best on its *worst* region — not the mean, which lets a strong WT score hide a bad ET score. That's
what needs porting here.

**What to do:**
1. There's already a standalone evaluator for this section:
   `01_model_federated/tools/eval_heldout_3d.py --checkpoint <path> --cache-path <96cube cache> --manifest <manifest.json>`
   — it prints per-region Dice. Read it before writing new evaluation code; don't duplicate it.
2. Add a `--eval-every N` flag to `run.py` (0 = never, matching today's behavior by default so nothing
   breaks for existing callers). Every `N` epochs, call the same scoring logic `eval_heldout_3d.py`
   uses, keep a running "best worst-region score", and when a new epoch beats it, copy that
   checkpoint to `<checkpoint_dir>/best.pt` (don't delete the per-epoch files — `prune_old_checkpoints`
   already manages those).
3. Print what changed (`"epoch 87: new best (worst region NC 0.71 > previous 0.68)"`) so it's visible
   in training logs, same spirit as `03`'s existing logging.

**What you can and can't test yourself:** the selection *logic* (given a stream of fake scores, does
it correctly track the best-worst-region one) is pure code — write it with unit tests, no data
needed. Wiring it to the real evaluator needs `D:/NeuroPeds AI/cache_96cube`, which lives locally on
this machine — write the integration, but the final real-data check happens with Ahmed, not on your
own machine, unless you already have a copy of that cache.

**Definition of done:** on `--data-mode dummy`, a short run with `--eval-every 2` produces a `best.pt`
in the checkpoint directory and the log lines showing when it changed. The selection function itself
has tests covering: first epoch always becomes best, a worse worst-region score doesn't overwrite a
better one, a tie doesn't flip-flop.

## One coordination note

Both fixes touch `run.py`'s argument list and the top of `main()`. If someone else is also working on
`HANDOFF_3D_CONFIG_SYSTEM.md` (a bigger change to the same file), coordinate on who merges first —
expect to rebase, not a real conflict, just worth a heads-up before you both open PRs.

## Who to ask

Ping Ahmed for access to `D:/NeuroPeds AI/cache_96cube` if you want to run the real integration check
yourself, or for the manifest paths if `00_shared/manifests/` doesn't already have what you need.
