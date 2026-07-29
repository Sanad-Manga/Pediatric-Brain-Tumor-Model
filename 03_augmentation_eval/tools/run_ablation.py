"""The ablation: one training run per condition, then one held-out score each.

This is section 03's deliverable and it cannot be produced any other way. Running
``run.py eval`` several times against a single checkpoint writes several rows
that differ only in the flag columns while the Dice columns come from the same
weights -- a table that looks like an ablation and measures nothing. Every row
here comes from its own training run.

Conditions are limited to what is actually implemented:

===========================  ================  ===========
condition                    use_augmentation  use_mixup
===========================  ================  ===========
``no_augmentation``          false             (forced off)
``augmentation_no_mixup``    true              false
``augmentation_and_mixup``   true              true
===========================  ================  ===========

``use_federation`` and ``use_domain_adaptation`` stay false in every row, and
that is a result rather than an omission: federated training was never run, and
hospitalA/hospitalB/heldout are a partition of a single cohort (53 + 92 + 82 =
227, the full subject count, with interleaved IDs), so there is no site shift for
domain adaptation to correct and no site labels with which to measure one. Say
that in the report instead of leaving the columns looking untested.

**The Dice values here are not the shipped model's.** These are short runs at a
fixed budget, chosen so the conditions are comparable to each other, not so they
reach the best achievable score. Quoting an ablation number as the model's
performance understates it. The comparison between rows is the finding; the
absolute values are not.

Usage::

    python tools/run_ablation.py --device cuda --epochs 8 \\
        --cache /content/data/pack_out_15k

Safe to interrupt: each condition resumes from its own checkpoint directory, and
conditions already present in the results CSV are skipped.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import evaluate, train                              # noqa: E402
from src.config import load_config                           # noqa: E402
from src.metrics import REGION_ORDER                         # noqa: E402

#: (experiment_name, use_augmentation, use_mixup)
CONDITIONS = [
    ("no_augmentation",        False, False),
    ("augmentation_no_mixup",  True,  False),
    ("augmentation_and_mixup", True,  True),
]


def completed_experiments(csv_path: Path) -> set[str]:
    """Experiment names already in the results CSV."""
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return set()
    import csv as _csv

    with open(csv_path, "r", encoding="utf-8", newline="") as fh:
        return {row["experiment_name"] for row in _csv.DictReader(fh)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default=None, help="override paths.cache_2d")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--epochs", type=int, default=8,
                    help="per condition; identical across conditions by design")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="cap steps per epoch to bound the total budget")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--val-fraction", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=1337,
                    help="shared by every condition; do not vary it per row")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--out-csv", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--rerun", action="store_true",
                    help="re-measure conditions already in the CSV")
    args = ap.parse_args()

    cache = Path(args.cache) if args.cache else None
    csv_path = Path(args.out_csv) if args.out_csv else load_config(args.config).resolve("results_csv")
    done = set() if args.rerun else completed_experiments(csv_path)

    print(f"ablation: {len(CONDITIONS)} conditions x {args.epochs} epochs, seed {args.seed}")
    print(f"results:  {csv_path}")
    if done:
        print(f"skipping (already measured): {', '.join(sorted(done))}")
    print()

    rows = []
    for name, use_aug, use_mixup in CONDITIONS:
        if name in done:
            continue

        print(f"--- {name}: use_augmentation={use_aug} use_mixup={use_mixup}")
        cfg = load_config(args.config, use_augmentation=use_aug, use_mixup=use_mixup)

        summary = train.run(
            cfg,
            run_id=f"ablation_{name}",
            epochs=args.epochs,
            batch_size=args.batch_size,
            val_fraction=args.val_fraction,
            max_steps_per_epoch=args.max_steps,
            cache_dir=cache,
            device=args.device,
            seed=args.seed,              # identical across conditions on purpose
            num_workers=args.num_workers,
        )
        checkpoint = Path(summary["checkpoint_dir"]) / "best.pt"

        # Score on held-out with post-processing OFF. A tuned threshold that
        # helped one condition and not another would be measuring the tuning,
        # not the condition.
        cfg_eval = load_config(args.config, use_augmentation=use_aug, use_mixup=use_mixup)
        cfg_eval.eval = dict(cfg_eval.eval)
        cfg_eval.eval["tta"] = False
        cfg_eval.eval["postproc"] = {}

        row = evaluate.run(
            cfg_eval,
            experiment_name=name,
            checkpoint=checkpoint,
            cache_dir=cache,
            csv_path=csv_path,
            device=args.device,
        )
        rows.append((name, row))
        print()

    if not rows:
        print("nothing to do; every condition is already in the CSV (--rerun to redo)")
        return 0

    width = max(len(n) for n, _ in rows)
    print("=" * (width + 40))
    print(f"{'condition':<{width}}  " + "  ".join(f"{r:>7}" for r in REGION_ORDER) + "     mean")
    print("-" * (width + 40))
    for name, row in rows:
        print(f"{name:<{width}}  " +
              "  ".join(f"{row[f'dice_{r}']:7.4f}" for r in REGION_ORDER) +
              f"  {row['mean_dice']:7.4f}")
    print("=" * (width + 40))
    print(f"\nwrote {csv_path}")
    print("These are short comparable runs, NOT the shipped model's scores.")
    print("Report the differences between rows, not the absolute values.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
