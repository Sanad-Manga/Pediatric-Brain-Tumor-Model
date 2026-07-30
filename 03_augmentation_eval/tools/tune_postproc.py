"""Sweep TTA / post-processing settings on VALIDATION patients only.

Why this exists as a script rather than a loop over ``run.py eval``: the tuning
set and the reporting set are different sets, and getting that wrong silently
produces a number that looks like a result but is not one. This script reads the
validation subject list *out of the checkpoint itself* (``train.py`` stores
``val_subjects`` in every checkpoint), so the split cannot drift from the one
the model actually trained under, and it refuses to touch the held-out manifest.

Usage::

    python tools/tune_postproc.py --checkpoint checkpoints/overnight_run/best.pt \\
        --cache D:/Processed_2D --device cpu

Then take the single best row and measure it ONCE on held-out::

    python run.py eval --experiment-name tta_postproc --checkpoint <ckpt> \\
        --cache <cache> --tta --et-boost 1.4 --min-component-voxels 50

If the held-out gain is much smaller than the validation gain, that is the
honest answer: the setting overfit the validation patients. Report it anyway.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import evaluate                                    # noqa: E402
from src.config import load_config                          # noqa: E402
from src.metrics import REGION_ORDER                        # noqa: E402

#: (label, eval.tta, eval.postproc overrides)
SWEEP = [
    ("baseline (argmax)",      False, {}),
    ("tta",                    True,  {}),
    ("tta + et_boost 1.2",     True,  {"et_boost": 1.2}),
    ("tta + et_boost 1.5",     True,  {"et_boost": 1.5}),
    ("tta + et_boost 2.0",     True,  {"et_boost": 2.0}),
    ("tta + drop <50 vox",     True,  {"min_component_voxels": 50}),
    ("tta + drop <200 vox",    True,  {"min_component_voxels": 200}),
    ("tta + largest lesion",   True,  {"keep_largest_wt": True}),
]


def read_val_subjects(checkpoint: Path) -> list[str]:
    """The validation patients this checkpoint was selected against.

    Newer checkpoints carry the split themselves. Older ones do not, so fall
    back to the ``history.json`` written alongside them -- which is where the
    split has always been recorded.
    """
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and payload.get("val_subjects"):
        return list(payload["val_subjects"])

    history = checkpoint.parent / "history.json"
    if history.exists():
        with open(history, "r", encoding="utf-8") as fh:
            subjects = json.load(fh).get("val_subjects")
        if subjects:
            print(f"(validation split recovered from {history.name})")
            return list(subjects)

    raise SystemExit(
        f"cannot recover the validation split for {checkpoint}: no 'val_subjects' "
        f"in the checkpoint and no usable {history}. Tuning without it would mean "
        "guessing at the split, so stop here rather than tuning on the wrong "
        "patients."
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--cache", default=None, help="override paths.cache_2d")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--config", default=None)
    ap.add_argument("--eval-plane", default=None,
                    choices=["axial", "coronal", "both"])
    args = ap.parse_args()

    checkpoint = Path(args.checkpoint)
    val_subjects = read_val_subjects(checkpoint)
    print(f"tuning on {len(val_subjects)} validation patients from {checkpoint.name}")
    print("these are NOT held-out patients; nothing printed here is a final result\n")

    rows = []
    for label, tta, postproc in SWEEP:
        cfg = load_config(args.config)
        if args.eval_plane:
            cfg.eval["plane"] = args.eval_plane
        cfg.eval["tta"] = tta
        cfg.eval["postproc"] = postproc

        row = evaluate.run(
            cfg,
            experiment_name=label,
            checkpoint=checkpoint,
            cache_dir=Path(args.cache) if args.cache else None,
            device=args.device,
            subjects=val_subjects,
            write_csv=False,          # tuning numbers never enter the ablation CSV
        )
        rows.append((label, row))

    base = rows[0][1]
    width = max(len(label) for label, _ in rows)
    print("\n" + "=" * (width + 46))
    print(f"{'setting':<{width}}  " + "  ".join(f"{r:>7}" for r in REGION_ORDER) +
          "     mean   d_mean")
    print("-" * (width + 46))
    for label, row in rows:
        delta = row["mean_dice"] - base["mean_dice"]
        print(f"{label:<{width}}  " +
              "  ".join(f"{row[f'dice_{r}']:7.4f}" for r in REGION_ORDER) +
              f"  {row['mean_dice']:7.4f}  {delta:+7.4f}")
    print("=" * (width + 46))
    print("\nPick ONE row. Measure it once on held-out via `run.py eval`.")
    print("Watch ET specifically -- mean_dice can rise while ET falls, which is")
    print("the exact failure that made epoch 25 look better than epoch 16.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
