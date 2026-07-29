"""Precompute ROC / AUC on the held-out patients and cache it to JSON.

Scoring 82 held-out subjects takes ~80s, which is far too slow to run inside a
Streamlit rerun. This script does it once; the pages read the cache. Re-run it
whenever the checkpoint changes:

    python -m utils.build_metrics_cache --checkpoint <path> --cache <slice cache>

The cache records which checkpoint produced it, so a page can refuse to show
numbers that belong to a different model rather than silently mismatching.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .inference import DEFAULT_CHECKPOINT, available_subjects, checkpoint_metadata
from .metrics import collect_scores

CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "roc_cache.json"


def held_out_subjects(manifest_dir: Path) -> list[str]:
    with open(manifest_dir / "heldout.json", "r", encoding="utf-8") as fh:
        blob = json.load(fh)
    return blob["subjects"] if isinstance(blob, dict) else blob


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    ap.add_argument("--cache", default="D:/pack_out", help="2D slice cache")
    ap.add_argument("--manifests", default=None)
    ap.add_argument("--plane", default="axial")
    ap.add_argument("--slices-per-subject", type=int, default=6)
    ap.add_argument("--max-pixels", type=int, default=4000)
    ap.add_argument("--out", default=str(CACHE_PATH))
    args = ap.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[2]
    manifests = Path(args.manifests) if args.manifests else repo_root / "00_shared" / "manifests"

    present = set(available_subjects(args.cache))
    subjects = [s for s in held_out_subjects(manifests) if s in present]
    if not subjects:
        raise SystemExit("no held-out subjects found in the cache")

    print(f"scoring {len(subjects)} held-out subjects ({args.plane})…")
    res = collect_scores(args.checkpoint, Path(args.cache), subjects,
                         plane=args.plane, slices_per_subject=args.slices_per_subject,
                         max_pixels_per_slice=args.max_pixels)

    meta = checkpoint_metadata(args.checkpoint)
    payload = {
        "checkpoint": {k: meta[k] for k in
                       ("epoch", "epochs_completed", "best_mean_dice", "architecture")},
        "plane": args.plane,
        "n_subjects": len(subjects),
        "n_slices": res["n_slices"],
        "slices_per_subject": args.slices_per_subject,
        "max_pixels_per_slice": args.max_pixels,
        "evaluated_on": "held-out patients (never trained on)",
        "regions": {},
    }
    for region, v in res["regions"].items():
        if v is None:
            payload["regions"][region] = None
            continue
        # thin the curve for storage; the AUC is computed from the full curve
        keep = np.unique(np.linspace(0, len(v["fpr"]) - 1, 300).astype(int))
        payload["regions"][region] = {
            "auc": v["auc"],
            "fpr": [round(float(x), 6) for x in np.asarray(v["fpr"])[keep]],
            "tpr": [round(float(x), 6) for x in np.asarray(v["tpr"])[keep]],
            "n_pixels": v["n_pixels"],
            "n_positive": v["n_positive"],
            "prevalence": v["prevalence"],
            "dice": v["dice"],
            "sensitivity": v["sensitivity"],
            "specificity": v["specificity"],
            "precision": v["precision"],
            "hd95_median_mm": v["hd95_median_mm"],
            "hd95_n_slices": v["hd95_n_slices"],
            "tp": v["tp"], "fp": v["fp"], "fn": v["fn"], "tn": v["tn"],
        }
        def fmt(x, suffix=""):
            return "n/a" if x is None else f"{x:.4f}{suffix}"

        print(f"  {region}: AUC={fmt(v['auc'])}  Dice={fmt(v['dice'])}  "
              f"sens={fmt(v['sensitivity'])}  spec={fmt(v['specificity'])}  "
              f"HD95={fmt(v['hd95_median_mm'], 'mm')}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(f"wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
