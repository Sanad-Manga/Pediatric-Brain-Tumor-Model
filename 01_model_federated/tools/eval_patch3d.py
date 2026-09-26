#!/usr/bin/env python
"""Held-out Dice for patch-trained 3D checkpoints, scored at FULL resolution.

Runs sliding-window inference (src/patch_infer.py) on whole volumes from the
full-resolution cache (tools/build_fullres_cache.py) and scores against the original
1 mm ground truth, so the numbers are directly comparable with the 2D pipeline's.
Several checkpoints are ensembled by averaging their class probabilities.

Also reports the ET cleanup rule: when the total predicted ET is tiny, relabel it as
non-enhancing tumour (label 1 -> 2). NC = {1,2,3} and WT = {1,2,3,4} contain both
labels, so only the ET score changes. T = 0 is "no rule".

Dice comes from 03_augmentation_eval/src/metrics.py, loaded by file path exactly as
tools/eval_heldout_3d.py does: that section and this one both have a top-level
package called `src`, and putting both on sys.path silently resolves to one of them.

Usage:
    python tools/eval_patch3d.py --checkpoint checkpoints/<run>/epoch_N.pt [more ...] \
        --cache-path "D:/NeuroPeds AI/cache_fullres" --manifest ../00_shared/manifests/heldout.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

SEC01 = Path(__file__).resolve().parent.parent
SEC03 = SEC01.parent / "03_augmentation_eval"
sys.path.insert(0, str(SEC01))

from src.data import load_manifest  # noqa: E402
from src.model import FederatedUNet3D  # noqa: E402
from src.patch_data import zscore_crop  # noqa: E402
from src.patch_infer import predict_volume  # noqa: E402

FLIP_VIEWS = ((), (0,), (1,), (2,))


def _load_metrics_module():
    path = SEC03 / "src" / "metrics.py"
    spec = importlib.util.spec_from_file_location("brats_peds_metrics_03", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_metrics = _load_metrics_module()
REGION_ORDER, aggregate, dice_regions = _metrics.REGION_ORDER, _metrics.aggregate, _metrics.dice_regions


def apply_et_min_voxels(pred: np.ndarray, min_voxels: int) -> np.ndarray:
    """Relabel predicted ET (1) as non-enhancing (2) when the total ET count is
    below `min_voxels`. Returns a new array; the input is never modified."""
    out = np.array(pred, copy=True)
    count = int((out == 1).sum())
    if min_voxels > 0 and 0 < count < min_voxels:
        out[out == 1] = 2
    return out


def load_model(checkpoint_path: Path, device) -> FederatedUNet3D:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = FederatedUNet3D()
    model.load_state_dict(payload["model_state"])
    return model.to(device).eval()


def evaluate(
    checkpoints,
    cache_path,
    manifest,
    roi_size=(128, 128, 128),
    flips=((),),
    et_min_voxels_list=(0, 500),
    save_probs_dir=None,
    device=None,
    limit=None,
) -> dict:
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    cache = Path(cache_path)
    checkpoints = [Path(c) for c in ([checkpoints] if isinstance(checkpoints, (str, Path)) else checkpoints)]
    models = [load_model(c, device) for c in checkpoints]
    subjects = load_manifest(str(manifest))
    if limit:
        subjects = subjects[:limit]
    if save_probs_dir:
        Path(save_probs_dir).mkdir(parents=True, exist_ok=True)

    per_threshold: dict[int, list[dict]] = {int(t): [] for t in et_min_voxels_list}
    missing: list[str] = []
    t0 = time.time()
    for n, sid in enumerate(subjects, 1):
        img_path, seg_path, stats_path = (cache / f"{sid}.{s}" for s in ("img.npy", "seg.npy", "stats.npy"))
        if not (img_path.is_file() and seg_path.is_file() and stats_path.is_file()):
            missing.append(sid)
            continue
        image = zscore_crop(np.load(img_path).astype(np.float32), np.load(stats_path))
        truth = np.load(seg_path).astype(np.int8)
        image_t = torch.from_numpy(image)
        probs = sum(predict_volume(m, image_t, roi_size, flips=flips, device=device) for m in models) / len(models)
        pred = torch.argmax(probs, dim=0).numpy().astype(np.int8)
        if save_probs_dir:
            with open(Path(save_probs_dir) / f"{sid}.npy", "wb") as fh:
                np.save(fh, probs.numpy().astype(np.float16))
        for threshold in per_threshold:
            per_threshold[threshold].append(dice_regions(apply_et_min_voxels(pred, threshold), truth))
        if n % 10 == 0 or n == len(subjects):
            print(f"  {n}/{len(subjects)} scored ({time.time() - t0:.0f}s)", file=sys.stderr, flush=True)

    scored = len(next(iter(per_threshold.values()))) if per_threshold else 0
    if not scored:
        raise RuntimeError("no subjects scored (cache missing?)")
    results = {}
    for threshold, rows in per_threshold.items():
        agg = aggregate(rows)
        mean = sum(agg[f"dice_{r}"] for r in REGION_ORDER) / len(REGION_ORDER)
        results[str(threshold)] = {"mean": mean, **{r: agg[f"dice_{r}"] for r in REGION_ORDER}}
    return {
        "checkpoints": [str(c) for c in checkpoints], "roi_size": list(roi_size),
        "n_flip_views": len(flips), "n_subjects_scored": scored, "n_subjects_missing": len(missing),
        "missing_subjects": missing, "results": results,
        "elapsed_seconds": round(time.time() - t0, 1),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", nargs="+", required=True)
    ap.add_argument("--cache-path", required=True)
    ap.add_argument("--manifest", default=str(SEC01.parent / "00_shared" / "manifests" / "heldout.json"))
    ap.add_argument("--roi-size", type=int, nargs=3, default=[128, 128, 128])
    ap.add_argument("--flips", choices=["none", "axes"], default="none",
                    help="'axes' = identity + a flip along each spatial axis (4 views)")
    ap.add_argument("--et-min-voxels", type=int, nargs="+", default=[0, 500])
    ap.add_argument("--save-probs-dir", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)

    result = evaluate(
        args.checkpoint, args.cache_path, args.manifest, roi_size=tuple(args.roi_size),
        flips=FLIP_VIEWS if args.flips == "axes" else ((),), et_min_voxels_list=args.et_min_voxels,
        save_probs_dir=args.save_probs_dir, device=args.device, limit=args.limit,
    )
    print("RESULT_JSON:" + json.dumps(result))
    for threshold, r in result["results"].items():
        print(f"ET rule T={threshold:>5}: mean={r['mean']:.4f}  ET={r['ET']:.3f} NC={r['NC']:.3f} WT={r['WT']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
