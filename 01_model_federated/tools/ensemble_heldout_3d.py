#!/usr/bin/env python
"""Held-out Dice for an ensemble of N 3D federated checkpoints, averaging
softmax probabilities (not weights, not logits) -- the same technique that
took the 2D pipeline from 0.71 (best single model) to 0.754 (ensemble).

Built to test the two checkpoints from the 2026-09-22/23 overnight run
(dice_ce and dice_focal, both epoch 80 -- neither variant ever beat this
epoch despite ~1000 more epochs each) without spending any more GPU time on
training first: this only costs one held-out pass.

Usage:
    python tools/ensemble_heldout_3d.py --checkpoints ckpt_a.pt ckpt_b.pt [...] \
        --cache-path "D:/NeuroPeds AI/cache_96cube"
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

from src.data import MODALITIES, _zscore_normalize, load_manifest  # noqa: E402
from src.model import FederatedUNet3D  # noqa: E402


def _load_metrics_module():
    """See eval_heldout_3d.py: 01_model_federated and 03_augmentation_eval
    both have a top-level package literally named `src` -- putting both on
    sys.path makes `import src.metrics` resolve to whichever one comes
    first, silently. Load 03's metrics.py by explicit file path instead."""
    path = SEC03 / "src" / "metrics.py"
    spec = importlib.util.spec_from_file_location("brats_peds_metrics_03", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_metrics = _load_metrics_module()
REGION_ORDER, aggregate, dice_regions = _metrics.REGION_ORDER, _metrics.aggregate, _metrics.dice_regions


def load_model(checkpoint_path: Path, device: str) -> FederatedUNet3D:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = FederatedUNet3D()
    model.load_state_dict(payload["model_state"])
    model.to(device).eval()
    print(f"loaded epoch {payload.get('epoch')} loss_kind={payload.get('loss_kind')} "
          f"| {checkpoint_path}", flush=True)
    return model


def ensemble_probs(models: list[FederatedUNet3D], x: torch.Tensor) -> torch.Tensor:
    probs_sum = None
    with torch.no_grad():
        for m in models:
            logits, _features = m(x)
            probs = torch.softmax(logits, dim=1)
            probs_sum = probs if probs_sum is None else probs_sum + probs
    return probs_sum / len(models)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--cache-path", required=True)
    ap.add_argument("--manifest", default=str(SEC01.parent / "00_shared" / "manifests" / "heldout.json"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    cache_path = Path(args.cache_path)
    models = [load_model(Path(c), device) for c in args.checkpoints]
    subjects = load_manifest(args.manifest)

    scores, missing = [], []
    t0 = time.time()
    for n, sid in enumerate(subjects, 1):
        npz_path = cache_path / f"{sid}.npz"
        if not npz_path.is_file():
            missing.append(sid)
            continue
        with np.load(npz_path) as data:
            x = np.stack([_zscore_normalize(data[m].astype(np.float32)) for m in MODALITIES], axis=0)
            true = data["seg"].astype(np.int64)
        x_t = torch.from_numpy(x)[None].to(device)
        probs = ensemble_probs(models, x_t)
        pred = torch.argmax(probs, dim=1)[0].cpu().numpy().astype(np.int64)
        scores.append(dice_regions(pred, true))
        if n % 20 == 0 or n == len(subjects):
            print(f"  {n}/{len(subjects)} subjects scored ({time.time()-t0:.0f}s elapsed)", flush=True)

    if not scores:
        print("no subjects scored (cache missing?)", file=sys.stderr)
        return 1

    agg = aggregate(scores)
    mean = sum(agg[f"dice_{r}"] for r in REGION_ORDER) / len(REGION_ORDER)
    result = {
        "checkpoints": args.checkpoints,
        "n_subjects_scored": len(scores),
        "n_subjects_missing_from_cache": len(missing),
        "missing_subjects": missing,
        "dice": {f"dice_{r}": agg[f"dice_{r}"] for r in REGION_ORDER},
        "mean": mean,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    print("RESULT_JSON:" + json.dumps(result))
    print(f"\n{'  '.join(f'{r}={agg[f'dice_{r}']:.4f}' for r in REGION_ORDER)}  mean={mean:.4f}  "
          f"({len(scores)}/{len(subjects)} subjects, {len(missing)} missing from cache)")
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
