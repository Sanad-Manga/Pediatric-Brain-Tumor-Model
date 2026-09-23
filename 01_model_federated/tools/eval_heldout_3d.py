#!/usr/bin/env python
"""Held-out Dice for a 3D federated checkpoint. This section (01_model_federated)
has never had an evaluation script -- train_single_client only ever returned a
training-loss curve, so there has been no way to tell whether a 3D run is
actually learning to segment anything. This fills that gap.

Reuses 03_augmentation_eval/src/metrics.py's dice_regions()/aggregate() --
those are pure numpy over integer label volumes, shape-agnostic, and already
match this project's official ET/NC/WT region definitions (00_shared/CONTRACTS.md),
so there is no reason to reimplement Dice a second time for 3D.

Usage:
    python tools/eval_heldout_3d.py --checkpoint checkpoints/<run_id>/epoch_N.pt \
        --cache-path "D:/NeuroPeds AI/cache_96cube" \
        --manifest ../00_shared/manifests/heldout.json
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

from src.data import MODALITIES, VOLUME_SIZE, _zscore_normalize, load_manifest  # noqa: E402
from src.model import FederatedUNet3D  # noqa: E402


def _load_metrics_module():
    """03_augmentation_eval and this section both have a top-level package
    literally named `src` -- putting both on sys.path makes `import src.metrics`
    resolve to whichever one sys.path lists first, silently, and if it picks
    the wrong one the failure looks like a missing module, not a name clash.
    Loading 03's metrics.py by explicit file path sidesteps the collision
    entirely instead of relying on sys.path ordering.
    """
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
    print(f"loaded epoch {payload.get('epoch')} | avg_loss {payload.get('avg_loss')} "
          f"| checkpoint {checkpoint_path}")
    return model


def score_subject(model: FederatedUNet3D, cache_path: Path, subject_id: str,
                  device: str) -> dict[str, float] | None:
    npz_path = cache_path / f"{subject_id}.npz"
    if not npz_path.is_file():
        return None
    with np.load(npz_path) as data:
        x = np.stack([_zscore_normalize(data[m].astype(np.float32)) for m in MODALITIES], axis=0)
        true = data["seg"].astype(np.int64)

    x_t = torch.from_numpy(x)[None].to(device)  # (1, 4, 96, 96, 96)
    with torch.no_grad():
        logits, _features = model(x_t)
        pred = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.int64)

    return dice_regions(pred, true)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--cache-path", required=True)
    ap.add_argument("--manifest", default=str(SEC01.parent / "00_shared" / "manifests" / "heldout.json"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-json", default=None, help="optional path to write the summary to")
    args = ap.parse_args()

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    cache_path = Path(args.cache_path)
    model = load_model(Path(args.checkpoint), device)
    subjects = load_manifest(args.manifest)

    scores, missing = [], []
    t0 = time.time()
    for n, sid in enumerate(subjects, 1):
        result = score_subject(model, cache_path, sid, device)
        if result is None:
            missing.append(sid)
            continue
        scores.append(result)
        if n % 20 == 0 or n == len(subjects):
            print(f"  {n}/{len(subjects)} subjects scored ({time.time()-t0:.0f}s elapsed)", flush=True)

    if not scores:
        print("no subjects scored (cache missing?)", file=sys.stderr)
        return 1

    agg = aggregate(scores)
    mean = sum(agg[f"dice_{r}"] for r in REGION_ORDER) / len(REGION_ORDER)
    result = {
        "checkpoint": str(args.checkpoint),
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
