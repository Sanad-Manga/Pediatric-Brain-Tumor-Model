#!/usr/bin/env python
"""Simulate what a real 2D-acquired clinical MRI stack would look like on the
96^3 grid this project trains on, for a first, honest look at how badly a
3D-trained model degrades on that kind of input -- WITHOUT real clinical scan
data, which is still pending ethics/institutional clearance (see conversation
2026-09-23: "a lot of politics and ethics" -- checking with the other doctors
later).

**This is a proxy, not a validated model of any real scanner protocol.**
Nothing here has been checked against an actual Egyptian clinic's DICOM
headers. Treat every number this produces as "roughly what to expect if the
real protocol is in this ballpark" -- a reason to keep pursuing the real
acquisition parameters, not a replacement for them.

What it simulates: our 3D pipeline trains on the 96^3 cache, resampled
(tools/build_96cube_cache.py) from near-isotropic ~1mm 3D-acquired BraTS
volumes. A real 2D-acquired clinical sequence (axial 2D FSE/TSE T1/T2/FLAIR,
the common routine protocol) instead has thick slices (commonly ~4-6mm) with
a gap between them (commonly ~1mm), while in-plane resolution stays close to
native. This function approximates that: along one axis of the 96^3 volume
(ASSUMPTION: axis 0 -- this project's resample script never verified which
array axis corresponds to which anatomical plane per subject, so "through-
plane" here is an assumption, not a checked fact), it averages together
groups of adjacent slices (partial-volume blur, standard effect of a thick
slice) and drops slices to simulate the gap, then nearest-fills back to 96
so the degraded volume still matches the shape the model expects (a real
clinical stack resampled onto this grid would have exactly this kind of
"real data every few slices, interpolated/duplicated in between" structure).

Usage:
    python tools/simulate_2d_degradation.py --checkpoint <path> \
        --cache-path "D:/NeuroPeds AI/cache_96cube" \
        --slices-per-thick-slab 4   # ASSUMPTION: ~1.6mm/voxel native (240/96,
                                     # 155/96 blended) * 4 ~= 6mm slice spacing,
                                     # in the ballpark of a real 5mm+1mm-gap
                                     # 2D protocol -- not a verified figure
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
    path = SEC03 / "src" / "metrics.py"
    spec = importlib.util.spec_from_file_location("brats_peds_metrics_03", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_metrics = _load_metrics_module()
REGION_ORDER, aggregate, dice_regions = _metrics.REGION_ORDER, _metrics.aggregate, _metrics.dice_regions


def degrade_volume(vol: np.ndarray, axis: int, slices_per_thick_slab: int) -> np.ndarray:
    """(D,H,W) -> same shape, degraded along `axis` to simulate thick, gapped
    2D-acquisition slices. Groups of `slices_per_thick_slab` real slices become
    one averaged "thick slice" (partial-volume blur); every other group is then
    dropped entirely (the gap) and nearest-filled from its surviving neighbour,
    so roughly half the through-plane information is genuinely gone, not just
    blurred -- closer to what a real sparse 2D stack resampled onto this grid
    would actually contain than blur alone would be.
    """
    vol = np.moveaxis(vol, axis, 0)
    n = vol.shape[0]
    out = np.empty_like(vol)
    slab = slices_per_thick_slab
    i = 0
    slab_idx = 0
    while i < n:
        j = min(i + slab, n)
        if slab_idx % 2 == 0:
            avg = vol[i:j].mean(axis=0, keepdims=True)
            out[i:j] = avg  # thick slice: real signal, blurred
        else:
            out[i:j] = out[i - 1:i]  # gap: nearest-fill from the last real thick slice
        i = j
        slab_idx += 1
    return np.moveaxis(out, 0, axis)


def degrade_label(seg: np.ndarray, axis: int, slices_per_thick_slab: int) -> np.ndarray:
    """Same slab/gap structure as degrade_volume, but majority-vote per slab
    (not mean -- labels are discrete) so the "ground truth as if reconstructed
    from a thick/gapped stack" stays a valid label map, and nearest-fill the
    gap slabs the same way."""
    seg = np.moveaxis(seg, axis, 0)
    n = seg.shape[0]
    out = np.empty_like(seg)
    slab = slices_per_thick_slab
    i = 0
    slab_idx = 0
    while i < n:
        j = min(i + slab, n)
        if slab_idx % 2 == 0:
            block = seg[i:j]
            flat = block.reshape(block.shape[0], -1)
            # per-voxel-column majority label across the slab
            maj = np.apply_along_axis(lambda col: np.bincount(col).argmax(), 0, flat)
            out[i:j] = maj.reshape((1,) + block.shape[1:])
        else:
            out[i:j] = out[i - 1:i]
        i = j
        slab_idx += 1
    return np.moveaxis(out, 0, axis)


def load_model(checkpoint_path: Path, device: str) -> FederatedUNet3D:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = FederatedUNet3D()
    model.load_state_dict(payload["model_state"])
    model.to(device).eval()
    print(f"loaded epoch {payload.get('epoch')} loss_kind={payload.get('loss_kind')} "
          f"| {checkpoint_path}", flush=True)
    return model


def score_subject(model, cache_path: Path, subject_id: str, device: str,
                  degrade: bool, axis: int, slab: int) -> dict | None:
    npz_path = cache_path / f"{subject_id}.npz"
    if not npz_path.is_file():
        return None
    with np.load(npz_path) as data:
        arrays = {m: data[m].astype(np.float32) for m in MODALITIES}
        true = data["seg"].astype(np.int64)
    if degrade:
        arrays = {m: degrade_volume(v, axis, slab) for m, v in arrays.items()}
        true_for_scoring = degrade_label(true, axis, slab)
    else:
        true_for_scoring = true

    x = np.stack([_zscore_normalize(arrays[m]) for m in MODALITIES], axis=0)
    x_t = torch.from_numpy(x)[None].to(device)
    with torch.no_grad():
        logits, _features = model(x_t)
        pred = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.int64)
    # Score against the ORIGINAL clean ground truth, not the degraded one --
    # the clinical question is "how well does the model still find the real
    # tumor", not "does it match a degraded version of the truth".
    return dice_regions(pred, true)


def run_eval(model, cache_path: Path, subjects: list[str], device: str,
            degrade: bool, axis: int, slab: int, label: str) -> dict | None:
    scores, missing = [], []
    t0 = time.time()
    for n, sid in enumerate(subjects, 1):
        result = score_subject(model, cache_path, sid, device, degrade, axis, slab)
        if result is None:
            missing.append(sid)
            continue
        scores.append(result)
        if n % 20 == 0 or n == len(subjects):
            print(f"  [{label}] {n}/{len(subjects)} scored ({time.time()-t0:.0f}s)", flush=True)
    if not scores:
        return None
    agg = aggregate(scores)
    mean = sum(agg[f"dice_{r}"] for r in REGION_ORDER) / len(REGION_ORDER)
    return {"label": label, "dice": {f"dice_{r}": agg[f"dice_{r}"] for r in REGION_ORDER},
           "mean": mean, "n_scored": len(scores), "n_missing": len(missing)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--cache-path", required=True)
    ap.add_argument("--manifest", default=str(SEC01.parent / "00_shared" / "manifests" / "heldout.json"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--axis", type=int, default=0,
                    help="ASSUMPTION: which array axis is 'through-plane' -- never verified per-subject")
    ap.add_argument("--slices-per-thick-slab", type=int, default=4,
                    help="ASSUMPTION: real slice-thickness/gap parameters unknown; this is a placeholder")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    cache_path = Path(args.cache_path)
    model = load_model(Path(args.checkpoint), device)
    subjects = load_manifest(args.manifest)

    clean = run_eval(model, cache_path, subjects, device, degrade=False,
                     axis=args.axis, slab=args.slices_per_thick_slab, label="clean_3d")
    degraded = run_eval(model, cache_path, subjects, device, degrade=True,
                        axis=args.axis, slab=args.slices_per_thick_slab, label="simulated_2d")

    result = {
        "checkpoint": str(args.checkpoint),
        "axis": args.axis,
        "slices_per_thick_slab": args.slices_per_thick_slab,
        "note": "simulated_2d is a proxy for a real 2D-acquired scan, NOT validated "
                "against any real clinic's acquisition parameters",
        "clean_3d": clean,
        "simulated_2d": degraded,
    }
    if clean and degraded:
        drop = clean["mean"] - degraded["mean"]
        result["mean_dice_drop"] = drop
        print("RESULT_JSON:" + json.dumps(result))
        print(f"\nclean 3D input      : {'  '.join(f'{k}={v:.4f}' for k,v in clean['dice'].items())}  mean={clean['mean']:.4f}")
        print(f"simulated 2D input   : {'  '.join(f'{k}={v:.4f}' for k,v in degraded['dice'].items())}  mean={degraded['mean']:.4f}")
        print(f"mean Dice drop under simulated 2D degradation: {drop:.4f}")
    else:
        print("RESULT_JSON:" + json.dumps(result))
        print("one or both passes failed to score any subjects", file=sys.stderr)
        return 1
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
