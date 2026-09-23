#!/usr/bin/env python
"""One-time resample of the raw BraTS-PEDs NIfTI volumes to the 96^3 cache
format src/data.py's real mode expects (see 00_shared/CONTRACTS.md "Data
pipeline"). CPU-only -- never touches the GPU, safe to run alongside anything
else using it.

Input layout (one folder per subject, standard BraTS-PEDs naming):
    <raw_dir>/<subject_id>/<subject_id>-{t1c,t1n,t2f,t2w,seg}.nii.gz

Output: one flat file per subject, <out_dir>/<subject_id>.npz, with arrays
"t1c" "t1n" "t2f" "t2w" (96x96x96 float16, RAW un-normalized intensities --
normalization happens in data.py, not here, per the contract) and "seg"
(96x96x96 uint8, labels 0-4).

Resampling: trilinear for the 4 image modalities, NEAREST for the
segmentation mask -- interpolating a label map would blend adjacent classes
into fractional, meaningless values at every boundary voxel, exactly where
Dice is decided.

Usage:
    python tools/build_96cube_cache.py --raw-dir "D:/NeuroPeds AI/PKG - BraTS-PEDs-v1/BraTS-PEDs-v1/Training" --out-dir "D:/NeuroPeds AI/cache_96cube"
    python tools/build_96cube_cache.py ... --subjects BraTS-PED-00001-000 BraTS-PED-00002-000   # smoke test a few first
    python tools/build_96cube_cache.py ... --workers 8   # default: all logical cores
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

VOLUME_SIZE = 96
MODALITIES = ("t1c", "t1n", "t2f", "t2w")
VALID_LABELS = {0, 1, 2, 3, 4}


def _load_nii(path: Path) -> np.ndarray:
    img = nib.load(str(path))
    return np.asarray(img.dataobj, dtype=np.float32)


def _resample(volume: np.ndarray, mode: str) -> np.ndarray:
    """(D0,H0,W0) -> (96,96,96) via torch's own interpolate; CPU tensor, no
    dependency beyond what's already installed for training."""
    t = torch.from_numpy(volume)[None, None]  # (1,1,D,H,W)
    kw = {"align_corners": False} if mode == "trilinear" else {}
    out = F.interpolate(t, size=(VOLUME_SIZE,) * 3, mode=mode, **kw)
    return out[0, 0].numpy()


def _subject_paths(raw_dir: Path, subject_id: str) -> dict[str, Path]:
    sdir = raw_dir / subject_id
    paths = {m: sdir / f"{subject_id}-{m}.nii.gz" for m in MODALITIES}
    paths["seg"] = sdir / f"{subject_id}-seg.nii.gz"
    return paths


def discover_subjects(raw_dir: Path) -> list[str]:
    return sorted(p.name for p in raw_dir.iterdir()
                  if p.is_dir() and (p / f"{p.name}-seg.nii.gz").exists())


def process_one(raw_dir_str: str, out_dir_str: str, subject_id: str) -> tuple[str, str, str]:
    """Returns (subject_id, status, detail). status in {"ok","skip","error"}."""
    raw_dir, out_dir = Path(raw_dir_str), Path(out_dir_str)
    out_path = out_dir / f"{subject_id}.npz"
    if out_path.exists():
        return subject_id, "skip", "already present"

    paths = _subject_paths(raw_dir, subject_id)
    missing = [k for k, p in paths.items() if not p.exists()]
    if missing:
        return subject_id, "error", f"missing files: {missing}"

    try:
        arrays: dict[str, np.ndarray] = {}
        native_shape = None
        for m in MODALITIES:
            vol = _load_nii(paths[m])
            if native_shape is None:
                native_shape = vol.shape
            elif vol.shape != native_shape:
                return subject_id, "error", (f"{m} shape {vol.shape} != {native_shape} "
                                             f"(modalities must share one grid)")
            arrays[m] = _resample(vol, "trilinear").astype(np.float16)

        seg = _load_nii(paths["seg"])
        if seg.shape != native_shape:
            return subject_id, "error", f"seg shape {seg.shape} != {native_shape}"
        seg_resampled = np.rint(_resample(seg, "nearest")).astype(np.uint8)
        bad = set(np.unique(seg_resampled).tolist()) - VALID_LABELS
        if bad:
            return subject_id, "error", f"seg has out-of-range labels after resample: {bad}"
        arrays["seg"] = seg_resampled

        for m in MODALITIES:
            if arrays[m].shape != (VOLUME_SIZE,) * 3:
                return subject_id, "error", f"{m} resampled to {arrays[m].shape}, expected 96^3"

        # np.savez_compressed silently APPENDS ".npz" to any path that doesn't
        # already end in ".npz" -- passing the ".tmp" path as a string would
        # actually write "<...>.npz.tmp.npz" and the rename below would then
        # fail to find it. Writing through an open file handle avoids that.
        tmp_path = out_dir / f"{subject_id}.npz.tmp"
        with open(tmp_path, "wb") as fh:
            np.savez_compressed(fh, **arrays)
        tmp_path.rename(out_path)  # atomic-ish on the same filesystem: no partial file survives a kill
        return subject_id, "ok", f"native {native_shape}"
    except Exception as exc:  # noqa: BLE001 - report per-subject, never abort the whole batch
        return subject_id, "error", f"{type(exc).__name__}: {exc}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", required=True, help="...BraTS-PEDs-v1/Training")
    ap.add_argument("--out-dir", required=True, help="e.g. D:/NeuroPeds AI/cache_96cube")
    ap.add_argument("--subjects", nargs="+", default=None,
                    help="only these subject IDs (smoke test); default: every subject with a seg file")
    ap.add_argument("--workers", type=int, default=None, help="default: all logical CPU cores")
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    if not raw_dir.is_dir():
        print(f"FATAL: raw dir not found: {raw_dir}", file=sys.stderr)
        return 2
    out_dir.mkdir(parents=True, exist_ok=True)

    subjects = args.subjects or discover_subjects(raw_dir)
    if not subjects:
        print(f"FATAL: no subjects with a seg file found under {raw_dir}", file=sys.stderr)
        return 2
    print(f"{len(subjects)} subjects | raw={raw_dir} | out={out_dir} | "
          f"workers={args.workers or 'all cores'}", flush=True)

    t0 = time.time()
    ok, skipped, errors = [], [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_one, str(raw_dir), str(out_dir), sid): sid for sid in subjects}
        for n, fut in enumerate(as_completed(futures), 1):
            sid, status, detail = fut.result()
            if status == "ok":
                ok.append(sid)
            elif status == "skip":
                skipped.append(sid)
            else:
                errors.append((sid, detail))
                print(f"  [{n}/{len(subjects)}] ERROR {sid}: {detail}", flush=True)
            if n % 20 == 0 or n == len(subjects):
                elapsed = time.time() - t0
                print(f"  [{n}/{len(subjects)}] ok={len(ok)} skip={len(skipped)} "
                      f"error={len(errors)} | {elapsed:.0f}s elapsed", flush=True)

    summary = {
        "raw_dir": str(raw_dir), "out_dir": str(out_dir),
        "requested": len(subjects), "ok": len(ok), "skipped": len(skipped),
        "errors": [{"subject_id": s, "detail": d} for s, d in errors],
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    (out_dir / "_build_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nDONE in {summary['elapsed_seconds']:.0f}s: {len(ok)} ok, {len(skipped)} already "
          f"present, {len(errors)} errors. Summary: {out_dir / '_build_summary.json'}")
    return 1 if errors and not ok and not skipped else 0


if __name__ == "__main__":
    sys.exit(main())
