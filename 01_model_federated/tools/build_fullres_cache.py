#!/usr/bin/env python
"""One-time conversion of the raw BraTS-PEDs NIfTI volumes into the FULL-RESOLUTION
cache that src/patch_data.py reads. CPU-only; never touches the GPU.

Unlike tools/build_96cube_cache.py there is NO resampling and NO reorientation:
the arrays keep the source grid (240x240x155 at 1 mm for BraTS-PEDs), so a patch
cut from the cache is exactly what a scanner would have produced, and predictions
made on it line up voxel-for-voxel with the original ground truth.

Input layout (one folder per subject, standard BraTS-PEDs naming):
    <raw_dir>/<sid>/<sid>-{t1c,t1n,t2f,t2w,seg}.nii.gz

Output, per subject, in <out_dir>:
    <sid>.img.npy    float16 (4, X, Y, Z), channels t1c,t1n,t2f,t2w, RAW intensities
    <sid>.seg.npy    uint8   (X, Y, Z), labels 0-4
    <sid>.stats.npy  float32 (4, 2): mean, std over the voxels > 0 of each raw
                     modality ((0, 1) when a modality has no voxel > 0). Stored
                     because normalising a small crop with its own statistics would
                     not match normalising the whole volume.

Usage:
    python tools/build_fullres_cache.py --raw-dir "D:/NeuroPeds AI/PKG - BraTS-PEDs-v1/BraTS-PEDs-v1/Training" \
        --out-dir "D:/NeuroPeds AI/cache_fullres"
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

MODALITIES = ("t1c", "t1n", "t2f", "t2w")
VALID_LABELS = {0, 1, 2, 3, 4}
SUFFIXES = ("img.npy", "seg.npy", "stats.npy")


def _load_nii(path: Path) -> np.ndarray:
    return np.asarray(nib.load(str(path)).dataobj, dtype=np.float32)


def _subject_paths(raw_dir: Path, subject_id: str) -> dict[str, Path]:
    sdir = raw_dir / subject_id
    paths = {m: sdir / f"{subject_id}-{m}.nii.gz" for m in MODALITIES}
    paths["seg"] = sdir / f"{subject_id}-seg.nii.gz"
    return paths


def discover_subjects(raw_dir: Path) -> list[str]:
    return sorted(p.name for p in raw_dir.iterdir()
                  if p.is_dir() and (p / f"{p.name}-seg.nii.gz").exists())


def compute_stats(image: np.ndarray) -> np.ndarray:
    """(4, X, Y, Z) raw -> (4, 2) float32 of [mean, std] over voxels > 0."""
    stats = np.zeros((image.shape[0], 2), dtype=np.float32)
    for c in range(image.shape[0]):
        brain = image[c] > 0
        if brain.any():
            values = image[c][brain].astype(np.float64)
            stats[c] = (values.mean(), values.std())
        else:
            stats[c] = (0.0, 1.0)
    return stats


def _output_paths(out_dir: Path, subject_id: str) -> list[Path]:
    return [out_dir / f"{subject_id}.{suffix}" for suffix in SUFFIXES]


def _tmp_path(final_path: Path) -> Path:
    return final_path.with_name(final_path.name + ".tmp")


def _write_tmp(tmp_path: Path, array: np.ndarray) -> None:
    # np.save appends ".npy" to a string path that lacks it, which would make the
    # rename later miss the file; writing through an open handle avoids that.
    with open(tmp_path, "wb") as fh:
        np.save(fh, array)


def process_one(raw_dir_str: str, out_dir_str: str, subject_id: str) -> tuple[str, str, str]:
    """Returns (subject_id, status, detail); status is one of ok / skip / error."""
    raw_dir, out_dir = Path(raw_dir_str), Path(out_dir_str)
    outputs = _output_paths(out_dir, subject_id)
    if all(p.exists() for p in outputs):
        return subject_id, "skip", "already present"

    tmp_files: list[Path] = []
    written: list[Path] = []
    try:
        paths = _subject_paths(raw_dir, subject_id)
        missing = [k for k, p in paths.items() if not p.exists()]
        if missing:
            return subject_id, "error", f"missing files: {missing}"

        volumes = [_load_nii(paths[m]) for m in MODALITIES]
        seg = _load_nii(paths["seg"])
        shapes = {v.shape for v in volumes} | {seg.shape}
        if len(shapes) != 1 or len(seg.shape) != 3:
            return subject_id, "error", f"shape mismatch among the five files: {sorted(shapes)}"

        seg_int = np.rint(seg).astype(np.int64)
        bad = set(np.unique(seg_int).tolist()) - VALID_LABELS
        if bad:
            return subject_id, "error", f"seg has labels outside 0-4: {sorted(bad)[:8]}"

        image = np.stack(volumes, axis=0)
        image16 = image.astype(np.float16)
        if not np.isfinite(image16).all():
            return subject_id, "error", "an intensity does not fit in float16"
        arrays = [image16, seg_int.astype(np.uint8), compute_stats(image)]

        out_dir.mkdir(parents=True, exist_ok=True)
        for final_path, array in zip(outputs, arrays):
            tmp_path = _tmp_path(final_path)
            tmp_files.append(tmp_path)  # registered BEFORE writing so a failed write is still cleaned up
            _write_tmp(tmp_path, array)
        for tmp_path, final_path in zip(tmp_files, outputs):
            tmp_path.replace(final_path)
            written.append(final_path)
        return subject_id, "ok", f"shape {tuple(seg.shape)}"
    except Exception as exc:  # noqa: BLE001 - one bad patient must not abort the batch
        for p in tmp_files + written:
            p.unlink(missing_ok=True)
        return subject_id, "error", f"{type(exc).__name__}: {exc}"


def build(raw_dir: Path, out_dir: Path, subjects: list[str] | None = None,
          workers: int | None = None) -> dict:
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    subjects = subjects or discover_subjects(raw_dir)

    t0 = time.time()
    ok, skipped, errors = [], [], []

    def _record(result: tuple[str, str, str]) -> None:
        sid, status, detail = result
        if status == "ok":
            ok.append(sid)
        elif status == "skip":
            skipped.append(sid)
        else:
            errors.append((sid, detail))
            print(f"  ERROR {sid}: {detail}", flush=True)

    if workers is not None and workers <= 1:
        for sid in subjects:
            _record(process_one(str(raw_dir), str(out_dir), sid))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(process_one, str(raw_dir), str(out_dir), sid) for sid in subjects]
            for n, fut in enumerate(as_completed(futures), 1):
                _record(fut.result())
                if n % 20 == 0 or n == len(subjects):
                    print(f"  [{n}/{len(subjects)}] ok={len(ok)} skip={len(skipped)} "
                          f"error={len(errors)} | {time.time() - t0:.0f}s", flush=True)

    summary = {
        "raw_dir": str(raw_dir), "out_dir": str(out_dir), "requested": len(subjects),
        "ok": len(ok), "skipped": len(skipped),
        "errors": [{"subject_id": s, "detail": d} for s, d in errors],
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    (out_dir / "_build_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--subjects", nargs="+", default=None, help="only these subject IDs (smoke test)")
    ap.add_argument("--workers", type=int, default=None, help="default: all logical CPU cores")
    args = ap.parse_args(argv)

    raw_dir = Path(args.raw_dir)
    if not raw_dir.is_dir():
        print(f"FATAL: raw dir not found: {raw_dir}", file=sys.stderr)
        return 2
    subjects = args.subjects or discover_subjects(raw_dir)
    if not subjects:
        print(f"FATAL: no subjects with a seg file under {raw_dir}", file=sys.stderr)
        return 2
    print(f"{len(subjects)} subjects | raw={raw_dir} | out={args.out_dir}", flush=True)
    summary = build(raw_dir, Path(args.out_dir), subjects, args.workers)
    print(f"\nDONE in {summary['elapsed_seconds']:.0f}s: {summary['ok']} ok, {summary['skipped']} skipped, "
          f"{len(summary['errors'])} errors. Summary: {Path(args.out_dir) / '_build_summary.json'}")
    return 1 if summary["errors"] and not summary["ok"] and not summary["skipped"] else 0


if __name__ == "__main__":
    sys.exit(main())
