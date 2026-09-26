"""python -m rstar --t1c a.nii.gz --t1n b.nii.gz --t2f c.nii.gz --t2w d.nii.gz --out seg.nii.gz [--json report.json] [--models-root DIR]

A sequence whose flag is omitted is treated as absent. Exit codes: 0 = written (status 'ok' or 'review'), 2 = the input broke the contract.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .config import SEQUENCES, RStarConfig
from .contract import ContractError


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rstar", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in SEQUENCES:
        p.add_argument(f"--{name}", default=None, help=f"NIfTI file for {name}; omit if the sequence was not acquired")
    p.add_argument("--out", required=True, help="output label NIfTI (uint8, labels 0-4)")
    p.add_argument("--json", default=None, help="write a JSON report (status, mode, warnings, diagnostics)")
    p.add_argument("--models-root", default=None, help="directory holding the checkpoints (default: env RSTAR_MODELS_ROOT, else the repo root)")
    return p


def main(argv=None, segmenter=None) -> int:
    args = build_parser().parse_args(argv)
    import nibabel as nib

    paths = {name: getattr(args, name) for name in SEQUENCES}
    try:
        if segmenter is None:
            from .pipeline import RStarSegmenter

            segmenter = RStarSegmenter(RStarConfig(models_root=Path(args.models_root) if args.models_root else None))
        result, affine = segmenter.segment_paths(paths)
    except ContractError as exc:
        print(f"rstar: input rejected: {exc}", file=sys.stderr)
        return 2
    nib.save(nib.Nifti1Image(result.labels.astype(np.uint8), affine), args.out)
    if args.json:
        report = {"status": result.status, "mode": result.mode, "warnings": result.warnings, "diagnostics": result.diagnostics}
        Path(args.json).write_text(json.dumps(_jsonable(report), indent=2), encoding="utf-8")
    for w in result.warnings:
        print(f"rstar: warning: {w}", file=sys.stderr)
    print(f"rstar: {result.mode}, status {result.status}, wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
