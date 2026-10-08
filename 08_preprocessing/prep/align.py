"""Rigid alignment of T1n, T2-FLAIR and T2w to the same patient's T1c (see ../SPEC.md).

R* assumes the four sequences line up voxel for voxel; on our stress test a 2 mm misalignment between T2w/FLAIR and T1c
cost R* 0.078 Dice. This step registers each sequence to T1c with a 6-parameter rigid transform (Mattes mutual
information, three resolution levels, initialised from image geometry) and resamples it onto the T1c grid.

    python -m prep.align --t1c a-t1c.nii.gz --t1n a-t1n.nii.gz --t2f a-t2f.nii.gz --t2w a-t2w.nii.gz --out-dir out/
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

SEQUENCES = ("t1c", "t1n", "t2f", "t2w")
REFERENCE = "t1c"
SEED = 20261008


def align_to_reference(moving: sitk.Image, fixed: sitk.Image, seed: int = SEED) -> tuple[sitk.Image, sitk.Euler3DTransform, float]:
    """Rigidly register `moving` to `fixed`; return (moving resampled onto fixed's grid, the transform, final metric).

    The transform maps points of the fixed (reference) space into the moving image: output(x) = moving(T(x)).
    Deterministic for the same inputs: fixed sampling seed and a single registration thread."""
    fixed_f = sitk.Cast(fixed, sitk.sitkFloat32)
    moving_f = sitk.Cast(moving, sitk.sitkFloat32)
    transform = sitk.Euler3DTransform(sitk.CenteredTransformInitializer(
        fixed_f, moving_f, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY))

    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.2, seed)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsRegularStepGradientDescent(learningRate=1.0, minStep=1e-4, numberOfIterations=300,
                                                 relaxationFactor=0.5, gradientMagnitudeTolerance=1e-8)
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetShrinkFactorsPerLevel([4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([2.0, 1.0, 0.0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    reg.SetInitialTransform(transform, inPlace=True)
    reg.SetNumberOfThreads(1)

    # Multi-threaded metric evaluation sums in a varying order (differences at ~1e-14 that can grow over iterations);
    # one thread for the whole registration makes repeated runs bit-identical.
    threads = sitk.ProcessObject.GetGlobalDefaultNumberOfThreads()
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    try:
        reg.Execute(fixed_f, moving_f)
    finally:
        sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(threads)
    metric = float(reg.GetMetricValue())
    if not math.isfinite(metric):
        raise RuntimeError("registration did not converge (metric is not finite)")
    aligned = sitk.Resample(moving, fixed, transform, sitk.sitkLinear, 0.0, moving.GetPixelID())
    return aligned, transform, metric


def transform_summary(t: sitk.Euler3DTransform) -> dict:
    rx, ry, rz, tx, ty, tz = t.GetParameters()
    return {"rotation_deg": [round(math.degrees(a), 3) for a in (rx, ry, rz)],
            "translation_mm": [round(v, 3) for v in (tx, ty, tz)], "center_mm": [round(v, 3) for v in t.GetCenter()]}


def align_sequences(paths: dict[str, str | Path], seed: int = SEED) -> tuple[dict[str, sitk.Image], dict]:
    """Align T1n, T2-FLAIR and T2w to T1c. Returns ({sequence: image on the T1c grid}, {sequence: report}).
    T1c is returned unchanged."""
    images = {}
    for name in SEQUENCES:
        p = Path(paths[name])
        if not p.is_file():
            raise FileNotFoundError(f"missing input: {p}")
        images[name] = sitk.ReadImage(str(p))
    fixed = images[REFERENCE]
    out, report = {REFERENCE: fixed}, {REFERENCE: {"reference": True}}
    for name in SEQUENCES:
        if name == REFERENCE:
            continue
        try:
            aligned, t, metric = align_to_reference(images[name], fixed, seed)
        except RuntimeError as exc:
            raise RuntimeError(f"{name}: {exc}") from exc
        out[name] = aligned
        report[name] = {**transform_summary(t), "metric": round(metric, 5)}
    return out, report


def residual_error_mm(points: np.ndarray, *transforms: sitk.Transform) -> np.ndarray:
    """Distance (mm) each physical point moves under the composition of `transforms` (applied in the given order).
    For a perfect recovery of a known misalignment the composition is the identity and every distance is 0."""
    out = np.empty(len(points))
    for i, p in enumerate(points):
        q = tuple(float(v) for v in p)
        for t in transforms:
            q = t.TransformPoint(q)
        out[i] = math.dist(q, p)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Rigidly align T1n, T2-FLAIR and T2w to T1c.")
    for name in SEQUENCES:
        ap.add_argument(f"--{name}", required=True, help=f"{name} NIfTI file")
    ap.add_argument("--out-dir", required=True, help="directory for the aligned files and alignment.json")
    ap.add_argument("--prefix", default="aligned", help="output file prefix (default: aligned)")
    args = ap.parse_args(argv)
    try:
        images, report = align_sequences({n: getattr(args, n) for n in SEQUENCES})
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, img in images.items():
        sitk.WriteImage(img, str(out / f"{args.prefix}-{name}.nii.gz"))
    (out / "alignment.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
