"""Issue #65: on one patient, does our alignment step or the scan's own alignment line FLAIR/T2w up with T1c better?

For each sequence this writes a figure a human can read (T1c in grey, the sequence's edges drawn over it, the scan's
own position next to our step's, plus a zoom on where the two disagree most) and a verdict from `prep.judge`, which
settles the comparison with evidence the step did not optimise. Numbers and picture go together: the picture shows
*where* the two differ, the verdict says *which is right*, because a sub-2 mm difference between a sharp scan and a
resampled one is not something the eye judges reliably.

    python tools/visual_check.py --t1c <p>-t1c.nii.gz --t1n <p>-t1n.nii.gz --t2f <p>-t2f.nii.gz \
        --t2w <p>-t2w.nii.gz --out-dir out/00127 --label BraTS-PED-00127-000

Writes out/<label>-<seq>.png per sequence and out/verdicts.json. `--ants` adds a second, slower opinion from a metric
that is not mutual information at all; `--seg` centres the slices on the tumour instead of the brain.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import SimpleITK as sitk

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from prep import align, judge  # noqa: E402

ORIGINAL_COLOUR = (1.0, 0.25, 0.25, 0.95)   # the scan's own alignment (BraTS)
ALIGNED_COLOUR = (0.25, 0.95, 1.0, 0.95)    # our step's alignment
EDGE_PERCENTILE = 90.0                      # strongest 10% of edges inside the brain are drawn
ZOOM_HALF_WIDTH = 28                        # voxels either side of the worst-disagreement point
PLANES = ("axial", "coronal", "sagittal")


def edges(img: sitk.Image, mask: np.ndarray, percentile: float = EDGE_PERCENTILE) -> tuple[np.ndarray, float]:
    """`img`'s edge strength, zeroed outside `mask`, and the level that keeps the strongest `100 - percentile` %."""
    g = judge.gradient_magnitude(img)
    inside = g[mask]
    g = np.where(mask, g, 0.0)
    return g, float(np.percentile(inside, percentile)) if inside.size else 1.0


def worst_disagreement(a_edges: np.ndarray, b_edges: np.ndarray, mask: np.ndarray) -> tuple[int, int, int]:
    """The (z, y, x) voxel where the two edge maps disagree most, blurred so a single voxel cannot win."""
    diff = (a_edges ^ b_edges).astype(np.float32)
    blurred = sitk.GetArrayFromImage(sitk.SmoothingRecursiveGaussian(sitk.GetImageFromArray(diff), 4.0))
    blurred[~mask] = -1.0
    return tuple(int(v) for v in np.unravel_index(int(np.argmax(blurred)), blurred.shape))


def _slice(vol: np.ndarray, plane: str, centre: tuple[int, int, int]) -> np.ndarray:
    z, y, x = centre
    if plane == "axial":
        return vol[z, :, :]
    if plane == "coronal":
        return vol[:, y, :]
    return vol[:, :, x]


def _panel(ax, base: np.ndarray, overlays, window: tuple[float, float], title: str, crop=None) -> None:
    """T1c in grey with each candidate's edges drawn over it as a contour line, as in an ITK-SNAP overlay."""
    if crop is not None:
        (r0, r1), (c0, c1) = crop
        base = base[r0:r1, c0:c1]
        overlays = [(g[r0:r1, c0:c1], level, colour) for g, level, colour in overlays]
    ax.imshow(base.T, cmap="gray", vmin=window[0], vmax=window[1], origin="lower", interpolation="nearest")
    for gradient, level, colour in overlays:
        if gradient.max() > level:
            ax.contour(gradient.T, levels=[level], colors=[colour], linewidths=0.7)
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])


def figure(path: Path, label: str, seq: str, t1c: sitk.Image, original: sitk.Image, aligned: sitk.Image,
           mask: np.ndarray, verdict: dict, centre: tuple[int, int, int] | None = None) -> tuple[int, int, int]:
    """Draw T1c with each candidate's edges over it, per plane, plus a zoom where the two differ most."""
    t1c_arr = sitk.GetArrayFromImage(sitk.Cast(t1c, sitk.sitkFloat32))
    window = tuple(np.percentile(t1c_arr[mask], [1, 99.5])) if mask.any() else (0.0, 1.0)
    g_orig, lv_orig = edges(original, mask)
    g_align, lv_align = edges(aligned, mask)
    worst = worst_disagreement(g_orig > lv_orig, g_align > lv_align, mask)
    centre = centre or worst

    def overlays(plane, at):
        return ((_slice(g_orig, plane, at), lv_orig, ORIGINAL_COLOUR),
                (_slice(g_align, plane, at), lv_align, ALIGNED_COLOUR))

    fig, axes = plt.subplots(4, 3, figsize=(10.5, 13.0))
    for row, plane in enumerate(PLANES):
        base = _slice(t1c_arr, plane, centre)
        o, a = overlays(plane, centre)
        _panel(axes[row][0], base, [o], window, f"{plane}: scan's own alignment")
        _panel(axes[row][1], base, [a], window, f"{plane}: our step's alignment")
        _panel(axes[row][2], base, [o, a], window, f"{plane}: both")

    z, y, x = worst
    half = ZOOM_HALF_WIDTH
    crop = ((max(0, y - half), min(t1c_arr.shape[1], y + half)), (max(0, x - half), min(t1c_arr.shape[2], x + half)))
    base = _slice(t1c_arr, "axial", worst)
    o, a = overlays("axial", worst)
    _panel(axes[3][0], base, [o], window, f"zoom z={z}: scan's own", crop)
    _panel(axes[3][1], base, [a], window, f"zoom z={z}: our step", crop)
    _panel(axes[3][2], base, [o, a], window, f"zoom z={z}: both", crop)

    scores = verdict["scores"]
    lines = ["  ".join(f"{k.replace('_', ' ')}: {scores['original'][k]:.4f} vs {scores['aligned'][k]:.4f}"
                       f" -> {verdict['criterion_votes'][k]}" for k in keys)
             for keys in (judge.CRITERIA[:2], judge.CRITERIA[2:])]
    opinions = "; ".join(f"re-registered ({m}): {o['gap_to_original_mm']} mm from the scan's own position, "
                         f"{o['gap_to_aligned_mm']} mm from ours -> {o['vote']}"
                         for m, o in verdict["second_opinions"].items())
    fig.suptitle(
        f"{label}  {seq.upper()} over T1c   |   VERDICT: {verdict['verdict'].upper()}\n"
        f"our step would move this sequence by {verdict['disagreement_mm']} mm   "
        f"(scan's own position vs ours, both resampled once)\n"
        + "\n".join(lines) + f"\n{opinions}\n"
        f"red = {seq} where the scan has it   cyan = {seq} where our step puts it",
        fontsize=8.5, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return worst


def check_patient(paths: dict[str, Path], out_dir: Path, label: str, sequences: tuple[str, ...],
                  reregister: tuple[str, ...], seg: Path | None = None) -> dict:
    t1c = sitk.ReadImage(str(paths["t1c"]))
    mask_img = judge.brain_mask(t1c)
    mask = sitk.GetArrayFromImage(mask_img) > 0
    centre = None
    if seg is not None:
        lab = sitk.GetArrayFromImage(sitk.ReadImage(str(seg))) > 0
        if lab.any():
            centre = tuple(int(round(v)) for v in np.argwhere(lab).mean(0))
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {"label": label, "sequences": {}}
    for seq in sequences:
        t0 = time.time()
        moving = sitk.ReadImage(str(paths[seq]))
        aligned, ours, _ = align.align_to_reference(moving, t1c)
        verdict = judge.judge(moving, t1c, ours, reregister=reregister)
        png = out_dir / f"{label}-{seq}.png"
        worst = figure(png, label, seq, t1c, moving, aligned, mask, verdict, centre)
        verdict.update({"our_transform": align.transform_summary(ours), "figure": png.name,
                        "worst_disagreement_zyx": list(worst), "seconds": round(time.time() - t0, 1)})
        results["sequences"][seq] = verdict
        print(f"{label} {seq}: {verdict['verdict']} "
              f"(step would move it {verdict['disagreement_mm']} mm) -> {png.name}")
    (out_dir / "verdicts.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in align.SEQUENCES:
        ap.add_argument(f"--{name}", required=name == "t1c", help=f"{name} NIfTI file")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--label", default="patient", help="name used in file names and figure titles")
    ap.add_argument("--sequences", default="t2f,t2w", help="comma-separated sequences to check (default t2f,t2w)")
    ap.add_argument("--ants", action="store_true",
                    help="also re-register with ANTS local correlation (slow; its value at both candidates is scored "
                         "either way)")
    ap.add_argument("--seg", help="optional segmentation; centres the slices on the tumour")
    args = ap.parse_args(argv)

    sequences = tuple(s.strip() for s in args.sequences.split(",") if s.strip())
    unknown = [s for s in sequences if s not in align.SEQUENCES or s == align.REFERENCE]
    if unknown:
        print(f"error: cannot check {unknown} (expected any of {align.SEQUENCES[1:]})", file=sys.stderr)
        return 1
    paths = {}
    for name in ("t1c",) + sequences:
        value = getattr(args, name)
        if value is None:
            print(f"error: --{name} is required to check {sequences}", file=sys.stderr)
            return 1
        paths[name] = Path(value)
        if not paths[name].is_file():
            print(f"error: missing input: {paths[name]}", file=sys.stderr)
            return 1
    try:
        check_patient(paths, Path(args.out_dir), args.label, sequences,
                      ("joint", "ants") if args.ants else ("joint",),
                      Path(args.seg) if args.seg else None)
    except (RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
