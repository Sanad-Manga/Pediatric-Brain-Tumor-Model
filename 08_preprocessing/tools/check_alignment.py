"""Verification of the alignment step on BraTS-PEDs (SPEC Req 9-12). CPU only.

A. Recovery: move T2w and T2-FLAIR of 10 fresh-30 patients by a known random rigid transform (1-3 mm per axis,
   2-5 degrees per axis), align to T1c, measure residual error over 20,000 brain points.
B. Do no harm: align the untouched sequences; the recovered transform must be close to identity.
C. End to end: deployed R* (rstar command-line tool, CPU) on 5 patients: clean, misaligned (T2w and FLAIR moved
   2 mm + 3 degrees), and realigned; mean Dice (ET after R*'s 500 mm^3 rule / TC / WT).

    <nnunet_env python> tools/check_alignment.py [--skip-c]
"""
import argparse
import json
import math
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import SimpleITK as sitk

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from prep import align  # noqa: E402

RAW = Path(r"D:\NeuroPeds AI\PKG - BraTS-PEDs-v1\BraTS-PEDs-v1\Training")
FRESH = Path(r"C:\Users\ahmed\neuropeds_overnight\unused30.json")
RSTAR_DIR = HERE.parent / "06_rstar_inference"
MODELS = Path(r"C:\Users\ahmed\Pediatric-Brain-Tumor-Model")
RSTAR_PY = r"C:\Users\ahmed\AppData\Local\Programs\Python\Python314\python.exe"  # CPU torch; the GPU stays with nnU-Net
N_POINTS = 20000


def rigid(ref, rot_deg, shift):
    t = sitk.Euler3DTransform()
    t.SetCenter(ref.TransformContinuousIndexToPhysicalPoint([(s - 1) / 2 for s in ref.GetSize()]))
    t.SetRotation(*[math.radians(a) for a in rot_deg])
    t.SetTranslation([float(v) for v in shift])
    return t


def brain_points(ref_t1n, n=N_POINTS, seed=0):
    arr = sitk.GetArrayFromImage(ref_t1n)
    idx = np.argwhere(arr > 0)
    pick = idx[np.random.default_rng(seed).choice(len(idx), min(n, len(idx)), replace=False)]
    return np.array([ref_t1n.TransformIndexToPhysicalPoint([int(i) for i in p[::-1]]) for p in pick])


def files(sid):
    return {m: RAW / sid / f"{sid}-{m}.nii.gz" for m in align.SEQUENCES}


def check_ab(sids):
    rng = np.random.default_rng(align.SEED)
    rows = []
    for sid in sids:
        f = files(sid)
        t1c, t1n = sitk.ReadImage(str(f["t1c"])), sitk.ReadImage(str(f["t1n"]))
        pts = brain_points(t1n)
        for seq in ("t2f", "t2w"):
            img = sitk.ReadImage(str(f[seq]))
            rot = rng.uniform(2, 5, 3) * rng.choice([-1, 1], 3)
            shift = rng.uniform(1, 3, 3) * rng.choice([-1, 1], 3)
            p = rigid(img, rot, shift)
            moved = sitk.Resample(img, img, p, sitk.sitkLinear, 0.0, img.GetPixelID())  # moved(x) = img(P(x))
            t0 = time.time()
            _, ta, _ = align.align_to_reference(moved, t1c)
            ea = align.residual_error_mm(pts, ta, p)
            _, tb, _ = align.align_to_reference(img, t1c)
            eb = align.residual_error_mm(pts, tb)
            # Consistency (descriptive, not a requirement): does the recovery in A land where B's alignment of the
            # untouched scan lands? Small = the step is precise; a non-zero B then reflects BraTS's own inter-sequence
            # alignment rather than an error of the step.
            pa = np.array([p.TransformPoint(ta.TransformPoint(tuple(map(float, q)))) for q in pts])
            pb = np.array([tb.TransformPoint(tuple(map(float, q))) for q in pts])
            eab = np.linalg.norm(pa - pb, axis=1)
            rows.append({"sid": sid, "seq": seq, "applied_rot_deg": [round(v, 2) for v in rot],
                         "applied_shift_mm": [round(v, 2) for v in shift],
                         "A_mean_mm": float(ea.mean()), "A_max_mm": float(ea.max()),
                         "B_mean_mm": float(eb.mean()), "B_max_mm": float(eb.max()),
                         "AB_mean_mm": float(eab.mean()), "AB_max_mm": float(eab.max()),
                         "B_rotation_deg": align.transform_summary(tb)["rotation_deg"],
                         "B_translation_mm": align.transform_summary(tb)["translation_mm"],
                         "seconds": round(time.time() - t0, 1)})
            print(f"{sid[10:15]} {seq}: A mean {ea.mean():.3f} max {ea.max():.3f} | B mean {eb.mean():.3f} max {eb.max():.3f}"
                  f" | A-vs-B mean {eab.mean():.3f} max {eab.max():.3f} mm | B transform {rows[-1]['B_rotation_deg']} deg"
                  f" {rows[-1]['B_translation_mm']} mm ({rows[-1]['seconds']} s)", flush=True)
    return rows


def dice(a, b):
    s = int(a.sum()) + int(b.sum())
    return 1.0 if s == 0 else 2 * int((a & b).sum()) / s


def rstar_mean_dice(paths, gt, work):
    out = work / "seg.nii.gz"
    cmd = [RSTAR_PY, "-m", "rstar", *sum(([f"--{m}", str(paths[m])] for m in align.SEQUENCES), []),
           "--out", str(out), "--models-root", str(MODELS)]
    r = subprocess.run(cmd, cwd=RSTAR_DIR, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"rstar failed: {r.stderr[-800:]}")
    lab = np.asarray(nib.load(str(out)).dataobj).astype(np.uint8)
    et, tc, wt = dice(lab == 1, gt == 1), dice(np.isin(lab, (1, 2, 3)), np.isin(gt, (1, 2, 3))), dice(lab > 0, gt > 0)
    return (et + tc + wt) / 3


def check_c(sids):
    rows = []
    for sid in sids:
        f = files(sid)
        gt = np.rint(np.asarray(nib.load(str(RAW / sid / f"{sid}-seg.nii.gz")).dataobj)).astype(np.uint8)
        with tempfile.TemporaryDirectory(prefix="align_c_") as tmp:
            tmp = Path(tmp)
            mis = dict(f)
            for seq in ("t2f", "t2w"):
                img = sitk.ReadImage(str(f[seq]))
                moved = sitk.Resample(img, img, rigid(img, (0, 0, 3), (2, 0, 0)), sitk.sitkLinear, 0.0, img.GetPixelID())
                mis[seq] = tmp / f"mis-{seq}.nii.gz"
                sitk.WriteImage(moved, str(mis[seq]))
            realigned, _ = align.align_sequences(mis)
            re = {}
            for seq, img in realigned.items():
                re[seq] = tmp / f"re-{seq}.nii.gz"
                sitk.WriteImage(img, str(re[seq]))
            row = {"sid": sid}
            for cond, paths in (("clean", f), ("misaligned", mis), ("realigned", re)):
                d = tmp / cond
                d.mkdir()
                row[cond] = rstar_mean_dice(paths, gt, d)
            rows.append(row)
            print(f"{sid[10:15]} R* mean Dice: clean {row['clean']:.4f} | misaligned {row['misaligned']:.4f} | realigned {row['realigned']:.4f}", flush=True)
    return rows


def report(ab, c):
    L = ["# Alignment step: verification results", "",
         "Generated by `tools/check_alignment.py` on BraTS-PEDs fresh-30 patients (sorted ids). Residual error = distance",
         "between where a brain voxel should be and where the recovered transform puts it, over 20,000 brain points.", ""]
    ok_a = all(r["A_mean_mm"] < 0.5 and r["A_max_mm"] < 1.0 for r in ab)
    ok_b = all(r["B_mean_mm"] < 0.5 for r in ab)
    L += ["## A. Recovery of a known misalignment (Req 9: mean < 0.5 mm and max < 1.0 mm for every case)", "",
          "| Patient | Sequence | Applied rotation (deg) | Applied shift (mm) | Mean residual (mm) | Max residual (mm) |", "|---|---|---|---|---|---|"]
    L += [f"| {r['sid'][10:15]} | {r['seq']} | {r['applied_rot_deg']} | {r['applied_shift_mm']} | {r['A_mean_mm']:.3f} | {r['A_max_mm']:.3f} |" for r in ab]
    L += ["", f"**Req 9: {'PASS' if ok_a else 'FAIL'}** (worst mean {max(r['A_mean_mm'] for r in ab):.3f} mm, worst max {max(r['A_max_mm'] for r in ab):.3f} mm)", "",
          "## B. Already-aligned scans left unchanged (Req 10: mean < 0.5 mm from identity for every case)", "",
          "| Patient | Sequence | Mean (mm) | Max (mm) |", "|---|---|---|---|"]
    L += [f"| {r['sid'][10:15]} | {r['seq']} | {r['B_mean_mm']:.3f} | {r['B_max_mm']:.3f} |" for r in ab]
    L += ["", f"**Req 10: {'PASS' if ok_b else 'FAIL'}** (worst mean {max(r['B_mean_mm'] for r in ab):.3f} mm)", "",
          "## Consistency of A with B (descriptive, not a requirement)", "",
          "Distance between where the recovery in A puts each brain point and where B's alignment of the untouched scan puts",
          "it. If this is small while B is not zero, the step is precise and B measures BraTS's own inter-sequence alignment.", "",
          "| Patient | Sequence | A-vs-B mean (mm) | A-vs-B max (mm) | B rotation (deg) | B translation (mm) |", "|---|---|---|---|---|---|"]
    L += [f"| {r['sid'][10:15]} | {r['seq']} | {r['AB_mean_mm']:.3f} | {r['AB_max_mm']:.3f} | {r['B_rotation_deg']} | {r['B_translation_mm']} |" for r in ab]
    L += [""]
    if c:
        mc, mm, mr = (np.mean([r[k] for r in c]) for k in ("clean", "misaligned", "realigned"))
        ok_c = abs(mr - mc) <= 0.01
        L += ["## C. End to end: R* on clean, misaligned (T2w + FLAIR moved 2 mm + 3 degrees) and realigned scans (Req 11)", "",
              "| Patient | Clean | Misaligned | Realigned |", "|---|---|---|---|"]
        L += [f"| {r['sid'][10:15]} | {r['clean']:.4f} | {r['misaligned']:.4f} | {r['realigned']:.4f} |" for r in c]
        L += [f"| **Mean** | **{mc:.4f}** | **{mm:.4f}** | **{mr:.4f}** |", "",
              f"**Req 11: {'PASS' if ok_c else 'FAIL'}** (realigned minus clean {mr - mc:+.4f}; misaligned minus clean {mm - mc:+.4f})", ""]
    (HERE / "RESULTS_alignment.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    json.dump({"ab": ab, "c": c}, open(HERE / "tools" / "alignment_rows.json", "w"), indent=1)
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-c", action="store_true")
    a = ap.parse_args()
    sids = sorted(json.load(open(FRESH)))
    ab = check_ab(sids[:10])
    c = [] if a.skip_c else check_c(sids[:5])
    report(ab, c)


if __name__ == "__main__":
    main()
