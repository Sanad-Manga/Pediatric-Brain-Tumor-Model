"""Diagnose verification C: why did R* collapse on realigned 00077, and why was 00115 rejected?"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import SimpleITK as sitk

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "tools"))
from prep import align  # noqa: E402
import check_alignment as ca  # noqa: E402

out = []
for sid in ("BraTS-PED-00115-000",):
    f = ca.files(sid)
    for m in align.SEQUENCES:
        a = np.asarray(nib.load(str(f[m])).dataobj)
        out.append(f"{sid[10:15]} raw {m}: dtype {a.dtype}, exactly-zero {np.mean(a == 0):.1%}, min {a.min():.1f}")

sid = "BraTS-PED-00077-000"
f = ca.files(sid)
gt = np.rint(np.asarray(nib.load(str(ca.RAW / sid / f"{sid}-seg.nii.gz")).dataobj)).astype(np.uint8)
with tempfile.TemporaryDirectory(prefix="diag_c_") as tmp:
    tmp = Path(tmp)
    mis = dict(f)
    for seq in ("t2f", "t2w"):
        img = sitk.ReadImage(str(f[seq]))
        moved = sitk.Resample(img, img, ca.rigid(img, (0, 0, 3), (2, 0, 0)), sitk.sitkLinear, 0.0, img.GetPixelID())
        mis[seq] = tmp / f"mis-{seq}.nii.gz"
        sitk.WriteImage(moved, str(mis[seq]))
    realigned, rep = align.align_sequences(mis)
    out.append("00077 alignment report: " + json.dumps(rep))
    re = {}
    for seq, img in realigned.items():
        re[seq] = tmp / f"re-{seq}.nii.gz"
        sitk.WriteImage(img, str(re[seq]))
    for m in align.SEQUENCES:
        raw = nib.load(str(f[m]))
        new = nib.load(str(re[m]))
        a, b = np.asarray(raw.dataobj), np.asarray(new.dataobj)
        out.append(f"00077 {m}: raw dtype {a.dtype} zero {np.mean(a == 0):.1%} | realigned dtype {b.dtype} zero {np.mean(b == 0):.1%}"
                   f" | affine equal {np.allclose(raw.affine, new.affine, atol=1e-3)} | corr with raw {np.corrcoef(a.ravel().astype(float), b.ravel().astype(float))[0, 1]:.4f}")
    # mixes: which realigned file breaks R*?
    mixes = {"raw t1c+t1n, realigned t2f+t2w": {**f, "t2f": re["t2f"], "t2w": re["t2w"]},
             "all four realigned": re}
    for name, paths in mixes.items():
        d = tmp / name.replace(" ", "_").replace("+", "").replace(",", "")
        d.mkdir()
        seg = d / "seg.nii.gz"
        cmd = [ca.RSTAR_PY, "-m", "rstar", *sum(([f"--{m}", str(paths[m])] for m in align.SEQUENCES), []),
               "--out", str(seg), "--json", str(d / "rep.json"), "--models-root", str(ca.MODELS)]
        r = subprocess.run(cmd, cwd=ca.RSTAR_DIR, capture_output=True, text=True)
        if r.returncode:
            out.append(f"{name}: rstar failed: {r.stderr[-300:]}")
            continue
        lab = np.asarray(nib.load(str(seg)).dataobj).astype(np.uint8)
        d3 = [ca.dice(lab == 1, gt == 1), ca.dice(np.isin(lab, (1, 2, 3)), np.isin(gt, (1, 2, 3))), ca.dice(lab > 0, gt > 0)]
        rj = json.loads((d / "rep.json").read_text())
        out.append(f"{name}: ET/TC/WT {[round(x, 3) for x in d3]} mean {np.mean(d3):.4f} | status {rj.get('status')} mode {rj.get('mode')} warnings {rj.get('warnings')}")
text = "\n".join(out)
print(text)
(HERE / "tools" / "diag_c.txt").write_text(text + "\n", encoding="utf-8")
