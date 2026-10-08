"""Addendum 1, Req 18: on untouched BraTS scans, sequences the registration would move < 1 mm are left identical."""
import json
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from prep import align  # noqa: E402

RAW = Path(r"D:\NeuroPeds AI\PKG - BraTS-PEDs-v1\BraTS-PEDs-v1\Training")
lines = []
for sid in ("BraTS-PED-00077-000", "BraTS-PED-00127-000", "BraTS-PED-00128-000"):
    paths = {m: RAW / sid / f"{sid}-{m}.nii.gz" for m in align.SEQUENCES}
    out, rep = align.align_sequences(paths)
    for m in ("t1n", "t2f", "t2w"):
        same = np.array_equal(sitk.GetArrayFromImage(out[m]), sitk.GetArrayFromImage(sitk.ReadImage(str(paths[m]))))
        r = rep[m]
        lines.append(f"{sid[10:15]} {m}: displacement {r['displacement_mm']:.3f} mm -> skipped {r['skipped']} | output identical to input {same}"
                     f" | rotation {r['rotation_deg']} deg, translation {r['translation_mm']} mm")
        print(lines[-1], flush=True)
ok = all(("skipped True" in l) == ("identical to input True" in l) for l in lines)
lines.append(f"Req 18: {'PASS' if ok else 'FAIL'} (every skipped sequence is identical to its input; every non-skipped one was resampled)")
(HERE / "tools" / "check_skip.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(lines[-1])
