r"""Build data/model_metrics.json: headline accuracy of the 2D model, the 3D family and R*, measured the same way.

Read by pages/Dashboard.py. Run on the machine that has the saved probability maps (not in git):

    python 05_frontend_demo/utils/build_model_metrics.py

Models, exactly as on the comparison page:
  2D  = shipped 2D ensemble, argmax
  3D  = deployed 4-member 96^3 family (flip-TTA), argmax, then the 500 mm^3 enhancing-tumour rule
  R*  = 2D 0.5 + 3D 0.5, background x0.5, argmax, then the 500 mm^3 rule (the deployed R* pipeline)
Regions: ET = label 1, TC = labels 1-3, WT = labels 1-4.

Two summaries per test set:
  per_patient_mean_dice : mean over patients of the (ET+TC+WT)/3 Dice (empty prediction of an empty region = 1)
  pooled                : Dice / sensitivity / precision over all voxels of all patients pooled together
  per_patient           : each patient's (ET+TC+WT)/3 Dice for the three models
Test sets: fresh-30 (never used for any choice; the clean test) and held-out 81 (some R* settings were tuned on it).
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "06_rstar_inference"))
from rstar import fusion  # noqa: E402

RAW = Path(r"D:\NeuroPeds AI\PKG - BraTS-PEDs-v1\BraTS-PEDs-v1\Training")
SETS = {
    "fresh30": (Path(r"D:\NeuroPeds AI\probs2d_fresh30"), Path(r"D:\NeuroPeds AI\probs3d_deployed\family_fresh30")),
    "heldout": (Path(r"D:\NeuroPeds AI\probs2d_heldout"), Path(r"D:\NeuroPeds AI\probs3d_deployed\family_heldout")),
}
SHAPE = (240, 240, 155)
REGIONS = {"ET": (1,), "TC": (1, 2, 3), "WT": (1, 2, 3, 4)}
OUT = Path(__file__).resolve().parents[1] / "data" / "model_metrics.json"


def _probs(path: Path) -> np.ndarray:
    t = torch.from_numpy(np.load(path).astype(np.float32))
    if tuple(t.shape[1:]) != SHAPE:
        t = F.interpolate(t[None], size=SHAPE, mode="trilinear", align_corners=False)[0]
    return t.numpy()


def _patient(args):
    set_name, sid = args
    d2, d3 = SETS[set_name]
    truth = np.rint(np.asarray(nib.load(str(RAW / sid / f"{sid}-seg.nii.gz")).dataobj)).astype(np.uint8)
    p2, p3 = _probs(d2 / f"{sid}.npy"), _probs(d3 / f"{sid}.npy")
    labels = {
        "2d": fusion.argmax_labels(p2),
        "3d": fusion.apply_small_et_rule(fusion.argmax_labels(p3), 500.0, 1.0)[0],
        "rstar": fusion.apply_small_et_rule(fusion.fuse(p2, p3, 0.5, 0.5), 500.0, 1.0)[0],
    }
    out = {"sid": sid}
    for model, lab in labels.items():
        for region, cls in REGIONS.items():
            p, t = np.isin(lab, cls), np.isin(truth, cls)
            out[(model, region)] = (int((p & t).sum()), int((p & ~t).sum()), int((~p & t).sum()))
    return out


def _dice(tp, fp, fn):
    return 1.0 if tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn)


def main() -> int:
    result = {"_about": __doc__.split("\n\n", 2)[2].strip(), "sets": {}}
    with ProcessPoolExecutor(6) as ex:
        for set_name, (d2, _) in SETS.items():
            ids = sorted(p.stem for p in d2.glob("*.npy"))
            rows = list(ex.map(_patient, [(set_name, i) for i in ids]))
            entry = {"n_patients": len(ids), "models": {}, "per_patient": []}
            for model in ("2d", "3d", "rstar"):
                per_patient = [np.mean([_dice(*r[(model, reg)]) for reg in REGIONS]) for r in rows]
                pooled = {}
                for reg in REGIONS:
                    tp, fp, fn = (sum(r[(model, reg)][k] for r in rows) for k in range(3))
                    pooled[reg] = {"dice": round(_dice(tp, fp, fn), 4), "sensitivity": round(tp / (tp + fn), 4),
                                   "precision": round(tp / (tp + fp), 4)}
                entry["models"][model] = {"per_patient_mean_dice": round(float(np.mean(per_patient)), 4), "pooled": pooled}
            for r in rows:                                   # per-patient mean Dice, for the Dashboard's dot chart
                entry["per_patient"].append({"patient": r["sid"], **{
                    m: round(float(np.mean([_dice(*r[(m, reg)]) for reg in REGIONS])), 6) for m in ("2d", "3d", "rstar")}})
            result["sets"][set_name] = entry
            print(set_name, {m: entry["models"][m]["per_patient_mean_dice"] for m in entry["models"]}, flush=True)
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
