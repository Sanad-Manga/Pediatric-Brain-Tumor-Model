r"""Precompute 2D-only, 3D-only and R* segmentations for the model-comparison Streamlit page.

Writes one compact package per patient to OUT_DIR:
  <patient>/labels_2d.npz   {"labels": uint8 (240,240,155)}
  <patient>/labels_3d.npz   {"labels": uint8 (240,240,155)}
  <patient>/labels_rstar.npz{"labels": uint8 (240,240,155)}
  <patient>/meta.json       per-model per-region Dice/precision/recall against ground truth,
                             plus mode/status/warnings from the R* run.

No checkpoints or raw NIfTI ship in this package -- only small label volumes and numbers, so it is
safe to zip and hand to someone without repo/data access. See HANDOFF_MODEL_COMPARISON.md.
"""
import json
import sys
from pathlib import Path

import numpy as np
import nibabel as nib

sys.path.insert(0, r"C:\Users\ahmed\Pediatric-Brain-Tumor-Model-rstar\06_rstar_inference")
from rstar import RStarSegmenter, RStarConfig  # noqa: E402
from rstar import fusion  # noqa: E402
from rstar.contract import ContractError  # noqa: E402

DATA_ROOT = Path(r"D:\NeuroPeds AI\PKG - BraTS-PEDs-v1\BraTS-PEDs-v1\Training")
MODELS_ROOT = Path(r"C:\Users\ahmed\Pediatric-Brain-Tumor-Model")
OUT_DIR = Path(r"C:\Users\ahmed\AppData\Local\Temp\claude\C--Users-ahmed\1099fabd-13c1-4247-845e-223d5116695f\scratchpad\report_v2\comparison_package")

PATIENTS = [
    "BraTS-PED-00021-000", "BraTS-PED-00028-000", "BraTS-PED-00030-000",
    "BraTS-PED-00051-000", "BraTS-PED-00093-000", "BraTS-PED-00099-000", "BraTS-PED-00230-000",
]
REGIONS = {"ET": (1,), "NC": (1, 2, 3), "WT": (1, 2, 3, 4)}


def region_rates(pred, truth, labels):
    p, t = np.isin(pred, labels), np.isin(truth, labels)
    tp, fp, fn = int((p & t).sum()), int((p & ~t).sum()), int((~p & t).sum())
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 1.0
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    return {"dice": round(dice, 4), "precision": round(precision, 4) if precision == precision else None,
            "recall": round(recall, 4) if recall == recall else None}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = RStarConfig(models_root=MODELS_ROOT, device="auto")
    seg = RStarSegmenter(cfg)  # loads + hash-verifies the shipped 2D ensemble and 3D family once

    skipped = []
    for pid in PATIENTS:
        pdir = DATA_ROOT / pid
        paths = {m: pdir / f"{pid}-{m}.nii.gz" for m in ("t1c", "t1n", "t2f", "t2w")}
        truth = np.asarray(nib.load(str(pdir / f"{pid}-seg.nii.gz")).dataobj).astype(np.int64)

        try:
            result, affine = seg.segment_paths(paths)      # R*: runs 2D + 3D internally
        except ContractError as e:
            print(f"{pid}: SKIPPED -- input contract rejected it: {e}")
            skipped.append({"patient_id": pid, "reason": str(e)})
            continue
        labels_rstar = result.labels

        # Independent 2D-alone and 3D-alone argmax, reusing the probabilities R* already computed
        # (re-deriving them the same way rstar's own pipeline does, so all three are apples-to-apples).
        vol = np.zeros((4, *cfg.expected_shape), dtype=np.float32)
        zooms = None
        for c, name in enumerate(("t1c", "t1n", "t2f", "t2w")):
            img = nib.load(str(paths[name]))
            vol[c] = np.asarray(img.dataobj, dtype=np.float32)
            if zooms is None:
                zooms = float(np.prod(img.header.get_zooms()[:3]))
        p2 = seg._probs_2d(vol)
        p3 = seg._probs_3d(vol, (True, True, True, True))
        labels_2d = fusion.argmax_labels(p2)
        labels_3d, _, _ = fusion.apply_small_et_rule(fusion.argmax_labels(p3), cfg.et_min_mm3, zooms)

        out = OUT_DIR / pid
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / "labels_2d.npz", labels=labels_2d.astype(np.uint8))
        np.savez_compressed(out / "labels_3d.npz", labels=labels_3d.astype(np.uint8))
        np.savez_compressed(out / "labels_rstar.npz", labels=labels_rstar.astype(np.uint8))

        meta = {
            "patient_id": pid,
            "rstar_mode": result.mode, "rstar_status": result.status, "rstar_warnings": result.warnings,
            "rstar_agreement": result.diagnostics.get("agreement"),
            "regions": {
                model: {region: region_rates(labels, truth, lab) for region, lab in REGIONS.items()}
                for model, labels in (("2d", labels_2d), ("3d", labels_3d), ("rstar", labels_rstar))
            },
        }
        (out / "meta.json").write_text(json.dumps(meta, indent=2))
        print(pid, "mean Dice  2D:", round(np.mean([meta["regions"]["2d"][r]["dice"] for r in REGIONS]), 3),
              " 3D:", round(np.mean([meta["regions"]["3d"][r]["dice"] for r in REGIONS]), 3),
              " R*:", round(np.mean([meta["regions"]["rstar"][r]["dice"] for r in REGIONS]), 3))

    if skipped:
        (OUT_DIR / "skipped.json").write_text(json.dumps(skipped, indent=2))
        print(f"\n{len(skipped)} patient(s) skipped -- see skipped.json")


if __name__ == "__main__":
    main()
