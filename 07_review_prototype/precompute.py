"""Build the local cache for the review-flag prototype (SPEC.md). Run once on the machine that has the saved
probability maps and raw NIfTI:

    python 07_review_prototype/precompute.py --out "D:/NeuroPeds AI/review_prototype"

Uses the deployed R* fusion and small-ET rule from 06_rstar_inference and the spot-level review flags
(rstar.review_flags) with the rule chosen in the pre-registered evaluation (review if mean ET prob < 0.7).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "06_rstar_inference"))
from rstar import fusion  # noqa: E402
from rstar.review_flags import Decision, decide, find_et_spots, review_outputs  # noqa: E402

PATIENTS_FROM = REPO / "05_frontend_demo" / "comparison_cache"
RAW = Path(r"D:\NeuroPeds AI\PKG - BraTS-PEDs-v1\BraTS-PEDs-v1\Training")
PROBS_2D = Path(r"D:\NeuroPeds AI\probs2d_heldout")
SAVED_3D = Path(r"C:\Users\ahmed\AppData\Local\Temp\claude\C--Users-ahmed\1099fabd-13c1-4247-845e-223d5116695f\scratchpad")
FAMILY_3D = SAVED_3D / "probs3d"                                    # deployed 4-member family, flip-TTA, 96^3
MEMBERS_3D = [SAVED_3D / f"probs3d_mem_old{m}_heldout" for m in "ACDE"]
SPECK_VOXELS = 50
PROB_CUT = 0.7
SHAPE = (240, 240, 155)


def _probs(path: Path) -> np.ndarray:
    """(5, 240, 240, 155) float32; 96^3 maps upsampled trilinearly exactly as the R* pipeline does."""
    t = torch.from_numpy(np.load(path).astype(np.float32))
    if tuple(t.shape[1:]) != SHAPE:
        t = F.interpolate(t[None], size=SHAPE, mode="trilinear", align_corners=False)[0]
    return t.numpy()


def _window(vol: np.ndarray) -> np.ndarray:
    brain = vol > 0
    if not brain.any():
        return np.zeros(vol.shape, np.uint8)
    lo, hi = np.percentile(vol[brain], [0.5, 99.5])
    out = np.clip((vol - lo) / max(hi - lo, 1e-6), 0, 1) * 255
    return np.where(brain, out, 0).round().astype(np.uint8)


def _slices(mask: np.ndarray, k: int) -> list[int]:
    z = np.where((mask == k).any(axis=(0, 1)))[0]
    return [int(z.min()), int(z.max())]


def build_patient(sid: str, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    for seq in ("t1c", "t2f"):
        img = np.asarray(nib.load(str(RAW / sid / f"{sid}-{seq}.nii.gz")).dataobj, dtype=np.float32)
        np.savez_compressed(out / f"{seq}.npz", img=_window(img))
    gt = np.rint(np.asarray(nib.load(str(RAW / sid / f"{sid}-seg.nii.gz")).dataobj)).astype(np.uint8)
    np.savez_compressed(out / "gt.npz", labels=gt)

    p2, p3 = _probs(PROBS_2D / f"{sid}.npy"), _probs(FAMILY_3D / f"{sid}.npy")
    flagged = fusion.fuse(p2, p3, 0.5, 0.5).astype(np.uint8)
    today, _, _ = fusion.apply_small_et_rule(flagged, 500.0, 1.0)
    np.savez_compressed(out / "flagged.npz", labels=flagged)
    np.savez_compressed(out / "today.npz", labels=today.astype(np.uint8))

    et_prob = np.clip(0.5 * p2[1] + 0.5 * p3[1], 0.0, 1.0)
    sources = [_probs(m / f"{sid}.npy").argmax(0) == 1 for m in MEMBERS_3D] + [p2.argmax(0) == 1]
    spot_ids, spots = find_et_spots(flagged, et_prob, sources)
    big = [s for s in spots if s.voxels >= SPECK_VOXELS]
    small = [s for s in spots if s.voxels < SPECK_VOXELS]
    decisions = decide(big, size_cut=0, prob_cut=PROB_CUT, agree_cut=0.0, small_cut=0)
    review_mask, review_spots = review_outputs(spot_ids, big, decisions)
    speck_mask, specks = review_outputs(spot_ids, small, [Decision(s.spot_id, "review", "speck") for s in small])
    np.savez_compressed(out / "review_mask.npz", mask=review_mask)
    np.savez_compressed(out / "specks.npz", mask=speck_mask)
    gt_et = gt == 1
    for mask, items in ((review_mask, review_spots), (speck_mask, specks)):
        for e in items:
            m = mask == e["spot_id"]
            e["real"] = bool((m & gt_et).any())
            e["slices"] = _slices(mask, e["spot_id"])
    (out / "review_spots.json").write_text(json.dumps(review_spots, indent=1), encoding="utf-8")
    (out / "specks.json").write_text(json.dumps(specks, indent=1), encoding="utf-8")
    meta = {"patient_id": sid, "rule": {"review_if_mean_et_prob_below": PROB_CUT, "speck_below_voxels": SPECK_VOXELS},
            "n_kept": sum(d.action == "keep" for d in decisions), "n_review": len(review_spots), "n_specks": len(specks),
            "today_et_voxels": int((today == 1).sum()), "flagged_et_voxels": int((flagged == 1).sum()),
            "expert_et_voxels": int(gt_et.sum())}
    (out / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=r"D:\NeuroPeds AI\review_prototype")
    args = ap.parse_args()
    out = Path(args.out)
    patients = sorted(p.name for p in PATIENTS_FROM.iterdir() if p.is_dir())
    for sid in patients:
        m = build_patient(sid, out / sid)
        print(f"{sid}: kept {m['n_kept']}, review {m['n_review']}, specks {m['n_specks']} | ET voxels today "
              f"{m['today_et_voxels']}, with flags {m['flagged_et_voxels']}, expert {m['expert_et_voxels']}", flush=True)
    (out / "patients.json").write_text(json.dumps(patients, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
