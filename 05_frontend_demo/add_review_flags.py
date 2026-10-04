r"""Add review flags (team decision Q2 = D, "flag uncertain enhancing tumour instead of deleting it") to the
comparison package, in the format the Model Comparison page reads (issue #44):

  comparison_cache/<patient>/review_mask.npz    {"mask": uint8 (240,240,155)}  0 = not flagged, k = flagged spot k
  comparison_cache/<patient>/review_spots.json  [{"spot_id", "voxels", "mean_et_prob", "models_agree", "reason"}]

Rule (chosen in neuropeds_overnight/PREREGISTERED_review_flags_2026-10-02.md, same as 07_review_prototype): take R*'s
enhancing tumour BEFORE the 500 mm^3 rule, split it into connected spots, ignore specks under 50 voxels, and flag
every spot whose mean enhancing-tumour probability (2D 0.5 + 3D 0.5) is below 0.7. Agreement = share of the spot that
each of the four 3D models and the 2D model also call enhancing, averaged.

Runs RStarSegmenter's own code path (same as precompute_comparison.py). Existing files are not changed.
Needs the GPU machine (raw NIfTI + pinned checkpoints, not in git).
"""
import json
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from precompute_comparison import DATA_ROOT, MODELS_ROOT, OUT_DIR  # noqa: E402
from rstar import RStarSegmenter, RStarConfig, fusion  # noqa: E402
from rstar.review_flags import decide, find_et_spots, review_outputs  # noqa: E402

SPECK_VOXELS = 50
PROB_CUT = 0.7
ALL_PRESENT = (True, True, True, True)


def main():
    cfg = RStarConfig(models_root=Path(os.environ.get("RSTAR_MODELS_ROOT", MODELS_ROOT)), device="auto")
    seg = RStarSegmenter(cfg)
    for out in sorted(p for p in OUT_DIR.iterdir() if p.is_dir()):
        pid = out.name
        vol = np.stack([np.asarray(nib.load(str(DATA_ROOT / pid / f"{pid}-{m}.nii.gz")).dataobj, dtype=np.float32)
                        for m in ("t1c", "t1n", "t2f", "t2w")])
        p2 = seg._probs_2d(vol)
        p3 = seg._probs_3d(vol, ALL_PRESENT)
        fused = fusion.fuse(p2, p3, cfg.w3d, cfg.background_scale)        # R* before the 500 mm^3 rule
        et_prob = np.clip(cfg.w3d * p3[1] + (1 - cfg.w3d) * p2[1], 0.0, 1.0)
        sources = [seg._probs_3d(vol, ALL_PRESENT, models=[m]).argmax(0) == 1 for m in seg.models_3d]
        sources.append(p2.argmax(0) == 1)
        spot_ids, spots = find_et_spots(fused, et_prob, sources)
        big = [s for s in spots if s.voxels >= SPECK_VOXELS]
        decisions = decide(big, size_cut=0, prob_cut=PROB_CUT, agree_cut=0.0, small_cut=0)
        mask, review_spots = review_outputs(spot_ids, big, decisions)
        np.savez_compressed(out / "review_mask.npz", mask=mask)
        (out / "review_spots.json").write_text(json.dumps(review_spots, indent=1), encoding="utf-8")
        kept = sum(d.action == "keep" for d in decisions)
        print(f"{pid}: {len(review_spots)} flagged, {kept} kept, {len(spots) - len(big)} specks ignored", flush=True)


if __name__ == "__main__":
    main()
