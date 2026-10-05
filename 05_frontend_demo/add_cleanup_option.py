r"""Add an EXTRA research option, "R* + fragment cleanup", to the comparison package. Nothing existing is changed.

Fragment cleanup (06_rstar_inference SPEC Addendum 3, PR #59): every connected piece of predicted tumour smaller than
200 voxels becomes background, and every enhancing-tumour piece smaller than 200 voxels becomes non-enhancing, before
the 500 mm^3 rule. Here it is applied to the saved R* labels and the 500 mm^3 rule is re-applied; that gives exactly the
labels the pipeline would produce (verified on all 111 evaluation patients: the rule only relabels enhancing tumour,
so it never changes which pieces exist). The review flags are unchanged (they are computed before the cleanup).

Writes per patient in comparison_cache/<patient>/:
  labels_rstar_clean.npz   {"labels": uint8 (240,240,155)}
  meta.json                adds "regions"["rstar_clean"], "rstar_clean_note" and "rstar_clean_changes"
CPU only; needs only the comparison package (no models, no raw data).
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "06_rstar_inference"))
from precompute_comparison import OUT_DIR, REGIONS, region_rates  # noqa: E402
from rstar.fusion import apply_small_et_rule  # noqa: E402

MIN_VOXELS = 200
STRUCTURE = np.ones((3, 3, 3), dtype=bool)
NOTE = ("Research option, not the deployed model: R* with tumour pieces under 200 voxels removed before the 500 mm3 "
        "rule. In tests on 111 patients it left the usual Dice unchanged and raised the official BraTS lesion-wise "
        "score by about 0.07, with false enhancing spots shown as tumour falling from 22 to 2 (held-out). The size was "
        "chosen on those patients; it needs cross-validation before it can replace R*.")


def remove_fragments(labels, min_voxels):
    """Same rule as rstar.fusion.remove_fragments (SPEC Addendum 3)."""
    out = np.array(labels, copy=True)
    for mask_value, new_value in ((None, 0), (1, 2)):
        mask = out > 0 if mask_value is None else out == mask_value
        comp, n = ndimage.label(mask, STRUCTURE)
        if n:
            sizes = np.bincount(comp.ravel())
            out[(sizes[comp] < min_voxels) & mask] = new_value
    return out


def main():
    for out_dir in sorted(p for p in OUT_DIR.iterdir() if p.is_dir()):
        expert = np.load(out_dir / "expert.npz")["labels"]
        rstar = np.load(out_dir / "labels_rstar.npz")["labels"]
        clean = apply_small_et_rule(remove_fragments(rstar, MIN_VOXELS), 500.0, 1.0)[0].astype(np.uint8)
        np.savez_compressed(out_dir / "labels_rstar_clean.npz", labels=clean)
        changed = rstar != clean
        meta = json.loads((out_dir / "meta.json").read_text())
        meta["regions"]["rstar_clean"] = {r: region_rates(clean, expert, lab) for r, lab in REGIONS.items()}
        meta["rstar_clean_note"] = NOTE
        meta["rstar_clean_changes"] = {"voxels_changed": int(changed.sum()),
                                       "tumour_voxels_removed": int(((rstar > 0) & (clean == 0)).sum()),
                                       "enhancing_voxels_relabelled": int(((rstar == 1) & (clean == 2)).sum())}
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        mean = lambda k: np.mean([meta["regions"][k][r]["dice"] for r in REGIONS])
        print(f"{out_dir.name}: changed {int(changed.sum()):6d} voxels | mean Dice R* {mean('rstar'):.4f} -> clean {mean('rstar_clean'):.4f}",
              flush=True)


if __name__ == "__main__":
    main()
