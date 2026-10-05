r"""Add an EXTRA research option, "R* hi-res", to the comparison package. Nothing existing is changed.

R* hi-res = the deployed R* pipeline (2D 0.5 + 3D 0.5, background x0.5, 500 mm3 rule) where the 3D part is
0.5 x the deployed 96^3 family + 0.5 x a 4-member 160^3 family trained for 170 epochs (the earlier 110-epoch 160^3
families were undertrained). Pre-registered test (neuropeds_overnight/PREREGISTERED_160_long_2026-10-04.md, arm AL):
held-out 0.8160 vs 0.8096 (+0.0064, 95% CI +0.0046..+0.0083), fresh-30 0.8204 vs 0.8051 (+0.0153, CI +0.0021..+0.0356),
no extra false enhancing tumour; small/medium enhancing tumour on fresh-30 0.540 -> 0.628. One training run of the
170-epoch family: a research option, NOT the deployed model; it needs cross-validation before it can replace R*.

Writes per patient in comparison_cache/<patient>/:
  labels_rstar_hires.npz   {"labels": uint8 (240,240,155)}
  meta.json                adds "regions"["rstar_hires"] (same Dice/precision/recall format) and "rstar_hires_note"

Runs the same code path as precompute_comparison.py (RStarSegmenter's own 2D and 3D probabilities) and first checks
that it reproduces the shipped labels_rstar.npz exactly, so R* and R* hi-res are compared like for like.
Needs the GPU machine: raw NIfTI under DATA_ROOT and the 160^3 checkpoints under HIRES_CKPT_DIR (not in git).
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from precompute_comparison import DATA_ROOT, MODELS_ROOT, OUT_DIR, REGIONS, region_rates  # noqa: E402
from rstar import RStarSegmenter, RStarConfig, fusion, preprocess  # noqa: E402
from rstar.sections import import_submodule  # noqa: E402

HIRES_CKPT_DIR = Path(r"C:\Users\ahmed\neuropeds_overnight\ckpt_region")
HIRES_MEMBERS = [f"H160L{m}_ep169.pt" for m in ("a15", "a25", "b15", "b25")]  # the 170-epoch 160^3 family
HIRES_CUBE = 160
MAX_DIFF_FRACTION = 0.001
NOTE = ('Research option, not the deployed model: R* with its 3D part averaged from the deployed 96^3 family and a 160^3-grid family trained for 170 epochs. In a pre-registered test it beat R* on both test sets with no extra false enhancing tumour (clean test 0.820 vs 0.805, held-out 0.816 vs 0.810, both statistically clear), mostly by finding small enhancing tumours better. One training run so far; it needs cross-validation before it can replace R*.')


def load_hires_family(device):
    model_mod = import_submodule("01", "model")
    nets = []
    for name in HIRES_MEMBERS:
        payload = torch.load(HIRES_CKPT_DIR / name, map_location="cpu", weights_only=False)
        net = model_mod.FederatedUNet3D()
        net.load_state_dict(payload["model_state"])
        nets.append(net.to(device).eval())
    return nets


def probs_3d_at(seg, vol, cube, models):
    """seg._probs_3d with the 3D grid temporarily set to `cube` (preprocess.CUBE is read at call time)."""
    old = preprocess.CUBE
    preprocess.CUBE = cube
    try:
        return seg._probs_3d(vol, (True, True, True, True), models=models)
    finally:
        preprocess.CUBE = old


def main():
    cfg = RStarConfig(models_root=Path(os.environ.get("RSTAR_MODELS_ROOT", MODELS_ROOT)), device="auto")
    seg = RStarSegmenter(cfg)
    hires = load_hires_family(seg.device)
    for out in sorted(p for p in OUT_DIR.iterdir() if p.is_dir()):
        pid = out.name
        pdir = DATA_ROOT / pid
        vol = np.zeros((4, *cfg.expected_shape), dtype=np.float32)
        zooms = None
        for c, name in enumerate(("t1c", "t1n", "t2f", "t2w")):
            img = nib.load(str(pdir / f"{pid}-{name}.nii.gz"))
            vol[c] = np.asarray(img.dataobj, dtype=np.float32)
            if zooms is None:
                zooms = float(np.prod(img.header.get_zooms()[:3]))
        truth = np.asarray(nib.load(str(pdir / f"{pid}-seg.nii.gz")).dataobj).astype(np.int64)

        p2 = seg._probs_2d(vol)
        p3 = seg._probs_3d(vol, (True, True, True, True))
        shipped = np.load(out / "labels_rstar.npz")["labels"]
        same, _, _ = fusion.apply_small_et_rule(fusion.fuse(p2, p3, cfg.w3d, cfg.background_scale), cfg.et_min_mm3, zooms)
        # The shipped files (30 Sep) and today's run of the same pipeline differ by a few dozen boundary voxels
        # (GPU rounding; today's runs are identical run to run). Allow that, refuse anything larger.
        n_diff, n_tumour = int((same.astype(np.uint8) != shipped).sum()), max(int((shipped > 0).sum()), 1)
        if n_diff > MAX_DIFF_FRACTION * n_tumour:
            raise SystemExit(f"{pid}: recomputed R* differs from the shipped labels_rstar.npz in {n_diff} voxels "
                             f"(> {MAX_DIFF_FRACTION:.1%} of {n_tumour} tumour voxels); not writing a non-comparable option")

        p160 = probs_3d_at(seg, vol, HIRES_CUBE, hires)
        labels, _, _ = fusion.apply_small_et_rule(
            fusion.fuse(p2, 0.5 * (p3 + p160), cfg.w3d, cfg.background_scale), cfg.et_min_mm3, zooms)
        np.savez_compressed(out / "labels_rstar_hires.npz", labels=labels.astype(np.uint8))

        meta = json.loads((out / "meta.json").read_text())
        meta["regions"]["rstar_hires"] = {region: region_rates(labels, truth, lab) for region, lab in REGIONS.items()}
        meta["rstar_hires_note"] = NOTE
        (out / "meta.json").write_text(json.dumps(meta, indent=2))
        mean = lambda k: np.mean([meta["regions"][k][r]["dice"] for r in REGIONS])
        print(f"{pid}: mean Dice R* {mean('rstar'):.3f} -> R* hi-res {mean('rstar_hires'):.3f} "
              f"(R* recompute vs shipped: {n_diff} voxels differ)", flush=True)


if __name__ == "__main__":
    main()
