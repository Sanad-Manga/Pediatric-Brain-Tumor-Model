r"""Add a CANDIDATE option, "R* + nnU-Net", to the comparison package. Nothing existing is changed.

R* + nnU-Net = (2D + 3D + nnU-Net) / 3, background x0.5, argmax, 500 mm^3 rule (06_rstar_inference SPEC Addendum 4,
`RStarConfig(nnunet=True)`). Pre-registered results (US BraTS-PEDs only): 30 untouched patients 0.844 vs R* 0.805;
5-fold cross-validation on 226 patients: nnU-Net +0.087 over the 3D family; at least as robust as R* under
misalignment, scanner differences and missing sequences. None of the demo patients was used to train nnU-Net (checked).

Writes per patient in comparison_cache/<patient>/:
  labels_rstar_nn.npz   {"labels": uint8 (240,240,155)}
  meta.json             adds "regions"["rstar_nn"], "rstar_nn_note", "rstar_nn_mode", "rstar_nn_status"
Needs the GPU, the checkpoints (RSTAR_MODELS_ROOT, default C:\Users\ahmed\Pediatric-Brain-Tumor-Model), nnunetv2 and
the raw BraTS-PEDs scans (precompute_comparison.DATA_ROOT).
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "06_rstar_inference"))
from precompute_comparison import DATA_ROOT, OUT_DIR, REGIONS, region_rates  # noqa: E402
from rstar import RStarConfig, RStarSegmenter  # noqa: E402

MODELS_ROOT = Path(os.environ.get("RSTAR_MODELS_ROOT", r"C:\Users\ahmed\Pediatric-Brain-Tumor-Model"))
TRAIN_MANIFEST = HERE.parent / "00_shared" / "manifests" / "hospitalA_plus_B.json"
NOTE = ("Candidate, not yet the deployed model: R* with a third model, nnU-Net, averaged in equally. In pre-registered "
        "tests it scored 0.844 vs R* 0.805 on 30 untouched patients, beat our 3D models by +0.087 in 5-fold "
        "cross-validation on 226 patients, and stayed at least as good as R* under misaligned sequences, scanner "
        "differences and missing sequences. US data only; flagged spots belong to the deployed R*.")


def main():
    trained_on = set(json.load(open(TRAIN_MANIFEST)))
    seg = RStarSegmenter(RStarConfig(models_root=MODELS_ROOT, nnunet=True))
    for out_dir in sorted(p for p in OUT_DIR.iterdir() if p.is_dir()):
        sid = out_dir.name
        assert sid not in trained_on, f"{sid} was in nnU-Net's training set"
        result, _ = seg.segment_paths({n: DATA_ROOT / sid / f"{sid}-{n}.nii.gz" for n in ("t1c", "t1n", "t2f", "t2w")})
        labels = result.labels.astype(np.uint8)
        expert = np.load(out_dir / "expert.npz")["labels"]
        np.savez_compressed(out_dir / "labels_rstar_nn.npz", labels=labels)
        meta = json.loads((out_dir / "meta.json").read_text())
        same_agreement = meta.get("rstar_agreement") is None or abs(result.diagnostics["agreement"] - meta["rstar_agreement"]) < 0.01
        meta["regions"]["rstar_nn"] = {r: region_rates(labels, expert, lab) for r, lab in REGIONS.items()}
        meta["rstar_nn_note"] = NOTE
        meta["rstar_nn_mode"] = result.mode
        meta["rstar_nn_status"] = result.status
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        mean = lambda k: np.mean([meta["regions"][k][r]["dice"] for r in REGIONS])
        print(f"{sid}: {result.mode} {result.status} | mean Dice R* {mean('rstar'):.4f} -> R*+nnU-Net {mean('rstar_nn'):.4f}"
              f" | 2D/3D agreement same as stored R*: {same_agreement}", flush=True)
        assert same_agreement, f"{sid}: the 2D/3D part differs from the stored R* run"


if __name__ == "__main__":
    main()
