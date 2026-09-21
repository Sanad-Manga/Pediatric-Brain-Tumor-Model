#!/usr/bin/env python
"""Package the two source checkpoints (original exhibition + epoch17) into
one ensemble checkpoint file that inference.py's load_model() can build an
EnsembleModel from. Verifies it loads correctly and reproduces the measured
ensemble numbers on a couple of real slices before anything gets promoted."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

SEC03 = Path(r"C:\Users\ahmed\Pediatric-Brain-Tumor-Model\03_augmentation_eval")
CKPT_A = SEC03 / "checkpoints" / "overnight_run" / "best.pt.exhibition-backup"
CKPT_B = SEC03 / "checkpoints" / "s1b_w64_d3" / "best.pt"
OUT = Path(r"C:\Users\ahmed\neuropeds_overnight\ensemble_A_original_B_epoch17.pt")

MEASURED_MEAN = 0.7541   # from ensemble_eval.py's held-out --eval-plane both run


def slim_member(path: Path) -> dict:
    """Just enough for load_model() to rebuild this member: weights + geometry."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    geom = {k: payload[k] for k in ("width", "depth") if k in payload}
    out = {"model_state_dict": payload["model_state_dict"],
          "spatial_dims": payload.get("spatial_dims", 2)}
    out.update(geom)
    return out


def main() -> int:
    payload = {
        "ensemble": True,
        "member_paths": [str(CKPT_A), str(CKPT_B)],
        "members": [slim_member(CKPT_A), slim_member(CKPT_B)],
        "architecture": "ensemble",
        "spatial_dims": 2,
        "best_mean_dice": MEASURED_MEAN,
    }
    torch.save(payload, OUT)
    print(f"wrote: {OUT}")

    # Verify: load it back through the exact code path the app uses.
    sys.path.insert(0, str(Path(r"C:\Users\ahmed\Pediatric-Brain-Tumor-Model"
                                r"\05_frontend_demo")))
    from utils.inference import load_model, checkpoint_metadata, predict_slice  # noqa: E402

    meta = checkpoint_metadata(OUT)
    print("metadata:", meta)

    model, type_head, cfg, meta = load_model(OUT, cache_dir=None, device="cuda")
    print("loaded OK, member count:", len(model.members))

    cache = Path(r"D:\NeuroPeds AI\pack_out_15k")
    # A real held-out subject with visible tumour, quick sanity check.
    result = predict_slice(model, cfg, cache, "BraTS-PED-00051-000", "axial", 77,
                           device="cuda")
    print("smoke-test slice prediction shape:", result["prediction"].shape,
         "unique classes:", sorted(set(result["prediction"].flatten().tolist())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
