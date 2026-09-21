#!/usr/bin/env python
"""Ensemble the original exhibition (strong ET) and epoch-17 s1b_w64_d3
(strong NC/WT) by averaging softmax probabilities, then score the ensemble
on the held-out set with --eval-plane both. If their errors are genuinely
different (not just correlated), the ensemble should beat both individually."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, r"C:\Users\ahmed\Pediatric-Brain-Tumor-Model\03_augmentation_eval")
import numpy as np  # noqa: E402
import torch  # noqa: E402
from src.config import load_config  # noqa: E402
from src.dummy import load_checkpoint  # noqa: E402
from src.evaluate import build_eval_dataset, predict_probs  # noqa: E402
from src.metrics import REGION_ORDER, aggregate, dice_regions  # noqa: E402
from src.model import build_model, infer_geometry  # noqa: E402
from src.postproc import probs_to_classes  # noqa: E402
from src.slices import unpad  # noqa: E402

CACHE = Path(r"D:\NeuroPeds AI\pack_out_15k")
CKPT_A = Path(r"C:\Users\ahmed\Pediatric-Brain-Tumor-Model\03_augmentation_eval"
             r"\checkpoints\overnight_run\best.pt.exhibition-backup")   # original, strong ET
CKPT_B = Path(r"C:\Users\ahmed\Pediatric-Brain-Tumor-Model\03_augmentation_eval"
             r"\checkpoints\s1b_w64_d3\best.pt")                        # epoch17, strong NC/WT
DEVICE = "cuda"


def load_model(checkpoint: Path, cfg):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    geom = {k: payload[k] for k in ("width", "depth") if k in payload}
    if not geom:
        geom = infer_geometry(payload["model_state_dict"])
    model = build_model(cfg, spatial_dims=int(payload.get("spatial_dims", 2)), **geom)
    return load_checkpoint(checkpoint, model).to(DEVICE).eval()


def restack_volume_probs(model, items, cfg):
    """Per-plane restacked probability volumes for one subject, one model.
    Mirrors evaluate_subject()'s internals but returns per-plane probs
    instead of collapsing planes -- caller does the cross-model averaging
    before the plane-averaging step."""
    from src.evaluate import restack_volume
    per_plane = {}
    true_volume = None
    for item in items:
        plane = item["plane"]
        indices = item.get("slice_indices")
        probs = predict_probs(model, item["images"], device=DEVICE, tta=False)
        probs = unpad(probs, item["pad"])
        labels = unpad(item["labels"], item["pad"])
        true_volume = restack_volume(labels, plane, cfg, indices=indices)
        volume_probs = restack_volume(probs.transpose(1, 0, 2, 3), plane, cfg,
                                      channelled=True, indices=indices)
        per_plane[plane] = volume_probs
    return per_plane, true_volume


def main() -> int:
    cfg = load_config(r"C:\Users\ahmed\Pediatric-Brain-Tumor-Model\03_augmentation_eval"
                      r"\config.overnight_s1b_w64_d3.yaml")
    cfg.eval["plane"] = "both"

    print("loading models...")
    model_a = load_model(CKPT_A, cfg)
    model_b = load_model(CKPT_B, cfg)

    dataset, subjects = build_eval_dataset(cfg, dummy_data=False, dummy_n=0,
                                           cache_dir=CACHE, tmp_dir=None, subjects=None)
    by_subject: dict[str, list] = {}
    for i in range(len(dataset)):
        try:
            item = dataset[i]
        except FileNotFoundError:
            continue
        by_subject.setdefault(item["subject_id"], []).append(item)

    ensemble_scores, a_scores, b_scores = [], [], []
    for n, sid in enumerate(subjects, 1):
        items = by_subject.get(sid)
        if not items:
            continue
        planes_a, true_vol = restack_volume_probs(model_a, items, cfg)
        planes_b, _ = restack_volume_probs(model_b, items, cfg)

        # Ensemble: average model A and B per plane, then average planes --
        # same "mean across planes" policy evaluate_subject() already uses.
        ens_per_plane = [0.5 * (planes_a[p] + planes_b[p]) for p in planes_a]
        ens_prob = sum(ens_per_plane) / len(ens_per_plane)
        ens_pred = probs_to_classes(ens_prob)
        ensemble_scores.append(dice_regions(ens_pred, true_vol))

        a_prob = sum(planes_a.values()) / len(planes_a)
        b_prob = sum(planes_b.values()) / len(planes_b)
        a_scores.append(dice_regions(probs_to_classes(a_prob), true_vol))
        b_scores.append(dice_regions(probs_to_classes(b_prob), true_vol))

        if n % 20 == 0:
            print(f"  {n}/{len(subjects)} subjects scored")

    for name, scores in (("original (A)", a_scores), ("epoch17 (B)", b_scores),
                        ("ENSEMBLE (avg)", ensemble_scores)):
        agg = aggregate(scores)
        mean = sum(agg[f"dice_{r}"] for r in REGION_ORDER) / len(REGION_ORDER)
        print(f"{name:16s} " + "  ".join(f"{r}={agg[f'dice_{r}']:.4f}" for r in REGION_ORDER)
              + f"  mean={mean:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
