#!/usr/bin/env python
"""Score an ensemble of N checkpoints on the held-out set, --eval-plane both.
Averages softmax probabilities across models (and across planes, same policy
evaluate_subject() already uses for a single model). Prints a JSON result
line prefixed with RESULT_JSON: for the caller to parse.

Usage: score_ensemble_cli.py <ckpt1> <ckpt2> [<ckpt3> ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SEC03 = Path(r"C:\Users\ahmed\Pediatric-Brain-Tumor-Model\03_augmentation_eval")
CACHE = Path(r"D:\NeuroPeds AI\pack_out_15k")
CFG_PATH = SEC03 / "config.overnight_s1b_w64_d3.yaml"
sys.path.insert(0, str(SEC03))

import torch  # noqa: E402
from src.config import load_config  # noqa: E402
from src.dummy import load_checkpoint  # noqa: E402
from src.evaluate import build_eval_dataset, predict_probs, restack_volume  # noqa: E402
from src.metrics import REGION_ORDER, aggregate, dice_regions  # noqa: E402
from src.model import build_model, infer_geometry  # noqa: E402
from src.postproc import probs_to_classes  # noqa: E402
from src.slices import unpad  # noqa: E402

DEVICE = "cuda"


def load_one(path: Path, cfg):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    geom = {k: payload[k] for k in ("width", "depth") if k in payload}
    if not geom:
        geom = infer_geometry(payload["model_state_dict"])
    m = build_model(cfg, spatial_dims=int(payload.get("spatial_dims", 2)), **geom)
    return load_checkpoint(path, m).to(DEVICE).eval()


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: score_ensemble_cli.py <ckpt1> <ckpt2> [...]", file=sys.stderr)
        return 2
    paths = [Path(a) for a in argv[1:]]

    cfg = load_config(CFG_PATH)
    cfg.eval["plane"] = "both"
    print(f"loading {len(paths)} member(s)...", flush=True)
    models = [load_one(p, cfg) for p in paths]

    dataset, subjects = build_eval_dataset(cfg, dummy_data=False, dummy_n=0,
                                           cache_dir=CACHE, tmp_dir=None, subjects=None)
    by_subject: dict[str, list] = {}
    for i in range(len(dataset)):
        try:
            item = dataset[i]
        except FileNotFoundError:
            continue
        by_subject.setdefault(item["subject_id"], []).append(item)

    scores = []
    for n, sid in enumerate(subjects, 1):
        items = by_subject.get(sid)
        if not items:
            continue
        per_model_planes = []
        true_vol = None
        for m in models:
            planes: dict = {}
            for item in items:
                plane = item["plane"]
                indices = item.get("slice_indices")
                probs = predict_probs(m, item["images"], device=DEVICE, tta=False)
                probs = unpad(probs, item["pad"])
                labels = unpad(item["labels"], item["pad"])
                true_vol = restack_volume(labels, plane, cfg, indices=indices)
                planes[plane] = restack_volume(probs.transpose(1, 0, 2, 3), plane, cfg,
                                               channelled=True, indices=indices)
            per_model_planes.append(planes)

        ens_probs_per_plane = []
        for plane in per_model_planes[0]:
            stacked = sum(pm[plane] for pm in per_model_planes) / len(per_model_planes)
            ens_probs_per_plane.append(stacked)
        ens_prob = sum(ens_probs_per_plane) / len(ens_probs_per_plane)
        pred = probs_to_classes(ens_prob)
        scores.append(dice_regions(pred, true_vol))
        if n % 20 == 0:
            print(f"  {n}/{len(subjects)} subjects scored", flush=True)

    if not scores:
        print("no subjects scored", file=sys.stderr)
        return 1

    agg = aggregate(scores)
    mean = sum(agg[f"dice_{r}"] for r in REGION_ORDER) / len(REGION_ORDER)
    result = {"dice": {f"dice_{r}": agg[f"dice_{r}"] for r in REGION_ORDER}, "mean": mean}
    print("RESULT_JSON:" + json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
