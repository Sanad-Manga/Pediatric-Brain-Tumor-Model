#!/usr/bin/env python
"""Per-patient held-out Dice for the original model, epoch-17 model and their
ensemble (--eval-plane both, same math as ensemble_eval.py), saved to CSV.

ensemble_eval.py computed these per-patient scores but only ever printed the
averages, so the spread across patients was never recorded. Also records how
many ground-truth voxels each region has per patient: dice_score() returns 1.0
when both masks are empty and 0.0 when only one is, so a patient with no ET in
the ground truth is a scoring artifact, not a real success or failure, and the
spread statistics must be able to exclude them.
"""
from __future__ import annotations

import csv
import ctypes
import sys
from pathlib import Path

sys.path.insert(0, r"C:\Users\ahmed\neuropeds_overnight")
import ensemble_eval as ee  # noqa: E402  (reuses its loaders; SEC03 is on sys.path via it)
from src.config import load_config  # noqa: E402
from src.evaluate import build_eval_dataset  # noqa: E402
from src.metrics import REGION_ORDER, dice_regions, region_mask  # noqa: E402
from src.postproc import probs_to_classes  # noqa: E402

OUT = Path(r"C:\Users\ahmed\neuropeds_overnight\per_patient_scores.csv")


def main() -> int:
    # Hold the machine awake (see orchestrate.prevent_sleep): ES_CONTINUOUS | ES_SYSTEM_REQUIRED.
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)

    cfg = load_config(r"C:\Users\ahmed\Pediatric-Brain-Tumor-Model\03_augmentation_eval"
                      r"\config.overnight_s1b_w64_d3.yaml")
    cfg.eval["plane"] = "both"

    print("loading models...", flush=True)
    model_a = ee.load_model(ee.CKPT_A, cfg)
    model_b = ee.load_model(ee.CKPT_B, cfg)

    dataset, subjects = build_eval_dataset(cfg, dummy_data=False, dummy_n=0,
                                           cache_dir=ee.CACHE, tmp_dir=None, subjects=None)
    by_subject: dict[str, list] = {}
    for i in range(len(dataset)):
        try:
            item = dataset[i]
        except FileNotFoundError:
            continue
        by_subject.setdefault(item["subject_id"], []).append(item)

    fields = ["subject_id"] + [f"true_vox_{r}" for r in REGION_ORDER]
    for model in ("original", "epoch17", "ensemble"):
        fields += [f"{model}_{r}" for r in REGION_ORDER]

    rows = []
    for n, sid in enumerate(subjects, 1):
        items = by_subject.get(sid)
        if not items:
            continue
        planes_a, true_vol = ee.restack_volume_probs(model_a, items, cfg)
        planes_b, _ = ee.restack_volume_probs(model_b, items, cfg)

        ens_prob = sum(0.5 * (planes_a[p] + planes_b[p]) for p in planes_a) / len(planes_a)
        preds = {
            "original": probs_to_classes(sum(planes_a.values()) / len(planes_a)),
            "epoch17": probs_to_classes(sum(planes_b.values()) / len(planes_b)),
            "ensemble": probs_to_classes(ens_prob),
        }
        row = {"subject_id": sid}
        for r in REGION_ORDER:
            row[f"true_vox_{r}"] = int(region_mask(true_vol, r).sum())
        for name, pred in preds.items():
            scores = dice_regions(pred, true_vol)
            for r in REGION_ORDER:
                row[f"{name}_{r}"] = round(float(scores[f"dice_{r}"]), 6)
        rows.append(row)

        if n % 10 == 0:
            print(f"  {n}/{len(subjects)} subjects scored", flush=True)
            with open(OUT, "w", newline="", encoding="utf-8") as fh:   # checkpoint partial progress
                w = csv.DictWriter(fh, fieldnames=fields)
                w.writeheader()
                w.writerows(rows)

    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} subjects -> {OUT}", flush=True)
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
    return 0


if __name__ == "__main__":
    sys.exit(main())
