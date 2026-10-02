"""Spot-level review flags and a per-spot error scorecard (SPEC.md Addendum 1).

Standalone: NOT used by pipeline.py. Splits predicted enhancing tumour (label 1) into separate 26-connected
spots, describes each (size, confidence, agreement between model sources) and marks it ``keep`` or ``review``
instead of silently relabelling small ET away. The scorecard counts errors per spot and per true lesion,
including SILENT errors: a false spot that is kept, or a true lesion covered by no kept or review spot.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import ndimage

_STRUCTURE = np.ones((3, 3, 3), dtype=bool)   # 26-connectivity, same for predictions and ground truth
REASONS = ("small", "borderline", "low_confidence", "models_disagree")


@dataclass(frozen=True)
class Spot:
    spot_id: int
    voxels: int
    mean_et_prob: float
    max_et_prob: float
    agreement: float | None


@dataclass(frozen=True)
class Decision:
    spot_id: int
    action: str            # "keep" | "review" | "drop" (drop = today's rule relabelled it away)
    reason: str | None


def _same_shape(a: np.ndarray, b: np.ndarray, what: str) -> None:
    if a.shape != b.shape:
        raise ValueError(f"{what}: shapes differ, {a.shape} vs {b.shape}")


def _components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    lab, n = ndimage.label(mask, structure=_STRUCTURE)
    return lab, int(n)


def find_et_spots(labels, et_prob, source_et_masks: Sequence = ()) -> tuple[np.ndarray, list[Spot]]:
    labels = np.asarray(labels)
    et_prob = np.asarray(et_prob, dtype=np.float64)
    _same_shape(labels, et_prob, "labels and et_prob")
    if np.isnan(et_prob).any() or et_prob.min(initial=0.0) < 0 or et_prob.max(initial=0.0) > 1:
        raise ValueError("et_prob must be finite and within [0, 1]")
    sources = [np.asarray(m, dtype=bool) for m in source_et_masks]
    for k, m in enumerate(sources):
        _same_shape(labels, m, f"labels and source mask {k}")
    lab, n = _components(labels == 1)
    spots = []
    for k in range(1, n + 1):
        m = lab == k
        agree = float(np.mean([src[m].mean() for src in sources])) if sources else None
        spots.append(Spot(spot_id=k, voxels=int(m.sum()), mean_et_prob=float(et_prob[m].mean()),
                          max_et_prob=float(et_prob[m].max()), agreement=agree))
    return lab.astype(np.uint16), spots


def _check_unit(name: str, value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or math.isnan(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be a number in [0, 1], got {value!r}")
    return float(value)


def decide(spots: Sequence[Spot], size_cut, prob_cut, agree_cut, small_cut=500) -> list[Decision]:
    for name, v in (("size_cut", size_cut), ("small_cut", small_cut)):
        if isinstance(v, bool) or not isinstance(v, (int, float)) or math.isnan(v) or v < 0:
            raise ValueError(f"{name} must be a non-negative number, got {v!r}")
    if small_cut > size_cut:
        raise ValueError(f"small_cut ({small_cut!r}) must not exceed size_cut ({size_cut!r})")
    prob_cut = _check_unit("prob_cut", prob_cut)
    agree_cut = _check_unit("agree_cut", agree_cut)
    out = []
    for s in spots:
        reason = None
        if s.voxels < size_cut:
            reason = "small" if s.voxels < small_cut else "borderline"
        elif s.mean_et_prob < prob_cut:
            reason = "low_confidence"
        elif s.agreement is not None and s.agreement < agree_cut:
            reason = "models_disagree"
        out.append(Decision(s.spot_id, "review" if reason else "keep", reason))
    return out


def t500_decisions(spots: Sequence[Spot], total_et_voxels: int, voxel_mm3: float = 1.0,
                   min_mm3: float = 500.0) -> list[Decision]:
    """Today's rule (fusion.apply_small_et_rule) as decisions: all spots dropped below min_mm3, else all kept."""
    action = "keep" if total_et_voxels * float(voxel_mm3) >= float(min_mm3) else "drop"
    return [Decision(s.spot_id, action, None) for s in spots]


def review_outputs(spot_ids: np.ndarray, spots: Sequence[Spot], decisions: Sequence[Decision]):
    by_id = {d.spot_id: d for d in decisions}
    review_mask = np.zeros(spot_ids.shape, dtype=np.uint8)
    review_spots = []
    for s in spots:
        d = by_id[s.spot_id]
        if d.action != "review":
            continue
        new_id = len(review_spots) + 1
        review_mask[spot_ids == s.spot_id] = new_id
        review_spots.append({"spot_id": new_id, "voxels": int(s.voxels), "mean_et_prob": float(s.mean_et_prob),
                             "models_agree": None if s.agreement is None else float(s.agreement),
                             "reason": d.reason})
    return review_mask, review_spots


def _dice(a: np.ndarray, b: np.ndarray) -> float:
    sa, sb = int(a.sum()), int(b.sum())
    if sa == 0 and sb == 0:
        return 1.0
    return 2.0 * int((a & b).sum()) / (sa + sb)


def score_patient(spot_ids: np.ndarray, spots: Sequence[Spot], decisions: Sequence[Decision], gt_labels) -> dict:
    gt_labels = np.asarray(gt_labels)
    _same_shape(spot_ids, gt_labels, "spot_ids and gt_labels")
    gt_et = gt_labels == 1
    by_id = {d.spot_id: d.action for d in decisions}
    kept = np.isin(spot_ids, [s.spot_id for s in spots if by_id[s.spot_id] == "keep"])
    review = np.isin(spot_ids, [s.spot_id for s in spots if by_id[s.spot_id] == "review"])
    c = dict(kept_real=0, kept_false=0, review_real=0, review_false=0)
    for s in spots:
        action = by_id[s.spot_id]
        if action == "drop":
            continue
        real = bool(gt_et[spot_ids == s.spot_id].any())
        c[f"{'kept' if action == 'keep' else 'review'}_{'real' if real else 'false'}"] += 1
    gt_lab, n_les = _components(gt_et)
    caught_keep = caught_review = missed = 0
    for k in range(1, n_les + 1):
        m = gt_lab == k
        if (m & kept).any():
            caught_keep += 1
        elif (m & review).any():
            caught_review += 1
        else:
            missed += 1
    return dict(c, lesions=n_les, lesions_caught_keep=caught_keep, lesions_caught_review=caught_review,
                lesions_missed=missed, silent_false=c["kept_false"], silent_missed=missed,
                et_dice_kept=_dice(kept, gt_et), et_dice_with_review=_dice(kept | review, gt_et))
