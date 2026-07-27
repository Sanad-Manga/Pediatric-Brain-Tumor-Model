"""BraTS-PEDs region Dice.

Label encoding (CONTRACTS.md, verified against the actual dataset — 4 subregions,
not 3):

    0 = background
    1 = Enhancing Tumor (ET)
    2 = Non-Enhancing Tumor (NET)
    3 = Cystic Component (CC)
    4 = Peritumoral Edema (ED)

Official evaluation regions:

    ET = {1}
    NC = {1, 2, 3}      enhancing + cystic + necrosis complex
    WT = {1, 2, 3, 4}   whole tumor, all non-zero labels

Empty-region conventions (both asserted in the tests):

    * region empty in BOTH prediction and ground truth -> Dice = 1.0
      (a correct prediction of absence is not a failure)
    * region empty in one but not the other             -> Dice = 0.0

The second case matters here in practice: DMG/DIPG subjects frequently have
little or no enhancing tumor, so ET is genuinely absent in some ground truths.
:func:`evaluate_subjects` therefore also reports how many subjects had an empty
ground-truth region, so the mean stays interpretable.
"""

from __future__ import annotations

import numpy as np
import torch

#: region name -> the class labels composing it
REGIONS: dict[str, tuple[int, ...]] = {
    "ET": (1,),
    "NC": (1, 2, 3),
    "WT": (1, 2, 3, 4),
}

REGION_ORDER = ("ET", "NC", "WT")


def _to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def region_mask(class_mask, region: str) -> np.ndarray:
    """Boolean mask of one BraTS region from an integer class mask."""
    if region not in REGIONS:
        raise KeyError(f"unknown region {region!r}; expected one of {sorted(REGIONS)}")
    arr = _to_numpy(class_mask)
    return np.isin(arr, REGIONS[region])


def dice_score(pred_bool: np.ndarray, true_bool: np.ndarray) -> float:
    """Dice for one binary region, with the empty-region conventions applied."""
    pred_sum = int(pred_bool.sum())
    true_sum = int(true_bool.sum())
    if pred_sum == 0 and true_sum == 0:
        return 1.0
    if pred_sum == 0 or true_sum == 0:
        return 0.0
    intersection = int(np.logical_and(pred_bool, true_bool).sum())
    return 2.0 * intersection / (pred_sum + true_sum)


def dice_regions(pred_classes, true_classes) -> dict[str, float]:
    """Per-region Dice for one subject.

    Both inputs are integer class masks (any broadcastable shape, typically
    ``(1, D, H, W)`` or ``(D, H, W)``). Returns
    ``{"dice_ET": ..., "dice_NC": ..., "dice_WT": ...}``.
    """
    pred = _to_numpy(pred_classes)
    true = _to_numpy(true_classes)
    if pred.shape != true.shape:
        raise ValueError(
            f"prediction shape {pred.shape} does not match ground truth shape {true.shape}"
        )
    return {
        f"dice_{r}": dice_score(region_mask(pred, r), region_mask(true, r))
        for r in REGION_ORDER
    }


def logits_to_classes(logits) -> np.ndarray:
    """Argmax over the class channel of ``(B, C, D, H, W)`` or ``(C, D, H, W)``."""
    arr = _to_numpy(logits)
    axis = 1 if arr.ndim == 5 else 0
    return np.argmax(arr, axis=axis)


def mean_dice(scores: dict[str, float]) -> float:
    """Mean of the three region Dice scores."""
    return float(sum(scores[f"dice_{r}"] for r in REGION_ORDER) / len(REGION_ORDER))


def aggregate(per_subject: list[dict[str, float]]) -> dict[str, float]:
    """Average per-subject region scores into the row-level summary."""
    if not per_subject:
        raise ValueError("cannot aggregate an empty list of per-subject scores")
    out = {
        f"dice_{r}": float(np.mean([s[f"dice_{r}"] for s in per_subject]))
        for r in REGION_ORDER
    }
    out["mean_dice"] = mean_dice(out)
    return out


def count_empty_ground_truth(true_masks) -> dict[str, int]:
    """How many subjects have each region absent from the ground truth."""
    counts = {r: 0 for r in REGION_ORDER}
    for mask in true_masks:
        for r in REGION_ORDER:
            if not region_mask(mask, r).any():
                counts[r] += 1
    return counts
