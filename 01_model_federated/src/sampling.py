"""Small-ET oversampling sampler (SPEC.md Addendum 9; team decision Q3 = "more practice on small tumours").

Patients whose label volume has 1..max_voxels enhancing-tumour (label 1) voxels are drawn `weight` times as
often as patients with more ET. Patients with NO ET keep exactly their uniform share of draws: seeing fewer
tumour-free examples is what produced extra false ET on tumour-free patients in the Addendum 8 run.
Epoch length is unchanged (one draw per patient, with replacement); the matched control is weight=1.0.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch.utils.data import Dataset, WeightedRandomSampler


def _check_weight(weight) -> float:
    if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight <= 0:
        raise ValueError(f"small-ET weight must be a positive finite number, got {weight!r}")
    return float(weight)


def _check_max_voxels(max_voxels) -> int:
    if isinstance(max_voxels, bool) or not isinstance(max_voxels, int) or max_voxels < 1:
        raise ValueError(f"small-ET max_voxels must be an integer >= 1, got {max_voxels!r}")
    return max_voxels


def count_et_voxels(dataset: Dataset) -> list[int]:
    """Label-1 voxel count per patient, read through the dataset itself so it matches what training sees."""
    return [int((dataset[i][1] == 1).sum()) for i in range(len(dataset))]


def small_et_sampling_weights(et_counts: Sequence[int], weight: float, max_voxels: int = 600) -> torch.Tensor:
    """Per-patient draw probabilities (float64, summing to 1).

    ET-free patients share n_free/N of the mass evenly; the rest goes to ET patients with relative weight
    `weight` for small ET (1..max_voxels, both ends included) and 1 for larger ET.
    """
    weight = _check_weight(weight)
    max_voxels = _check_max_voxels(max_voxels)
    n = len(et_counts)
    if n == 0:
        raise ValueError("et_counts is empty")
    free = [c == 0 for c in et_counts]
    small = [0 < c <= max_voxels for c in et_counts]
    n_free = sum(free)
    probs = torch.zeros(n, dtype=torch.float64)
    if n_free:
        probs[torch.tensor(free)] = 1.0 / n                                    # n_free/N in total, spread evenly
    rel = torch.tensor([0.0 if f else (weight if s else 1.0) for f, s in zip(free, small)], dtype=torch.float64)
    if rel.sum() > 0:
        probs += rel / rel.sum() * ((n - n_free) / n)
    return probs


def build_small_et_sampler(dataset: Dataset, weight: float, max_voxels: int = 600):
    """(sampler, counts): a WeightedRandomSampler drawing len(dataset) patients per epoch with replacement."""
    counts = count_et_voxels(dataset)
    probs = small_et_sampling_weights(counts, weight, max_voxels)
    sampler = WeightedRandomSampler(weights=probs, num_samples=len(dataset), replacement=True)
    return sampler, counts
