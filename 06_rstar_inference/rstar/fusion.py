"""The R* decision rule: probability fusion, the small-ET rule and the 2D-vs-3D agreement."""
from __future__ import annotations

import numpy as np


def argmax_labels(probs: np.ndarray) -> np.ndarray:
    """Class with the highest probability per voxel; ties go to the lowest class index. (C, ...) -> uint8 (...)."""
    return np.argmax(probs, axis=0).astype(np.uint8)


def fuse(p2: np.ndarray, p3: np.ndarray, w3d: float, background_scale: float) -> np.ndarray:
    """probs = w3d * p3 + (1 - w3d) * p2, background probability multiplied by `background_scale`, then argmax."""
    p2 = np.asarray(p2, dtype=np.float32)
    p3 = np.asarray(p3, dtype=np.float32)
    if p2.shape != p3.shape:
        raise ValueError(f"the 2D and 3D probability volumes must have the same shape, got {p2.shape} and {p3.shape}")
    acc = np.float32(w3d) * p3 + np.float32(1.0 - w3d) * p2
    acc[0] *= np.float32(background_scale)
    return argmax_labels(acc)


def apply_small_et_rule(labels: np.ndarray, min_mm3: float, voxel_mm3: float):
    """Relabel enhancing tumour (1) as non-enhancing (2) when 0 < total ET volume < min_mm3.
    Returns (new labels, ET voxels before the rule, relabelled?). The input is never modified."""
    out = np.array(labels, copy=True)
    is_et = out == 1
    n_et = int(is_et.sum())
    relabelled = 0 < n_et * float(voxel_mm3) < float(min_mm3)
    if relabelled:
        out[is_et] = 2
    return out, n_et, relabelled


def whole_tumour_dice(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a) > 0
    b = np.asarray(b) > 0
    sa, sb = int(a.sum()), int(b.sum())
    if sa == 0 and sb == 0:
        return 1.0
    if sa == 0 or sb == 0:
        return 0.0
    return 2.0 * int((a & b).sum()) / (sa + sb)


def agreement(p2: np.ndarray, p3: np.ndarray) -> float:
    """Whole-tumour Dice between the 2D-alone and the 3D-alone argmax. No ground truth needed."""
    return whole_tumour_dice(np.argmax(p2, axis=0), np.argmax(p3, axis=0))
