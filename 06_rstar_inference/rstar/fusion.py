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


def fuse_weighted(probs_list, weights, background_scale: float) -> np.ndarray:
    """probs = sum(w_i * p_i), background probability multiplied by `background_scale`, then argmax (SPEC Addendum 4).
    The weights must be non-negative, one per volume, and sum to 1."""
    probs_list = [np.asarray(p, dtype=np.float32) for p in probs_list]
    weights = [float(w) for w in weights]
    if len(probs_list) != len(weights) or not probs_list:
        raise ValueError(f"need one weight per probability volume, got {len(weights)} weights for {len(probs_list)} volumes")
    if any(p.shape != probs_list[0].shape for p in probs_list):
        raise ValueError(f"probability volumes must have the same shape, got {[p.shape for p in probs_list]}")
    if any(not np.isfinite(w) or w < 0 for w in weights) or abs(sum(weights) - 1.0) > 1e-6:
        raise ValueError(f"weights must be non-negative and sum to 1, got {weights}")
    acc = np.zeros_like(probs_list[0])
    for p, w in zip(probs_list, weights):
        acc += np.float32(w) * p
    acc[0] *= np.float32(background_scale)
    return argmax_labels(acc)


def cyst_override(labels: np.ndarray, nnunet_probs: np.ndarray):
    """SPEC Addendum 4b: every voxel where nnU-Net's own argmax is cyst (label 3) becomes 3.
    Returns (new labels, voxels changed). The input is never modified."""
    labels = np.asarray(labels)
    nnunet_probs = np.asarray(nnunet_probs)
    if nnunet_probs.shape[1:] != labels.shape:
        raise ValueError(f"nnU-Net probabilities {nnunet_probs.shape} do not match the labels {labels.shape}")
    out = np.array(labels, copy=True)
    cyst = np.argmax(nnunet_probs, axis=0) == 3
    changed = int((cyst & (out != 3)).sum())
    out[cyst] = 3
    return out, changed


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


_STRUCTURE_26 = np.ones((3, 3, 3), dtype=bool)


def remove_fragments(labels: np.ndarray, min_voxels: int):
    """SPEC Addendum 3. Whole-tumour (labels 1-4) components with fewer than min_voxels voxels become 0; then
    enhancing-tumour (1) components with fewer than min_voxels voxels become 2. 26-connectivity.
    Returns (new labels, whole-tumour voxels removed, ET voxels relabelled). The input is never modified."""
    from scipy import ndimage  # local import: keeps `import rstar` light

    out = np.array(labels, copy=True)
    removed = relabelled = 0
    if min_voxels <= 0:
        return out, removed, relabelled
    for mask_value, new_value in ((None, 0), (1, 2)):
        mask = out > 0 if mask_value is None else out == mask_value
        comp, n = ndimage.label(mask, _STRUCTURE_26)
        if n == 0:
            continue
        sizes = np.bincount(comp.ravel())
        small = (sizes[comp] < min_voxels) & mask
        count = int(small.sum())
        out[small] = new_value
        if new_value == 0:
            removed = count
        else:
            relabelled = count
    return out, removed, relabelled
