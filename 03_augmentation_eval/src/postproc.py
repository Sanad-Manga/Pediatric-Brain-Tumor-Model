"""Inference-time post-processing: class re-weighting and component cleanup.

Nothing here retrains anything. It reshapes the decision the network has
already made, which is the right lever for *this* model: it under-segments
(ET precision 0.757 against sensitivity 0.496), so it is losing Dice by being
too conservative, not by over-calling tumour.

Two knobs, both off by default so leaving the config keys out reproduces plain
argmax exactly:

``et_boost``
    Multiply the enhancing-tumour probability channel before argmax. ``1.0`` is
    plain argmax. Above 1.0 trades precision for sensitivity on ET only. This is
    a *decision threshold* shift, not a model change -- it cannot invent tumour
    the network gave zero probability, it only lowers the bar for tumour the
    network already suspected.

``keep_largest_wt`` / ``min_component_voxels``
    Drop spurious specks. Pediatric tumours are single connected lesions far
    more often than adult glioma, so isolated blobs a long way from the main
    mass are usually false positives. ``keep_largest_wt`` is the aggressive
    version (keep one lesion); ``min_component_voxels`` is the mild version
    (drop anything under N voxels) and is the safer default of the two --
    a genuine multifocal subject loses real tumour under ``keep_largest_wt``.

**These must be tuned on validation subjects and measured on the held-out set
exactly once.** Sweeping on held-out and reporting the best value produces a
number that is not a generalisation estimate. ``tools/tune_postproc.py``
enforces the split; use it rather than hand-running ``run.py eval`` in a loop.
"""

from __future__ import annotations

import numpy as np

#: class index of enhancing tumour in the 0-4 label encoding (see metrics.py)
ET_CLASS = 1

#: no-op settings; equal to plain argmax over the probability volume
DEFAULT_POSTPROC = {
    "et_boost": 1.0,
    "keep_largest_wt": False,
    "min_component_voxels": 0,
}


def _label_components(foreground: np.ndarray):
    """Connected-component labelling with full 3D (26-)connectivity."""
    try:
        from scipy import ndimage
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError(
            "component cleanup needs scipy (`pip install scipy`). Run without "
            "--keep-largest-wt / --min-component-voxels to skip it."
        ) from exc
    structure = np.ones((3, 3, 3), dtype=bool)
    return ndimage.label(foreground, structure=structure)


def clean_components(classes: np.ndarray, keep_largest_wt: bool = False,
                     min_component_voxels: int = 0) -> np.ndarray:
    """Remove small or non-primary connected components from a class volume.

    Components are found on the *whole tumour* (any non-zero class) so a lesion
    is never split apart just because its core and its edema are different
    classes. Removed voxels become background.
    """
    if not keep_largest_wt and min_component_voxels <= 0:
        return classes

    foreground = classes > 0
    if not foreground.any():
        return classes

    labelled, n = _label_components(foreground)
    if n <= 1 and not min_component_voxels:
        return classes

    # index 0 is background; sizes[i] is the voxel count of component i
    sizes = np.bincount(labelled.ravel())
    keep = np.ones(len(sizes), dtype=bool)
    keep[0] = False

    if min_component_voxels > 0:
        keep &= sizes >= min_component_voxels
    if keep_largest_wt:
        largest = int(np.argmax(sizes[1:])) + 1
        only_largest = np.zeros_like(keep)
        only_largest[largest] = True
        keep &= only_largest

    if not keep.any():
        # Every component was filtered out. Returning an empty volume would score
        # Dice 0.0 against a non-empty ground truth, which is strictly worse than
        # the un-cleaned prediction -- so decline to clean rather than destroy it.
        return classes

    return np.where(keep[labelled], classes, 0).astype(classes.dtype)


def probs_to_classes(prob_volume: np.ndarray, et_boost: float = 1.0,
                     keep_largest_wt: bool = False,
                     min_component_voxels: int = 0) -> np.ndarray:
    """``(C, X, Y, Z)`` probabilities -> ``(X, Y, Z)`` class labels.

    With all defaults this is exactly ``np.argmax(prob_volume, axis=0)``.
    """
    probs = prob_volume
    if et_boost != 1.0:
        probs = probs.copy()
        probs[ET_CLASS] *= float(et_boost)

    classes = np.argmax(probs, axis=0).astype(np.uint8)
    return clean_components(classes, keep_largest_wt, min_component_voxels)
