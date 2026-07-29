"""Post-processing and TTA behaviour.

The property that matters most is the first one: with the new knobs off, the
prediction must be identical to plain argmax. Everything measured before this
file existed has to stay valid.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.postproc import DEFAULT_POSTPROC, clean_components, probs_to_classes

scipy = pytest.importorskip("scipy", reason="component cleanup needs scipy")


def _probs(classes: np.ndarray, num_classes: int = 5) -> np.ndarray:
    """One-hot-ish probability volume that argmaxes back to ``classes``."""
    out = np.full((num_classes, *classes.shape), 0.1, dtype=np.float32)
    for c in range(num_classes):
        out[c][classes == c] = 0.9
    return out


def test_defaults_are_plain_argmax():
    rng = np.random.default_rng(0)
    probs = rng.random((5, 12, 12, 8)).astype(np.float32)
    assert np.array_equal(probs_to_classes(probs, **DEFAULT_POSTPROC),
                          np.argmax(probs, axis=0))


def test_et_boost_only_moves_et():
    probs = np.zeros((5, 4, 4, 4), dtype=np.float32)
    probs[0] = 0.50      # background
    probs[1] = 0.30      # ET, losing to background
    probs[4] = 0.20      # edema
    assert (probs_to_classes(probs) == 0).all()

    # 0.30 * 2.0 = 0.60 > 0.50, so ET wins; edema is untouched either way
    boosted = probs_to_classes(probs, et_boost=2.0)
    assert (boosted == 1).all()


def test_et_boost_cannot_create_tumour_from_zero_probability():
    probs = np.zeros((5, 4, 4, 4), dtype=np.float32)
    probs[0] = 1.0
    assert (probs_to_classes(probs, et_boost=100.0) == 0).all()


def test_min_component_voxels_drops_specks_and_keeps_the_mass():
    classes = np.zeros((10, 10, 10), dtype=np.uint8)
    classes[2:7, 2:7, 2:7] = 1          # 125-voxel lesion
    classes[9, 9, 9] = 1                # 1-voxel speck

    cleaned = clean_components(classes, min_component_voxels=10)
    assert cleaned[9, 9, 9] == 0
    assert cleaned[2:7, 2:7, 2:7].all()


def test_keep_largest_wt_keeps_one_lesion():
    classes = np.zeros((10, 10, 10), dtype=np.uint8)
    classes[1:4, 1:4, 1:4] = 1          # 27 voxels
    classes[6:8, 6:8, 6:8] = 2          # 8 voxels

    cleaned = clean_components(classes, keep_largest_wt=True)
    assert cleaned[1:4, 1:4, 1:4].all()
    assert not cleaned[6:8, 6:8, 6:8].any()


def test_components_span_classes_so_a_lesion_is_not_split():
    """Core and its surrounding edema are one lesion, not two components."""
    classes = np.zeros((10, 10, 10), dtype=np.uint8)
    classes[2:8, 2:8, 2:8] = 4          # edema shell
    classes[4:6, 4:6, 4:6] = 1          # enhancing core inside it

    cleaned = clean_components(classes, keep_largest_wt=True)
    assert np.array_equal(cleaned, classes)


def test_cleanup_declines_rather_than_emptying_the_volume():
    """Filtering everything away would score Dice 0.0; return the input instead."""
    classes = np.zeros((6, 6, 6), dtype=np.uint8)
    classes[1, 1, 1] = 1

    assert np.array_equal(clean_components(classes, min_component_voxels=10_000),
                          classes)


def test_empty_prediction_survives_cleanup():
    classes = np.zeros((6, 6, 6), dtype=np.uint8)
    assert not clean_components(classes, keep_largest_wt=True).any()


class _FlipSensitiveModel:
    """Returns logits that depend on left-right position, so flipping matters."""

    def __call__(self, x):
        import torch

        n, _c, h, w = x.shape
        logits = torch.zeros(n, 5, h, w)
        logits[:, 1, :, : w // 2] = 5.0      # ET on the left half only
        return logits, None

    def eval(self):
        return self

    def to(self, _device):
        return self


def test_tta_averages_the_two_views():
    from src.evaluate import predict_probs

    images = np.zeros((2, 4, 8, 8), dtype=np.float32)
    model = _FlipSensitiveModel()

    plain = predict_probs(model, images, tta=False)
    tta = predict_probs(model, images, tta=True)

    # Plain: ET confident on the left, absent on the right.
    assert plain[0, 1, 0, 0] > plain[0, 1, 0, 7]
    # Flip-averaged: the asymmetry is gone and both halves match.
    assert tta[0, 1, 0, 0] == pytest.approx(tta[0, 1, 0, 7], abs=1e-6)
    # Still a probability distribution.
    assert tta.sum(axis=1) == pytest.approx(np.ones((2, 8, 8)), abs=1e-5)
