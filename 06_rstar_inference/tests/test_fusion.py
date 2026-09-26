"""Req 3, 4, 5 and 10: the decision rule, the agreement measure and probability averaging."""
import numpy as np
import pytest
import torch
from _helpers import make_volume, tiny_config

from rstar import RStarSegmenter
from rstar.fusion import agreement, apply_small_et_rule, argmax_labels, fuse, whole_tumour_dice


def _labels_with_et(n_et, shape=(20, 20, 20), seed=0):
    rng = np.random.default_rng(seed)
    labels = rng.choice([0, 2, 3, 4], size=shape).astype(np.uint8)
    flat = labels.reshape(-1)
    flat[:n_et] = 1
    return labels


# ------------------------------------------------------------------------------------------------ Req 3
def test_et_below_the_threshold_becomes_non_enhancing_and_nothing_else_moves():
    labels = _labels_with_et(100)
    out, n_et, relabelled = apply_small_et_rule(labels, 500.0, 1.0)
    assert relabelled and n_et == 100
    assert not (out == 1).any()
    assert np.array_equal(out == 2, (labels == 2) | (labels == 1))          # exactly the old 1s joined the 2s
    keep = labels != 1
    assert np.array_equal(out[keep], labels[keep])
    for region in ((1, 2, 3), (1, 2, 3, 4)):                              # tumour core and whole tumour do not change
        assert np.array_equal(np.isin(out, region), np.isin(labels, region))


@pytest.mark.parametrize("n_et, relabelled", [(499, True), (500, False), (501, False), (1, True)])
def test_the_threshold_is_exclusive(n_et, relabelled):
    labels = _labels_with_et(n_et, shape=(30, 30, 30))
    out, count, flag = apply_small_et_rule(labels, 500.0, 1.0)
    assert flag is relabelled and count == n_et
    assert ((out == 1).sum() == 0) == relabelled


def test_no_et_is_unchanged():
    labels = _labels_with_et(0)
    out, count, flag = apply_small_et_rule(labels, 500.0, 1.0)
    assert not flag and count == 0 and np.array_equal(out, labels)


@pytest.mark.parametrize("n_et, voxel, relabelled", [(300, 2.0, False), (900, 0.5, True), (250, 2.0, False), (249, 2.0, True)])
def test_the_rule_is_in_cubic_millimetres(n_et, voxel, relabelled):
    labels = _labels_with_et(n_et, shape=(30, 30, 30))
    assert apply_small_et_rule(labels, 500.0, voxel)[2] is relabelled


def test_the_input_is_never_modified_and_a_new_array_is_returned():
    labels = _labels_with_et(100)
    before = labels.copy()
    out, _, _ = apply_small_et_rule(labels, 500.0, 1.0)
    assert out is not labels and np.array_equal(labels, before)
    out2, _, flag = apply_small_et_rule(labels, 0.0, 1.0)                     # nothing relabelled: still a distinct copy
    assert not flag and out2 is not labels


# ------------------------------------------------------------------------------------------------ Req 4
def _random_probs(shape, seed):
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=(5, *shape)).astype(np.float32)
    e = np.exp(logits - logits.max(axis=0))
    return (e / e.sum(axis=0)).astype(np.float32)


@pytest.mark.parametrize("w3d, scale", [(0.5, 0.5), (0.3, 0.7), (0.8, 1.0), (0.5, 0.2)])
def test_fusion_equals_an_independent_computation(w3d, scale):
    p2, p3 = _random_probs((4, 5, 6), 1), _random_probs((4, 5, 6), 2)
    acc = (np.float32(w3d) * p3.astype(np.float64) + np.float32(1 - w3d) * p2.astype(np.float64))
    acc[0] *= scale
    expected = np.argmax(acc, axis=0)
    out = fuse(p2, p3, w3d, scale)
    assert out.dtype == np.uint8 and out.shape == (4, 5, 6)
    assert np.array_equal(out, expected)


def test_w3d_extremes_reduce_to_a_single_model():
    p2, p3 = _random_probs((4, 5, 6), 3), _random_probs((4, 5, 6), 4)
    assert np.array_equal(fuse(p2, p3, 1.0, 1.0), np.argmax(p3, axis=0))
    assert np.array_equal(fuse(p2, p3, 0.0, 1.0), np.argmax(p2, axis=0))
    assert np.array_equal(argmax_labels(p3), np.argmax(p3, axis=0))


def test_ties_go_to_the_lowest_class_index():
    uniform = np.full((5, 3, 3, 3), 0.2, dtype=np.float32)
    assert (fuse(uniform, uniform, 0.5, 1.0) == 0).all()                       # all tied -> class 0
    assert (fuse(uniform, uniform, 0.5, 0.5) == 1).all()                       # background halved -> classes 1-4 tied -> class 1


def test_the_background_scale_actually_changes_the_decision():
    p = np.zeros((5, 1, 1, 1), np.float32)
    p[0], p[2] = 0.55, 0.45
    assert fuse(p, p, 0.5, 1.0)[0, 0, 0] == 0
    assert fuse(p, p, 0.5, 0.5)[0, 0, 0] == 2


def test_fusion_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        fuse(_random_probs((3, 3, 3), 1), _random_probs((3, 3, 4), 2), 0.5, 0.5)


# ------------------------------------------------------------------------------------------------ Req 5
def _onehot(mask):
    p = np.zeros((5, *mask.shape), np.float32)
    p[0] = ~mask
    p[2] = mask
    return p


def test_agreement_edge_cases():
    rng = np.random.default_rng(0)
    a = rng.random((8, 8, 8)) > 0.6
    assert agreement(_onehot(a), _onehot(a)) == 1.0
    assert agreement(_onehot(a), _onehot(~a)) == 0.0
    empty = np.zeros((8, 8, 8), bool)
    assert agreement(_onehot(empty), _onehot(empty)) == 1.0
    assert agreement(_onehot(empty), _onehot(a)) == 0.0 and agreement(_onehot(a), _onehot(empty)) == 0.0


def test_agreement_equals_an_independent_dice_and_is_symmetric():
    rng = np.random.default_rng(1)
    a, b = rng.random((10, 10, 10)) > 0.5, rng.random((10, 10, 10)) > 0.7
    expected = 2.0 * np.logical_and(a, b).sum() / (a.sum() + b.sum())
    assert agreement(_onehot(a), _onehot(b)) == pytest.approx(expected)
    assert agreement(_onehot(a), _onehot(b)) == pytest.approx(agreement(_onehot(b), _onehot(a)))
    assert whole_tumour_dice(a, b) == pytest.approx(expected)


def test_agreement_uses_the_argmax_of_all_tumour_classes_not_just_one():
    a = np.zeros((6, 6, 6), bool); a[:3] = True
    p2, p3 = _onehot(a), _onehot(a)
    p2[2] = 0.0; p2[4] = a.astype(np.float32); p2[0] = ~a                     # tumour drawn as class 4 in one model, class 2 in the other
    assert agreement(p2, p3) == 1.0


# ------------------------------------------------------------------------------------------------ Req 10
class _KnownLogits(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.register_buffer("logits", torch.tensor(logits, dtype=torch.float32))

    def forward(self, x):
        n = x.shape[0]
        out = self.logits.reshape(1, 5, *([1] * (x.ndim - 2))).expand(n, 5, *x.shape[2:])
        return out.clone(), None


def _expected_mean(members):
    probs = [torch.softmax(torch.tensor(m, dtype=torch.float32), dim=0).numpy() for m in members]
    return np.mean(probs, axis=0)


def test_ensemble_probabilities_are_the_mean_of_the_members_softmax_in_both_branches():
    members = [[1.0, 2.0, 0.0, -1.0, 0.5], [0.0, -2.0, 3.0, 0.2, 0.1]]
    vol, _ = make_volume()
    seg = RStarSegmenter(tiny_config(), models_2d=[_KnownLogits(m) for m in members], models_3d=[_KnownLogits(m) for m in members])
    expected = _expected_mean(members)
    p3 = seg._probs_3d(vol, (True,) * 4)
    inside = vol[1] > 0
    assert np.abs(p3[:, inside] - expected[:, None]).max() < 1e-5             # every brain voxel, not just the mean
    assert np.allclose(p3.sum(axis=0), 1.0, atol=1e-4)
    p2 = seg._probs_2d(vol)
    # brain voxels are covered by both the axial and the coronal slices, so the plane average equals the member average
    assert np.abs(p2[:, inside] - expected[:, None]).max() < 1e-5
    assert np.allclose(p2[:, inside].sum(axis=0), 1.0, atol=1e-4)
