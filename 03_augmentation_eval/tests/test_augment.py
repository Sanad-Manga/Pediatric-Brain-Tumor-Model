"""Augmentation determinism, label integrity, mixup, and the config gate."""

from __future__ import annotations

import numpy as np
import pytest

from src.augment import (
    apply_augmentation,
    build_transforms,
    mixup_batch,
    one_hot,
    segmentation_mixup,
)
from src.dataset import SliceDataset


def _sample(cfg, seed=0):
    rng = np.random.default_rng(seed)
    h, w = cfg.common_size
    image = rng.normal(size=(len(cfg.modalities), h, w)).astype(np.float32)
    label = rng.integers(0, 5, size=(1, h, w)).astype(np.int64)
    return image, label


# ----------------------------------------------------------------- Req 13, 14
def test_stack_contains_the_specified_transforms_in_order(cfg):
    cfg.use_augmentation = True
    names = [type(t).__name__ for t in build_transforms(cfg).transforms]
    assert names[:7] == [
        "RandFlipd", "RandFlipd", "RandRotated", "RandZoomd",
        "RandScaleIntensityd", "RandShiftIntensityd", "RandGaussianNoised",
    ]


def test_spatial_transforms_use_nearest_neighbour_on_the_mask(cfg):
    cfg.use_augmentation = True
    for transform in build_transforms(cfg).transforms:
        mode = getattr(transform, "mode", None)
        if isinstance(mode, (tuple, list)) and len(mode) == 2:
            assert str(mode[1]) == "nearest"


def test_intensity_transforms_touch_the_image_only(cfg):
    cfg.use_augmentation = True
    for transform in build_transforms(cfg).transforms:
        if type(transform).__name__ in (
            "RandScaleIntensityd", "RandShiftIntensityd", "RandGaussianNoised"
        ):
            assert tuple(transform.keys) == ("image",)


# --------------------------------------------------------------- Req 15, 16
def test_labels_stay_within_the_valid_set_and_shapes_hold(cfg):
    cfg.use_augmentation = True
    transforms = build_transforms(cfg)
    image, label = _sample(cfg)

    for seed in range(12):
        aug_image, aug_label = apply_augmentation(transforms, image, label, seed=seed)
        assert aug_image.shape == image.shape
        assert aug_label.shape == label.shape
        assert set(np.unique(aug_label).tolist()) <= cfg.valid_labels


def test_a_fractional_label_raises_inside_the_pipeline(cfg):
    """The in-pipeline assertion, not just a test-side check."""
    from src.augment import AssertLabelValuesd

    guard = AssertLabelValuesd(keys="label", valid_labels=cfg.valid_labels)
    with pytest.raises(AssertionError, match="outside"):
        guard({"label": np.array([[[0.5, 1.0]]], dtype=np.float32)})


# ----------------------------------------------------------------- Req 17
def test_augmentation_off_gives_bitwise_identical_repeated_loads(cfg, cache, entry_factory):
    cfg.use_augmentation = False
    entries = [entry_factory(is_augmented=True, aug_seed=5)]
    ds = SliceDataset(entries, cfg, cache_dir=cache)

    first, second = ds[0], ds[0]
    assert np.array_equal(first["image"], second["image"])
    assert np.array_equal(first["label"], second["label"])
    assert ds.transforms is None


# ----------------------------------------------------------------- Req 18
def test_augmentation_on_varies_while_shapes_and_labels_hold(cfg, cache, entry_factory):
    cfg.use_augmentation = True
    ds_a = SliceDataset([entry_factory(is_augmented=True, aug_seed=1)], cfg, cache_dir=cache)
    ds_b = SliceDataset([entry_factory(is_augmented=True, aug_seed=999)], cfg, cache_dir=cache)

    a, b = ds_a[0], ds_b[0]
    assert a["image"].shape == b["image"].shape
    assert a["label"].shape == b["label"].shape
    assert set(np.unique(a["label"]).tolist()) <= cfg.valid_labels
    assert not np.array_equal(a["image"], b["image"])


# ----------------------------------------------------------------- Req 19
def test_the_same_seed_reproduces_the_same_augmentation(cfg):
    cfg.use_augmentation = True
    transforms = build_transforms(cfg)
    image, label = _sample(cfg)

    first = apply_augmentation(transforms, image, label, seed=42)
    second = apply_augmentation(transforms, image, label, seed=42)
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])


# ------------------------------------------------------------- Req 20, 21
def test_mixup_shares_one_lambda_across_image_and_one_hot(cfg):
    rng = np.random.default_rng(3)
    h, w = 16, 16
    img_a = np.zeros((4, h, w), dtype=np.float32)
    img_b = np.ones((4, h, w), dtype=np.float32)
    oh_a = one_hot(np.zeros((1, h, w), dtype=np.int64), cfg.num_classes)
    oh_b = one_hot(np.full((1, h, w), 3, dtype=np.int64), cfg.num_classes)

    mixed_img, mixed_oh, lam = segmentation_mixup(
        img_a, oh_a, img_b, oh_b, alpha=0.4, rng=rng, plane_a="axial", plane_b="axial"
    )

    assert mixed_img.shape == (4, h, w)
    assert mixed_oh.shape == (cfg.num_classes, h, w)
    # The same lambda produced both mixtures.
    assert np.allclose(mixed_img, 1.0 - lam, atol=1e-6)
    assert np.allclose(mixed_oh[0], lam, atol=1e-6)
    assert np.allclose(mixed_oh[3], 1.0 - lam, atol=1e-6)


def test_mixed_one_hot_sums_to_one_everywhere(cfg):
    rng = np.random.default_rng(4)
    h, w = 12, 12
    for trial in range(10):
        la = rng.integers(0, 5, size=(1, h, w)).astype(np.int64)
        lb = rng.integers(0, 5, size=(1, h, w)).astype(np.int64)
        _, mixed_oh, _ = segmentation_mixup(
            rng.normal(size=(4, h, w)).astype(np.float32), one_hot(la, 5),
            rng.normal(size=(4, h, w)).astype(np.float32), one_hot(lb, 5),
            alpha=0.4, rng=rng, plane_a="axial", plane_b="axial",
        )
        assert np.allclose(mixed_oh.sum(axis=0), 1.0, atol=1e-5)


def test_mixup_never_mixes_scalar_labels(cfg):
    """The mixed target is a per-pixel distribution, not a blended class index."""
    h, w = 8, 8
    oh_a = one_hot(np.zeros((1, h, w), dtype=np.int64), 5)
    oh_b = one_hot(np.full((1, h, w), 4, dtype=np.int64), 5)
    _, mixed_oh, lam = segmentation_mixup(
        np.zeros((4, h, w), np.float32), oh_a,
        np.zeros((4, h, w), np.float32), oh_b,
        alpha=0.4, rng=np.random.default_rng(0), plane_a="axial", plane_b="axial",
    )
    # Mass sits on classes 0 and 4 only -- never on the average class 2.
    assert np.allclose(mixed_oh[2], 0.0)
    assert np.allclose(mixed_oh[0] + mixed_oh[4], 1.0, atol=1e-6)


# ----------------------------------------------------------------- Req 22
def test_cross_plane_mixup_raises(cfg):
    h, w = 8, 8
    args = (np.zeros((4, h, w), np.float32), one_hot(np.zeros((1, h, w), np.int64), 5),
            np.zeros((4, h, w), np.float32), one_hot(np.zeros((1, h, w), np.int64), 5))
    with pytest.raises(ValueError, match="same plane"):
        segmentation_mixup(*args, alpha=0.4, plane_a="axial", plane_b="coronal")


def test_batch_mixup_pairs_within_a_plane_only(cfg):
    rng = np.random.default_rng(5)
    planes = ["axial", "axial", "coronal", "coronal"]
    images = rng.normal(size=(4, 4, 8, 8)).astype(np.float32)
    onehots = np.stack([one_hot(rng.integers(0, 5, (1, 8, 8)).astype(np.int64), 5)
                        for _ in range(4)])

    mixed_img, mixed_oh, lams = mixup_batch(images, onehots, planes, alpha=0.4, rng=rng)
    assert mixed_img.shape == images.shape
    assert np.allclose(mixed_oh.sum(axis=1), 1.0, atol=1e-5)
    assert len(lams) == 4


def test_mixup_rejects_a_non_positive_alpha(cfg):
    h, w = 4, 4
    args = (np.zeros((4, h, w), np.float32), one_hot(np.zeros((1, h, w), np.int64), 5),
            np.zeros((4, h, w), np.float32), one_hot(np.zeros((1, h, w), np.int64), 5))
    with pytest.raises(ValueError, match="alpha must be > 0"):
        segmentation_mixup(*args, alpha=0.0)


# ------------------------------------------------------------- Req 23, 49
@pytest.mark.parametrize(
    "use_augmentation,use_mixup,expected",
    [(False, False, False), (False, True, False), (True, False, False), (True, True, True)],
)
def test_all_four_flag_combinations_are_reachable(cfg, use_augmentation, use_mixup, expected):
    """use_augmentation: false disables mixup even when use_mixup: true."""
    cfg.use_augmentation = use_augmentation
    cfg.use_mixup = use_mixup
    assert cfg.mixup_enabled is expected
    assert (build_transforms(cfg) is not None) is use_augmentation
