"""Slice extraction, cache format, normalisation stats and pad/unpad."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.slices import (
    compute_norm_stats,
    compute_pad,
    extract_plane_slices,
    index_cache,
    load_subject_plane,
    normalize,
    pad_to,
    restack_plane_slices,
    unpad,
)


# ----------------------------------------------------------------- Req 5
def test_axial_and_coronal_shapes_match_the_measured_counts(cfg):
    """240x240x155 -> 155 axial slices of 240x240, 240 coronal of 240x155."""
    volume = np.zeros((240, 240, 155), dtype=np.float32)

    axial = extract_plane_slices(volume, cfg.plane_axis("axial"))
    coronal = extract_plane_slices(volume, cfg.plane_axis("coronal"))

    assert axial.shape == (155, 240, 240)
    assert coronal.shape == (240, 240, 155)


def test_sagittal_is_never_emitted(cfg, cache, manifests):
    assert "sagittal" not in cfg.planes
    for path in Path(cache).glob("*.npz"):
        assert "sagittal" not in path.name


def test_restack_is_the_exact_inverse_of_slicing(cfg):
    rng = np.random.default_rng(0)
    volume = rng.random((12, 10, 8)).astype(np.float32)
    for plane in ("axial", "coronal"):
        axis = cfg.plane_axis(plane)
        restacked = restack_plane_slices(extract_plane_slices(volume, axis), axis)
        assert np.array_equal(restacked, volume)


# ----------------------------------------------------------------- Req 6
def test_cache_file_holds_every_required_array(cfg, cache, manifests):
    subject = manifests["hospitalA"][0]
    data = load_subject_plane(cache, subject, "axial")

    for mod in cfg.modalities:
        assert data[mod].dtype == np.float16
    assert data["seg"].dtype == np.uint8
    assert set(np.unique(data["seg"]).tolist()) <= {0, 1, 2, 3, 4}
    assert data["norm_mean"].shape == (len(cfg.modalities),)
    assert data["norm_std"].shape == (len(cfg.modalities),)
    assert data["has_tumor"].dtype == bool
    assert data["has_tumor"].shape == (data["seg"].shape[0],)
    assert str(data["subject_id"]) == subject
    assert str(data["plane"]) == "axial"
    assert str(data["hospital"]) == "hospitalA"


# ----------------------------------------------------------------- Req 7
def test_normalisation_uses_volume_level_stats_not_per_slice(cfg, cache, manifests):
    """Two slices of one volume must be normalised by identical constants."""
    subject = manifests["hospitalA"][0]
    data = load_subject_plane(cache, subject, "axial")
    mean, std = float(data["norm_mean"][0]), float(data["norm_std"][0])

    raw_a, raw_b = data["t1c"][2], data["t1c"][5]
    norm_a = normalize(raw_a, mean, std, cfg.min_std)
    norm_b = normalize(raw_b, mean, std, cfg.min_std)

    # Recovering the constants from either slice gives the same answer.
    scale_a = (raw_a.astype(np.float32) - norm_a * np.float32(std)).mean()
    scale_b = (raw_b.astype(np.float32) - norm_b * np.float32(std)).mean()
    assert np.isclose(scale_a, scale_b, atol=1e-2)
    assert np.isclose(scale_a, mean, atol=1e-2)


def test_norm_stats_ignore_background_zeros():
    volume = np.zeros((8, 8, 8), dtype=np.float32)
    volume[2:6, 2:6, 2:6] = 100.0
    mean, std = compute_norm_stats(volume)
    assert mean == pytest.approx(100.0)   # zeros excluded, not averaged in


# ----------------------------------------------------------------- Req 8
def test_all_zero_volume_normalises_to_zero_without_nan():
    volume = np.zeros((6, 6, 6), dtype=np.float32)
    mean, std = compute_norm_stats(volume, min_std=1e-6)
    assert std >= 1e-6
    out = normalize(volume[0], mean, std, min_std=1e-6)
    assert np.all(out == 0.0)
    assert not np.isnan(out).any()


# ---------------------------------------------------------------- Req 10, 11
@pytest.mark.parametrize("shape", [(240, 240), (240, 155)])
def test_pad_round_trip_is_bitwise_identical(shape):
    rng = np.random.default_rng(1)
    arr = rng.random((4, *shape)).astype(np.float32)
    padded, pad = pad_to(arr, (256, 256))
    assert padded.shape == (4, 256, 256)
    assert np.array_equal(unpad(padded, pad), arr)


def test_padding_is_centred_with_the_remainder_at_the_end():
    assert compute_pad((240, 155), (256, 256)) == ((8, 8), (50, 51))


def test_padding_never_resizes_the_content():
    arr = np.ones((1, 240, 155), dtype=np.float32)
    padded, pad = pad_to(arr, (256, 256))
    assert padded.sum() == arr.sum()          # only zeros were added
    assert np.array_equal(unpad(padded, pad), arr)


# ----------------------------------------------------------------- Req 12
def test_slice_larger_than_common_size_is_a_hard_error():
    arr = np.zeros((1, 300, 300), dtype=np.float32)
    with pytest.raises(ValueError, match="exceeds common_size"):
        pad_to(arr, (256, 256))


# ------------------------------------------------------------ cache indexing
def test_index_cache_reports_every_subject_and_plane(cfg, cache, manifests):
    index = index_cache(cache, cfg.planes)
    all_subjects = {sid for ids in manifests.values() for sid in ids}
    assert set(index) == all_subjects
    for planes in index.values():
        assert set(planes) == set(cfg.planes)


def test_missing_cache_names_the_path_and_the_config_key(tmp_path):
    with pytest.raises(FileNotFoundError, match="paths.cache_2d"):
        index_cache(tmp_path / "nope")
