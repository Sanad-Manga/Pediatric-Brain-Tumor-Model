"""Cache layout, indexing, and pad/unpad."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.slices import (
    INDEX_FILENAME,
    MODALITY_ORDER,
    build_index,
    compute_pad,
    extract_plane_slices,
    index_cache,
    list_slice_files,
    list_subjects,
    load_slice,
    pad_to,
    read_index,
    slice_path,
    unpad,
    write_index,
)


# ------------------------------------------------------------- cache layout
def test_cache_uses_one_file_per_slice(cfg, cache, manifests):
    """<subject>/<plane>/slice_NNN.npz -- the format the team produces."""
    subject = manifests["hospitalA"][0]
    for plane in cfg.planes:
        files = list_slice_files(cache, subject, plane)
        assert files, f"no slice files for {plane}"
        assert files[0].name == "slice_000.npz"
        assert files[0].parent.name == plane
        assert files[0].parent.parent.name == subject


def test_slice_holds_image_and_mask_only(cfg, cache, manifests):
    subject = manifests["hospitalA"][0]
    data = load_slice(cache, subject, "axial", 0)

    assert set(data) == {"image", "mask"}
    assert data["image"].dtype == np.float32
    assert data["image"].shape[0] == len(MODALITY_ORDER) == len(cfg.modalities)
    assert data["mask"].dtype == np.uint8
    assert set(np.unique(data["mask"]).tolist()) <= {0, 1, 2, 3, 4}


def test_channel_order_is_t1c_t1n_t2f_t2w(cfg):
    """Confirmed with the team; the config must agree with the cache."""
    assert tuple(cfg.modalities) == MODALITY_ORDER == ("t1c", "t1n", "t2f", "t2w")


def test_plane_slice_counts_and_shapes_differ_as_briefed(cfg, cache, manifests):
    """Axial and coronal genuinely differ in count and shape."""
    subject = manifests["hospitalA"][0]
    ax = load_slice(cache, subject, "axial", 0)["image"]
    co = load_slice(cache, subject, "coronal", 0)["image"]

    assert len(list_slice_files(cache, subject, "axial")) != \
           len(list_slice_files(cache, subject, "coronal"))
    assert ax.shape[-2:] != co.shape[-2:]


def test_sagittal_is_never_present(cfg, cache, manifests):
    assert "sagittal" not in cfg.planes
    for subject in manifests["hospitalA"]:
        assert not (Path(cache) / subject / "sagittal").exists()


def test_missing_slice_names_the_path(cfg, cache, manifests):
    with pytest.raises(FileNotFoundError, match="slice file not found"):
        load_slice(cache, manifests["hospitalA"][0], "axial", 9999)


def test_missing_cache_names_the_config_key(tmp_path):
    with pytest.raises(FileNotFoundError, match="paths.cache_2d"):
        list_subjects(tmp_path / "nope")


# ----------------------------------------------------------------- indexing
def test_index_records_tumour_and_et_per_slice(cfg, cache, manifests):
    index = index_cache(cache, cfg.planes)
    all_subjects = {sid for ids in manifests.values() for sid in ids}
    assert set(index) == all_subjects

    for subject, planes in index.items():
        assert set(planes) == set(cfg.planes)
        for plane, summary in planes.items():
            indices = summary["indices"]
            assert len(indices) == summary["n_slices"]
            assert indices == sorted(indices)
            # every list holds real slice indices, all of them present
            assert set(summary["tumor"]) <= set(indices)
            assert set(summary["et"]) <= set(summary["tumor"])   # label 1 is a tumour label
            assert set(summary["empty"]) == set(indices) - set(summary["tumor"])


def test_index_matches_a_direct_scan_of_the_masks(cfg, cache, manifests):
    """The index is a cache of truth -- verify it against the masks themselves."""
    index = index_cache(cache, cfg.planes)
    subject = manifests["hospitalA"][0]

    for plane in cfg.planes:
        summary = index[subject][plane]
        tumor, et = set(summary["tumor"]), set(summary["et"])
        for i in summary["indices"]:
            mask = load_slice(cache, subject, plane, i)["mask"]
            assert (i in tumor) == bool((mask != 0).any()), (plane, i)
            assert (i in et) == bool((mask == 1).any()), (plane, i)


def test_index_is_written_and_read_back(cfg, cache):
    assert (Path(cache) / INDEX_FILENAME).exists()
    index = read_index(cache)
    assert index["modalities"] == list(MODALITY_ORDER)
    assert index["n_subjects"] > 0
    assert index["n_slice_files"] > 0


def test_missing_index_tells_you_how_to_build_it(cfg, tmp_path, manifests):
    from src.dummy import make_dummy_cache

    bare = tmp_path / "bare"
    make_dummy_cache(bare, manifests["hospitalA"][:1], cfg,
                     volume_shape=(8, 8, 6), seed=1, with_index=False)
    with pytest.raises(FileNotFoundError, match="run.py index"):
        index_cache(bare, cfg.planes)


def test_index_is_deterministic(cfg, cache):
    a = build_index(cache, planes=cfg.planes, progress_every=0)
    b = build_index(cache, planes=cfg.planes, progress_every=0)
    assert a == b


def test_index_round_trips_through_json(cfg, cache, tmp_path):
    index = build_index(cache, planes=cfg.planes, progress_every=0)
    write_index(index, tmp_path)
    assert read_index(tmp_path) == index


# ---------------------------------------------------------- pad / unpad
@pytest.mark.parametrize("shape", [(240, 240), (240, 155), (24, 24), (24, 16)])
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


def test_slice_larger_than_common_size_is_a_hard_error():
    arr = np.zeros((1, 300, 300), dtype=np.float32)
    with pytest.raises(ValueError, match="exceeds common_size"):
        pad_to(arr, (256, 256))


# ------------------------------------------------------------- plane axes
def test_axial_and_coronal_axes_match_the_measured_counts(cfg):
    """240x240x155 -> 155 axial slices of 240x240, 240 coronal of 240x155."""
    volume = np.zeros((240, 240, 155), dtype=np.float32)
    assert extract_plane_slices(volume, cfg.plane_axis("axial")).shape == (155, 240, 240)
    assert extract_plane_slices(volume, cfg.plane_axis("coronal")).shape == (240, 240, 155)


def test_slice_path_is_zero_padded():
    p = slice_path("/tmp/cache", "SUBJ", "axial", 7)
    assert p.name == "slice_007.npz"
