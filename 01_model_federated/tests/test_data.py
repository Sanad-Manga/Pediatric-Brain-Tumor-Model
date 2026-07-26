import pytest

from src.data import build_dataset, load_manifest


def test_dummy_dataset_sample_shape(small_manifest):
    manifest_path = small_manifest("hospA", 3)
    ds = build_dataset(manifest_path, data_mode="dummy")
    assert len(ds) == 3
    x, y = ds[0]
    assert x.shape == (4, 96, 96, 96)
    assert y.shape == (96, 96, 96)
    assert y.min() >= 0 and y.max() <= 4


def test_dummy_dataset_matches_real_manifest_counts(real_manifest_path):
    assert len(load_manifest(real_manifest_path("hospitalA"))) == 53
    assert len(load_manifest(real_manifest_path("hospitalB"))) == 92
    assert len(load_manifest(real_manifest_path("heldout"))) == 82

    ds_a = build_dataset(real_manifest_path("hospitalA"), data_mode="dummy")
    ds_b = build_dataset(real_manifest_path("hospitalB"), data_mode="dummy")
    assert len(ds_a) == 53
    assert len(ds_b) == 92


def test_missing_manifest_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_manifest(str(tmp_path / "does_not_exist.json"))


def test_empty_manifest_raises(tmp_path):
    path = tmp_path / "empty.json"
    path.write_text("[]")
    with pytest.raises(ValueError):
        load_manifest(str(path))


def test_real_mode_without_cache_path_raises(small_manifest):
    manifest_path = small_manifest("hospA", 2)
    with pytest.raises(ValueError):
        build_dataset(manifest_path, data_mode="real", cache_path=None)
