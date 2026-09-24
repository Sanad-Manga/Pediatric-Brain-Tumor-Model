import numpy as np

from tools.simulate_2d_degradation import degrade_label, degrade_volume


def test_degrade_volume_preserves_shape():
    vol = np.random.rand(8, 6, 6).astype(np.float32)
    out = degrade_volume(vol, axis=0, slices_per_thick_slab=2)
    assert out.shape == vol.shape
    assert out.dtype == vol.dtype


def test_degrade_volume_thick_slabs_are_uniform_and_averaged():
    """A distinct value per slice along the simulated axis: kept ("real")
    slabs must become their own average, and gap slabs must copy the
    preceding kept slab's value -- not remain at the original per-slice
    values, or the "degradation" would be a no-op."""
    vol = np.arange(8).reshape(8, 1, 1).astype(np.float32) * np.ones((8, 3, 3), np.float32)
    out = degrade_volume(vol, axis=0, slices_per_thick_slab=2)
    per_slice = out.mean(axis=(1, 2))
    assert per_slice.tolist() == [0.5, 0.5, 0.5, 0.5, 4.5, 4.5, 4.5, 4.5]
    # every voxel within one slab must be identical (uniform blur)
    for i in range(0, 8, 2):
        assert np.all(out[i] == out[i + 1])


def test_degrade_volume_actually_changes_the_data():
    vol = np.random.rand(12, 6, 6).astype(np.float32)
    out = degrade_volume(vol, axis=0, slices_per_thick_slab=3)
    assert not np.array_equal(out, vol)


def test_degrade_volume_works_on_any_axis():
    vol = np.random.rand(6, 9, 6).astype(np.float32)
    for axis in (0, 1, 2):
        out = degrade_volume(vol, axis=axis, slices_per_thick_slab=3)
        assert out.shape == vol.shape


def test_degrade_volume_handles_length_not_a_multiple_of_slab():
    vol = np.random.rand(9, 4, 4).astype(np.float32)
    out = degrade_volume(vol, axis=0, slices_per_thick_slab=4)
    assert out.shape == vol.shape
    assert np.isfinite(out).all()


def test_degrade_label_majority_votes_and_stays_in_the_valid_set():
    seg = np.array([0, 0, 1, 1, 2, 2, 3, 3]).reshape(8, 1, 1).astype(np.int64) \
        * np.ones((8, 1, 1), np.int64)
    out = degrade_label(seg, axis=0, slices_per_thick_slab=2)
    assert out.flatten().tolist() == [0, 0, 0, 0, 2, 2, 2, 2]
    assert set(np.unique(out).tolist()) <= set(np.unique(seg).tolist())


def test_degrade_label_shape_and_dtype_preserved():
    seg = np.random.randint(0, 5, size=(10, 5, 5)).astype(np.int64)
    out = degrade_label(seg, axis=1, slices_per_thick_slab=3)
    assert out.shape == seg.shape
    assert out.dtype == seg.dtype
