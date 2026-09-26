"""Req 6-9: 3D and 2D pre-processing against independent computations, and the frame/geometry of both paths."""
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from _helpers import TINY, make_volume, tiny_config

from rstar import RStarSegmenter, preprocess
from rstar.guards import IdentityStub, center_of_mass
from rstar.sections import import_submodule

ALL = (True, True, True, True)


# ------------------------------------------------------------------------------------------------ Req 6
def test_the_96_cube_has_the_right_shape_dtype_and_float16_values():
    vol, _ = make_volume()
    cube = preprocess.to_96cube(vol, ALL)
    assert cube.shape == (4, 96, 96, 96) and cube.dtype == np.float32
    assert np.array_equal(cube.astype(np.float16).astype(np.float32), cube)


def test_the_resample_equals_an_independent_interpolate_call():
    vol, _ = make_volume(seed=3)
    cube = preprocess.to_96cube(vol, ALL)
    for c in range(4):
        expected = F.interpolate(torch.from_numpy(vol[c])[None, None], size=(96, 96, 96), mode="trilinear", align_corners=False)[0, 0]
        assert np.array_equal(cube[c], expected.numpy().astype(np.float16).astype(np.float32))


def test_zscore_leaves_background_at_zero_and_normalises_the_brain():
    vol, _ = make_volume(seed=1)
    z = preprocess.zscore_3d(preprocess.to_96cube(vol, ALL)[0])
    raw = preprocess.to_96cube(vol, ALL)[0]
    assert (z[raw == 0] == 0).all()
    brain = raw > 0
    assert abs(float(z[brain].mean())) < 1e-4 and abs(float(z[brain].std()) - 1.0) < 1e-4


def test_zscore_handles_a_constant_and_an_all_zero_channel_without_nan():
    const = np.zeros((8, 8, 8), np.float32); const[2:6, 2:6, 2:6] = 7.0
    out = preprocess.zscore_3d(const)
    assert np.isfinite(out).all() and (out == 0).all()
    zero = np.zeros((8, 8, 8), np.float32)
    assert np.array_equal(preprocess.zscore_3d(zero), zero)


def test_the_3d_input_matches_section_01_and_absent_channels_are_exactly_zero():
    vol, _ = make_volume(seed=2)
    present = (True, False, True, True)
    x = preprocess.prepare_3d_input(vol, present)
    assert x.shape == (1, 4, 96, 96, 96) and x.dtype == np.float32
    assert (x[0, 1] == 0).all(), "an absent channel must be exactly zero AFTER the z-score"
    reference = import_submodule("01", "data")._zscore_normalize
    cube = preprocess.to_96cube(vol, ALL)
    for c in (0, 2, 3):
        assert np.array_equal(x[0, c], reference(cube[c]))
    assert (preprocess.to_96cube(vol, present)[1] == 0).all()


# ------------------------------------------------------------------------------------------------ Req 7
def _asymmetric_volume():
    rng = np.random.default_rng(5)
    vol = rng.random((4, 12, 10, 8)).astype(np.float32) + 0.1
    vol[:, :, :, :2] = 0.0             # axial slices 0, 1 without brain
    vol[:, :, 7:, :] = 0.0             # coronal slices j = 7, 8, 9 without brain
    vol[:, :3, :, :] = 0.0
    return vol


def test_axial_slices_equal_independent_indexing():
    vol = _asymmetric_volume()
    norm, brain = preprocess.normalize_2d(vol)
    indices, images = preprocess.extract_slices(norm, brain, "axial")
    assert indices == [k for k in range(8) if (vol[:, :, :, k] > 0).any()] and indices[0] == 2
    for i, k in enumerate(indices):
        expected = np.flip(np.flip(norm[:, :, :, k], axis=1), axis=2)
        assert np.array_equal(images[i], expected.astype(np.float16).astype(np.float32))


def test_coronal_slices_equal_independent_indexing():
    vol = _asymmetric_volume()
    norm, brain = preprocess.normalize_2d(vol)
    indices, images = preprocess.extract_slices(norm, brain, "coronal")
    assert indices == [k for k in range(10) if (vol[:, :, 10 - 1 - k, :] > 0).any()]
    assert len(indices) == 7 and indices[0] == 3                 # slices j = 7, 8, 9 (k = 2, 1, 0) hold no brain
    for i, k in enumerate(indices):
        j = 10 - 1 - k
        expected = np.flip(norm[:, :, j, :], axis=1)
        assert np.array_equal(images[i], expected.astype(np.float16).astype(np.float32))
    assert images.dtype == np.float32 and images.shape[1:] == (4, 12, 8)


def test_the_2d_zscore_uses_brain_voxels_only():
    vol = _asymmetric_volume()
    norm, _ = preprocess.normalize_2d(vol)
    for c in range(4):
        brain = vol[c] > 0
        assert (norm[c][~brain] == 0).all()
        assert abs(float(norm[c][brain].mean())) < 1e-5 and abs(float(norm[c][brain].std()) - 1.0) < 1e-4


@pytest.mark.parametrize("plane", ["axial", "coronal"])
def test_restack_inverts_extraction_up_to_the_frame_flip(plane):
    vol = _asymmetric_volume()
    norm, brain = preprocess.normalize_2d(vol)
    indices, images = preprocess.extract_slices(norm, brain, plane)
    out = preprocess.restack(images.transpose(1, 0, 2, 3), plane, indices, vol.shape[1:])
    expected = np.zeros_like(out)
    f16 = norm.astype(np.float16).astype(np.float32)
    flipped = np.flip(f16, axis=(1, 2))
    if plane == "axial":
        expected[:, :, :, indices] = flipped[:, :, :, indices]
    else:
        expected[:, :, indices, :] = flipped[:, :, indices, :]
    assert np.array_equal(out, expected)
    axis_len = out.shape[3] if plane == "axial" else out.shape[2]
    empty = [k for k in range(axis_len) if k not in indices]
    assert empty, "the test volume must contain slices without brain"
    left_out = out[:, :, :, empty] if plane == "axial" else out[:, :, empty, :]
    assert (left_out == 0).all()


def test_padding_matches_section_03_and_is_invertible():
    slices = import_submodule("03", "slices")
    x = np.random.default_rng(0).random((4, 155, 240)).astype(np.float32)
    ours, our_pad = preprocess.pad_to(x, (256, 256))
    theirs, their_pad = slices.pad_to(x, (256, 256))
    assert np.array_equal(ours, theirs) and our_pad == their_pad == ((50, 51), (8, 8))
    assert np.array_equal(preprocess.unpad(ours, our_pad), x)
    with pytest.raises(ValueError):
        preprocess.pad_to(np.zeros((4, 300, 10)), (256, 256))


# ------------------------------------------------------------------------------------------------ Req 8
def test_the_frame_constant_is_the_verified_one():
    assert preprocess.FRAME_FLIPS == (0, 1)


def test_the_2d_path_puts_an_asymmetric_blob_back_exactly_where_it_was():
    vol, blob = make_volume()
    seg = RStarSegmenter(tiny_config(), models_2d=[IdentityStub()], models_3d=[IdentityStub()])
    p2 = seg._probs_2d(vol)
    assert p2.shape == (5, *TINY)
    assert np.array_equal(p2[1] > 0.5, blob)


def test_a_wrong_frame_constant_is_caught(monkeypatch):
    vol, blob = make_volume()
    seg = RStarSegmenter(tiny_config(), models_2d=[IdentityStub()], models_3d=[IdentityStub()])
    monkeypatch.setattr(preprocess, "FRAME_FLIPS", (0,))
    assert not np.array_equal(seg._probs_2d(vol)[1] > 0.5, blob)


# ------------------------------------------------------------------------------------------------ Req 9
def test_the_3d_path_keeps_the_blob_within_tolerance():
    vol, blob = make_volume()
    seg = RStarSegmenter(tiny_config(), models_2d=[IdentityStub()], models_3d=[IdentityStub()])
    predicted = seg._probs_3d(vol, ALL)[1] > 0.5
    assert predicted.any()
    offset = float(np.linalg.norm(center_of_mass(predicted) - center_of_mass(blob)))
    assert offset <= max(1.0, 3.0 * max(TINY) / 240.0)


class _Pointwise(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.conv = torch.nn.Conv3d(4, 5, kernel_size=1)

    def forward(self, x):
        return self.conv(x), None


def test_flip_tta_equals_no_tta_for_a_pointwise_model():
    vol, _ = make_volume(seed=4)
    model = _Pointwise().eval()
    seg = RStarSegmenter(tiny_config(), models_2d=[IdentityStub()], models_3d=[model])
    x = torch.from_numpy(preprocess.prepare_3d_input(vol, ALL))
    with torch.no_grad():
        direct = torch.softmax(model(x)[0], dim=1)
        direct = F.interpolate(direct, size=TINY, mode="trilinear", align_corners=False)[0].numpy()
    assert np.allclose(seg._probs_3d(vol, ALL), direct, atol=1e-5)
