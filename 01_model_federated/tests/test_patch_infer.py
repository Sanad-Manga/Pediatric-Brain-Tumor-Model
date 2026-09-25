import pytest
import torch
import torch.nn as nn

from src.patch_infer import predict_volume


class PointwiseStub(nn.Module):
    """1x1x1 convolution: every voxel's prediction depends only on that voxel's input, so
    ANY correct stitching or flip bookkeeping must reproduce the direct full-volume result."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(4, 5, kernel_size=1)

    def forward(self, x):
        return self.conv(x), torch.zeros(x.shape[0], 8)


def _direct(model, image):
    with torch.no_grad():
        return torch.softmax(model(image[None])[0], dim=1)[0]


@pytest.mark.parametrize("roi", [(16, 16, 16), (16, 12, 8)])
@pytest.mark.parametrize("shape", [(37, 29, 23), (16, 16, 16), (10, 12, 9)])
def test_sliding_window_equals_direct_full_volume_softmax(shape, roi):  # Req 43
    torch.manual_seed(0)
    model, image = PointwiseStub(), torch.randn(4, *shape)
    out = predict_volume(model, image, roi_size=roi, overlap=0.5, device="cpu")
    assert tuple(out.shape) == (5, *shape)
    assert out.dtype == torch.float32 and out.device.type == "cpu"
    assert torch.allclose(out, _direct(model, image), atol=1e-5)
    assert torch.allclose(out.sum(dim=0), torch.ones(shape), atol=1e-5)


def test_flip_tta_is_aligned_with_no_tta():  # Req 44
    torch.manual_seed(1)
    model, image = PointwiseStub(), torch.randn(4, 30, 26, 22)
    plain = predict_volume(model, image, roi_size=(16, 16, 16), device="cpu")
    for flips in ([(), (0,), (1,), (2,)], [(0, 1)], [(0, 1, 2), ()]):
        tta = predict_volume(model, image, roi_size=(16, 16, 16), flips=flips, device="cpu")
        assert torch.allclose(tta, plain, atol=1e-5), flips


class CountingStub(PointwiseStub):
    def __init__(self):
        super().__init__()
        self.window_shapes, self.n_windows = set(), 0

    def forward(self, x):
        self.window_shapes.add(tuple(x.shape[2:]))
        self.n_windows += x.shape[0]
        return super().forward(x)


def test_windows_have_the_requested_size_and_overlap_is_honoured():  # Req 43
    """A pointwise model is correct under ANY tiling, so equality with the direct result cannot show that
    roi_size / overlap are actually used; count what the model is really shown."""
    image = torch.randn(4, 37, 29, 23)
    dense, sparse = CountingStub(), CountingStub()
    predict_volume(dense, image, roi_size=(16, 12, 8), overlap=0.5, device="cpu")
    predict_volume(sparse, image, roi_size=(16, 12, 8), overlap=0.0, device="cpu")
    assert dense.window_shapes == {(16, 12, 8)} and sparse.window_shapes == {(16, 12, 8)}
    assert dense.n_windows > sparse.n_windows
