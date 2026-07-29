"""A compact U-Net honouring the shared model interface, 2D or 3D.

CONTRACTS.md fixes the interface::

    model(x) -> (seg_logits, features)

``features`` is the pooled bottleneck embedding that section 02's CORAL and the
PCA/LDA plots consume. It is returned here so the interface is real rather than
theoretical, even though this section does not implement domain adaptation.

**The same network takes 2D or 3D input.** ``spatial_dims=2`` gives
``(B, 4, H, W) -> (B, C, H, W)``; ``spatial_dims=3`` gives
``(B, 4, D, H, W) -> (B, C, D, H, W)``. Only the convolution / norm / pooling
operators change -- the topology, the interface and the bottleneck embedding are
identical, so a checkpoint trained either way flows through the same evaluation
path. :func:`build_model` infers the rank from an example batch when it is not
stated explicitly.

Section 03's *data* path is 2D slices (that is the dataset now). The 3D mode
exists because the model contract allows either.

Section 01 owns the production architecture. This exists so section 03 can train
and score end to end without waiting on it -- keep it small and boring, and swap
it out when section 01 lands.
"""

from __future__ import annotations

import torch
import torch.nn as nn

#: operator table, indexed by spatial rank
_OPS = {
    2: (nn.Conv2d, nn.ConvTranspose2d, nn.InstanceNorm2d, nn.MaxPool2d, nn.AdaptiveAvgPool2d),
    3: (nn.Conv3d, nn.ConvTranspose3d, nn.InstanceNorm3d, nn.MaxPool3d, nn.AdaptiveAvgPool3d),
}


def _block(conv, norm, in_ch: int, out_ch: int) -> nn.Sequential:
    """Two 3x3(x3) convolutions with instance norm -- batch-size-agnostic."""
    return nn.Sequential(
        conv(in_ch, out_ch, 3, padding=1, bias=False),
        norm(out_ch, affine=True),
        nn.LeakyReLU(0.01, inplace=True),
        conv(out_ch, out_ch, 3, padding=1, bias=False),
        norm(out_ch, affine=True),
        nn.LeakyReLU(0.01, inplace=True),
    )


class UNet(nn.Module):
    """U-Net over a slice or a volume, depending on ``spatial_dims``.

    ``width`` scales every level. The default (16, three downsamples) is chosen
    to be trainable on CPU; raise it once a GPU is available.
    """

    def __init__(self, in_channels: int = 4, num_classes: int = 5,
                 width: int = 16, depth: int = 3, spatial_dims: int = 2):
        super().__init__()
        if spatial_dims not in _OPS:
            raise ValueError(f"spatial_dims must be 2 or 3, got {spatial_dims}")
        self.spatial_dims = spatial_dims
        conv, deconv, norm, pool, gap = _OPS[spatial_dims]

        chans = [width * (2 ** i) for i in range(depth + 1)]

        self.encoders = nn.ModuleList()
        prev = in_channels
        for c in chans[:-1]:
            self.encoders.append(_block(conv, norm, prev, c))
            prev = c
        self.bottleneck = _block(conv, norm, prev, chans[-1])
        self.pool = pool(2)

        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for c in reversed(chans[:-1]):
            self.ups.append(deconv(c * 2, c, 2, stride=2))
            self.decoders.append(_block(conv, norm, c * 2, c))

        self.head = conv(chans[0], num_classes, 1)
        self.embed = gap(1)

    def forward(self, x):
        expected = self.spatial_dims + 2
        if x.ndim != expected:
            raise ValueError(
                f"spatial_dims={self.spatial_dims} expects a {expected}D input "
                f"(B, C, {'D, ' if self.spatial_dims == 3 else ''}H, W), "
                f"got shape {tuple(x.shape)}"
            )

        skips = []
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        features = self.embed(x).flatten(1)      # (B, chans[-1]) for CORAL / PCA

        for up, dec, skip in zip(self.ups, self.decoders, reversed(skips)):
            x = up(x)
            x = dec(torch.cat([x, skip], dim=1))

        return self.head(x), features


#: the 2D case is what section 03's data path uses
UNet2D = UNet


class TumorTypeHead(nn.Module):
    """Auxiliary classifier over the model's bottleneck ``features``.

    Deliberately outside ``UNet.forward`` and the shared ``model(x) ->
    (seg_logits, features)`` interface in CONTRACTS.md -- other sections may
    unpack that tuple assuming exactly two elements. This head consumes the
    already-exposed ``features`` embedding as a separate module, so nothing
    about the shared contract changes; a caller who doesn't know this head
    exists is unaffected.

    Trained against the geometric dmg_like/astrocytoma_like proxy
    (`tumor_type.py`) as a genuine second task: the point is to test whether
    the network's own features carry enough location/enhancement information
    to predict tumour type, not to hand it the label directly.
    """

    def __init__(self, in_features: int, num_types: int = 2, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(hidden, num_types),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def build_model(cfg, spatial_dims: int | None = None, example=None, **kwargs) -> UNet:
    """Construct the network, inferring the rank from ``example`` if given."""
    if spatial_dims is None:
        if example is not None:
            spatial_dims = int(example.ndim) - 2
        else:
            spatial_dims = int(cfg.data.get("spatial_dims", 2))
    return UNet(
        in_channels=len(cfg.modalities),
        num_classes=cfg.num_classes,
        spatial_dims=spatial_dims,
        **kwargs,
    )
