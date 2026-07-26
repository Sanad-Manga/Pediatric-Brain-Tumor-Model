"""3D U-Net wrapper exposing model(x) -> (seg_logits, features) per 00_shared/CONTRACTS.md.

features is a pooled bottleneck embedding, consumed later by the domain-adaptation
(CORAL) and PCA/LDA sections — unused here but always returned.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn
from monai.networks.nets import UNet

IN_CHANNELS = 4  # t1c, t1n, t2f, t2w
NUM_CLASSES = 5  # background + ET, NET, CC, ED
FEATURE_DIM = 256


class FederatedUNet3D(nn.Module):
    """MONAI 3D U-Net, trained from scratch, with a pooled bottleneck feature head."""

    def __init__(
        self,
        in_channels: int = IN_CHANNELS,
        num_classes: int = NUM_CLASSES,
        feature_dim: int = FEATURE_DIM,
    ) -> None:
        super().__init__()
        self._init_args = (in_channels, num_classes, feature_dim)
        self.backbone = UNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=num_classes,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )
        # No pretrained weights are loaded anywhere in this class — MONAI's UNet
        # initializes its own layers randomly by construction.
        self._bottleneck_channels = 256
        self.feature_pool = nn.AdaptiveAvgPool3d(1)
        self.feature_proj = nn.Linear(self._bottleneck_channels, feature_dim)

        self._bottleneck_activation: torch.Tensor | None = None
        self._register_bottleneck_hook()

    def _register_bottleneck_hook(self) -> None:
        # MONAI's UNet builds a recursive nn.Sequential; the deepest submodel's
        # first submodule is the bottleneck conv block. Grab it via a forward hook
        # so we can pool its activation without altering the backbone's own output.
        bottleneck_module = self._find_bottleneck_module(self.backbone)

        def _hook(_module: nn.Module, _inp: tuple, out: torch.Tensor) -> None:
            self._bottleneck_activation = out

        bottleneck_module.register_forward_hook(_hook)

    @staticmethod
    def _find_bottleneck_module(root: nn.Module) -> nn.Module:
        deepest = None
        max_depth = -1

        def _walk(mod: nn.Module, depth: int) -> None:
            nonlocal deepest, max_depth
            children = list(mod.children())
            if not children:
                return
            if depth > max_depth:
                max_depth = depth
                deepest = children[0]
            for child in children:
                _walk(child, depth + 1)

        _walk(root, 0)
        if deepest is None:
            raise RuntimeError("Could not locate a bottleneck module in the UNet backbone")
        return deepest

    def __deepcopy__(self, memo: dict) -> "FederatedUNet3D":
        # The bottleneck forward-hook closes over `self` at construction time,
        # so a naive deepcopy would leave the copy's hook writing into the
        # original instance's `_bottleneck_activation`. Rebuilding fresh (with
        # its own correctly-bound hook) and loading the copied weights avoids
        # that entirely.
        new_model = FederatedUNet3D(*self._init_args)
        new_model.load_state_dict(copy.deepcopy(self.state_dict(), memo))
        return new_model

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seg_logits = self.backbone(x)
        if self._bottleneck_activation is None:
            raise RuntimeError("Bottleneck hook did not fire during forward pass")
        pooled = self.feature_pool(self._bottleneck_activation).flatten(1)
        features = self.feature_proj(pooled)
        return seg_logits, features


def build_model() -> FederatedUNet3D:
    return FederatedUNet3D()
