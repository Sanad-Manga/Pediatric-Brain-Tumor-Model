"""A throwaway segmentation network so the ablation runner is testable today.

`01_model_federated` owns the real 3D U-Net. This module exists only so the
runner, the metrics and the CSV writing can be exercised end-to-end before that
model lands. It honours the model interface from CONTRACTS.md:

    model(x) -> (seg_logits, features)

so swapping in the real network is a one-line change in the runner.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


class DummySegNet(nn.Module):
    """Tiny random conv net: ``(B,4,D,H,W) -> ((B,5,D,H,W), (B,F))``."""

    def __init__(self, in_channels: int = 4, num_classes: int = 5, features: int = 8):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.encoder = nn.Conv3d(in_channels, features, kernel_size=3, padding=1)
        self.act = nn.ReLU(inplace=True)
        self.head = nn.Conv3d(features, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.act(self.encoder(x))
        seg_logits = self.head(h)
        # bottleneck embedding, matching the (seg_logits, features) contract
        feats = torch.mean(h, dim=(2, 3, 4))
        return seg_logits, feats


def save_dummy_checkpoint(path: str | Path, in_channels: int = 4, num_classes: int = 5,
                          seed: int = 0) -> Path:
    """Write a randomly-initialized checkpoint for testing the runner."""
    torch.manual_seed(seed)
    model = DummySegNet(in_channels=in_channels, num_classes=num_classes)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "arch": "DummySegNet",
            "in_channels": in_channels,
            "num_classes": num_classes,
        },
        path,
    )
    return path


def load_checkpoint(path: str | Path, model: nn.Module) -> nn.Module:
    """Load a checkpoint into ``model``, failing loudly on any key mismatch.

    ``strict=True`` is deliberate: silently tolerating missing or unexpected keys
    would let a mismatched checkpoint produce meaningless Dice numbers that look
    perfectly plausible in the results CSV.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"checkpoint not found: {path}. Pass --dummy-checkpoint to run against "
            f"a randomly-initialized DummySegNet instead."
        )
    blob = torch.load(path, map_location="cpu", weights_only=False)
    state = blob.get("model_state_dict", blob) if isinstance(blob, dict) else blob

    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"checkpoint {path} does not match the model architecture.\n{exc}"
        ) from exc
    return model
