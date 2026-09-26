"""Sliding-window inference for models trained on patches (src/patch_data.py).

A patch-trained network only ever saw crops, so a whole volume is scored by sliding
a window of the training-patch size across it, blending the overlapping windows'
class probabilities with a gaussian weight (voxels near a window's centre count
more than voxels at its edge), and optionally averaging over flipped copies of the
volume (test-time augmentation).
"""
from __future__ import annotations

from typing import Sequence

import torch
from monai.inferers import sliding_window_inference


@torch.no_grad()
def predict_volume(
    model: torch.nn.Module,
    image: torch.Tensor,
    roi_size: Sequence[int],
    overlap: float = 0.5,
    flips: Sequence[Sequence[int]] = ((),),
    sw_batch_size: int = 2,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """Class probabilities for one normalised volume.

    image: float tensor (4, X, Y, Z), already z-scored like the training patches.
    flips: spatial axes (0, 1, 2) to flip for each test-time-augmentation view;
        `((),)` is a single unflipped pass. Each view's probabilities are flipped
        back before averaging.
    Returns float32 probabilities (5, X, Y, Z) on the CPU. A volume smaller than
    the window on some axis is padded internally and cropped back.
    """
    if image.ndim != 4:
        raise ValueError(f"image must be (C, X, Y, Z), got shape {tuple(image.shape)}")
    if len(roi_size) != 3:
        raise ValueError(f"roi_size must have three entries, got {tuple(roi_size)}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    model = model.to(device).eval()
    volume = image.detach().float().unsqueeze(0).cpu()  # (1, C, X, Y, Z)

    def predictor(window: torch.Tensor) -> torch.Tensor:
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            logits = model(window)[0]
        return torch.softmax(logits.float(), dim=1)

    total = None
    for flip_axes in flips:
        dims = [int(a) + 2 for a in flip_axes]
        view = torch.flip(volume, dims=dims) if dims else volume
        probs = sliding_window_inference(
            view, roi_size=tuple(int(r) for r in roi_size), sw_batch_size=sw_batch_size,
            predictor=predictor, overlap=overlap, mode="gaussian",
            sw_device=device, device=torch.device("cpu"),
        )
        probs = torch.flip(probs, dims=dims) if dims else probs
        total = probs if total is None else total + probs
    return (total / len(flips))[0].float().cpu()
