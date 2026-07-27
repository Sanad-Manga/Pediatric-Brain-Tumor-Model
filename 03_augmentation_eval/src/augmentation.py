"""MONAI 3D augmentation stack for 96-cube full-volume pediatric brain MRI.

Everything here is gated by the single `use_augmentation` flag (CONTRACTS.md).
When it is off, :func:`build_transforms` returns ``None`` and the Dataset does a
plain deterministic load.

Spatial transforms are applied jointly to `image` and `label`; the label always
uses nearest-neighbour interpolation so no fractional class values appear.
Intensity transforms touch `image` only.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from monai.transforms import (
    Compose,
    RandAffined,
    RandFlipd,
    RandGaussianNoised,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandZoomd,
)

KEYS = ("image", "label")


def build_transforms(cfg, spatial_size: tuple[int, int, int] = (96, 96, 96)) -> Compose | None:
    """Build the augmentation pipeline, or ``None`` when augmentation is off.

    Returning ``None`` rather than an empty ``Compose`` keeps the "off" path
    genuinely deterministic — no transform object touches the data at all.
    """
    if not cfg.use_augmentation:
        return None

    a: dict[str, Any] = cfg.augmentation
    rotate = math.radians(float(a.get("rotate_range_deg", 10.0)))

    transforms = [
        # --- spatial: image + label together, label always nearest-neighbour ---
        RandFlipd(keys=KEYS, prob=float(a.get("flip_prob", 0.5)), spatial_axis=0),
        RandFlipd(keys=KEYS, prob=float(a.get("flip_prob", 0.5)), spatial_axis=1),
        RandFlipd(keys=KEYS, prob=float(a.get("flip_prob", 0.5)), spatial_axis=2),
        RandAffined(
            keys=KEYS,
            prob=float(a.get("affine_prob", 0.3)),
            rotate_range=(rotate, rotate, rotate),
            mode=("bilinear", "nearest"),
            padding_mode="zeros",
            spatial_size=spatial_size,
        ),
        RandZoomd(
            keys=KEYS,
            prob=float(a.get("zoom_prob", 0.3)),
            min_zoom=float(a.get("zoom_min", 0.9)),
            max_zoom=float(a.get("zoom_max", 1.1)),
            mode=("trilinear", "nearest"),
            align_corners=(True, None),
            keep_size=True,
        ),
        # --- intensity: image only ---
        RandScaleIntensityd(
            keys="image",
            prob=float(a.get("scale_intensity_prob", 0.3)),
            factors=float(a.get("scale_intensity_factor", 0.1)),
        ),
        RandShiftIntensityd(
            keys="image",
            prob=float(a.get("shift_intensity_prob", 0.3)),
            offsets=float(a.get("shift_intensity_offset", 0.1)),
        ),
        RandGaussianNoised(
            keys="image",
            prob=float(a.get("gaussian_noise_prob", 0.2)),
            mean=0.0,
            std=float(a.get("gaussian_noise_std", 0.05)),
        ),
    ]
    return Compose(transforms)


def apply_transforms(
    transforms: Compose | None,
    image: np.ndarray | torch.Tensor,
    label: np.ndarray | torch.Tensor,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the stack on one sample.

    ``image`` is ``(C, D, H, W)`` float, ``label`` is ``(1, D, H, W)`` integer.
    Passing the same ``seed`` twice reproduces the same output bit-for-bit.
    Returns plain ``torch.Tensor`` (not MetaTensor), label re-cast to int64.
    """
    img_t = torch.as_tensor(np.asarray(image), dtype=torch.float32)
    lab_t = torch.as_tensor(np.asarray(label), dtype=torch.float32)

    if transforms is None:
        return img_t, lab_t.round().to(torch.int64)

    if seed is not None:
        transforms.set_random_state(seed=int(seed))

    out = transforms({"image": img_t, "label": lab_t})

    img_out = torch.as_tensor(np.asarray(out["image"])).to(torch.float32)
    # Nearest-neighbour keeps values on the class grid; round() defends against
    # float round-trip drift (e.g. 2.9999997) before the int cast.
    lab_out = torch.as_tensor(np.asarray(out["label"])).round().to(torch.int64)
    return img_out, lab_out
