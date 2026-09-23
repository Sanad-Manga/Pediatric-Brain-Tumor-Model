"""3D MONAI augmentation stack for the federated U-Net pipeline.

Mirrors 03_augmentation_eval/src/augment.py's two rules, adapted from 2D
slices to full (D, H, W) volumes:

1. Spatial transforms (flip / rotate / zoom) apply to the image **and** the
   mask together, and the mask always uses nearest-neighbour interpolation,
   so no fractional labels can appear. Label values are asserted to stay
   within {0,1,2,3,4} inside the pipeline itself, not only in the tests --
   the resample script (tools/build_96cube_cache.py) had exactly this class
   of bug (nearest vs. bilinear on a label volume) and it would have gone
   unnoticed without an explicit check.
2. Nothing here mixes two different subjects' data (no mixup): unlike the
   2D pipeline, a 3D volume-level mixup would blend anatomy at every voxel,
   not just a handful of 2D slices, and this project has never validated
   that that is even a sane thing to do for a full 3D segmentation target.
   Left out until it is, rather than ported speculatively.

01_model_federated/BRIEF.md always said augmentation was a separate
section's job and its own code only ever provided the hook
(train_single.py's `augmentation_transform` argument) -- this builds the
first real implementation for it, run.py wires it to --use-augmentation.

`use_augmentation: false` (or --use-augmentation not passed) must remain
bitwise-identical to today's behaviour: build_transforms3d() is only ever
called when the flag is on, and returning None short-circuits the training
loop's `if use_augmentation and transform is not None` check in
train_single.py, same as the 2D pipeline's convention.
"""
from __future__ import annotations

import numpy as np
import torch
from monai.data import set_track_meta
from monai.transforms import (
    Compose,
    MapTransform,
    RandFlipd,
    RandGaussianNoised,
    RandRotated,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandZoomd,
)

# Plain tensors out, not MetaTensors -- nothing downstream reads MONAI
# metadata, same reasoning as the 2D module.
set_track_meta(False)

IMAGE_KEY = "image"
LABEL_KEY = "label"
VALID_LABELS = (0, 1, 2, 3, 4)


class AssertLabelValuesd(MapTransform):
    """Fail loudly if a transform introduced a label outside the valid set."""

    def __init__(self, keys, valid_labels):
        super().__init__(keys)
        self.valid_labels = set(int(v) for v in valid_labels)

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            arr = d[key]
            arr = arr.detach().cpu().numpy() if isinstance(arr, torch.Tensor) else np.asarray(arr)
            found = set(np.unique(arr).tolist())
            bad = {v for v in found if v not in self.valid_labels}
            if bad:
                raise AssertionError(
                    f"augmentation produced label values outside "
                    f"{sorted(self.valid_labels)}: {sorted(bad)[:8]}"
                )
        return d


def build_transforms3d(
    flip_prob: float = 0.5,
    rotate_prob: float = 0.3,
    rotate_range_deg: float = 10.0,
    zoom_prob: float = 0.3,
    zoom_min: float = 0.9,
    zoom_max: float = 1.1,
    scale_intensity_factor: float = 0.1,
    scale_intensity_prob: float = 0.3,
    shift_intensity_offset: float = 0.1,
    shift_intensity_prob: float = 0.3,
    gaussian_noise_prob: float = 0.2,
    gaussian_noise_std: float = 0.05,
    seed: int | None = None,
) -> Compose:
    """Build the 3D augmentation stack. Always returns a real Compose --
    the "off" case is handled by the caller never building/using one, same
    convention as the 2D module.
    """
    keys = [IMAGE_KEY, LABEL_KEY]
    rotate_rad = float(np.deg2rad(rotate_range_deg))

    transforms = [
        # --- spatial: image and mask together, mask always nearest-neighbour
        RandFlipd(keys=keys, prob=flip_prob, spatial_axis=0),
        RandFlipd(keys=keys, prob=flip_prob, spatial_axis=1),
        RandFlipd(keys=keys, prob=flip_prob, spatial_axis=2),
        RandRotated(
            keys=keys,
            range_x=rotate_rad, range_y=rotate_rad, range_z=rotate_rad,
            prob=rotate_prob,
            mode=("bilinear", "nearest"),
            padding_mode="zeros",
            keep_size=True,
        ),
        RandZoomd(
            keys=keys,
            prob=zoom_prob,
            min_zoom=zoom_min,
            max_zoom=zoom_max,
            mode=("bilinear", "nearest"),
            keep_size=True,
        ),
        # --- intensity: image only
        RandScaleIntensityd(keys=IMAGE_KEY, factors=scale_intensity_factor, prob=scale_intensity_prob),
        RandShiftIntensityd(keys=IMAGE_KEY, offsets=shift_intensity_offset, prob=shift_intensity_prob),
        RandGaussianNoised(keys=IMAGE_KEY, prob=gaussian_noise_prob, mean=0.0, std=gaussian_noise_std),
        AssertLabelValuesd(keys=LABEL_KEY, valid_labels=VALID_LABELS),
    ]

    compose = Compose(transforms)
    if seed is not None:
        compose.set_random_state(seed=int(seed))
    return compose


class Augment3D:
    """Callable matching train_single.py's `transform(x, y) -> (x, y)`
    contract (see `_apply_augmentation`).

    x: (B, 4, D, H, W) float, y: (B, D, H, W) integer labels -- the shapes
    DataLoader yields before train_single.py's own `y.unsqueeze(1)`.
    batch_size is fixed to 1 everywhere in this section (00_shared/
    CONTRACTS.md), but this loops per-sample rather than assuming that, so
    it keeps working if that constraint is ever relaxed.
    """

    def __init__(self, **transform_kwargs):
        self._compose = build_transforms3d(**transform_kwargs)

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out_x, out_y = [], []
        for i in range(x.shape[0]):
            sample = {IMAGE_KEY: x[i].float(), LABEL_KEY: y[i][None].float()}
            aug = self._compose(sample)
            out_x.append(torch.as_tensor(np.asarray(aug[IMAGE_KEY]), dtype=x.dtype))
            label = np.rint(np.asarray(aug[LABEL_KEY])[0]).astype(np.int64)
            out_y.append(torch.as_tensor(label, dtype=y.dtype))
        return torch.stack(out_x), torch.stack(out_y)
