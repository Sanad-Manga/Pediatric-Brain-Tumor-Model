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
    RandomizableTransform,
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


def _validate_dropout_prob(p: float) -> float:
    p = float(p)
    if not (0.0 <= p < 1.0):  # also rejects NaN
        raise ValueError(f"modality_dropout_prob must be in [0.0, 1.0), got {p!r}")
    return p


class RandModalityDropoutd(RandomizableTransform, MapTransform):
    """Zero whole input channels (MRI sequences) at random, to teach the model
    to cope with a sequence a clinic did not acquire.

    Each channel is dropped independently with probability `drop_prob`. If a
    draw would drop every channel, one channel chosen uniformly at random is
    kept instead, so the model never sees an all-zero input. A dropped channel
    is exactly 0.0: the hook receives already z-scored tensors, so "sequence
    missing" is encoded as an all-zero channel -- the same encoding deployment
    has to use for an absent sequence.

    Must run after every other image transform in the chain: anything that
    scales/shifts/adds noise afterwards would re-introduce non-zero values into
    a dropped channel.
    """

    def __init__(self, keys, drop_prob: float, allow_missing_keys: bool = False):
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, 1.0)
        self.drop_prob = _validate_dropout_prob(drop_prob)

    def randomize(self, n_channels: int) -> np.ndarray:
        drop = self.R.random_sample(n_channels) < self.drop_prob
        if drop.all():
            drop[self.R.randint(n_channels)] = False
        return drop

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            img = d[key]
            drop = self.randomize(img.shape[0])
            # copy: earlier no-op transforms can hand back the caller's own tensor
            img = img.clone() if isinstance(img, torch.Tensor) else np.array(img, copy=True)
            for c in np.flatnonzero(drop):
                img[c] = 0.0
            d[key] = img
        return d


def _validate_shift_prob(p: float) -> float:
    p = float(p)
    if not (0.0 <= p <= 1.0):  # also rejects NaN
        raise ValueError(f"sequence_shift_prob must be in [0.0, 1.0], got {p!r}")
    return p


def _validate_max_shift(v: float) -> float:
    v = float(v)
    if not np.isfinite(v) or v < 0.0:
        raise ValueError(f"sequence_shift_max_voxels must be a finite number >= 0, got {v!r}")
    return v


def _translate_channel(chan: torch.Tensor, shift) -> torch.Tensor:
    """Translate one (D, H, W) float32 channel by `shift` voxels per axis: out[x] = in[x - s].

    Bilinear resampling with torch's grid_sample; a voxel whose source location lies outside the
    volume is exactly 0 (forced, so vacated voxels never carry interpolation noise). An integer
    shift therefore equals np.roll on the interior, and a sub-voxel shift interpolates linearly.
    """
    axes = []
    valid = []
    for n, s in zip(chan.shape, shift):
        pos = torch.arange(n, dtype=torch.float64) - float(s)           # source index for every output index
        valid.append((pos >= 0.0) & (pos <= n - 1))
        norm = 2.0 * pos / (n - 1) - 1.0 if n > 1 else torch.zeros(n, dtype=torch.float64)
        axes.append(norm.to(torch.float32))
    gz, gy, gx = torch.meshgrid(axes[0], axes[1], axes[2], indexing="ij")
    grid = torch.stack([gx, gy, gz], dim=-1)[None]                       # (1, D, H, W, 3), last dim = (x=W, y=H, z=D)
    out = torch.nn.functional.grid_sample(
        chan[None, None], grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )[0, 0]
    inside = valid[0][:, None, None] & valid[1][None, :, None] & valid[2][None, None, :]
    return torch.where(inside, out, torch.zeros_like(out))


class RandSequenceShiftd(RandomizableTransform, MapTransform):
    """Simulate imperfect co-registration between MRI sequences.

    With probability `prob` (one draw per call) one reference channel is chosen uniformly at random
    and never moved; each other channel is selected independently with probability 0.5 (if none is,
    one of the others is chosen uniformly, so an applied call always moves at least one channel).
    Every selected channel gets its own translation, uniform in [-max_shift_voxels, +max_shift_voxels]
    on each axis (float, sub-voxel allowed). The label is not an input of this transform.

    Runs after the intensity transforms and before RandModalityDropoutd, which must stay last so a
    dropped channel is still exactly 0. The input is z-scored with background 0, so the zero fill
    of the vacated voxels matches the background.
    """

    def __init__(self, keys, prob: float, max_shift_voxels: float, allow_missing_keys: bool = False):
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, 1.0)
        self.prob = _validate_shift_prob(prob)
        self.max_shift_voxels = _validate_max_shift(max_shift_voxels)

    def randomize(self, n_channels: int):
        """Draw the plan: (reference index, (C, 3) shifts). Not applied -> (-1, all zeros)."""
        if n_channels < 2:
            raise ValueError(f"sequence shift needs at least 2 channels, got {n_channels}")
        shifts = np.zeros((n_channels, 3), dtype=np.float64)
        if not self.R.random_sample() < self.prob:
            return -1, shifts
        reference = int(self.R.randint(n_channels))
        others = [c for c in range(n_channels) if c != reference]
        hits = self.R.random_sample(n_channels) < 0.5
        selected = [c for c in others if hits[c]]
        if not selected:
            selected = [others[int(self.R.randint(len(others)))]]
        vectors = self.R.uniform(-self.max_shift_voxels, self.max_shift_voxels, size=(n_channels, 3))
        for c in selected:
            shifts[c] = vectors[c]
        return reference, shifts

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            img = d[key]
            n_channels = img.shape[0]
            if n_channels < 2:
                raise ValueError(f"sequence shift needs at least 2 channels, got {n_channels}")
            _reference, shifts = self.randomize(n_channels)
            is_tensor = isinstance(img, torch.Tensor)
            out = img.clone() if is_tensor else np.array(img, copy=True)
            for c in np.flatnonzero(np.any(shifts != 0.0, axis=1)):
                chan = img[c] if is_tensor else torch.from_numpy(np.ascontiguousarray(img[c]))
                moved = _translate_channel(chan.detach().to(torch.float32).cpu(), shifts[c])
                out[c] = moved.to(img.dtype).to(img.device) if is_tensor else moved.numpy().astype(img.dtype)
            d[key] = out
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
    modality_dropout_prob: float = 0.0,
    sequence_shift_prob: float = 0.0,
    sequence_shift_max_voxels: float = 1.2,
) -> Compose:
    """Build the 3D augmentation stack. Always returns a real Compose --
    the "off" case is handled by the caller never building/using one, same
    convention as the 2D module.

    `modality_dropout_prob` > 0 appends RandModalityDropoutd as the last image
    transform; the default 0.0 adds nothing, leaving the chain exactly as it
    was before sequence dropout existed.

    `sequence_shift_prob` > 0 inserts RandSequenceShiftd after the intensity
    transforms and before RandModalityDropoutd (Addendum 4); the shift is in voxels of
    the grid being trained on (1.2 voxels is about 3 mm in-plane on the 96^3 grid). The
    default 0.0 adds nothing.
    """
    modality_dropout_prob = _validate_dropout_prob(modality_dropout_prob)
    sequence_shift_prob = _validate_shift_prob(sequence_shift_prob)
    sequence_shift_max_voxels = _validate_max_shift(sequence_shift_max_voxels)
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
    ]
    if sequence_shift_prob > 0.0:
        transforms.append(RandSequenceShiftd(
            keys=IMAGE_KEY, prob=sequence_shift_prob, max_shift_voxels=sequence_shift_max_voxels))
    if modality_dropout_prob > 0.0:
        transforms.append(RandModalityDropoutd(keys=IMAGE_KEY, drop_prob=modality_dropout_prob))
    transforms.append(AssertLabelValuesd(keys=LABEL_KEY, valid_labels=VALID_LABELS))

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
