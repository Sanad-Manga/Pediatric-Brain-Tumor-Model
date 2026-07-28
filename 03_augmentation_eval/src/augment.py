"""2D MONAI augmentation stack and segmentation mixup.

Two rules drive everything here:

1. Spatial transforms (flip / rotate / zoom) apply to the image **and** the mask
   together, and the mask always uses nearest-neighbour interpolation, so no
   fractional labels can appear. The label value set is asserted to stay within
   {0,1,2,3,4} inside the pipeline itself, not only in the tests.
2. Mixup here is *segmentation* mixup: the same lambda mixes the image and the
   **one-hot mask**. Scalar labels are never mixed, and only slices from the same
   plane may be paired -- blending an axial with a coronal slice produces an
   anatomically meaningless image.

The whole branch is gated by ``use_augmentation`` (see :func:`build_transforms`
and ``Config.mixup_enabled``).
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

from .config import Config

# Plain tensors out, not MetaTensors -- nothing downstream reads MONAI metadata
# and bitwise-identity checks are cleaner without it.
set_track_meta(False)

IMAGE_KEY = "image"
LABEL_KEY = "label"


class AssertLabelValuesd(MapTransform):
    """Fail loudly if a transform introduced a label outside the valid set.

    This is the guard that catches an interpolation mode regression: bilinear on
    a mask silently produces fractional labels that no later stage would notice.
    """

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


def build_transforms(cfg: Config, seed: int | None = None) -> Compose | None:
    """Build the 2D augmentation stack, or ``None`` when augmentation is off.

    Returning ``None`` rather than an empty ``Compose`` is what makes
    ``use_augmentation: false`` bitwise-identical across repeated loads: no
    transform touches the arrays at all.
    """
    if not cfg.use_augmentation:
        return None

    a = cfg.augmentation
    keys = [IMAGE_KEY, LABEL_KEY]
    rotate_rad = float(np.deg2rad(a.get("rotate_range_deg", 10.0)))

    transforms = [
        # --- spatial: image and mask together, mask always nearest-neighbour
        RandFlipd(keys=keys, prob=a.get("flip_prob", 0.5), spatial_axis=0),
        RandFlipd(keys=keys, prob=a.get("flip_prob", 0.5), spatial_axis=1),
        RandRotated(
            keys=keys,
            range_x=rotate_rad,
            prob=a.get("rotate_prob", 0.3),
            mode=("bilinear", "nearest"),
            padding_mode="zeros",
            keep_size=True,
        ),
        RandZoomd(
            keys=keys,
            prob=a.get("zoom_prob", 0.3),
            min_zoom=a.get("zoom_min", 0.9),
            max_zoom=a.get("zoom_max", 1.1),
            mode=("bilinear", "nearest"),
            keep_size=True,
        ),
        # --- intensity: image only
        RandScaleIntensityd(
            keys=IMAGE_KEY,
            factors=a.get("scale_intensity_factor", 0.1),
            prob=a.get("scale_intensity_prob", 0.3),
        ),
        RandShiftIntensityd(
            keys=IMAGE_KEY,
            offsets=a.get("shift_intensity_offset", 0.1),
            prob=a.get("shift_intensity_prob", 0.3),
        ),
        RandGaussianNoised(
            keys=IMAGE_KEY,
            prob=a.get("gaussian_noise_prob", 0.2),
            mean=0.0,
            std=a.get("gaussian_noise_std", 0.05),
        ),
        AssertLabelValuesd(keys=LABEL_KEY, valid_labels=cfg.valid_labels),
    ]

    compose = Compose(transforms)
    if seed is not None:
        compose.set_random_state(seed=int(seed))
    return compose


def apply_augmentation(transforms, image: np.ndarray, label: np.ndarray, seed: int | None = None):
    """Run the stack on one sample, returning ``(image, label)`` as numpy.

    ``image`` is ``(C, H, W)`` float32 and ``label`` is ``(1, H, W)`` integer.
    Passing the same ``seed`` twice gives bitwise-identical output.
    """
    if transforms is None:
        return image, label
    if seed is not None:
        transforms.set_random_state(seed=int(seed))

    out = transforms({
        IMAGE_KEY: torch.as_tensor(np.ascontiguousarray(image), dtype=torch.float32),
        LABEL_KEY: torch.as_tensor(np.ascontiguousarray(label), dtype=torch.float32),
    })
    aug_image = np.asarray(out[IMAGE_KEY], dtype=np.float32)
    aug_label = np.rint(np.asarray(out[LABEL_KEY], dtype=np.float32)).astype(label.dtype)
    return aug_image, aug_label


# ----------------------------------------------------------------- one-hot ops
def one_hot(label: np.ndarray, num_classes: int) -> np.ndarray:
    """``(1, H, W)`` or ``(H, W)`` integer mask -> ``(num_classes, H, W)`` float32."""
    arr = np.asarray(label)
    if arr.ndim == 3:
        if arr.shape[0] != 1:
            raise ValueError(f"expected a single-channel label, got shape {arr.shape}")
        arr = arr[0]
    arr = arr.astype(np.int64)
    bad = set(np.unique(arr).tolist()) - set(range(num_classes))
    if bad:
        raise ValueError(f"label values {sorted(bad)} are outside 0..{num_classes - 1}")
    return np.eye(num_classes, dtype=np.float32)[arr].transpose(2, 0, 1)


# ---------------------------------------------------------------------- mixup
def segmentation_mixup(
    img_a: np.ndarray,
    oh_a: np.ndarray,
    img_b: np.ndarray,
    oh_b: np.ndarray,
    alpha: float,
    rng: np.random.Generator | None = None,
    plane_a: str | None = None,
    plane_b: str | None = None,
):
    """Mix two slices and their one-hot masks with one shared lambda.

    ``mixed_image  = lam * img_a + (1 - lam) * img_b``
    ``mixed_onehot = lam * oh_a  + (1 - lam) * oh_b``

    Because both one-hots sum to 1 across the channel axis, so does the mixture --
    that identity is what keeps the soft target a valid distribution, and it is
    asserted here rather than trusted.

    Returns ``(mixed_image, mixed_onehot, lam)``.
    """
    if alpha <= 0:
        raise ValueError("mixup.alpha must be > 0")
    if plane_a is not None and plane_b is not None and plane_a != plane_b:
        raise ValueError(
            f"mixup may only pair slices from the same plane, got "
            f"{plane_a!r} and {plane_b!r}; blending an axial with a coronal "
            f"slice is anatomically meaningless"
        )
    if img_a.shape != img_b.shape:
        raise ValueError(f"image shapes differ: {img_a.shape} vs {img_b.shape}")
    if oh_a.shape != oh_b.shape:
        raise ValueError(f"one-hot shapes differ: {oh_a.shape} vs {oh_b.shape}")

    rng = rng if rng is not None else np.random.default_rng()
    lam = float(rng.beta(alpha, alpha))

    mixed_image = (lam * img_a + (1.0 - lam) * img_b).astype(np.float32)
    mixed_onehot = (lam * oh_a + (1.0 - lam) * oh_b).astype(np.float32)

    channel_sums = mixed_onehot.sum(axis=0)
    if not np.allclose(channel_sums, 1.0, atol=1e-5):
        raise AssertionError(
            f"mixed one-hot does not sum to 1.0 across channels "
            f"(min {channel_sums.min():.6f}, max {channel_sums.max():.6f})"
        )
    return mixed_image, mixed_onehot, lam


def mixup_batch(images, onehots, planes, alpha: float, rng: np.random.Generator | None = None):
    """Mix a batch in place against a same-plane permutation of itself.

    Samples whose plane has no partner in the batch are returned unmixed rather
    than paired across planes.
    """
    rng = rng if rng is not None else np.random.default_rng()
    images = np.asarray(images)
    onehots = np.asarray(onehots)
    planes = list(planes)

    mixed_img = images.copy()
    mixed_oh = onehots.copy()
    lams = np.ones(len(planes), dtype=np.float32)

    by_plane: dict[str, list[int]] = {}
    for i, plane in enumerate(planes):
        by_plane.setdefault(plane, []).append(i)

    for plane, idx in by_plane.items():
        if len(idx) < 2:
            continue
        partners = list(rng.permutation(idx))
        for i, j in zip(idx, partners):
            mixed_img[i], mixed_oh[i], lams[i] = segmentation_mixup(
                images[i], onehots[i], images[j], onehots[j],
                alpha, rng, plane_a=plane, plane_b=planes[j],
            )
    return mixed_img, mixed_oh, lams
