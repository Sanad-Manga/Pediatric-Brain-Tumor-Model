"""Mixup for 3D volumetric segmentation.

This is NOT classification mixup. There is no scalar class label to blend —
the target is a dense mask volume. The mask is one-hot encoded across the 5
classes (background + labels 1-4) and mixed with the *same* lambda as the image,
so the result stays a valid soft target for Dice / Dice+CE:

    mixed_image = lam * image_a + (1 - lam) * image_b
    mixed_onehot = lam * onehot_a + (1 - lam) * onehot_b

The mixed one-hot still sums to 1.0 across the channel axis at every voxel,
which is what makes it a legal soft segmentation target.

The dataset runs at batch_size=1 (CONTRACTS.md), where a within-batch partner
does not exist. :class:`MixupBuffer` solves this by carrying the previous sample
forward: the first step passes through unmixed, every later step mixes the
current sample with the buffered one.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def one_hot_seg(label: torch.Tensor, num_classes: int = 5) -> torch.Tensor:
    """Convert an integer mask to one-hot.

    ``(B, 1, D, H, W)`` -> ``(B, C, D, H, W)`` (or the unbatched equivalent).
    """
    lab = label.to(torch.int64)
    if lab.dim() >= 2 and lab.shape[-4 if lab.dim() == 5 else 0] == 1:
        # drop the singleton channel axis before one-hot
        lab = lab.squeeze(-4) if lab.dim() == 5 else lab.squeeze(0)
    onehot = F.one_hot(lab, num_classes=num_classes)
    # one_hot appends the class axis last; move it to the channel position
    return onehot.movedim(-1, -4 if onehot.dim() == 5 else 0).to(torch.float32)


def sample_lambda(alpha: float, rng: np.random.Generator | None = None) -> float:
    """Draw lambda ~ Beta(alpha, alpha).

    A seeded ``numpy.random.Generator`` makes the draw reproducible; ``None``
    falls back to NumPy's global default generator.
    """
    if alpha <= 0:
        raise ValueError(f"mixup alpha must be > 0, got {alpha}")
    rng = rng if rng is not None else np.random.default_rng()
    return float(rng.beta(alpha, alpha))


def mixup_3d(
    images: torch.Tensor,
    labels_onehot: torch.Tensor,
    alpha: float = 0.4,
    rng: np.random.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Mix a batch of 3D volumes with their one-hot masks.

    ``images`` is ``(B, C_in, D, H, W)``, ``labels_onehot`` is
    ``(B, C_cls, D, H, W)``. Returns the mixed pair (same shapes) and lambda.

    With ``B == 1`` there is no partner to mix with, so the batch is returned
    unchanged with ``lam = 1.0``. Use :class:`MixupBuffer` for the batch_size=1
    training loop.
    """
    if alpha <= 0:
        raise ValueError(f"mixup alpha must be > 0, got {alpha}")
    if images.shape[0] != labels_onehot.shape[0]:
        raise ValueError(
            f"batch size mismatch: images {images.shape[0]} vs labels {labels_onehot.shape[0]}"
        )

    batch = images.shape[0]
    if batch < 2:
        return images, labels_onehot, 1.0

    rng = rng if rng is not None else np.random.default_rng()
    lam = sample_lambda(alpha, rng)
    perm = torch.as_tensor(rng.permutation(batch), dtype=torch.long)

    mixed_images = lam * images + (1.0 - lam) * images[perm]
    mixed_labels = lam * labels_onehot + (1.0 - lam) * labels_onehot[perm]
    return mixed_images, mixed_labels, lam


class MixupBuffer:
    """Segmentation mixup that works at ``batch_size=1``.

    Holds the previous sample and mixes it with the current one. The very first
    call has nothing buffered, so it returns the sample unmixed (``lam = 1.0``).

    ``enabled=False`` turns the whole thing into a pass-through — this is how
    ``use_augmentation: false`` switches mixup off regardless of ``use_mixup``.
    """

    def __init__(self, alpha: float = 0.4, enabled: bool = True, seed: int | None = None):
        if enabled and alpha <= 0:
            raise ValueError(f"mixup alpha must be > 0, got {alpha}")
        self.alpha = alpha
        self.enabled = enabled
        self._rng = np.random.default_rng(int(seed) if seed is not None else None)
        self._buffer: tuple[torch.Tensor, torch.Tensor] | None = None

    def reset(self) -> None:
        self._buffer = None

    def step(
        self, images: torch.Tensor, labels_onehot: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        """Mix the incoming batch with the buffered one; returns ``(img, lab, lam)``."""
        if not self.enabled:
            return images, labels_onehot, 1.0

        if images.shape[0] >= 2:
            self._buffer = (images[-1:].clone(), labels_onehot[-1:].clone())
            return mixup_3d(images, labels_onehot, self.alpha, self._rng)

        if self._buffer is None:
            self._buffer = (images.clone(), labels_onehot.clone())
            return images, labels_onehot, 1.0

        prev_img, prev_lab = self._buffer
        lam = sample_lambda(self.alpha, self._rng)
        mixed_images = lam * images + (1.0 - lam) * prev_img
        mixed_labels = lam * labels_onehot + (1.0 - lam) * prev_lab

        self._buffer = (images.clone(), labels_onehot.clone())
        return mixed_images, mixed_labels, lam


def build_mixup(cfg) -> MixupBuffer:
    """Construct the mixup buffer from config.

    Per Req 14, mixup is enabled only when ``use_augmentation`` AND ``use_mixup``
    are both true — that is what :attr:`Config.mixup_enabled` encodes.
    """
    return MixupBuffer(
        alpha=float(cfg.mixup.get("alpha", 0.4)),
        enabled=cfg.mixup_enabled,
        seed=cfg.expansion.get("seed"),
    )
