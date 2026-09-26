"""Soft guards (agreement status) and the geometry self-check."""
from __future__ import annotations

import numpy as np
import torch

from .config import RStarConfig


class SelfCheckError(RuntimeError):
    """The module's own geometry is wrong (a frame/orientation bug); nothing may be segmented until this is fixed."""


def agreement_status(agreement: float, cfg: RStarConfig):
    """('ok' | 'review', warnings). Labels are never withheld: a genuine 2D collapse on a correctly aligned patient gave agreement 0.000."""
    if agreement >= cfg.agreement_review:
        return "ok", []
    if agreement < cfg.agreement_strong:
        return "review", [
            f"the 2D and 3D outlines barely overlap (agreement {agreement:.2f}): possible 2D collapse or a frame/orientation "
            "error; verify the outline before using it"]
    return "review", [
        f"the 2D and 3D outlines disagree (agreement {agreement:.2f} < {cfg.agreement_review:.2f}); "
        "a weak signal, but cases like this scored lower on average"]


# ------------------------------------------------------------------------------------------------- self check
class IdentityStub(torch.nn.Module):
    """A model whose class-1 logit is +k*x[:, 0] and class-0 logit is -k*x[:, 0] + bias: it 'segments' whatever is bright in
    channel 0, and calls empty (all-zero) voxels background."""

    def __init__(self, gain: float = 5.0, bias: float = 10.0):
        super().__init__()
        self.gain = gain
        self.bias = bias

    def forward(self, x):
        x0 = x[:, 0]
        zeros = torch.zeros_like(x0)
        logits = torch.stack([-self.gain * x0 + self.bias, self.gain * x0, zeros, zeros, zeros], dim=1)
        return logits, None


def synthetic_case(shape=(48, 56, 40)):
    """An asymmetric synthetic scan: a brain ellipsoid with a bright blob off-centre on every axis. Returns (volume (4,*shape), blob mask)."""
    x, y, z = np.meshgrid(*[np.arange(n, dtype=np.float32) for n in shape], indexing="ij")
    cx, cy, cz = (s / 2.0 for s in shape)
    brain = ((x - cx) / (0.42 * shape[0])) ** 2 + ((y - cy) / (0.42 * shape[1])) ** 2 + ((z - cz) / (0.42 * shape[2])) ** 2 < 1.0
    blob = ((x - 0.34 * shape[0]) ** 2 + (y - 0.63 * shape[1]) ** 2 + (z - 0.40 * shape[2]) ** 2) < (0.11 * min(shape)) ** 2
    blob &= brain
    vol = np.zeros((4, *shape), dtype=np.float32)
    for c in range(4):
        vol[c][brain] = 100.0 + 10.0 * c
    vol[0][blob] = 400.0
    return vol, blob


def center_of_mass(mask: np.ndarray) -> np.ndarray:
    return np.argwhere(mask).mean(axis=0)


def run_self_check(probs_2d, probs_3d, shape=(48, 56, 40)) -> bool:
    """probs_2d(volume) / probs_3d(volume, present) are the pipeline's own path functions bound to identity stub models.
    The blob must come out where it went in: exactly through the 2D path, within 3 voxels (scaled to the grid) through the 3D path."""
    vol, blob = synthetic_case(shape)
    present = (True, True, True, True)
    p2 = probs_2d(vol)
    if not np.array_equal(p2[1] > 0.5, blob):
        raise SelfCheckError(
            "2D path self-check failed: the synthetic blob did not come back at its input position "
            "(a frame/orientation constant is wrong)")
    p3 = probs_3d(vol, present)
    predicted = p3[1] > 0.5
    if not predicted.any():
        raise SelfCheckError("3D path self-check failed: the synthetic blob was lost")
    tolerance = max(1.0, 3.0 * max(shape) / 240.0)
    offset = float(np.linalg.norm(center_of_mass(predicted) - center_of_mass(blob)))
    if offset > tolerance:
        raise SelfCheckError(f"3D path self-check failed: the blob moved by {offset:.1f} voxels (tolerance {tolerance:.1f})")
    return True
