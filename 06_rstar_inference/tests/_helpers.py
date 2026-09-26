"""Synthetic scans and stub models shared by the tests. Nothing here needs a checkpoint, real data or a GPU."""
import numpy as np
import torch

from rstar import RStarConfig
from rstar.guards import IdentityStub, synthetic_case

TINY = (48, 56, 40)


def tiny_config(**overrides) -> RStarConfig:
    kwargs = dict(expected_shape=TINY, device="cpu", verify_hashes=False)
    kwargs.update(overrides)
    return RStarConfig(**kwargs)


def make_volume(shape=TINY, seed=0):
    """(4, *shape) BraTS-like scan: a brain ellipsoid (about 70% background) with a bright blob in channel 0. Returns (volume, blob mask)."""
    vol, blob = synthetic_case(shape)
    rng = np.random.default_rng(seed)
    brain = vol[1] > 0
    for c in range(4):
        vol[c][brain] += rng.uniform(0.0, 30.0, size=int(brain.sum())).astype(np.float32)
    vol[0][blob] = 400.0 + rng.uniform(0.0, 10.0, size=int(blob.sum())).astype(np.float32)
    return vol, blob


class ConstantStub(torch.nn.Module):
    """Always predicts class `label` inside the brain (|channel 0| > 0) and background elsewhere; `label=0` predicts nothing."""

    def __init__(self, label: int):
        super().__init__()
        self.label = label

    def forward(self, x):
        inside = (x[:, 0].abs() > 1e-6).float()
        zeros = torch.zeros_like(inside)
        logits = [zeros] * 5
        logits[0] = 5.0 * (1.0 - inside) if self.label else 5.0 + zeros
        if self.label:
            logits[self.label] = 5.0 * inside
        return torch.stack(logits, dim=1), None


class CountingStub(IdentityStub):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return super().forward(x)


class RecordingStub(IdentityStub):
    """Identity stub that keeps the last input it saw."""

    def __init__(self):
        super().__init__()
        self.last = None

    def forward(self, x):
        self.last = x.detach().clone()
        return super().forward(x)
