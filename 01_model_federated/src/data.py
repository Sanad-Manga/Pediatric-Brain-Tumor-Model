"""Manifest-driven dataset: dummy-tensor mode, and real mode reading the shared
96^3 resampled-cache .npz files (see 00_shared/CONTRACTS.md "Data pipeline").
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .model import IN_CHANNELS

VOLUME_SIZE = 96
NUM_LABELS = 5  # 0=background, 1=ET, 2=NET, 3=CC, 4=ED
MODALITIES = ("t1c", "t1n", "t2f", "t2w")


def load_manifest(manifest_path: str) -> list[str]:
    path = Path(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    with open(path) as f:
        subject_ids = json.load(f)
    if not subject_ids:
        raise ValueError(f"Manifest is empty: {manifest_path}")
    return subject_ids


class BraTSPedsDataset(Dataset):
    """Yields (input, label) pairs for the subject IDs listed in a manifest.

    mode="dummy": generates deterministic-per-subject random tensors of the
    contractual shape, so the pipeline is fully testable before the real
    resampled cache exists.

    mode="real": loads `<cache_path>/<subject_id>.npz`, one file per subject,
    containing arrays keyed "t1c", "t1n", "t2f", "t2w", "seg" (the shared
    resampled cache format).
    """

    def __init__(
        self,
        manifest_path: str,
        mode: str = "dummy",
        cache_path: str | None = None,
        seed: int = 42,
    ) -> None:
        if mode not in ("dummy", "real"):
            raise ValueError(f"mode must be 'dummy' or 'real', got {mode!r}")
        if mode == "real" and not cache_path:
            raise ValueError("mode='real' requires a cache_path")

        self.subject_ids = load_manifest(manifest_path)
        self.mode = mode
        self.cache_path = cache_path
        self.seed = seed

    def __len__(self) -> int:
        return len(self.subject_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        subject_id = self.subject_ids[idx]
        if self.mode == "dummy":
            return self._dummy_sample(subject_id)
        return self._real_sample(subject_id)

    def _dummy_sample(self, subject_id: str) -> tuple[torch.Tensor, torch.Tensor]:
        rng = np.random.default_rng(abs(hash((self.seed, subject_id))) % (2**32))
        x = rng.standard_normal(
            (IN_CHANNELS, VOLUME_SIZE, VOLUME_SIZE, VOLUME_SIZE)
        ).astype(np.float32)
        y = rng.integers(
            0, NUM_LABELS, size=(VOLUME_SIZE, VOLUME_SIZE, VOLUME_SIZE)
        ).astype(np.int64)
        return torch.from_numpy(x), torch.from_numpy(y)

    def _real_sample(self, subject_id: str) -> tuple[torch.Tensor, torch.Tensor]:
        npz_path = os.path.join(self.cache_path, f"{subject_id}.npz")
        if not os.path.isfile(npz_path):
            raise FileNotFoundError(f"Missing cached subject file: {npz_path}")

        with np.load(npz_path) as data:
            x = np.stack([data[m].astype(np.float32) for m in MODALITIES], axis=0)
            y = data["seg"].astype(np.int64)

        return torch.from_numpy(x), torch.from_numpy(y)


def build_dataset(
    manifest_path: str,
    data_mode: str = "dummy",
    cache_path: str | None = None,
    seed: int = 42,
) -> BraTSPedsDataset:
    return BraTSPedsDataset(manifest_path, mode=data_mode, cache_path=cache_path, seed=seed)
