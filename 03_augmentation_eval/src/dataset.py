"""Datasets over an expansion plan (training) or a manifest (evaluation).

Two hard rules are enforced structurally rather than by convention:

* Hospital provenance survives to the sample level — every training item carries
  its `hospital`, `stratum` and `source_subject_id`, so A and B are separable at
  any point downstream and are never pooled here.
* The held-out institution is never augmented. :class:`EvalDataset` force-disables
  augmentation regardless of the config flag, so an eval run cannot accidentally
  inherit a training config.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from torch.utils.data import Dataset

from .augmentation import apply_transforms, build_transforms
from .config import Config
from .data import load_subject

LOGGER = logging.getLogger(__name__)


class ExpansionDataset(Dataset):
    """Training dataset for one hospital, driven by its expansion plan.

    Augmented entries are materialized on the fly from
    ``(source_subject_id, aug_seed)`` — the plan is the provenance record, the
    volumes are never duplicated on disk.

    * ``is_augmented=True``  -> transforms run under the entry's fixed seed
      (reproducible: same seed, same output).
    * ``is_augmented=False`` -> the original subject. It still receives live
      random augmentation when ``use_augmentation`` is on, and is returned
      untouched when it is off.
    """

    def __init__(self, plan: dict, cfg: Config, cache_dir=None):
        self.plan = plan
        self.cfg = cfg
        self.entries = plan["entries"]
        self.hospital = plan["hospital"]
        self.cache_dir = cache_dir if cache_dir is not None else cfg.resolve("cache")
        self.transforms = build_transforms(cfg, cfg.spatial_size)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict:
        entry = self.entries[idx]
        image, label = load_subject(
            self.cache_dir,
            entry["source_subject_id"],
            modalities=self.cfg.modalities,
            valid_labels=self.cfg.valid_labels,
        )

        if self.transforms is None:
            img_t = torch.as_tensor(image, dtype=torch.float32)
            lab_t = torch.as_tensor(label, dtype=torch.int64)
        else:
            # Fixed seed for planned augmented copies; live randomness for originals.
            seed = entry["aug_seed"] if entry["is_augmented"] else None
            img_t, lab_t = apply_transforms(self.transforms, image, label, seed=seed)

        return {
            "image": img_t,
            "label": lab_t,
            "sample_id": entry["sample_id"],
            "source_subject_id": entry["source_subject_id"],
            "hospital": entry["hospital"],
            "stratum": entry["stratum"],
            "is_augmented": bool(entry["is_augmented"]),
        }


class EvalDataset(Dataset):
    """Evaluation dataset over a manifest. Augmentation is always off.

    Constructing this with an augmentation-enabled config logs a warning and
    proceeds without augmentation — evaluation results must not depend on the
    training flag.
    """

    def __init__(self, subject_ids: list[str], cfg: Config, cache_dir=None):
        self.subject_ids = list(subject_ids)
        self.cfg = cfg
        self.cache_dir = cache_dir if cache_dir is not None else cfg.resolve("cache")
        if cfg.use_augmentation:
            LOGGER.warning(
                "use_augmentation is true but this is an evaluation set; "
                "augmentation is force-disabled for the held-out institution"
            )

    def __len__(self) -> int:
        return len(self.subject_ids)

    def __getitem__(self, idx: int) -> dict:
        sid = self.subject_ids[idx]
        image, label = load_subject(
            self.cache_dir,
            sid,
            modalities=self.cfg.modalities,
            valid_labels=self.cfg.valid_labels,
        )
        return {
            "image": torch.as_tensor(image, dtype=torch.float32),
            "label": torch.as_tensor(label, dtype=torch.int64),
            "sample_id": sid,
            "source_subject_id": sid,
        }


class DummyDataset(Dataset):
    """Random tensors shaped like the real thing, for tests and ``--dummy-data``.

    Reads nothing from disk, so every module can be exercised before the real
    model or cache is available.
    """

    def __init__(self, n: int = 4, cfg: Config | None = None, spatial_size=(96, 96, 96),
                 channels: int = 4, num_classes: int = 5, seed: int = 0):
        if cfg is not None:
            spatial_size = cfg.spatial_size
            channels = len(cfg.modalities)
            num_classes = cfg.num_classes
        self.n = n
        self.spatial_size = tuple(spatial_size)
        self.channels = channels
        self.num_classes = num_classes
        self.seed = seed

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict:
        rng = np.random.default_rng(self.seed + idx)
        image = rng.standard_normal((self.channels, *self.spatial_size), dtype=np.float32)
        label = rng.integers(0, self.num_classes, size=(1, *self.spatial_size), dtype=np.int64)
        return {
            "image": torch.as_tensor(image),
            "label": torch.as_tensor(label),
            "sample_id": f"dummy-{idx:04d}",
            "source_subject_id": f"dummy-{idx:04d}",
        }
