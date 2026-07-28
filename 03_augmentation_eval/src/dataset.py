"""Datasets and the CORAL-enabling batch sampler.

``SliceDataset`` glues a plan entry to the cache: load the raw slice, normalise
it with the *volume-level* statistics stored in the cache, pad it to the common
network size, and augment it only when the entry says so and the config allows
it.

``EvalSubjectDataset`` yields whole subjects rather than loose slices, because
Dice must be computed per patient from a restacked volume. It never augments,
regardless of ``use_augmentation`` -- held-out data is never augmented.

``PatientBalancedBatchSampler`` exists for section 02's CORAL: slices from one
patient are near-duplicates and give a rank-deficient covariance, so a batch has
to draw few slices from many patients. It also keeps every batch
plane-homogeneous, since axial vs coronal differ more than hospital A vs B and
mixing them makes CORAL align the plane difference instead of the scanner one.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset, Sampler

from .augment import apply_augmentation, build_transforms
from .config import Config
from .slices import load_subject_plane, normalize, pad_to


class _CacheReader:
    """Tiny LRU over decompressed cache files.

    One subject/plane file holds every slice of that volume, so re-reading it
    per slice would decompress ~70MB each time. Two entries is enough for the
    access patterns here (sequential within a subject).
    """

    def __init__(self, cache_dir: Path, maxsize: int = 2):
        self.cache_dir = Path(cache_dir)
        self.maxsize = maxsize
        self._store: OrderedDict[tuple[str, str], dict] = OrderedDict()

    def get(self, subject_id: str, plane: str) -> dict:
        key = (subject_id, plane)
        if key in self._store:
            self._store.move_to_end(key)
            return self._store[key]
        data = load_subject_plane(self.cache_dir, subject_id, plane)
        self._store[key] = data
        if len(self._store) > self.maxsize:
            self._store.popitem(last=False)
        return data


def _load_slice(reader: _CacheReader, cfg: Config, subject_id, plane, slice_index):
    """One normalised, padded slice: ``(image (C,H,W), label (1,H,W), pad, orig_shape)``."""
    data = reader.get(subject_id, plane)
    modalities = cfg.modalities
    mean, std = data["norm_mean"], data["norm_std"]

    channels = [
        normalize(data[mod][slice_index], float(mean[m]), float(std[m]), cfg.min_std)
        for m, mod in enumerate(modalities)
    ]
    image = np.stack(channels, axis=0)
    label = data["seg"][slice_index][None].astype(np.int64)

    orig_shape = tuple(image.shape[-2:])
    image, pad = pad_to(image, cfg.common_size)
    label, _ = pad_to(label, cfg.common_size)
    return image.astype(np.float32), label, pad, orig_shape


class SliceDataset(Dataset):
    """Training slices described by a plan.

    With ``use_augmentation: false`` no transform is constructed at all, so two
    loads of the same entry are bitwise-identical.
    """

    def __init__(self, entries, cfg: Config, cache_dir: Path | None = None):
        self.entries = list(entries)
        self.cfg = cfg
        self.reader = _CacheReader(Path(cache_dir) if cache_dir else cfg.resolve("cache_2d"))
        self.transforms = build_transforms(cfg)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict:
        entry = self.entries[idx]
        image, label, pad, orig_shape = _load_slice(
            self.reader, self.cfg,
            entry["source_subject_id"], entry["plane"], entry["slice_index"],
        )

        if self.transforms is not None and entry.get("is_augmented"):
            image, label = apply_augmentation(
                self.transforms, image, label, seed=entry.get("aug_seed")
            )

        return {
            "image": image,
            "label": label,
            "pad": pad,
            "orig_shape": orig_shape,
            **{k: entry[k] for k in
               ("hospital", "source_subject_id", "plane", "slice_index", "is_augmented")},
        }


class EvalSubjectDataset(Dataset):
    """One item per (subject, plane): every slice of that plane, never augmented."""

    def __init__(self, subject_ids, cfg: Config, cache_dir: Path | None = None, planes=None):
        self.cfg = cfg
        self.cache_dir = Path(cache_dir) if cache_dir else cfg.resolve("cache_2d")
        self.reader = _CacheReader(self.cache_dir)
        plane_choice = cfg.eval_plane
        self.planes = list(planes) if planes else (
            list(cfg.planes) if plane_choice == "both" else [plane_choice]
        )
        self.items = [(sid, plane) for sid in subject_ids for plane in self.planes]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        subject_id, plane = self.items[idx]
        data = self.reader.get(subject_id, plane)
        n_slices = int(data["seg"].shape[0])

        images, labels, pads = [], [], []
        orig_shape = None
        for i in range(n_slices):
            image, label, pad, orig_shape = _load_slice(
                self.reader, self.cfg, subject_id, plane, i
            )
            images.append(image)
            labels.append(label[0])
            pads.append(pad)

        return {
            "subject_id": subject_id,
            "plane": plane,
            "images": np.stack(images, axis=0),      # (N, C, H, W) padded
            "labels": np.stack(labels, axis=0),      # (N, H, W) padded
            "pad": pads[0],
            "orig_shape": orig_shape,
            "n_slices": n_slices,
        }


class PatientBalancedBatchSampler(Sampler):
    """Batches of distinct patients, one plane per batch.

    Both properties are requirements from section 02, not preferences: many
    patients per batch keeps the feature covariance full-rank, and a single
    plane per batch lets CORAL pair axial-A with axial-B.
    """

    def __init__(self, entries, batch_size: int, cfg: Config | None = None,
                 max_per_patient: int | None = None, seed: int = 0, drop_last: bool = True):
        self.entries = list(entries)
        self.batch_size = int(batch_size)
        self.drop_last = drop_last
        self.seed = int(seed)
        if max_per_patient is None:
            sampler_cfg = cfg.sampler if cfg is not None else {}
            max_per_patient = int(sampler_cfg.get("max_slices_per_patient_per_batch", 1))
        self.max_per_patient = max(1, int(max_per_patient))
        self.epoch = 0

    def __iter__(self):
        rng = np.random.default_rng([self.seed, self.epoch])
        self.epoch += 1

        # Group by plane first: a batch never spans planes.
        by_plane: dict[str, dict[str, list[int]]] = {}
        for i, entry in enumerate(self.entries):
            by_plane.setdefault(entry["plane"], {}) \
                    .setdefault(entry["source_subject_id"], []).append(i)

        batches = []
        for by_patient in by_plane.values():
            pools = {pid: [int(v) for v in rng.permutation(idx)]
                     for pid, idx in by_patient.items()}
            while True:
                available = [pid for pid, pool in pools.items() if pool]
                if len(available) < self.batch_size:
                    if self.drop_last or not available:
                        break
                    chosen = available
                else:
                    chosen = [available[k] for k in
                              rng.permutation(len(available))[: self.batch_size]]

                batch = []
                for pid in chosen:
                    for _ in range(self.max_per_patient):
                        if len(batch) >= self.batch_size or not pools[pid]:
                            break
                        batch.append(pools[pid].pop())
                if not batch:
                    break
                batches.append(batch)

        for k in rng.permutation(len(batches)):
            yield batches[k]

    def __len__(self) -> int:
        total = 0
        patients: dict[str, set] = {}
        counts: dict[str, int] = {}
        for entry in self.entries:
            plane = entry["plane"]
            patients.setdefault(plane, set()).add(entry["source_subject_id"])
            counts[plane] = counts.get(plane, 0) + 1
        for plane, pids in patients.items():
            if len(pids) >= self.batch_size:
                total += counts[plane] // self.batch_size
        return total
