"""Full-resolution, tumour/ET-balanced random crops ("patches") for 3D training.

The 96^3 pipeline shows the network ONE downsampled volume per patient per epoch.
The shipped 2D pipeline shows it ~244 full-detail slices per patient, sampled so
that enough of them contain enhancing tumour (ET) and not too many are tumour-free.
PatchDataset gives the 3D network the same two things: many different views of each
patient at the original 1 mm resolution, drawn with the same style of balance.

Reads the cache written by tools/build_fullres_cache.py:
    <cache>/<sid>.img.npy    float16 (4, X, Y, Z) raw intensities
    <cache>/<sid>.seg.npy    uint8   (X, Y, Z)    labels 0-4
    <cache>/<sid>.stats.npy  float32 (4, 2)       mean, std over voxels > 0 per modality

Sampling is category-first: pick a category ('et', 'tumor' or 'random') by
`fractions`, then a patient among those ELIGIBLE for it, then a centre voxel, then
jitter and clip so the patch always lies fully inside the volume. An 'et'/'tumor'
centre voxel is therefore always inside its own patch (clipping only ever moves the
patch toward the interior, and the jitter is at most a quarter of the patch).

Each __getitem__ draws its OWN patch (the index is ignored) so every epoch sees new
patches. Use num_workers=0: the internal generator is a single stream, and it is not
restored on resume.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .data import load_manifest

CATEGORIES = ("et", "tumor", "random")
PATCH_DIVISOR = 16  # the UNet has 4 stride-2 stages
MAX_INDEX_VOXELS = 4000
ET_LABEL = 1


def zscore_crop(crop: np.ndarray, stats: np.ndarray) -> np.ndarray:
    """(C, *spatial) raw float32 -> normalised, using the stored full-volume stats.

    Same semantics as src/data.py::_zscore_normalize: voxels > 0 become
    (v - mean) / std (or v - mean when std < 1e-8); every other voxel is exactly 0.
    """
    out = np.zeros_like(crop, dtype=np.float32)
    for c in range(crop.shape[0]):
        mean, std = float(stats[c, 0]), float(stats[c, 1])
        brain = crop[c] > 0
        if std < 1e-8:
            out[c][brain] = crop[c][brain] - mean
        else:
            out[c][brain] = (crop[c][brain] - mean) / std
    return out


def _validate_patch_size(patch_size) -> tuple[int, int, int]:
    try:
        size = tuple(patch_size)
    except TypeError as exc:
        raise ValueError(f"patch_size must be three integers, got {patch_size!r}") from exc
    if len(size) != 3 or not all(isinstance(v, (int, np.integer)) and not isinstance(v, bool) and v > 0
                                 for v in size):
        raise ValueError(f"patch_size must be three positive integers, got {patch_size!r}")
    size = tuple(int(v) for v in size)
    bad = [v for v in size if v % PATCH_DIVISOR]
    if bad:
        raise ValueError(f"every patch dimension must be a multiple of {PATCH_DIVISOR} "
                         f"(the UNet has 4 stride-2 stages); got {size}")
    return size


def _validate_fractions(fractions) -> tuple[float, float, float]:
    try:
        values = tuple(float(f) for f in fractions)
    except TypeError as exc:
        raise ValueError(f"fractions must be three numbers, got {fractions!r}") from exc
    if len(values) != len(CATEGORIES):
        raise ValueError(f"fractions must have {len(CATEGORIES)} entries {CATEGORIES}, got {fractions!r}")
    if any(f < 0 for f in values):
        raise ValueError(f"fractions must be non-negative, got {values}")
    total = sum(values)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"fractions must sum to 1 (tolerance 1e-6), got {values} (sum {total})")
    return tuple(f / total for f in values)


class PatchDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        cache_path: str,
        patch_size=(128, 128, 128),
        patches_per_epoch: int = 580,
        fractions=(0.35, 0.45, 0.20),
        seed: int = 42,
    ) -> None:
        self.patch_size = _validate_patch_size(patch_size)
        if not isinstance(patches_per_epoch, (int, np.integer)) or patches_per_epoch < 1:
            raise ValueError(f"patches_per_epoch must be a positive integer, got {patches_per_epoch!r}")
        self.patches_per_epoch = int(patches_per_epoch)
        self.fractions = _validate_fractions(fractions)
        self.cache_path = Path(cache_path)
        self.seed = seed
        self.subject_ids = load_manifest(manifest_path)

        self._shapes: list[tuple[int, int, int]] = []
        self._et_coords: list[np.ndarray] = []
        self._tumor_coords: list[np.ndarray] = []
        self._stats: list[np.ndarray] = []
        self._img_paths: list[Path] = []
        self._seg_paths: list[Path] = []
        self._img_mm: dict[int, np.ndarray] = {}
        self._seg_mm: dict[int, np.ndarray] = {}

        for i, sid in enumerate(self.subject_ids):
            img_path = self.cache_path / f"{sid}.img.npy"
            seg_path = self.cache_path / f"{sid}.seg.npy"
            stats_path = self.cache_path / f"{sid}.stats.npy"
            for p in (img_path, seg_path, stats_path):
                if not p.is_file():
                    raise FileNotFoundError(f"Missing cached file: {p}")
            seg = np.load(seg_path)
            shape = tuple(int(s) for s in seg.shape)
            for axis, (p_dim, v_dim) in enumerate(zip(self.patch_size, shape)):
                if p_dim > v_dim:
                    raise ValueError(f"patch_size {self.patch_size} is larger than the volume {shape} "
                                     f"of {sid} on axis {axis}")
            rng = np.random.default_rng([abs(int(seed)), i])
            self._shapes.append(shape)
            self._et_coords.append(self._subsample(np.argwhere(seg == ET_LABEL), rng))
            self._tumor_coords.append(self._subsample(np.argwhere(seg > 0), rng))
            self._stats.append(np.load(stats_path).astype(np.float32))
            self._img_paths.append(img_path)
            self._seg_paths.append(seg_path)

        self._eligible = {
            "et": [i for i, c in enumerate(self._et_coords) if len(c)],
            "tumor": [i for i, c in enumerate(self._tumor_coords) if len(c)],
            "random": list(range(len(self.subject_ids))),
        }
        for category, fraction in zip(CATEGORIES, self.fractions):
            if fraction > 0 and not self._eligible[category]:
                raise ValueError(f"category {category!r} has fraction {fraction} but no patient in "
                                 f"the manifest is eligible for it")

        self._rng = np.random.default_rng(seed)
        self.last_draw: dict | None = None

    @staticmethod
    def _subsample(coords: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if len(coords) > MAX_INDEX_VOXELS:
            coords = coords[np.sort(rng.choice(len(coords), MAX_INDEX_VOXELS, replace=False))]
        return coords.astype(np.int32)

    def __len__(self) -> int:
        return self.patches_per_epoch

    def draw(self) -> dict:
        """Choose a patch without reading any image data: {category, sid, start}."""
        category = CATEGORIES[int(self._rng.choice(len(CATEGORIES), p=self.fractions))]
        pidx = int(self._rng.choice(self._eligible[category]))
        shape = np.array(self._shapes[pidx])
        patch = np.array(self.patch_size)
        if category == "et":
            centre = self._et_coords[pidx][int(self._rng.integers(len(self._et_coords[pidx])))]
        elif category == "tumor":
            centre = self._tumor_coords[pidx][int(self._rng.integers(len(self._tumor_coords[pidx])))]
        else:
            centre = self._rng.integers(0, shape)
        jitter = self._rng.integers(-(patch // 4), patch // 4 + 1)
        start = np.clip(centre.astype(np.int64) - patch // 2 + jitter, 0, shape - patch)
        return {"category": category, "sid": self.subject_ids[pidx], "patient_index": pidx,
                "start": tuple(int(s) for s in start)}

    def _memmaps(self, pidx: int) -> tuple[np.ndarray, np.ndarray]:
        if pidx not in self._img_mm:
            self._img_mm[pidx] = np.load(self._img_paths[pidx], mmap_mode="r")
            self._seg_mm[pidx] = np.load(self._seg_paths[pidx], mmap_mode="r")
        return self._img_mm[pidx], self._seg_mm[pidx]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        d = self.draw()
        self.last_draw = d
        pidx = d["patient_index"]
        img, seg = self._memmaps(pidx)
        window = tuple(slice(s, s + p) for s, p in zip(d["start"], self.patch_size))
        crop = np.asarray(img[(slice(None),) + window], dtype=np.float32)
        x = zscore_crop(crop, self._stats[pidx])
        y = np.asarray(seg[window]).astype(np.int64)
        return torch.from_numpy(x), torch.from_numpy(y)
