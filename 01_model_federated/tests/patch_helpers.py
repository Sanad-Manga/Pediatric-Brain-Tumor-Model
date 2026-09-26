"""Tiny synthetic full-resolution cache in the REAL file format, for the patch tests.

Nothing here needs the real data or a GPU. Statistics are computed independently
with plain numpy (not by importing the builder) so the tests check the code
against a second implementation rather than against itself.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SHAPE = (48, 48, 40)


@dataclass
class Cohort:
    manifest: str
    cache: str
    sids: list[str]
    seg: dict[str, np.ndarray] = field(default_factory=dict)
    img: dict[str, np.ndarray] = field(default_factory=dict)      # float32 copy of the STORED float16
    stats: dict[str, np.ndarray] = field(default_factory=dict)
    et_patients: set[str] = field(default_factory=set)
    tumor_patients: set[str] = field(default_factory=set)


def independent_stats(image: np.ndarray) -> np.ndarray:
    stats = np.zeros((image.shape[0], 2), dtype=np.float32)
    for c in range(image.shape[0]):
        values = image[c][image[c] > 0].astype(np.float64)
        stats[c] = (values.mean(), values.std()) if values.size else (0.0, 1.0)
    return stats


def make_cohort(tmp_path: Path, n_et: int = 3, n_tumor_only: int = 2, n_empty: int = 1,
                shape: tuple[int, int, int] = SHAPE, positional: bool = False, seed: int = 0) -> Cohort:
    """n_et patients have ET (label 1) inside edema; n_tumor_only have tumour but no ET;
    n_empty have no tumour at all. positional=True makes the first three channels encode
    the voxel's x / y / z coordinate (+1) so a crop's origin can be read back from it."""
    cache = Path(tmp_path) / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    cohort = Cohort(manifest=str(Path(tmp_path) / "manifest.json"), cache=str(cache), sids=[])

    for i in range(n_et + n_tumor_only + n_empty):
        sid = f"P{i:02d}"
        seg = np.zeros(shape, dtype=np.uint8)
        if i < n_et:
            seg[10:22, 12:24, 8:18] = 4
            seg[14:18, 16:20, 10:14] = 1
        elif i < n_et + n_tumor_only:
            seg[24:36, 22:34, 12:24] = 4
            seg[28:31, 26:29, 15:18] = 2

        image = rng.random((4,) + shape).astype(np.float32) * 100.0 + 1.0
        image[:, ::6, :, :] = 0.0  # zero planes so "background exactly 0" is actually exercised
        if positional:
            xs, ys, zs = np.meshgrid(*[np.arange(s) for s in shape], indexing="ij")
            image[0], image[1], image[2] = xs + 1.0, ys + 1.0, zs + 1.0

        stored = image.astype(np.float16)
        stored_f32 = stored.astype(np.float32)
        np.save(cache / f"{sid}.img.npy", stored)
        np.save(cache / f"{sid}.seg.npy", seg)
        stats = independent_stats(stored_f32)
        np.save(cache / f"{sid}.stats.npy", stats)

        cohort.sids.append(sid)
        cohort.seg[sid], cohort.img[sid], cohort.stats[sid] = seg, stored_f32, stats
        if (seg == 1).any():
            cohort.et_patients.add(sid)
        if (seg > 0).any():
            cohort.tumor_patients.add(sid)

    Path(cohort.manifest).write_text(json.dumps(cohort.sids), encoding="utf-8")
    return cohort
