"""Tumour-type proxy, ported to 2D, for an auxiliary classification head.

    !! IMPORTANT !!
    No tumour-type ground truth exists anywhere in this dataset. The manifests
    are flat subject-ID lists and the slice cache holds only `image` and `mask`.
    Nothing states which subjects are high-grade astrocytoma and which are
    diffuse midline glioma / DIPG (BraTS-PEDs' two broad categories).

This is an IMAGING PROXY derived from segmentation geometry alone -- ported
directly from the 3D version's `strata.py` (git show 968b810). It is NOT
histology and must never be reported as a tumour-type classification result.

Proxy features, computed from a subject's restacked mask volume:

    midline_offset : |centroid along the L-R axis - image centre| / centre.
                     DMG is by definition midline, so this is small for DMG.
    inferior_frac  : fraction of tumour voxels below the axial midpoint.
                     DIPG sits in the pons/brainstem, i.e. inferior.
    et_frac        : label-1 voxels / all tumour voxels. DMG/DIPG frequently
                     enhances little or not at all.

A subject is `dmg_like` when all three conditions hold; otherwise
`astrocytoma_like`.

Where Dr. Rana's point comes in: the point of using this as an AUXILIARY
CLASSIFICATION HEAD rather than a data-balancing signal is that the head reads
the model's own bottleneck `features` (already part of the shared
`model(x) -> (seg_logits, features)` interface) -- so the network is pushed to
actually extract location/enhancement information from the image, rather than
being handed the type as a label. This module only computes the proxy label to
train that head against; it does not touch the model interface at all.

Reconstructing a volume from 2D slices for this purpose is legitimate here
because it has been verified exact: restacking a real subject's masks from
axial and from coronal slices gives bitwise-identical volumes.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from .config import Config
from .slices import list_slice_indices, load_slice

LOGGER = logging.getLogger(__name__)

DMG_LIKE = "dmg_like"
ASTROCYTOMA_LIKE = "astrocytoma_like"
TYPE_LABELS = (ASTROCYTOMA_LIKE, DMG_LIKE)   # index 0 / 1, fixed order for the head

#: proxy thresholds -- same values and rationale as the 3D version
MIDLINE_OFFSET_MAX = 0.15
INFERIOR_FRAC_MIN = 0.60
ET_FRAC_MAX = 0.10


def subject_mask_volume(cache_dir, subject_id: str, cfg: Config,
                        plane: str = "axial") -> np.ndarray:
    """Reconstruct one subject's full mask volume from its 2D slices.

    Uses whichever plane is requested (axial by default, since it has the
    higher tumour-bearing fraction and needs no channel dimension). Dropped
    slices are placed at their true index and left as background -- consistent
    with how evaluation restacks predictions.
    """
    indices = list_slice_indices(cache_dir, subject_id, plane)
    if not indices:
        raise FileNotFoundError(f"no {plane} slices for {subject_id} under {cache_dir}")

    axis = cfg.plane_axis(plane)
    vs = cfg.volume_shape
    volume = np.zeros(vs, dtype=np.uint8)
    dest = np.moveaxis(volume, axis, 0)
    for i in indices:
        dest[i] = load_slice(cache_dir, subject_id, plane, i)["mask"][0]
    return volume


def seg_features(seg: np.ndarray, lr_axis: int = 0, is_axis: int = 2) -> dict:
    """The three proxy features from a ``(X, Y, Z)`` class mask volume."""
    seg = np.asarray(seg)
    tumor = seg > 0
    n_tumor = int(tumor.sum())
    if n_tumor == 0:
        # No tumour: features are undefined. Fall on the astrocytoma_like side
        # rather than inventing a DMG call from nothing.
        return {"midline_offset": 1.0, "inferior_frac": 0.0, "et_frac": 0.0, "n_tumor": 0}

    coords = np.argwhere(tumor)
    centre_lr = (seg.shape[lr_axis] - 1) / 2.0
    midline_offset = abs(float(coords[:, lr_axis].mean()) - centre_lr) / max(centre_lr, 1e-6)

    midpoint_is = (seg.shape[is_axis] - 1) / 2.0
    inferior_frac = float((coords[:, is_axis] < midpoint_is).mean())

    et_frac = float((seg == 1).sum()) / float(n_tumor)

    return {"midline_offset": midline_offset, "inferior_frac": inferior_frac,
            "et_frac": et_frac, "n_tumor": n_tumor}


def classify(features: dict) -> str:
    """Apply the three thresholds. All must hold for a DMG-like call."""
    return DMG_LIKE if (
        features["n_tumor"] > 0
        and features["midline_offset"] <= MIDLINE_OFFSET_MAX
        and features["inferior_frac"] >= INFERIOR_FRAC_MIN
        and features["et_frac"] <= ET_FRAC_MAX
    ) else ASTROCYTOMA_LIKE


def label_index(label: str) -> int:
    return TYPE_LABELS.index(label)


def classify_subject(cache_dir, subject_id: str, cfg: Config, plane: str = "axial") -> dict:
    """Full pipeline for one subject: reconstruct volume -> features -> label."""
    volume = subject_mask_volume(cache_dir, subject_id, cfg, plane)
    feats = seg_features(volume, cfg.data.get("lr_axis", 0), cfg.data.get("is_axis", 2))
    return {"subject_id": subject_id, "stratum": classify(feats), **feats}


TYPE_INDEX_FILENAME = "tumor_type.json"


def build_type_index(cache_dir, subjects, cfg: Config, plane: str = "axial",
                     progress_every: int = 25) -> dict:
    """Classify every subject once. Reused by `plan`, `train` and reports."""
    cache_dir = Path(cache_dir)
    out = {}
    for n, sid in enumerate(subjects, 1):
        try:
            out[sid] = classify_subject(cache_dir, sid, cfg, plane)
        except FileNotFoundError as exc:
            LOGGER.warning("skipping %s: %s", sid, exc)
        if progress_every and n % progress_every == 0:
            LOGGER.info("classified %d/%d subjects", n, len(subjects))

    counts = {t: sum(1 for v in out.values() if v["stratum"] == t) for t in TYPE_LABELS}
    LOGGER.info("tumour-type proxy: %s", counts)
    return {"plane": plane, "thresholds": {
                "midline_offset_max": MIDLINE_OFFSET_MAX,
                "inferior_frac_min": INFERIOR_FRAC_MIN,
                "et_frac_max": ET_FRAC_MAX},
            "subjects": out, "counts": counts}


def write_type_index(index: dict, cache_dir) -> Path:
    path = Path(cache_dir) / TYPE_INDEX_FILENAME
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(index, fh, indent=1, sort_keys=True)
        fh.write("\n")
    return path


def read_type_index(cache_dir) -> dict:
    path = Path(cache_dir) / TYPE_INDEX_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"tumour-type index not found: {path}\n"
            f"  build it with: python run.py tumor-type"
        )
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)
