"""Cache access + manifest handling for the BraTS-PEDs 96-cube cache.

Cache format (per CONTRACTS.md, verified against the real folder):
one flat file per subject, ``<cache>/<subject_id>.npz``, holding

    t1c, t1n, t2f, t2w : 96x96x96 float16, RAW un-normalized intensities
    seg                : 96x96x96 uint8, labels 0-4

Intensities are *not* pre-normalized, so this module z-scores each modality over
brain (non-zero) voxels only. Background stays exactly 0.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

LOGGER = logging.getLogger(__name__)

HOSPITAL_MANIFESTS = {"A": "hospitalA.json", "B": "hospitalB.json"}
HELDOUT_MANIFEST = "heldout.json"


# --------------------------------------------------------------------- manifests
def load_manifest(manifest_dir: str | Path, name: str) -> list[str]:
    """Load a manifest by file name (e.g. ``hospitalA.json``) as a list of IDs."""
    path = Path(manifest_dir) / name
    if not path.exists():
        raise FileNotFoundError(f"manifest not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        ids = json.load(fh)
    if not isinstance(ids, list):
        raise ValueError(f"manifest {path} must be a JSON list of subject IDs")
    return [str(s) for s in ids]


def load_hospital(manifest_dir: str | Path, hospital: str) -> list[str]:
    """Load Hospital ``A`` or ``B``."""
    key = hospital.upper()
    if key not in HOSPITAL_MANIFESTS:
        raise KeyError(f"unknown hospital {hospital!r}; expected one of {sorted(HOSPITAL_MANIFESTS)}")
    return load_manifest(manifest_dir, HOSPITAL_MANIFESTS[key])


def load_heldout(manifest_dir: str | Path) -> list[str]:
    return load_manifest(manifest_dir, HELDOUT_MANIFEST)


def load_all_manifests(manifest_dir: str | Path) -> dict[str, list[str]]:
    """Return ``{"A": [...], "B": [...], "heldout": [...]}``."""
    return {
        "A": load_hospital(manifest_dir, "A"),
        "B": load_hospital(manifest_dir, "B"),
        "heldout": load_heldout(manifest_dir),
    }


def subject_path(cache_dir: str | Path, subject_id: str) -> Path:
    return Path(cache_dir) / f"{subject_id}.npz"


# --------------------------------------------------------------- normalization
def znorm_channel(channel: np.ndarray) -> np.ndarray:
    """Z-score one modality over its non-zero (brain) voxels; background stays 0.

    An all-background channel is returned unchanged (no division by a zero std).
    """
    out = channel.astype(np.float32, copy=True)
    mask = out != 0
    if not mask.any():
        return out
    values = out[mask]
    std = float(values.std())
    if std == 0.0:
        # Constant non-zero channel: centring it would blank the brain mask, so
        # leave it alone and let the caller's warning surface the oddity.
        return out
    out[mask] = (values - float(values.mean())) / std
    return out


# ------------------------------------------------------------------- loading
def load_subject_raw(cache_dir: str | Path, subject_id: str, modalities: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Load the raw, un-normalized arrays for one subject.

    Returns ``(image, label)`` with shapes ``(C, D, H, W)`` and ``(1, D, H, W)``.
    """
    path = subject_path(cache_dir, subject_id)
    if not path.exists():
        raise FileNotFoundError(f"cached subject {subject_id!r} not found at {path}")

    with np.load(path) as npz:
        available = set(npz.files)
        channels = []
        for key in modalities:
            if key not in available:
                raise KeyError(
                    f"subject {subject_id!r} ({path}) is missing modality {key!r}; "
                    f"available keys: {sorted(available)}"
                )
            channels.append(np.asarray(npz[key], dtype=np.float32))
        if "seg" not in available:
            raise KeyError(
                f"subject {subject_id!r} ({path}) is missing key 'seg'; "
                f"available keys: {sorted(available)}"
            )
        label = np.asarray(npz["seg"])

    image = np.stack(channels, axis=0)
    label = label.astype(np.int64)[None, ...]
    return image, label


def load_subject(
    cache_dir: str | Path,
    subject_id: str,
    modalities: list[str] | None = None,
    valid_labels: set[int] | None = None,
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Load one subject as ``(image (C,D,H,W) float32, label (1,D,H,W) int64)``.

    Raises ``ValueError`` if the segmentation carries a label outside
    ``valid_labels``.
    """
    modalities = modalities or ["t1c", "t1n", "t2f", "t2w"]
    valid_labels = valid_labels if valid_labels is not None else {0, 1, 2, 3, 4}

    image, label = load_subject_raw(cache_dir, subject_id, modalities)

    found = set(np.unique(label).tolist())
    bad = found - set(valid_labels)
    if bad:
        raise ValueError(
            f"subject {subject_id!r} has segmentation label(s) {sorted(bad)} "
            f"outside the allowed set {sorted(valid_labels)}"
        )

    if normalize:
        for c, name in enumerate(modalities):
            if not (image[c] != 0).any():
                LOGGER.warning(
                    "subject %s modality %s is entirely background; skipping normalization",
                    subject_id,
                    name,
                )
                continue
            image[c] = znorm_channel(image[c])

    return image, label


def brain_mask(image: np.ndarray) -> np.ndarray:
    """Per-channel non-zero mask, matching the mask used for normalization."""
    return image != 0


# ------------------------------------------------------------------ validation
def verify_label_ids(
    cache_dir: str | Path,
    subject_id: str,
    valid_labels: set[int] | None = None,
) -> set[int]:
    """Assert one real sample's label IDs sit inside the documented set.

    Per CONTRACTS.md the labels are 1=ET, 2=NET, 3=CC, 4=ED over a 0 background.
    Returns the set of labels actually found, and raises loudly (naming the
    subject) if anything unexpected is present.
    """
    valid_labels = valid_labels if valid_labels is not None else {0, 1, 2, 3, 4}
    path = subject_path(cache_dir, subject_id)
    if not path.exists():
        raise FileNotFoundError(f"cached subject {subject_id!r} not found at {path}")
    with np.load(path) as npz:
        if "seg" not in set(npz.files):
            raise KeyError(f"subject {subject_id!r} ({path}) is missing key 'seg'")
        found = set(np.unique(np.asarray(npz["seg"])).tolist())
    bad = found - set(valid_labels)
    if bad:
        raise ValueError(
            f"subject {subject_id!r} has segmentation label(s) {sorted(bad)} "
            f"outside the allowed set {sorted(valid_labels)}"
        )
    return found
