"""The 2D slice cache: layout, indexing, and shape harmonisation.

The team produces the slices directly from the original NIfTI volumes, and that
is the dataset from now on -- this section consumes them, it does not create
them. Layout, one file per slice::

    <cache_2d>/<subject_id>/<plane>/slice_<NNN>.npz     plane in {axial, coronal}
      image : (4, H, W) float32  -- t1c, t1n, t2f, t2w, in that channel order,
                                    ALREADY z-scored per volume over brain voxels
      mask  : (1, H, W) uint8    -- labels 0-4

    axial   : 155 files of 240 x 240
    coronal : 240 files of 240 x 155

Two consequences of that format:

* **Normalisation is already applied**, and applied per volume rather than per
  slice (verified: brain-voxel statistics drift across slices of a subject
  instead of being pinned to 0/1). Nothing here re-normalises.
* **Tumour presence is not recorded**, but plan building needs it *before*
  selecting slices. Deriving it means opening ~90,000 mask files, so it is
  computed once by `run.py index` into `<cache_2d>/index.json` and read from
  there afterwards.

Axial (240x240) and coronal (240x155) slices are brought to one network size by
zero-padding only. Resizing would stretch coronal slices, and aspect distortion
changes tumour morphology. Padding is also exactly invertible, which the
per-patient restacking in `evaluate.py` depends on.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import numpy as np

from .config import Config

LOGGER = logging.getLogger(__name__)

#: channel order inside `image`, confirmed with the team
MODALITY_ORDER = ("t1c", "t1n", "t2f", "t2w")

INDEX_FILENAME = "index.json"
SLICE_STEM = "slice_"


# ------------------------------------------------------------------ cache paths
def subject_dir(cache_dir, subject_id: str) -> Path:
    return Path(cache_dir) / subject_id


def plane_dir(cache_dir, subject_id: str, plane: str) -> Path:
    return Path(cache_dir) / subject_id / plane


def slice_path(cache_dir, subject_id: str, plane: str, slice_index: int) -> Path:
    return plane_dir(cache_dir, subject_id, plane) / f"{SLICE_STEM}{int(slice_index):03d}.npz"


def slice_number(path) -> int:
    """Slice index encoded in a filename: ``slice_008.npz`` -> ``8``."""
    m = re.search(r"(\d+)", Path(path).stem)
    if m is None:
        raise ValueError(f"cannot read a slice index from {path}")
    return int(m.group(1))


def list_slice_files(cache_dir, subject_id: str, plane: str) -> list[Path]:
    """Slice files for one subject/plane, ordered by slice index.

    Indices are **not** guaranteed to start at 0 or be contiguous with the
    filesystem order: subjects whose blank edge slices were dropped run e.g.
    ``slice_008..slice_134``. Everything downstream must use the number in the
    filename, never the position in this list -- restacking a prediction volume
    depends on it.
    """
    d = plane_dir(cache_dir, subject_id, plane)
    if not d.is_dir():
        return []
    return sorted(d.glob(f"{SLICE_STEM}*.npz"), key=slice_number)


def list_slice_indices(cache_dir, subject_id: str, plane: str) -> list[int]:
    """The actual slice indices present for one subject/plane, ascending."""
    return [slice_number(p) for p in list_slice_files(cache_dir, subject_id, plane)]


def list_subjects(cache_dir) -> list[str]:
    """Subject IDs present in the cache, sorted."""
    cache_dir = Path(cache_dir)
    if not cache_dir.is_dir():
        raise FileNotFoundError(
            f"2D slice cache not found: {cache_dir}\n"
            f"  set paths.cache_2d in config.yaml"
        )
    return sorted(p.name for p in cache_dir.iterdir() if p.is_dir())


# -------------------------------------------------------------------- slice I/O
def load_slice(cache_dir, subject_id: str, plane: str, slice_index: int) -> dict:
    """Read one slice file. Returns ``{"image": (4,H,W), "mask": (1,H,W)}``."""
    path = slice_path(cache_dir, subject_id, plane, slice_index)
    if not path.exists():
        raise FileNotFoundError(f"slice file not found: {path}")
    with np.load(path, allow_pickle=False) as npz:
        missing = {"image", "mask"} - set(npz.files)
        if missing:
            raise ValueError(
                f"{path} is missing {sorted(missing)}; expected keys 'image' and "
                f"'mask' (found {sorted(npz.files)})"
            )
        return {"image": npz["image"], "mask": npz["mask"]}


def write_slice(cache_dir, subject_id: str, plane: str, slice_index: int,
                image: np.ndarray, mask: np.ndarray) -> Path:
    """Write one slice file in the cache format. Used by the dummy generator."""
    path = slice_path(cache_dir, subject_id, plane, slice_index)
    path.parent.mkdir(parents=True, exist_ok=True)

    mask = np.asarray(mask, dtype=np.uint8)
    if mask.ndim == 2:
        mask = mask[None]
    bad = set(np.unique(mask).tolist()) - {0, 1, 2, 3, 4}
    if bad:
        raise ValueError(f"{subject_id} {plane} {slice_index}: labels outside 0-4: {sorted(bad)}")

    np.savez_compressed(path,
                        image=np.asarray(image, dtype=np.float32),
                        mask=mask)
    return path


# --------------------------------------------------------- shape harmonisation
def compute_pad(shape, target) -> tuple[tuple[int, int], ...]:
    """Centred pad widths taking ``shape`` up to ``target``.

    An odd remainder puts the extra row/column at the end, so 155 -> 256 pads
    (50, 51). A source slice larger than the target is a hard error: silently
    centre-cropping it would delete tumour.
    """
    pads = []
    for size, tgt in zip(shape, target):
        if size > tgt:
            raise ValueError(
                f"slice shape {tuple(shape)} exceeds common_size {tuple(target)} "
                f"along an axis ({size} > {tgt}); padding cannot shrink a slice"
            )
        total = tgt - size
        before = total // 2
        pads.append((before, total - before))
    return tuple(pads)


def pad_to(arr: np.ndarray, target) -> tuple[np.ndarray, tuple]:
    """Zero-pad the trailing two axes of ``arr`` to ``target``.

    Returns ``(padded, pad)`` where ``pad`` is ``((top, bottom), (left, right))``
    and is exactly what :func:`unpad` needs to undo this.
    """
    pad = compute_pad(arr.shape[-2:], target)
    lead = ((0, 0),) * (arr.ndim - 2)
    return np.pad(arr, lead + pad, mode="constant", constant_values=0), pad


def unpad(arr: np.ndarray, pad) -> np.ndarray:
    """Undo :func:`pad_to`. ``unpad(pad_to(x)[0], pad) == x`` bitwise."""
    (top, bottom), (left, right) = pad
    h = arr.shape[-2] - bottom
    w = arr.shape[-1] - right
    return arr[..., top:h, left:w]


def extract_plane_slices(volume: np.ndarray, axis: int) -> np.ndarray:
    """Slice a volume along ``axis`` into ``(N, H, W)``. Used by the dummy cache."""
    if volume.ndim != 3:
        raise ValueError(f"expected a 3D volume, got shape {volume.shape}")
    return np.moveaxis(volume, axis, 0)


# ---------------------------------------------------------------- index building
def _normalisation_looks_per_slice(samples) -> bool:
    """Heuristic guard against a per-slice-normalised cache.

    Per-slice normalisation pins every slice's brain voxels to mean 0 / std 1,
    which changes what an intensity means from slice to slice. Per-volume
    statistics drift instead. Cheap to check, expensive to discover later.
    """
    if len(samples) < 3:
        return False
    means = [m for m, _ in samples]
    stds = [s for _, s in samples]
    return (max(means) - min(means)) < 0.02 and all(abs(s - 1.0) < 0.02 for s in stds)


def build_index(cache_dir, planes=None, subjects=None, progress_every: int = 25) -> dict:
    """Scan every mask once and record which slices contain tumour / ET.

    This is the expensive pass (~90,000 files for the full cohort) and the whole
    reason the index exists: `plan` needs tumour presence *before* it selects
    slices, and re-deriving it per run would dominate runtime.
    """
    cache_dir = Path(cache_dir)
    planes = list(planes) if planes else ["axial", "coronal"]
    subjects = list(subjects) if subjects else list_subjects(cache_dir)

    index = {"modalities": list(MODALITY_ORDER), "planes": planes, "subjects": {}}
    norm_samples = []
    total_files = 0

    for n, subject_id in enumerate(subjects, 1):
        entry = {}
        for plane in planes:
            files = list_slice_files(cache_dir, subject_id, plane)
            if not files:
                continue
            indices, tumor, et, shape = [], [], [], None
            for position, path in enumerate(files):
                # The number in the filename, never the position: subjects with
                # dropped edge slices run e.g. 8..134, and a prediction volume
                # restacked on positions would be silently misaligned.
                i = slice_number(path)
                indices.append(i)
                with np.load(path, allow_pickle=False) as npz:
                    mask = npz["mask"]
                    if shape is None:
                        shape = list(mask.shape[-2:])
                    flat = mask.reshape(-1)
                    if flat.any():
                        tumor.append(i)
                        if (flat == 1).any():
                            et.append(i)
                    # sample a few images to sanity-check the normalisation
                    if len(norm_samples) < 8 and position and position % 40 == 0:
                        img = npz["image"][0]
                        brain = img[img != 0]
                        if brain.size > 100:
                            norm_samples.append((float(brain.mean()), float(brain.std())))
            total_files += len(files)
            entry[plane] = {"n_slices": len(files), "shape": shape,
                            "indices": indices, "tumor": tumor, "et": et}
        if entry:
            index["subjects"][subject_id] = entry
        if progress_every and n % progress_every == 0:
            LOGGER.info("indexed %d/%d subjects (%d slice files)", n, len(subjects), total_files)

    if _normalisation_looks_per_slice(norm_samples):
        LOGGER.warning(
            "cache appears to be normalised PER SLICE (brain mean/std pinned to "
            "0/1 on every slice). Per-volume normalisation is expected -- per-slice "
            "changes what an intensity means from slice to slice."
        )

    index["n_subjects"] = len(index["subjects"])
    index["n_slice_files"] = total_files
    return index


def write_index(index: dict, cache_dir) -> Path:
    path = Path(cache_dir) / INDEX_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(index, fh, indent=1, sort_keys=True)
        fh.write("\n")
    return path


def read_index(cache_dir) -> dict:
    path = Path(cache_dir) / INDEX_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"slice index not found: {path}\n"
            f"  build it once with: python run.py index --cache {cache_dir}"
        )
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def index_cache(cache_dir, planes=None) -> dict[str, dict[str, dict]]:
    """``{subject_id: {plane: summary}}`` from the index.

    Every list in a summary holds **actual slice indices**, not positions, so
    callers can pass them straight to :func:`load_slice`::

        indices : every slice present, ascending
        tumor   : those whose mask has any non-zero label
        et      : those whose mask contains label 1
    """
    index = read_index(cache_dir)
    wanted = set(planes) if planes else None

    out: dict[str, dict[str, dict]] = {}
    for subject_id, per_plane in index.get("subjects", {}).items():
        for plane, summary in per_plane.items():
            if wanted is not None and plane not in wanted:
                continue
            indices = [int(i) for i in summary["indices"]]
            tumor = set(int(i) for i in summary["tumor"])
            et = set(int(i) for i in summary["et"])
            out.setdefault(subject_id, {})[plane] = {
                "n_slices": int(summary["n_slices"]),
                "shape": tuple(summary.get("shape") or ()),
                "indices": indices,
                "tumor": sorted(tumor),
                "et": sorted(et),
                "empty": [i for i in indices if i not in tumor],
            }
    if not out:
        raise ValueError(f"slice index at {cache_dir} lists no subjects")
    return out


# ------------------------------------------------------------ manifest access
def load_manifest(manifest_dir, name: str) -> list[str]:
    """Read a flat JSON list of subject IDs."""
    path = Path(manifest_dir) / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"manifest not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        ids = json.load(fh)
    if not isinstance(ids, list):
        raise ValueError(f"manifest {path} is not a flat JSON list")
    return [str(i) for i in ids]


def subject_hospital_map(manifest_dir) -> dict[str, str]:
    """``{subject_id: 'hospitalA' | 'hospitalB' | 'heldout'}``."""
    out: dict[str, str] = {}
    for name in ("hospitalA", "hospitalB", "heldout"):
        for sid in load_manifest(manifest_dir, name):
            out[sid] = name
    return out
