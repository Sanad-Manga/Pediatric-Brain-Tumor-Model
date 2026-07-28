"""2D slice extraction, the slice cache, and shape harmonisation.

Source volumes are 240 x 240 x 155 at 1mm isotropic. Slicing the native data
(rather than resampling to 96 cubed as the 3D version did) removes the
resampling ceiling entirely -- round-tripping ground truth through 96 cubed
capped achievable Dice at ~0.93 WT / ~0.74-0.87 ET.

Plane / axis map (the only assignment consistent with the measured counts):

    axis 0 = sagittal   -- dropped, never emitted
    axis 1 = coronal    -- 240 slices of 240 x 155
    axis 2 = axial      -- 155 slices of 240 x 240

Cache layout, one file per subject per plane::

    <cache_2d>/<subject_id>_<plane>.npz
      t1c, t1n, t2f, t2w : (N, H, W) float16, raw un-normalized
      seg                : (N, H, W) uint8, labels 0-4
      norm_mean/norm_std : (4,) float32, per-modality z-score stats computed
                           over non-zero voxels of the WHOLE VOLUME
      has_tumor          : (N,) bool
      subject_id / plane / hospital : scalar strings

Normalisation statistics are computed per volume and stored in the cache, not
recomputed per slice: per-slice normalisation changes what an intensity means
from slice to slice and hurts the model. Storing them here is what makes
volume-level normalisation enforceable at load time without holding the volume
in memory.

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

#: cache filenames look like `BraTS-PED-00030-000_axial.npz`
_CACHE_RE = re.compile(r"^(?P<subject>.+)_(?P<plane>axial|coronal)\.npz$")


# --------------------------------------------------------------- slicing core
def extract_plane_slices(volume: np.ndarray, axis: int) -> np.ndarray:
    """Slice ``volume`` along ``axis``, returning ``(N, H, W)``.

    ``np.moveaxis`` is used rather than a Python loop so the operation stays a
    view where possible, and so :func:`restack_plane_slices` is its exact
    inverse.
    """
    if volume.ndim != 3:
        raise ValueError(f"expected a 3D volume, got shape {volume.shape}")
    return np.moveaxis(volume, axis, 0)


def restack_plane_slices(slices: np.ndarray, axis: int) -> np.ndarray:
    """Inverse of :func:`extract_plane_slices`: ``(N, H, W)`` -> volume."""
    return np.moveaxis(slices, 0, axis)


def compute_norm_stats(volume: np.ndarray, min_std: float = 1e-6) -> tuple[float, float]:
    """Z-score statistics over non-zero (brain) voxels of a whole volume.

    The volumes are skull-stripped, so zero is background and excluding it keeps
    the statistics on actual tissue. A volume that is entirely zero yields
    ``(0.0, min_std)``, so normalising it gives all zeros rather than NaN.
    """
    brain = volume[volume != 0]
    if brain.size == 0:
        return 0.0, float(min_std)
    return float(brain.mean()), float(max(brain.std(), min_std))


def normalize(slice_arr: np.ndarray, mean: float, std: float, min_std: float = 1e-6) -> np.ndarray:
    """Apply volume-level z-score statistics to one slice."""
    return (slice_arr.astype(np.float32) - np.float32(mean)) / np.float32(max(std, min_std))


# --------------------------------------------------------- shape harmonisation
def compute_pad(shape: tuple[int, int], target: tuple[int, int]) -> tuple[tuple[int, int], ...]:
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


def pad_to(arr: np.ndarray, target: tuple[int, int]) -> tuple[np.ndarray, tuple]:
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


# ------------------------------------------------------------------ cache I/O
def cache_filename(subject_id: str, plane: str) -> str:
    return f"{subject_id}_{plane}.npz"


def parse_cache_filename(name: str) -> tuple[str, str] | None:
    """``'sub_axial.npz'`` -> ``('sub', 'axial')``; ``None`` if it does not match."""
    m = _CACHE_RE.match(name)
    return (m.group("subject"), m.group("plane")) if m else None


def write_subject_plane(
    out_dir: Path,
    subject_id: str,
    plane: str,
    hospital: str,
    modality_slices: dict[str, np.ndarray],
    seg_slices: np.ndarray,
    norm_mean: np.ndarray,
    norm_std: np.ndarray,
) -> Path:
    """Write one ``<subject>_<plane>.npz`` in the cache format."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / cache_filename(subject_id, plane)

    seg = seg_slices.astype(np.uint8)
    bad = set(np.unique(seg).tolist()) - {0, 1, 2, 3, 4}
    if bad:
        raise ValueError(f"{subject_id} {plane}: segmentation has labels outside 0-4: {sorted(bad)}")

    flat = seg.reshape(seg.shape[0], -1)
    arrays = {m: v.astype(np.float16) for m, v in modality_slices.items()}
    np.savez_compressed(
        path,
        **arrays,
        seg=seg,
        norm_mean=np.asarray(norm_mean, dtype=np.float32),
        norm_std=np.asarray(norm_std, dtype=np.float32),
        has_tumor=(flat != 0).any(axis=1),
        # Enhancing tumour (label 1) tracked separately: it is the smallest of
        # the three scored regions, it is absent entirely in many subjects, and
        # a tumour-vs-empty balance alone would leave it under-represented.
        has_et=(flat == 1).any(axis=1),
        subject_id=np.asarray(subject_id),
        plane=np.asarray(plane),
        hospital=np.asarray(hospital),
    )
    return path


def load_subject_plane(cache_dir: Path, subject_id: str, plane: str) -> dict:
    """Read one cache file into a dict of arrays. Raises if it is missing."""
    path = Path(cache_dir) / cache_filename(subject_id, plane)
    if not path.exists():
        raise FileNotFoundError(f"cache file not found: {path}")
    with np.load(path, allow_pickle=False) as npz:
        return {k: npz[k] for k in npz.files}


def index_cache(cache_dir: Path, planes=None) -> dict[str, dict[str, dict]]:
    """Scan a cache directory, returning ``{subject_id: {plane: summary}}``.

    The summary holds only what plan building needs -- ``n_slices`` and the
    ``has_tumor`` mask -- so indexing does not decompress the image arrays.
    """
    cache_dir = Path(cache_dir)
    if not cache_dir.is_dir():
        raise FileNotFoundError(
            f"2D slice cache not found: {cache_dir}\n"
            f"  set paths.cache_2d in config.yaml, or run: "
            f"python run.py slices --out <dir>"
        )

    wanted = set(planes) if planes else None
    index: dict[str, dict[str, dict]] = {}
    for path in sorted(cache_dir.glob("*.npz")):
        parsed = parse_cache_filename(path.name)
        if parsed is None:
            continue
        subject_id, plane = parsed
        if wanted is not None and plane not in wanted:
            continue
        with np.load(path, allow_pickle=False) as npz:
            if "has_et" not in npz.files:
                raise ValueError(
                    f"{path.name} predates the has_et field; re-export the cache with "
                    f"`python run.py slices --out <dir>`"
                )
            has_tumor = np.asarray(npz["has_tumor"], dtype=bool)
            has_et = np.asarray(npz["has_et"], dtype=bool)
        index.setdefault(subject_id, {})[plane] = {
            "n_slices": int(has_tumor.size),
            "has_tumor": has_tumor,
            "has_et": has_et,
        }
    return index


# ------------------------------------------------------------ manifest access
def load_manifest(manifest_dir: Path, name: str) -> list[str]:
    """Read a flat JSON list of subject IDs."""
    path = Path(manifest_dir) / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"manifest not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        ids = json.load(fh)
    if not isinstance(ids, list):
        raise ValueError(f"manifest {path} is not a flat JSON list")
    return [str(i) for i in ids]


def subject_hospital_map(manifest_dir: Path) -> dict[str, str]:
    """``{subject_id: 'hospitalA' | 'hospitalB' | 'heldout'}``."""
    out: dict[str, str] = {}
    for name in ("hospitalA", "hospitalB", "heldout"):
        for sid in load_manifest(manifest_dir, name):
            out[sid] = name
    return out


# ------------------------------------------------------------- NIfTI -> cache
def _find_nifti(subject_dir: Path, subject_id: str, suffix: str) -> Path | None:
    """Locate ``<subject>-<suffix>.nii.gz`` tolerating minor naming variants."""
    for pattern in (f"*{suffix}.nii.gz", f"*{suffix}.nii"):
        hits = sorted(subject_dir.glob(pattern))
        if hits:
            return hits[0]
    return None


def export_subject(
    nifti_root: Path,
    out_dir: Path,
    subject_id: str,
    hospital: str,
    cfg: Config,
    planes=None,
) -> list[Path]:
    """Slice one subject's NIfTI volumes into the 2D cache.

    Sagittal is never emitted. Normalisation statistics are computed once per
    volume here and stored alongside the raw slices.
    """
    try:
        import nibabel as nib
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "slices requires nibabel (pip install -r requirements.txt)"
        ) from exc

    subject_dir = Path(nifti_root) / subject_id
    if not subject_dir.is_dir():
        raise FileNotFoundError(f"subject folder not found: {subject_dir}")

    modalities = cfg.modalities
    volumes: dict[str, np.ndarray] = {}
    means, stds = [], []
    for mod in modalities:
        path = _find_nifti(subject_dir, subject_id, mod)
        if path is None:
            raise FileNotFoundError(f"{subject_id}: no {mod} volume in {subject_dir}")
        vol = np.asanyarray(nib.load(str(path)).dataobj).astype(np.float32)
        volumes[mod] = vol
        mean, std = compute_norm_stats(vol, cfg.min_std)
        if std <= cfg.min_std:
            LOGGER.warning("%s %s: volume is entirely zero; normalised slices will be zero",
                           subject_id, mod)
        means.append(mean)
        stds.append(std)

    seg_path = _find_nifti(subject_dir, subject_id, "seg")
    if seg_path is None:
        raise FileNotFoundError(f"{subject_id}: no seg volume in {subject_dir}")
    seg_vol = np.asanyarray(nib.load(str(seg_path)).dataobj).astype(np.uint8)

    expected = cfg.volume_shape
    first = volumes[modalities[0]]
    if tuple(first.shape) != tuple(expected):
        LOGGER.warning("%s: volume shape %s differs from the expected %s; slicing anyway",
                       subject_id, tuple(first.shape), tuple(expected))
    if tuple(seg_vol.shape) != tuple(first.shape):
        raise ValueError(
            f"{subject_id}: seg shape {seg_vol.shape} does not match image shape {first.shape}"
        )

    written = []
    for plane in (planes or cfg.planes):
        axis = cfg.plane_axis(plane)
        written.append(
            write_subject_plane(
                Path(out_dir),
                subject_id,
                plane,
                hospital,
                {m: extract_plane_slices(v, axis) for m, v in volumes.items()},
                extract_plane_slices(seg_vol, axis),
                np.asarray(means, dtype=np.float32),
                np.asarray(stds, dtype=np.float32),
            )
        )
    return written


def export_cache(
    cfg: Config,
    out_dir: Path,
    manifest: str = "all",
    limit: int | None = None,
    planes=None,
) -> dict:
    """Export the 2D slice cache for one manifest (or all three).

    Returns a summary dict. A missing NIfTI root is a hard error before anything
    is written -- a half-written cache is worse than none.
    """
    nifti_root = cfg.resolve("nifti_root")
    if not nifti_root.is_dir():
        raise FileNotFoundError(
            f"raw NIfTI dataset not found: {nifti_root}\n"
            f"  set paths.nifti_root in config.yaml to the BraTS-PEDs folder"
        )

    manifest_dir = cfg.resolve("manifests")
    hospitals = subject_hospital_map(manifest_dir)
    names = ["hospitalA", "hospitalB", "heldout"] if manifest == "all" else [manifest]

    subjects: list[str] = []
    for name in names:
        subjects.extend(load_manifest(manifest_dir, name))
    if limit is not None:
        subjects = subjects[:limit]

    out_dir = Path(out_dir)
    written, skipped = [], []
    for sid in subjects:
        try:
            written.extend(export_subject(nifti_root, out_dir, sid, hospitals.get(sid, "unknown"),
                                          cfg, planes))
        except FileNotFoundError as exc:
            LOGGER.warning("skipping %s: %s", sid, exc)
            skipped.append(sid)

    return {
        "out_dir": out_dir,
        "subjects_requested": len(subjects),
        "subjects_written": len(subjects) - len(skipped),
        "subjects_skipped": skipped,
        "files_written": written,
        "planes": list(planes or cfg.planes),
    }
