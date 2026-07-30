"""Data + checkpoint loading for the demo.

These were placeholders returning `np.random.rand(...)` and a hardcoded Dice.
They now read the real slice cache and real trained checkpoints. Nothing here
fabricates a metric: if the cache or a checkpoint is absent, the caller gets an
exception or an explicit "unavailable" status to surface in the UI.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from .inference import (
    DEFAULT_CHECKPOINT,
    available_slices,
    available_subjects,
    checkpoint_metadata,
    find_tumor_slices,
)

#: Where the 2D slice cache lives on the machine serving the demo. Override with
#: the NEUROFED_CACHE_2D environment variable rather than editing this file.
DEFAULT_CACHE_2D = os.environ.get(
    "NEUROFED_CACHE_2D",
    str(Path(__file__).resolve().parents[1] / "demo_cache"),
)


def cache_dir() -> Path:
    return Path(DEFAULT_CACHE_2D)


def cache_available() -> bool:
    return cache_dir().is_dir()


def list_demo_subjects(limit: int | None = None) -> list[str]:
    """Subjects actually present in the cache."""
    subs = available_subjects(cache_dir())
    return subs[:limit] if limit else subs


def load_slice_image(subject_id: str, plane: str, slice_index: int) -> np.ndarray:
    """One real slice's four modalities, shape ``(4, H, W)``."""
    from src.slices import load_slice  # section 03, on sys.path via .inference

    return load_slice(cache_dir(), subject_id, plane, slice_index)["image"]


def load_mri_volume(subject_id: str, plane: str = "axial") -> np.ndarray:
    """Restack a subject's real slices into ``(4, H, W, N)``.

    Replaces the previous placeholder that returned random noise. Slices dropped
    from the cache stay zero-filled at their true index, matching how section 03
    restacks volumes for evaluation.
    """
    from src.slices import load_slice

    root = cache_dir()
    indices = available_slices(root, subject_id, plane)
    if not indices:
        raise FileNotFoundError(f"no {plane} slices cached for {subject_id} under {root}")

    first = load_slice(root, subject_id, plane, indices[0])["image"]
    c, h, w = first.shape
    volume = np.zeros((c, h, w, max(indices) + 1), dtype=first.dtype)
    for i in indices:
        volume[..., i] = load_slice(root, subject_id, plane, i)["image"]
    return volume


def tumor_slice_indices(subject_id: str, plane: str = "axial", limit: int | None = None):
    """Indices of slices whose ground truth actually contains tumour."""
    return find_tumor_slices(cache_dir(), subject_id, plane, limit=limit)


def load_checkpoint_status(checkpoint_path=DEFAULT_CHECKPOINT) -> dict:
    """Real provenance for a trained checkpoint.

    Replaces `load_federated_weights()`, which returned a fabricated
    ``{"global_dice": 0.924}``. The Dice reported here is the one the training
    run actually measured on held-out validation patients.
    """
    try:
        meta = checkpoint_metadata(checkpoint_path)
    except FileNotFoundError as exc:
        return {"status": "Unavailable", "detail": str(exc), "available": False}

    return {
        "status": "Loaded",
        "available": True,
        "epochs_completed": meta["epochs_completed"],
        "best_mean_dice": meta["best_mean_dice"],
        "architecture": meta["architecture"],
        "use_augmentation": meta["use_augmentation"],
        "use_mixup": meta["use_mixup"],
        "has_type_head": meta["has_type_head"],
        "path": meta["path"],
    }


def load_training_history(checkpoint_path=DEFAULT_CHECKPOINT) -> list[dict]:
    """Per-epoch loss / Dice history written alongside the checkpoint.

    Empty list when the run hasn't written one -- callers should render "no
    history yet" rather than invent a curve.
    """
    history_path = Path(checkpoint_path).parent / "history.json"
    if not history_path.exists():
        return []
    with open(history_path, "r", encoding="utf-8") as fh:
        blob = json.load(fh)
    # train.py writes a summary object wrapping the per-epoch records; older
    # files (and hand-made ones) may be the bare list.
    return blob.get("history", []) if isinstance(blob, dict) else blob


def load_metrics_cache(path: str | Path | None = None) -> dict | None:
    """Measured ROC / Dice / sensitivity / specificity / HD95, or None.

    Returns None rather than raising so a page can render an explicit "not
    computed yet" state. Build the cache with:
        python -m utils.build_metrics_cache
    """
    p = Path(path) if path else (Path(__file__).resolve().parent.parent / "data" / "roc_cache.json")
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_ablation_results(results_csv: str | Path | None = None) -> list[dict]:
    """Rows from section 03's ablation CSV, if it has been populated."""
    import csv

    path = Path(results_csv) if results_csv else (
        Path(__file__).resolve().parents[2] / "03_augmentation_eval" / "results" / "ablation_results.csv"
    )
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))
