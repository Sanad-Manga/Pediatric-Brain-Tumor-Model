"""Real model inference for the demo -- loads a section 03 checkpoint and runs it.

This is the bridge between the trained segmentation model and the Streamlit
pages. Everything here reads real files: an actual `.pt` checkpoint written by
`03_augmentation_eval/run.py train`, and actual slices from the 2D cache.

Nothing in this module invents numbers. If a checkpoint or the slice cache is
missing, the loader raises and the page is expected to say so plainly rather
than fall back to placeholder values -- a demo that silently shows made-up
metrics for a medical model is worse than one that shows an error.

Preprocessing deliberately reuses section 03's own `_load_slice`, so a slice
goes through the *identical* padding path used during training. Re-implementing
it here would risk a silent train/inference mismatch.
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

# --- make section 03's package importable ------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SECTION_03 = _REPO_ROOT / "03_augmentation_eval"
if str(_SECTION_03) not in sys.path:
    sys.path.insert(0, str(_SECTION_03))

from src.config import load_config                      # noqa: E402
from src.dataset import _load_slice                     # noqa: E402
from src.model import TumorTypeHead, build_model        # noqa: E402
from src.slices import list_slice_indices, list_subjects, unpad  # noqa: E402
from src.tumor_type import TYPE_LABELS                  # noqa: E402

#: label id -> (display name, hex colour). Matches the cache's 0-4 encoding.
LABEL_NAMES = {
    1: ("Enhancing Tumor (ET)", "#EF4444"),
    2: ("Non-enhancing Tumor Core (NETC)", "#10B981"),
    3: ("Cystic Component (CC)", "#3B82F6"),
    4: ("Peritumoral Edema (ED)", "#EAB308"),
}

DEFAULT_CHECKPOINT = _SECTION_03 / "checkpoints" / "overnight_run" / "best.pt"


def default_config(cache_dir: str | Path | None = None):
    """Section 03's config, optionally with the slice cache path overridden.

    The committed `config.yaml` points `cache_2d` wherever training last ran
    (e.g. a Colab Drive mount), which will not exist on a machine serving the
    demo. Callers pass the local cache explicitly.
    """
    cfg = load_config(_SECTION_03 / "config.yaml")
    if cache_dir is not None:
        cfg.paths["cache_2d"] = str(cache_dir)
    return cfg


def checkpoint_metadata(checkpoint_path: str | Path) -> dict:
    """Read a checkpoint's recorded provenance without building the model.

    These are the honest numbers to show in the UI: the epoch actually reached
    and the best validation Dice actually measured during that run.
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(
            f"no checkpoint at {path}\n"
            f"  train one with: python run.py train --run-id <id>"
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "path": str(path),
        "epoch": int(payload.get("epoch", -1)),
        "epochs_completed": int(payload.get("epoch", -1)) + 1,
        "best_mean_dice": payload.get("best_mean_dice"),
        "architecture": payload.get("architecture", "unet"),
        "spatial_dims": int(payload.get("spatial_dims", 2)),
        "use_augmentation": payload.get("use_augmentation"),
        "use_mixup": payload.get("use_mixup"),
        "has_type_head": "type_head_state_dict" in payload,
    }


@lru_cache(maxsize=4)
def load_model(checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
               cache_dir: str | Path | None = None,
               device: str = "cpu"):
    """Load the trained segmentation model (and tumour-type head, if present).

    Cached so Streamlit reruns don't re-read the checkpoint on every widget
    interaction.
    """
    path = Path(checkpoint_path)
    meta = checkpoint_metadata(path)
    cfg = default_config(cache_dir)

    payload = torch.load(path, map_location=device, weights_only=False)
    model = build_model(cfg, spatial_dims=meta["spatial_dims"])
    model.load_state_dict(payload["model_state_dict"])
    model.to(device).eval()

    type_head = None
    if meta["has_type_head"]:
        # Size the head from the saved weights rather than guessing: train.py
        # builds it from a real feature tensor's width, so the checkpoint is
        # the only authority on what that width was.
        state = payload["type_head_state_dict"]
        in_features = int(state["net.0.weight"].shape[1])
        hidden = int(state["net.0.weight"].shape[0])
        num_types = int(state["net.2.weight"].shape[0])
        type_head = TumorTypeHead(in_features, num_types=num_types, hidden=hidden)
        type_head.load_state_dict(state)
        type_head.to(device).eval()

    return model, type_head, cfg, meta


# ------------------------------------------------------------------ inference
@torch.no_grad()
def predict_slice(model, cfg, cache_dir, subject_id: str, plane: str,
                  slice_index: int, device: str = "cpu", type_head=None) -> dict:
    """Segment one cached slice. Returns real arrays, unpadded to native size.

    ``probs`` is the softmax over classes -- ``confidence`` below is the mean
    probability the model assigned to the class it actually predicted, i.e. a
    measured quantity, not a hand-picked number.
    """
    image, label, pad, orig_shape = _load_slice(cache_dir, cfg, subject_id, plane, slice_index)

    x = torch.from_numpy(image[None]).to(device)
    seg_logits, features = model(x)
    probs = torch.softmax(seg_logits, dim=1)[0]
    pred = probs.argmax(dim=0).cpu().numpy().astype(np.uint8)
    probs_np = probs.cpu().numpy()

    pred = unpad(pred, pad)
    truth = unpad(label[0], pad)
    image_native = np.stack([unpad(c, pad) for c in image])
    # Per-class probabilities, unpadded like everything else -- ROC/AUC needs
    # the continuous scores, not just the argmax.
    probs_native = np.stack([unpad(c, pad) for c in probs_np])
    confidence = float(probs_native.max(axis=0).mean())

    out = {
        "subject_id": subject_id,
        "plane": plane,
        "slice_index": slice_index,
        "image": image_native,       # (4, H, W) the four modalities
        "prediction": pred,          # (H, W) labels 0-4
        "ground_truth": truth,       # (H, W) labels 0-4
        "probabilities": probs_native,  # (5, H, W) softmax over classes
        "confidence": confidence,
        "features": features[0].cpu().numpy(),
    }

    if type_head is not None:
        type_logits = type_head(features)
        type_probs = torch.softmax(type_logits, dim=1)[0].cpu().numpy()
        idx = int(type_probs.argmax())
        out["tumor_type"] = {
            "label": TYPE_LABELS[idx],
            "confidence": float(type_probs[idx]),
            "probabilities": {name: float(p) for name, p in zip(TYPE_LABELS, type_probs)},
        }
    return out


def region_breakdown(mask: np.ndarray) -> list[dict]:
    """Per-label share of the segmented tumour, measured from a real mask."""
    total = int((mask > 0).sum())
    rows = []
    for label_id, (name, colour) in LABEL_NAMES.items():
        voxels = int((mask == label_id).sum())
        rows.append({
            "label": label_id,
            "name": name,
            "colour": colour,
            "voxels": voxels,
            "percent": (100.0 * voxels / total) if total else 0.0,
        })
    return rows


def dice_per_region(pred: np.ndarray, truth: np.ndarray) -> dict:
    """Dice of prediction vs ground truth for the three scored regions.

    Regions follow the BraTS convention used by section 03's evaluator:
    ET = label 1, TC (tumour core) = labels 1+3, WT (whole tumour) = any label.
    A region absent from *both* masks is reported as None, not 1.0 -- scoring a
    region nobody has as perfect would inflate the mean.
    """
    regions = {
        "ET": ({1}, {1}),
        "TC": ({1, 3}, {1, 3}),
        "WT": ({1, 2, 3, 4}, {1, 2, 3, 4}),
    }
    out = {}
    for name, (p_labels, t_labels) in regions.items():
        p = np.isin(pred, list(p_labels))
        t = np.isin(truth, list(t_labels))
        denom = p.sum() + t.sum()
        out[name] = (2.0 * (p & t).sum() / denom) if denom else None
    scored = [v for v in out.values() if v is not None]
    out["mean"] = float(np.mean(scored)) if scored else None
    return out


# ------------------------------------------------------------------ discovery
def available_subjects(cache_dir) -> list[str]:
    """Subject ids actually present in the slice cache."""
    return list_subjects(cache_dir)


def available_slices(cache_dir, subject_id: str, plane: str = "axial") -> list[int]:
    """Slice indices actually present for one subject/plane."""
    return list_slice_indices(cache_dir, subject_id, plane)


def find_tumor_slices(cache_dir, subject_id: str, plane: str = "axial",
                      limit: int | None = None) -> list[int]:
    """Slice indices whose ground-truth mask contains any tumour.

    Reads the cache's `index.json` when it exists (written by `run.py index`),
    which is far faster than opening every mask over a network filesystem.
    """
    import json

    index_path = Path(cache_dir) / "index.json"
    if index_path.exists():
        with open(index_path, "r", encoding="utf-8") as fh:
            index = json.load(fh)
        entry = index.get("subjects", {}).get(subject_id, {}).get(plane)
        if entry and "tumor" in entry:
            found = [int(i) for i in entry["tumor"]]
            return found[:limit] if limit else found

    from src.slices import load_slice
    found = []
    for i in list_slice_indices(cache_dir, subject_id, plane):
        if load_slice(cache_dir, subject_id, plane, i)["mask"].any():
            found.append(i)
            if limit and len(found) >= limit:
                break
    return found
