"""Bridge between the Streamlit demo and the real dataset + real model.

Before this module the demo ran entirely on `np.random` — synthetic noise with a
drawn circle standing in for a tumor. This connects it to the actual pipeline:

* real 96-cube volumes and ground-truth masks from the shared cache
* the real `FederatedUNet3D` from `01_model_federated` (MONAI 3D U-Net)
* the real ET / NC / WT region Dice from `03_augmentation_eval`

    !! READ THIS BEFORE DEMOING !!
    No trained checkpoint exists yet. `01_model_federated` has the architecture
    and the training loop, but nothing on disk has been trained. So unless a
    checkpoint is supplied, `run_inference` runs RANDOMLY INITIALIZED weights and
    the segmentation it returns is meaningless noise.

    Every consumer must surface `model_status()` so nobody mistakes untrained
    output for a result. Ground-truth overlays are real and safe to show; model
    predictions are not, until a checkpoint lands.

Point `NEUROFED_CACHE` at the cache directory to override the default path.
"""

from __future__ import annotations

import importlib.util
import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

# --------------------------------------------------------------------- paths
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent

DEFAULT_CACHE = Path("D:/Medical AI Workshop/cache_96cube")
CACHE_DIR = Path(os.environ.get("NEUROFED_CACHE", DEFAULT_CACHE))
MANIFEST_DIR = _REPO_ROOT / "00_shared" / "manifests"
CHECKPOINT_ENV = "NEUROFED_CHECKPOINT"

MODALITIES = ("t1c", "t1n", "t2f", "t2w")
MODALITY_LABELS = {
    "t1c": "T1c (post-contrast)",
    "t1n": "T1 (native)",
    "t2f": "T2-FLAIR",
    "t2w": "T2-weighted",
}

# Label encoding, per 00_shared/CONTRACTS.md (4 subregions, not 3).
CLASS_NAMES = {
    1: "Enhancing Tumor (ET)",
    2: "Non-Enhancing Tumor (NET)",
    3: "Cystic Component (CC)",
    4: "Peritumoral Edema (ED)",
}
CLASS_COLORS = {1: "#EF4444", 2: "#10B981", 3: "#3B82F6", 4: "#EAB308"}

# The cache is 96^3 resampled from the BraTS 240x240x155 @ 1mm grid, so a cached
# voxel spans roughly 2.5 x 2.5 x 1.61 mm. Volumes in mL are therefore an
# ESTIMATE that assumes no cropping happened before resampling. Voxel counts and
# percentages below are exact regardless.
VOXEL_MM3 = (240 / 96) * (240 / 96) * (155 / 96)


def _load_module(name: str, path: Path):
    """Import a module by file path.

    Needed because the section directories start with digits
    (`01_model_federated`), which are not importable package names.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------ manifests
@lru_cache(maxsize=1)
def list_subjects() -> dict[str, list[str]]:
    """`{"Hospital A": [...], "Hospital B": [...], "Held-out": [...]}`."""
    files = {
        "Hospital A": "hospitalA.json",
        "Hospital B": "hospitalB.json",
        "Held-out": "heldout.json",
    }
    out: dict[str, list[str]] = {}
    for label, fname in files.items():
        path = MANIFEST_DIR / fname
        if not path.exists():
            raise FileNotFoundError(f"manifest not found: {path}")
        with open(path, "r", encoding="utf-8") as fh:
            out[label] = sorted(json.load(fh))
    return out


def cache_available() -> bool:
    return CACHE_DIR.is_dir()


def subject_cohort(subject_id: str) -> str:
    for cohort, ids in list_subjects().items():
        if subject_id in ids:
            return cohort
    return "unknown"


# ------------------------------------------------------------------- loading
def _znorm(channel: np.ndarray) -> np.ndarray:
    """Z-score over brain (non-zero) voxels only; background stays 0."""
    out = channel.astype(np.float32, copy=True)
    mask = out != 0
    if not mask.any():
        return out
    vals = out[mask]
    std = float(vals.std())
    if std == 0.0:
        return out
    out[mask] = (vals - float(vals.mean())) / std
    return out


@dataclass
class Volume:
    """One subject's real volumes plus its ground-truth mask."""

    subject_id: str
    cohort: str
    images: dict[str, np.ndarray]  # modality -> (96,96,96) normalized float32
    raw: dict[str, np.ndarray]     # modality -> (96,96,96) raw float32, for display
    seg: np.ndarray                # (96,96,96) uint8, labels 0-4

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.seg.shape

    def stack(self) -> np.ndarray:
        """`(4,96,96,96)` normalized, in the channel order the model expects."""
        return np.stack([self.images[m] for m in MODALITIES], axis=0)


def load_volume(subject_id: str) -> Volume:
    """Load one real subject from the shared cache."""
    path = CACHE_DIR / f"{subject_id}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"subject {subject_id!r} not found at {path}. "
            f"Set {'NEUROFED_CACHE'} to the shared cache directory."
        )
    raw: dict[str, np.ndarray] = {}
    with np.load(path) as npz:
        available = set(npz.files)
        for m in MODALITIES:
            if m not in available:
                raise KeyError(f"subject {subject_id!r} is missing modality {m!r}")
            raw[m] = np.asarray(npz[m], dtype=np.float32)
        if "seg" not in available:
            raise KeyError(f"subject {subject_id!r} is missing key 'seg'")
        seg = np.asarray(npz["seg"]).astype(np.uint8)

    images = {m: _znorm(arr) for m, arr in raw.items()}
    return Volume(subject_id, subject_cohort(subject_id), images, raw, seg)


# ----------------------------------------------------------------- statistics
def region_masks(seg: np.ndarray) -> dict[str, np.ndarray]:
    """Official BraTS-PEDs evaluation regions."""
    return {
        "ET": np.isin(seg, (1,)),
        "NC": np.isin(seg, (1, 2, 3)),
        "WT": np.isin(seg, (1, 2, 3, 4)),
    }


def subregion_stats(seg: np.ndarray) -> list[dict]:
    """Per-class voxel counts, share of tumor, and estimated volume."""
    tumor = int((seg > 0).sum())
    rows = []
    for label, name in CLASS_NAMES.items():
        count = int((seg == label).sum())
        rows.append(
            {
                "label": label,
                "name": name,
                "color": CLASS_COLORS[label],
                "voxels": count,
                "pct_of_tumor": (100.0 * count / tumor) if tumor else 0.0,
                "volume_ml": count * VOXEL_MM3 / 1000.0,
            }
        )
    return rows


def region_stats(seg: np.ndarray) -> list[dict]:
    """ET / NC / WT voxel counts and estimated volumes."""
    return [
        {
            "region": name,
            "voxels": int(mask.sum()),
            "volume_ml": int(mask.sum()) * VOXEL_MM3 / 1000.0,
            "present": bool(mask.any()),
        }
        for name, mask in region_masks(seg).items()
    ]


def dice(pred_bool: np.ndarray, true_bool: np.ndarray) -> float:
    """Dice with the documented empty-region conventions."""
    p, t = int(pred_bool.sum()), int(true_bool.sum())
    if p == 0 and t == 0:
        return 1.0
    if p == 0 or t == 0:
        return 0.0
    return 2.0 * int(np.logical_and(pred_bool, true_bool).sum()) / (p + t)


def region_dice(pred_seg: np.ndarray, true_seg: np.ndarray) -> dict[str, float]:
    pm, tm = region_masks(pred_seg), region_masks(true_seg)
    return {r: dice(pm[r], tm[r]) for r in ("ET", "NC", "WT")}


# ---------------------------------------------------------------------- model
def checkpoint_path() -> Path | None:
    """Explicit checkpoint via env var, else the newest `epoch_*.pt` under checkpoints/."""
    env = os.environ.get(CHECKPOINT_ENV)
    if env:
        p = Path(env)
        return p if p.exists() else None

    roots = [_REPO_ROOT / "checkpoints", Path("D:/Medical AI Workshop/checkpoints")]
    found: list[Path] = []
    for root in roots:
        if root.is_dir():
            found.extend(root.rglob("*.pt"))
    if not found:
        return None
    return max(found, key=lambda p: p.stat().st_mtime)


@dataclass
class ModelStatus:
    trained: bool
    checkpoint: Path | None
    detail: str


def model_status() -> ModelStatus:
    """Whether the loaded model has trained weights. Surface this in the UI."""
    ckpt = checkpoint_path()
    if ckpt is None:
        return ModelStatus(
            trained=False,
            checkpoint=None,
            detail=(
                "No trained checkpoint found. The real 3D U-Net architecture is loaded "
                "but its weights are randomly initialized, so predicted masks are "
                "meaningless. Ground-truth overlays are real."
            ),
        )
    return ModelStatus(
        trained=True,
        checkpoint=ckpt,
        detail=f"Loaded trained weights from {ckpt.name}.",
    )


@lru_cache(maxsize=1)
def load_model():
    """Build the real `FederatedUNet3D`, loading a checkpoint when one exists."""
    model_py = _REPO_ROOT / "01_model_federated" / "src" / "model.py"
    if not model_py.exists():
        raise FileNotFoundError(f"model definition not found at {model_py}")
    mod = _load_module("federated_model", model_py)
    model = mod.build_model()

    ckpt = checkpoint_path()
    if ckpt is not None:
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
        state = blob.get("model_state", blob.get("model_state_dict", blob))
        model.load_state_dict(state, strict=True)

    return model.eval()


@torch.no_grad()
def run_inference(volume: Volume) -> dict:
    """Real forward pass over a real volume.

    Returns the predicted class mask, the bottleneck feature vector, and the
    per-region Dice against ground truth. Read `model_status()` before showing
    any of it as a result — with no checkpoint, the mask is noise.
    """
    model = load_model()
    x = torch.from_numpy(volume.stack()).unsqueeze(0)  # (1,4,96,96,96)
    seg_logits, features = model(x)
    pred = torch.argmax(seg_logits, dim=1)[0].numpy().astype(np.uint8)
    return {
        "pred_seg": pred,
        "features": features[0].numpy(),
        "dice": region_dice(pred, volume.seg),
        "status": model_status(),
    }


# --------------------------------------------------------------- ROC / AUC
#: probability composition per evaluation region — the softmax classes that make
#: up each BraTS region, so the "score" for that region is their summed probability
REGION_CLASSES = {"ET": (1,), "NC": (1, 2, 3), "WT": (1, 2, 3, 4)}


@torch.no_grad()
def roc_data(volume: Volume, max_points: int = 400) -> dict:
    """Real voxel-level ROC + AUC per region for one subject.

    The score for a region is the summed softmax probability of its constituent
    classes; the target is that region's ground-truth mask. Both are real — this
    is a genuine ROC, not an illustration.

    With an untrained checkpoint the curve sits on the diagonal and AUC lands near
    0.5. That is the correct result for random weights, not a bug.

    A region absent from the ground truth has no positive class, so ROC/AUC are
    undefined for it; those regions come back with ``auc=None``.
    """
    from sklearn.metrics import auc as _auc
    from sklearn.metrics import roc_curve

    model = load_model()
    x = torch.from_numpy(volume.stack()).unsqueeze(0)
    seg_logits, _ = model(x)
    probs = torch.softmax(seg_logits, dim=1)[0].numpy()  # (5, D, H, W)

    truth = volume.seg.reshape(-1)
    out: dict[str, dict] = {}

    for region, classes in REGION_CLASSES.items():
        score = probs[list(classes)].sum(axis=0).reshape(-1).astype(np.float64)
        target = np.isin(truth, classes).astype(np.uint8)

        n_pos = int(target.sum())
        if n_pos == 0 or n_pos == target.size:
            out[region] = {"fpr": None, "tpr": None, "auc": None,
                           "n_pos": n_pos, "n_neg": int(target.size - n_pos)}
            continue

        fpr, tpr, _ = roc_curve(target, score)
        region_auc = float(_auc(fpr, tpr))

        # Thin the curve for plotting; ROC from ~885k voxels has far more points
        # than a chart needs. Endpoints are always kept so the curve spans 0..1.
        if len(fpr) > max_points:
            idx = np.unique(
                np.concatenate([
                    np.linspace(0, len(fpr) - 1, max_points).astype(int),
                    [0, len(fpr) - 1],
                ])
            )
            fpr, tpr = fpr[idx], tpr[idx]

        out[region] = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "auc": region_auc,
            "n_pos": n_pos,
            "n_neg": int(target.size - n_pos),
        }

    out["_status"] = model_status()
    return out


# --------------------------------------------------------- ablation results
ABLATION_CSV = _REPO_ROOT / "03_augmentation_eval" / "results" / "ablation_results.csv"


def ablation_results() -> list[dict]:
    """Rows from `03_augmentation_eval`'s results CSV, or `[]` if it doesn't exist.

    This is the only legitimate source of held-out performance numbers: they come
    from the ablation runner scoring a real checkpoint. Nothing here invents a
    figure — an empty list means no evaluated run exists yet.
    """
    if not ABLATION_CSV.exists():
        return []
    import csv

    with open(ABLATION_CSV, "r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def heldout_dice() -> dict[str, float] | None:
    """Per-region Dice from the most recent evaluated run, or None if there is none."""
    rows = ablation_results()
    if not rows:
        return None
    last = rows[-1]
    try:
        return {r: float(last[f"dice_{r}"]) for r in ("ET", "NC", "WT")}
    except (KeyError, ValueError):
        return None


# ------------------------------------------------------------------ display
def slice_of(arr: np.ndarray, index: int, plane: str = "Axial") -> np.ndarray:
    """Extract a 2D slice. `plane` is Axial, Coronal or Sagittal."""
    plane = plane.lower()
    if plane.startswith("sag"):
        return arr[index, :, :]
    if plane.startswith("cor"):
        return arr[:, index, :]
    return arr[:, :, index]


def plane_extent(shape: tuple[int, int, int], plane: str) -> int:
    plane = plane.lower()
    if plane.startswith("sag"):
        return shape[0]
    if plane.startswith("cor"):
        return shape[1]
    return shape[2]


def busiest_slice(seg: np.ndarray, plane: str = "Axial") -> int:
    """Index of the slice holding the most tumor — a useful default view."""
    tumor = seg > 0
    if not tumor.any():
        return plane_extent(seg.shape, plane) // 2
    plane = plane.lower()
    if plane.startswith("sag"):
        counts = tumor.sum(axis=(1, 2))
    elif plane.startswith("cor"):
        counts = tumor.sum(axis=(0, 2))
    else:
        counts = tumor.sum(axis=(0, 1))
    return int(np.argmax(counts))
