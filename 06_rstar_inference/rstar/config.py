"""Configuration for the R* inference module and the pinned-checkpoint manifest."""
from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath

PACKAGE_ROOT = Path(__file__).resolve().parents[1]          # .../06_rstar_inference
REPO_ROOT = PACKAGE_ROOT.parent                             # repository root (holds sections 01 and 03)
DEFAULT_MANIFEST = PACKAGE_ROOT / "config" / "models.default.json"
SEQUENCES = ("t1c", "t1n", "t2f", "t2w")                     # channel order of every array this module accepts
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


@dataclass
class RStarConfig:
    """Every number in the recipe, with the values measured on 2026-09-26 as defaults."""

    w3d: float = 0.5                       # weight of the 3D family in the probability average
    background_scale: float = 0.5          # multiplies the background probability before the argmax
    et_min_mm3: float = 500.0              # relabel enhancing tumour as non-enhancing below this total volume
    voxel_mm3: float = 1.0                 # volume of one input voxel (BraTS grid: 1 mm isotropic)
    agreement_review: float = 0.70         # 2D-vs-3D whole-tumour agreement below this => status 'review'
    agreement_strong: float = 0.10         # below this => 'review' with the stronger warning
    expected_shape: tuple = (240, 240, 155)
    verify_hashes: bool = True
    device: str = "auto"
    review_flags: bool = False             # also return enhancing-tumour spots to review (SPEC Addendum 2); labels unchanged
    review_prob_cut: float = 0.7           # flag a spot whose mean ET probability is below this (pre-registered study, 2026-10-02)
    review_min_voxels: int = 50            # spots smaller than this are neither kept nor flagged
    models_root: Path | None = None        # where the (untracked) checkpoints live; default: env RSTAR_MODELS_ROOT, else the repo root
    manifest_path: Path | None = None

    def __post_init__(self) -> None:
        def finite(name: str, value: float) -> float:
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number, got {value!r}")
            return value

        if not 0.0 <= finite("w3d", self.w3d) <= 1.0:
            raise ValueError(f"w3d must be in [0, 1], got {self.w3d!r}")
        if not finite("background_scale", self.background_scale) > 0.0:
            raise ValueError(f"background_scale must be positive, got {self.background_scale!r}")
        if finite("et_min_mm3", self.et_min_mm3) < 0.0:
            raise ValueError(f"et_min_mm3 must be >= 0, got {self.et_min_mm3!r}")
        if not finite("voxel_mm3", self.voxel_mm3) > 0.0:
            raise ValueError(f"voxel_mm3 must be positive, got {self.voxel_mm3!r}")
        for name in ("agreement_review", "agreement_strong"):
            if not 0.0 <= finite(name, getattr(self, name)) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {getattr(self, name)!r}")
        if self.agreement_strong > self.agreement_review:
            raise ValueError(f"agreement_strong ({self.agreement_strong}) must not exceed agreement_review ({self.agreement_review})")
        if not isinstance(self.review_flags, bool):
            raise ValueError(f"review_flags must be True or False, got {self.review_flags!r}")
        if not 0.0 <= finite("review_prob_cut", self.review_prob_cut) <= 1.0:
            raise ValueError(f"review_prob_cut must be in [0, 1], got {self.review_prob_cut!r}")
        if isinstance(self.review_min_voxels, bool) or not isinstance(self.review_min_voxels, int) or self.review_min_voxels < 0:
            raise ValueError(f"review_min_voxels must be a non-negative integer, got {self.review_min_voxels!r}")
        shape = tuple(self.expected_shape)
        if len(shape) != 3 or any(int(s) != s or s < 1 for s in shape):
            raise ValueError(f"expected_shape must be three positive integers, got {self.expected_shape!r}")
        self.expected_shape = tuple(int(s) for s in shape)

    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        import torch  # local import: keeps `import rstar` light

        return "cuda" if torch.cuda.is_available() else "cpu"

    def resolved_models_root(self) -> Path:
        if self.models_root is not None:
            return Path(self.models_root)
        env = os.environ.get("RSTAR_MODELS_ROOT")
        return Path(env) if env else REPO_ROOT


def _check_entry(entry: dict, where: str) -> None:
    for key in ("name", "path", "sha256"):
        if key not in entry:
            raise ValueError(f"manifest entry {where} is missing '{key}'")
    path = str(entry["path"])
    if Path(path).is_absolute() or PureWindowsPath(path).drive or path.startswith(("/", "\\")):
        raise ValueError(f"manifest entry {where} must use a path relative to the models root, got {path!r}")
    if not _HEX64.match(str(entry["sha256"])):
        raise ValueError(f"manifest entry {where} needs a 64-hex-digit lowercase sha256, got {entry['sha256']!r}")


def validate_manifest(manifest: dict) -> dict:
    """The manifest lists exactly one 2D ensemble file and four 3D files, each with a relative path and a SHA-256."""
    if "ensemble_2d" not in manifest or "family_3d" not in manifest:
        raise ValueError("manifest needs 'ensemble_2d' and 'family_3d'")
    _check_entry(manifest["ensemble_2d"], "ensemble_2d")
    if len(manifest["family_3d"]) != 4:
        raise ValueError(f"manifest must list exactly 4 3D members, found {len(manifest['family_3d'])}")
    for i, entry in enumerate(manifest["family_3d"]):
        _check_entry(entry, f"family_3d[{i}]")
    return manifest


def load_manifest(path: str | Path | None = None) -> dict:
    path = Path(path) if path is not None else DEFAULT_MANIFEST
    if not path.exists():
        raise FileNotFoundError(f"models manifest not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        return validate_manifest(json.load(fh))
