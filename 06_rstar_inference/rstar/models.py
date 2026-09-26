"""Hash-verified loading of the 2D ensemble and the 3D family.

The checkpoints are pickled dicts (`torch.load(..., weights_only=False)`), so loading an unverified file would execute
arbitrary code. Every file is therefore checked against the SHA-256 pinned in the manifest BEFORE it is opened.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from .config import REPO_ROOT, RStarConfig, load_manifest
from .sections import import_submodule


class ModelIntegrityError(RuntimeError):
    """A checkpoint's SHA-256 differs from the pinned value."""


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def verify_sha256(path: str | Path, expected: str) -> str:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ModelIntegrityError(
            f"checkpoint {path} does not match the pinned hash: expected {expected[:12]}..., found {actual[:12]}... "
            "(pass verify_hashes=False only if you know why the file changed)")
    return actual


def _resolve(entry: dict, cfg: RStarConfig) -> Path:
    path = cfg.resolved_models_root() / entry["path"]
    if cfg.verify_hashes:
        verify_sha256(path, entry["sha256"])
    elif not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    return path


def load_2d_ensemble(cfg: RStarConfig):
    """The shipped 2D ensemble: an 'ensemble' payload whose members are averaged in probability space.
    Returns (models, info) with info = [{name, path, sha256}]."""
    manifest = load_manifest(cfg.manifest_path)
    entry = manifest["ensemble_2d"]
    path = _resolve(entry, cfg)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not (isinstance(payload, dict) and payload.get("ensemble") and payload.get("members")):
        raise ModelIntegrityError(f"{path} is not an ensemble checkpoint (no 'ensemble'/'members' entries)")
    config_mod = import_submodule("03", "config")
    model_mod = import_submodule("03", "model")
    cfg2d = config_mod.load_config(REPO_ROOT / "03_augmentation_eval" / "config.yaml")
    device = cfg.resolved_device()
    models = []
    for member in payload["members"]:
        geom = {k: member[k] for k in ("width", "depth") if k in member} or model_mod.infer_geometry(member["model_state_dict"])
        net = model_mod.build_model(cfg2d, spatial_dims=int(payload.get("spatial_dims", 2)), **geom)
        net.load_state_dict(member["model_state_dict"])
        models.append(net.to(device).eval())
    return models, [dict(entry, resolved_path=str(path))]


def load_3d_family(cfg: RStarConfig):
    """The four 3D dropout members. Returns (models, info)."""
    manifest = load_manifest(cfg.manifest_path)
    model_mod = import_submodule("01", "model")
    device = cfg.resolved_device()
    models, info = [], []
    for entry in manifest["family_3d"]:
        path = _resolve(entry, cfg)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        net = model_mod.FederatedUNet3D()
        net.load_state_dict(payload["model_state"])
        models.append(net.to(device).eval())
        info.append(dict(entry, resolved_path=str(path)))
    return models, info
