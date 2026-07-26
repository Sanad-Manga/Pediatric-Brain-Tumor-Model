"""Per-epoch checkpoint save/load/resume. Colab disconnects — resume is not optional."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import torch

_CKPT_RE = re.compile(r"^epoch_(\d+)\.pt$")


def checkpoint_dir_for(base_dir: str, run_id: str) -> Path:
    d = Path(base_dir) / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_checkpoint(
    base_dir: str,
    run_id: str,
    epoch: int,
    model_state: dict[str, Any],
    optimizer_state: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> Path:
    d = checkpoint_dir_for(base_dir, run_id)
    path = d / f"epoch_{epoch}.pt"
    payload = {
        "epoch": epoch,
        "model_state": model_state,
        "optimizer_state": optimizer_state,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    return path


def latest_checkpoint(base_dir: str, run_id: str) -> Path | None:
    d = Path(base_dir) / run_id
    if not d.is_dir():
        return None
    best_epoch = -1
    best_path = None
    for entry in d.iterdir():
        match = _CKPT_RE.match(entry.name)
        if not match:
            continue
        epoch = int(match.group(1))
        if epoch > best_epoch:
            best_epoch = epoch
            best_path = entry
    return best_path


def load_checkpoint(base_dir: str, run_id: str) -> dict[str, Any] | None:
    path = latest_checkpoint(base_dir, run_id)
    if path is None:
        return None
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise RuntimeError(f"Corrupted or unreadable checkpoint: {path}") from exc
