"""CORAL domain alignment utilities.

The physical image batch remains one 96^3 subject.  A detached feature queue
supplies previous subjects for a stable covariance estimate while the newest
embedding in each domain keeps its gradient.
"""
from __future__ import annotations

from collections import deque
import torch
from torch.utils.data import DataLoader

from .config import TrainConfig
from .data import build_dataset
from .model import FederatedUNet3D


def covariance(features: torch.Tensor, eps: float = 0.0) -> torch.Tensor:
    """Return the unbiased feature covariance for a (subjects, features) tensor."""
    if features.ndim != 2:
        raise ValueError(f"features must have shape (N, D), got {tuple(features.shape)}")
    if features.shape[0] < 2:
        raise ValueError("CORAL covariance requires at least two subjects")
    centered = features - features.mean(dim=0, keepdim=True)
    cov = centered.T @ centered / (features.shape[0] - 1)
    if eps:
        cov = cov + torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype) * eps
    return cov


def coral_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Deep CORAL loss: squared Frobenius distance between domain covariances."""
    if source.ndim != 2 or target.ndim != 2:
        raise ValueError("source and target must both have shape (N, D)")
    if source.shape[1] != target.shape[1]:
        raise ValueError("source and target feature dimensions must match")
    d = source.shape[1]
    # Covariance is numerically sensitive; keep it FP32 even when the 3D model
    # forward pass uses mixed precision.
    source_cov = covariance(source.float())
    target_cov = covariance(target.float())
    return (source_cov - target_cov).square().sum() / (4.0 * d * d)


class FeatureQueue:
    """Fixed-size detached history with a gradient-carrying current embedding."""

    def __init__(self, max_size: int) -> None:
        if max_size < 2:
            raise ValueError("CORAL feature queue must hold at least two subjects")
        self._items: deque[torch.Tensor] = deque(maxlen=max_size - 1)

    def with_current(self, current: torch.Tensor) -> torch.Tensor | None:
        if current.ndim != 2 or current.shape[0] != 1:
            raise ValueError("queue expects one embedding with shape (1, D)")
        if not self._items:
            return None
        return torch.cat([*self._items, current], dim=0)

    def append(self, current: torch.Tensor) -> None:
        self._items.append(current.detach())


def train_coral_alignment(
    model: FederatedUNet3D,
    config: TrainConfig,
    source_manifest_path: str,
    target_manifest_path: str,
) -> float:
    """Run one memory-safe Hospital A/B alignment phase on the global model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    use_amp = device.type == "cuda"
    source = DataLoader(
        build_dataset(source_manifest_path, config.data_mode, config.cache_path, config.seed),
        batch_size=1,
        shuffle=True,
    )
    target = DataLoader(
        build_dataset(target_manifest_path, config.data_mode, config.cache_path, config.seed + 1),
        batch_size=1,
        shuffle=True,
    )
    steps = config.coral_steps_per_round or max(len(source), len(target))
    source_iter, target_iter = iter(source), iter(target)
    source_queue = FeatureQueue(config.coral_queue_size)
    target_queue = FeatureQueue(config.coral_queue_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    losses: list[float] = []
    model.train()

    for _ in range(steps):
        try:
            source_x, _ = next(source_iter)
        except StopIteration:
            source_iter = iter(source)
            source_x, _ = next(source_iter)
        try:
            target_x, _ = next(target_iter)
        except StopIteration:
            target_iter = iter(target)
            target_x, _ = next(target_iter)
        source_x, target_x = source_x.to(device), target_x.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=use_amp):
            _, source_current = model(source_x)
            _, target_current = model(target_x)
            source_features = source_queue.with_current(source_current)
            target_features = target_queue.with_current(target_current)
            loss = (
                None
                if source_features is None
                else config.coral_weight * coral_loss(source_features, target_features)
            )

        if loss is not None:
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            losses.append(float(loss.detach()))
        source_queue.append(source_current)
        target_queue.append(target_current)

    return sum(losses) / max(len(losses), 1)
