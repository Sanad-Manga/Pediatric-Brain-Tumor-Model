"""Single-client training loop: sanity check on one manifest before FedAvg."""
from __future__ import annotations

from typing import Callable

import torch
from torch.utils.data import DataLoader
from monai.losses import DiceCELoss

from .checkpoint import load_checkpoint, save_checkpoint
from .config import TrainConfig
from .data import build_dataset
from .model import FederatedUNet3D, build_model


def _apply_augmentation(
    x: torch.Tensor,
    y: torch.Tensor,
    use_augmentation: bool,
    transform: Callable | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if use_augmentation and transform is not None:
        return transform(x, y)
    return x, y


def train_single_client(
    config: TrainConfig,
    manifest_path: str,
    num_epochs: int,
    model: FederatedUNet3D | None = None,
    augmentation_transform: Callable | None = None,
    resume: bool = False,
) -> tuple[FederatedUNet3D, list[float]]:
    """Trains `model` (or a fresh one) on the given manifest for num_epochs.

    Returns (model, per-epoch loss list). Checkpoints after every epoch to
    <config.checkpoint_dir>/<config.run_id>/epoch_<N>.pt. If resume=True and a
    checkpoint exists, continues from the next epoch instead of epoch 0.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    model = (model or build_model()).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True, include_background=True)

    dataset = build_dataset(manifest_path, data_mode=config.data_mode, cache_path=config.cache_path)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)

    start_epoch = 0
    losses: list[float] = []
    if resume:
        ckpt = load_checkpoint(config.checkpoint_dir, config.run_id)
        if ckpt is not None:
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            start_epoch = ckpt["epoch"] + 1
        else:
            import warnings

            warnings.warn(
                f"No checkpoint found for run_id={config.run_id!r}; starting from epoch 0."
            )

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    for epoch in range(start_epoch, start_epoch + num_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for x, y in loader:
            x, y = _apply_augmentation(x, y, config.use_augmentation, augmentation_transform)
            x = x.to(device)
            y = y.to(device).unsqueeze(1)  # (B, 1, D, H, W) for DiceCELoss one-hot target

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                seg_logits, _features = model(x)
                loss = loss_fn(seg_logits, y)

            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        losses.append(avg_loss)

        save_checkpoint(
            config.checkpoint_dir,
            config.run_id,
            epoch,
            model.state_dict(),
            optimizer.state_dict(),
        )

    return model, losses
