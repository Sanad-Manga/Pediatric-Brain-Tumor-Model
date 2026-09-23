"""Single-client training loop: sanity check on one manifest before FedAvg."""
from __future__ import annotations

import time
import warnings
from typing import Callable

import torch
from torch.utils.data import DataLoader
from monai.losses import DiceCELoss, DiceFocalLoss

from .checkpoint import load_checkpoint, prune_old_checkpoints, save_checkpoint
from .config import TrainConfig
from .data import build_dataset
from .model import FederatedUNet3D, build_model


def _step_scheduler_clamped(scheduler, lr_horizon: int) -> None:
    """CosineAnnealingLR is PERIODIC past T_max, not a floor -- calling
    .step() past it makes lr climb back toward the original (already proven
    unstable at 1e-3 / risky even at 3e-4) rather than staying at eta_min.
    Measured live: at epoch 2*T_max it's back to the exact starting lr. A real
    overnight run crossed this boundary before it was caught. Freeze instead:
    once scheduler.last_epoch has reached the horizon, stop stepping --
    last_epoch stays at lr_horizon forever and lr stays at eta_min forever.
    """
    if scheduler.last_epoch < lr_horizon:
        scheduler.step()


def _build_loss(loss_kind: str):
    """dice_ce (original): plain CrossEntropy has no notion of a rare class --
    ET is the smallest region by voxel count of the 4, and a real overnight run
    on real data (2026-09-22) showed ET-specific held-out Dice declining while
    NC/WT kept improving, alongside training loss plateauing -- the same
    under-segmentation this project's 2D pipeline already has (precision >>
    sensitivity, see HANDOFF.md). dice_focal down-weights voxels the model
    already classifies confidently (gamma=2.0, MONAI's default) so the loss
    stops being dominated by the easy majority-class voxels ET is drowned out
    by; a standard fix for exactly this class-imbalance symptom.
    """
    if loss_kind == "dice_ce":
        return DiceCELoss(to_onehot_y=True, softmax=True, include_background=True)
    if loss_kind == "dice_focal":
        return DiceFocalLoss(to_onehot_y=True, softmax=True, include_background=True, gamma=2.0)
    raise ValueError(f"loss_kind must be 'dice_ce' or 'dice_focal', got {loss_kind!r}")


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
    deadline_unix: float | None = None,
    lr_horizon: int | None = None,
    loss_kind: str = "dice_ce",
) -> tuple[FederatedUNet3D, list[float]]:
    """Trains `model` (or a fresh one) on the given manifest for num_epochs.

    Returns (model, per-epoch loss list). Checkpoints after every epoch to
    <config.checkpoint_dir>/<config.run_id>/epoch_<N>.pt. If resume=True and a
    checkpoint exists, continues from the next epoch instead of epoch 0.

    deadline_unix: if given, the epoch loop stops (checkpoint already saved
    for every completed epoch) once time.time() passes it, checked between
    epochs -- same reason 03_augmentation_eval/src/train.py has one: a fixed
    epoch count either wastes budget finishing early or overruns it, and this
    is untested on real data for the first time tonight.

    lr_horizon: cosine-anneals lr -> 1e-5 over this many epochs (built once,
    fixed regardless of num_epochs, replayed correctly on resume). Set this
    to a realistic estimate of how many epochs will actually complete before
    deadline_unix, NOT to num_epochs itself -- num_epochs here is meant as a
    generous safety cap, not a real target, and a cosine tied to a cap far
    higher than what actually runs barely decays before the deadline arrives.
    This is the exact bug 03_augmentation_eval/src/train.py's --lr-horizon
    fixes; same cause, same fix, reapplied here since the failure mode
    doesn't care which section's training loop it's in. None = no schedule
    (flat lr the whole run, the original behaviour).

    Non-finite batches (loss is nan/inf) are skipped rather than applied --
    the 2D pipeline hit exactly this on real data early on and losing the
    whole run to one bad batch's gradient is a worse outcome than one skipped
    step. Gradients are clipped for the same reason: a from-scratch 3D UNet
    with no batch norm and real, unbounded-range MRI intensities is a
    plausible source of an early large gradient.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    model = (model or build_model()).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    loss_fn = _build_loss(loss_kind)

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
            warnings.warn(
                f"No checkpoint found for run_id={config.run_id!r}; starting from epoch 0."
            )

    scheduler = None
    if lr_horizon is not None:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(lr_horizon, 1), eta_min=1e-5)
        # optimizer.load_state_dict() above (if resumed) restored the ALREADY-
        # decayed lr; stepping the scheduler from there would decay it a
        # second time. Restart the replay from base_lr, same fix as
        # 03_augmentation_eval/src/train.py.
        for group, base_lr in zip(optimizer.param_groups, scheduler.base_lrs):
            group["lr"] = base_lr
        for _ in range(start_epoch):
            _step_scheduler_clamped(scheduler, lr_horizon)

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    for epoch in range(start_epoch, start_epoch + num_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        n_skipped = 0
        for x, y in loader:
            x, y = _apply_augmentation(x, y, config.use_augmentation, augmentation_transform)
            x = x.to(device)
            y = y.to(device).unsqueeze(1)  # (B, 1, D, H, W) for DiceCELoss one-hot target

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                seg_logits, _features = model(x)
                loss = loss_fn(seg_logits, y)

            if not torch.isfinite(loss):
                n_skipped += 1
                continue

            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        losses.append(avg_loss)
        if n_skipped:
            warnings.warn(f"epoch {epoch}: skipped {n_skipped} non-finite batch(es) "
                          f"out of {n_batches + n_skipped}")

        current_lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            _step_scheduler_clamped(scheduler, lr_horizon)

        save_checkpoint(
            config.checkpoint_dir,
            config.run_id,
            epoch,
            model.state_dict(),
            optimizer.state_dict(),
            extra={"avg_loss": avg_loss, "n_skipped_nonfinite": n_skipped, "lr": current_lr,
                  "loss_kind": loss_kind},
        )
        prune_old_checkpoints(config.checkpoint_dir, config.run_id)

        if deadline_unix is not None and time.time() >= deadline_unix:
            warnings.warn(f"epoch {epoch}: stopping, past deadline_unix={deadline_unix}")
            break

    return model, losses
