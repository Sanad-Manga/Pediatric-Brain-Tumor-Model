"""FedAvg orchestration: local training per client + subject-count-weighted aggregation."""
from __future__ import annotations

import copy
from typing import Callable

import torch

from .checkpoint import load_checkpoint, save_checkpoint
from .config import TrainConfig
from .data import load_manifest
from .model import FederatedUNet3D, build_model
from .train_single import train_single_client
from .domain_adaptation import train_coral_alignment


def weighted_average_state_dicts(
    state_dicts: list[dict[str, torch.Tensor]], weights: list[float]
) -> dict[str, torch.Tensor]:
    """Subject-count-weighted average of a list of model state dicts."""
    if len(state_dicts) != len(weights):
        raise ValueError("state_dicts and weights must have the same length")
    total = sum(weights)
    if total <= 0:
        raise ValueError("Sum of weights must be positive")
    norm_weights = [w / total for w in weights]

    avg_state: dict[str, torch.Tensor] = {}
    for key in state_dicts[0].keys():
        stacked = torch.stack(
            [sd[key].float() * w for sd, w in zip(state_dicts, norm_weights)], dim=0
        )
        avg_state[key] = stacked.sum(dim=0).to(state_dicts[0][key].dtype)
    return avg_state


def train_federated(
    config: TrainConfig,
    client_manifest_paths: list[str],
    num_rounds: int,
    local_epochs: int,
    augmentation_transform: Callable | None = None,
    resume: bool = False,
) -> tuple[FederatedUNet3D, list[list[float]]]:
    """Runs FedAvg across the given client manifests.

    Each round: every client trains a copy of the global model locally for
    `local_epochs` epochs, then client weights are averaged (weighted by each
    client's subject count) into a new global model. Checkpointed every round
    under <config.checkpoint_dir>/<config.run_id>/epoch_<round>.pt.
    """
    if len(client_manifest_paths) < 2:
        raise ValueError("Federated training requires at least 2 client manifests")

    client_subject_counts = [len(load_manifest(p)) for p in client_manifest_paths]

    global_model = build_model()
    start_round = 0
    round_losses: list[list[float]] = []

    if resume:
        ckpt = load_checkpoint(config.checkpoint_dir, config.run_id)
        if ckpt is not None:
            global_model.load_state_dict(ckpt["model_state"])
            start_round = ckpt["epoch"] + 1
        else:
            import warnings

            warnings.warn(
                f"No checkpoint found for run_id={config.run_id!r}; starting from round 0."
            )

    dummy_optimizer = torch.optim.Adam(global_model.parameters(), lr=config.lr)

    for round_idx in range(start_round, start_round + num_rounds):
        client_state_dicts = []
        this_round_losses = []

        for manifest_path in client_manifest_paths:
            local_model = copy.deepcopy(global_model)
            trained_model, losses = train_single_client(
                config=config,
                manifest_path=manifest_path,
                num_epochs=local_epochs,
                model=local_model,
                augmentation_transform=augmentation_transform,
                resume=False,
            )
            client_state_dicts.append(trained_model.state_dict())
            this_round_losses.append(losses)

        round_losses.append([l for client_losses in this_round_losses for l in client_losses])

        new_global_state = weighted_average_state_dicts(client_state_dicts, client_subject_counts)
        global_model.load_state_dict(new_global_state)

        coral_round_loss = None
        if config.use_domain_adaptation:
            if len(client_manifest_paths) != 2:
                raise ValueError(
                    "Domain adaptation currently supports exactly 2 client manifests (Hospital A and Hospital B)."
                )
            coral_round_loss = train_coral_alignment(
                global_model, config, client_manifest_paths[0], client_manifest_paths[1]
            )

        save_checkpoint(
            config.checkpoint_dir,
            config.run_id,
            round_idx,
            global_model.state_dict(),
            dummy_optimizer.state_dict(),
            extra={"round": round_idx, "coral_loss": coral_round_loss},
        )

    return global_model, round_losses
