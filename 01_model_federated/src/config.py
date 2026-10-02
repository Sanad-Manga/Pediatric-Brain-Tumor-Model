"""Config dataclass shared by single-client and federated training loops."""
from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass, field

import yaml


DEFAULT_CONFIG = {
    "model": {"width": 16, "depth": 5},
    "loss": {"kind": "dice_ce", "class_weights": None},
    "schedule": {"kind": "none", "min_lr": 1.0e-5},
    "augmentation": {
        "flip_prob": 0.5,
        "rotate_prob": 0.3,
        "rotate_range_deg": 10.0,
        "zoom_prob": 0.3,
        "zoom_min": 0.9,
        "zoom_max": 1.1,
        "scale_intensity_factor": 0.1,
        "scale_intensity_prob": 0.3,
        "shift_intensity_offset": 0.1,
        "shift_intensity_prob": 0.3,
        "gaussian_noise_prob": 0.2,
        "gaussian_noise_std": 0.05,
        "modality_dropout_prob": 0.0,
        "sequence_shift_prob": 0.0,
        "sequence_shift_max_voxels": 1.2,
    },
}


def load_config(path: str | Path) -> dict:
    """Load and validate the model, loss, schedule, and augmentation YAML."""
    with Path(path).open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}

    config = {
        section: {**defaults, **(raw.get(section) or {})}
        for section, defaults in DEFAULT_CONFIG.items()
    }
    model = config["model"]
    if not isinstance(model["width"], int) or isinstance(model["width"], bool) or model["width"] < 1:
        raise ValueError("model.width must be an integer >= 1")
    if not isinstance(model["depth"], int) or isinstance(model["depth"], bool) or model["depth"] < 2:
        raise ValueError("model.depth must be an integer >= 2")

    loss = config["loss"]
    if loss["kind"] not in {"dice_ce", "dice_focal", "region_dice_bce"}:
        raise ValueError(f"loss.kind must be dice_ce, dice_focal, or region_dice_bce; got {loss['kind']!r}")
    weights = loss["class_weights"]
    if weights is not None and (
        not isinstance(weights, list)
        or len(weights) != 5
        or any(not isinstance(weight, (int, float)) or isinstance(weight, bool) for weight in weights)
    ):
        raise ValueError("loss.class_weights must be null or a list of exactly 5 numbers")

    if config["schedule"]["kind"] not in {"cosine", "none"}:
        raise ValueError(f"schedule.kind must be cosine or none; got {config['schedule']['kind']!r}")
    return config


@dataclass
class TrainConfig:
    # Contract flags (00_shared/CONTRACTS.md)
    use_augmentation: bool = False
    use_federation: bool = False
    use_domain_adaptation: bool = False

    # Data
    data_mode: str = "dummy"  # "dummy" | "real" | "patch"
    cache_path: str | None = None
    manifest_paths: list[str] = field(default_factory=list)

    # Patch mode (data_mode="patch"): full-resolution random crops, see src/patch_data.py
    patch_size: tuple[int, int, int] = (128, 128, 128)
    patches_per_epoch: int = 580
    patch_fractions: tuple[float, float, float] = (0.35, 0.45, 0.20)

    # Training
    batch_size: int = 1
    lr: float = 1e-3
    local_epochs: int = 1
    num_rounds: int = 2
    coral_weight: float = 1.0
    coral_queue_size: int = 8
    coral_steps_per_round: int | None = None
    model_width: int = 16
    model_depth: int = 5
    class_weights: list[float] | None = None
    schedule_kind: str = "none"
    schedule_min_lr: float = 1.0e-5
    lr_horizon: int | None = None

    # Checkpointing
    run_id: str = "default_run"
    checkpoint_dir: str = "checkpoints"

    # Misc
    seed: int = 42

    def __post_init__(self) -> None:
        if self.batch_size != 1:
            raise ValueError(
                f"batch_size must be 1 (fixed by 00_shared/CONTRACTS.md), got {self.batch_size}"
            )
        if self.data_mode not in ("dummy", "real", "patch"):
            raise ValueError(f"data_mode must be 'dummy', 'real' or 'patch', got {self.data_mode!r}")
        if self.data_mode in ("real", "patch") and not self.cache_path:
            raise ValueError(f"data_mode={self.data_mode!r} requires an explicit cache_path")
        self.patch_size = tuple(self.patch_size)
        self.patch_fractions = tuple(self.patch_fractions)
        if len(self.patch_size) != 3 or not all(
            isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in self.patch_size
        ):
            raise ValueError(f"patch_size must be three positive integers, got {self.patch_size!r}")
        if not isinstance(self.patches_per_epoch, int) or self.patches_per_epoch < 1:
            raise ValueError(f"patches_per_epoch must be a positive integer, got {self.patches_per_epoch!r}")
        if (len(self.patch_fractions) != 3 or any(f < 0 for f in self.patch_fractions)
                or abs(sum(self.patch_fractions) - 1.0) > 1e-6):
            raise ValueError(f"patch_fractions must be three non-negative numbers summing to 1, "
                             f"got {self.patch_fractions!r}")
        if self.coral_weight < 0:
            raise ValueError("coral_weight must be non-negative")
        if self.coral_queue_size < 2:
            raise ValueError("coral_queue_size must be at least 2")
        if self.coral_steps_per_round is not None and self.coral_steps_per_round < 2:
            raise ValueError("coral_steps_per_round must be at least 2 when set")
