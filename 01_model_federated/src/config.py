"""Config dataclass shared by single-client and federated training loops."""
from __future__ import annotations

from dataclasses import dataclass, field


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
