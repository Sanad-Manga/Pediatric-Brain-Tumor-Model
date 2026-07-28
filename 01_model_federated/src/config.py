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
    data_mode: str = "dummy"  # "dummy" | "real"
    cache_path: str | None = None
    manifest_paths: list[str] = field(default_factory=list)

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
        if self.data_mode not in ("dummy", "real"):
            raise ValueError(f"data_mode must be 'dummy' or 'real', got {self.data_mode!r}")
        if self.data_mode == "real" and not self.cache_path:
            raise ValueError("data_mode='real' requires an explicit cache_path")
        if self.coral_weight < 0:
            raise ValueError("coral_weight must be non-negative")
        if self.coral_queue_size < 2:
            raise ValueError("coral_queue_size must be at least 2")
        if self.coral_steps_per_round is not None and self.coral_steps_per_round < 2:
            raise ValueError("coral_steps_per_round must be at least 2 when set")
