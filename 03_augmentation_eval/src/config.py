"""Configuration loading for the 2D augmentation + ablation pipeline.

A single YAML file drives everything. The flag that matters most is
``use_augmentation``: per CONTRACTS.md it gates the entire augmentation branch
(MONAI transform stack *and* mixup) with no code changes required, and
``use_mixup`` is nested inside it.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"

#: the two planes this section works with; sagittal is deliberately absent
PLANES = ("axial", "coronal")


@dataclass
class Config:
    """Parsed pipeline configuration.

    Attributes mirror the top-level keys of ``config.yaml``. Nested sections are
    kept as plain dicts so new knobs can be added to the YAML without touching
    this class.
    """

    use_augmentation: bool = True
    use_mixup: bool = True
    use_federation: bool = False
    use_domain_adaptation: bool = False

    paths: dict = field(default_factory=dict)
    data: dict = field(default_factory=dict)
    expansion: dict = field(default_factory=dict)
    sampler: dict = field(default_factory=dict)
    eval: dict = field(default_factory=dict)
    tumor_type: dict = field(default_factory=dict)
    augmentation: dict = field(default_factory=dict)
    mixup: dict = field(default_factory=dict)

    #: directory the config file was loaded from; relative paths resolve against it
    root: Path = field(default_factory=lambda: DEFAULT_CONFIG_PATH.parent)

    # ---------------------------------------------------------------- gating
    @property
    def mixup_enabled(self) -> bool:
        """Mixup runs only when the master augmentation flag is also on.

        This is the contract in SPEC.md Req 23: ``use_augmentation: false``
        disables mixup regardless of ``use_mixup``.
        """
        return bool(self.use_augmentation and self.use_mixup)

    # ----------------------------------------------------------------- paths
    def resolve(self, key: str) -> Path:
        """Resolve a ``paths.<key>`` entry to an absolute path."""
        if key not in self.paths:
            raise KeyError(f"paths.{key} is not defined in the config")
        p = Path(self.paths[key])
        return p if p.is_absolute() else (self.root / p).resolve()

    # ------------------------------------------------------------------ data
    @property
    def modalities(self) -> list:
        return list(self.data.get("modalities", ["t1c", "t1n", "t2f", "t2w"]))

    @property
    def planes(self) -> list:
        return list(self.data.get("planes", list(PLANES)))

    @property
    def common_size(self) -> tuple[int, int]:
        h, w = self.data.get("common_size", [256, 256])
        return int(h), int(w)

    @property
    def resize_mode(self) -> str:
        return str(self.data.get("resize_mode", "pad"))

    @property
    def volume_shape(self) -> tuple[int, int, int]:
        x, y, z = self.data.get("volume_shape", [240, 240, 155])
        return int(x), int(y), int(z)

    @property
    def num_classes(self) -> int:
        return int(self.data.get("num_classes", 5))

    @property
    def valid_labels(self) -> set:
        return set(self.data.get("valid_labels", [0, 1, 2, 3, 4]))

    @property
    def min_std(self) -> float:
        return float(self.data.get("min_std", 1e-6))

    def plane_axis(self, plane: str) -> int:
        """Array axis that ``plane`` slices along.

        The default map (sagittal=0, coronal=1, axial=2) is the only assignment
        consistent with the measured slice counts: 155 axial slices of 240x240
        and 240 coronal slices of 240x155 from a 240x240x155 volume.
        """
        axes = self.data.get("plane_axis", {"sagittal": 0, "coronal": 1, "axial": 2})
        if plane not in axes:
            raise KeyError(f"unknown plane {plane!r}; expected one of {sorted(axes)}")
        return int(axes[plane])

    # ------------------------------------------------------------- evaluation
    @property
    def tumor_type_enabled(self) -> bool:
        return bool(self.tumor_type.get("enabled", False))

    @property
    def tumor_type_loss_weight(self) -> float:
        return float(self.tumor_type.get("loss_weight", 0.2))

    @property
    def eval_plane(self) -> str:
        plane = str(self.eval.get("plane", "axial"))
        if plane not in (*PLANES, "both"):
            raise ValueError(
                f"eval.plane must be one of {[*PLANES, 'both']}, got {plane!r}"
            )
        return plane


def load_config(path: str | Path | None = None, **overrides: Any) -> Config:
    """Load ``config.yaml`` and apply keyword overrides.

    Overrides map to top-level scalar flags, e.g.
    ``load_config(use_augmentation=False)``. This is how CLI flags reach the
    config without editing the YAML. ``None`` overrides are ignored so argparse
    tri-state flags pass straight through.
    """
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(f"config file not found: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as fh:
        raw: Mapping[str, Any] = yaml.safe_load(fh) or {}

    cfg = Config(
        use_augmentation=bool(raw.get("use_augmentation", True)),
        use_mixup=bool(raw.get("use_mixup", True)),
        use_federation=bool(raw.get("use_federation", False)),
        use_domain_adaptation=bool(raw.get("use_domain_adaptation", False)),
        paths=copy.deepcopy(dict(raw.get("paths", {}))),
        data=copy.deepcopy(dict(raw.get("data", {}))),
        expansion=copy.deepcopy(dict(raw.get("expansion", {}))),
        sampler=copy.deepcopy(dict(raw.get("sampler", {}))),
        eval=copy.deepcopy(dict(raw.get("eval", {}))),
        tumor_type=copy.deepcopy(dict(raw.get("tumor_type", {}))),
        augmentation=copy.deepcopy(dict(raw.get("augmentation", {}))),
        mixup=copy.deepcopy(dict(raw.get("mixup", {}))),
        root=cfg_path.resolve().parent,
    )

    for key, value in overrides.items():
        if value is None:
            continue
        if not hasattr(cfg, key):
            raise KeyError(f"unknown config override: {key}")
        setattr(cfg, key, value)

    return cfg
