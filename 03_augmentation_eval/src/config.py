"""Configuration loading for the augmentation + ablation pipeline.

A single YAML file drives everything. The one flag that matters most is
``use_augmentation`` — per CONTRACTS.md it gates the entire augmentation branch
(MONAI transform stack *and* mixup) with no code changes required.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


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
    strata: dict = field(default_factory=dict)
    augmentation: dict = field(default_factory=dict)
    mixup: dict = field(default_factory=dict)

    #: directory the config file was loaded from; relative paths resolve against it
    root: Path = field(default_factory=lambda: DEFAULT_CONFIG_PATH.parent)

    # ---------------------------------------------------------------- helpers
    @property
    def mixup_enabled(self) -> bool:
        """Mixup runs only when the master augmentation flag is also on.

        This is the contract in Req 14: ``use_augmentation: false`` disables
        mixup regardless of ``use_mixup``.
        """
        return bool(self.use_augmentation and self.use_mixup)

    def resolve(self, key: str) -> Path:
        """Resolve a ``paths.<key>`` entry to an absolute path."""
        if key not in self.paths:
            raise KeyError(f"paths.{key} is not defined in the config")
        p = Path(self.paths[key])
        return p if p.is_absolute() else (self.root / p).resolve()

    @property
    def spatial_size(self) -> tuple:
        return tuple(self.data.get("spatial_size", [96, 96, 96]))

    @property
    def modalities(self) -> list:
        return list(self.data.get("modalities", ["t1c", "t1n", "t2f", "t2w"]))

    @property
    def num_classes(self) -> int:
        return int(self.data.get("num_classes", 5))

    @property
    def valid_labels(self) -> set:
        return set(self.data.get("valid_labels", [0, 1, 2, 3, 4]))


def load_config(path: str | Path | None = None, **overrides: Any) -> Config:
    """Load ``config.yaml`` and apply keyword overrides.

    Overrides map to top-level scalar flags, e.g.
    ``load_config(use_augmentation=False)``. This is how CLI flags reach the
    config without editing the YAML.
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
        strata=copy.deepcopy(dict(raw.get("strata", {}))),
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
