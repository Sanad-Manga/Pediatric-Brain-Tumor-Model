"""Shared fixtures: a tiny synthetic cache, no real data, CPU only.

Volume shape (24, 24, 16) reproduces the real axial/coronal shape mismatch in
miniature -- axial slices are 24x24 and coronal are 24x16, the same asymmetry as
240x240 vs 240x155 -- so the padding and restacking paths are genuinely
exercised without a 32GB dataset.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.dummy import make_dummy_cache, make_dummy_manifests  # noqa: E402

DUMMY_VOLUME = (24, 24, 16)


@pytest.fixture
def cfg(tmp_path):
    """Config pointed at temp dirs, with a network size that fits the tiny slices."""
    c = load_config()
    c.data["common_size"] = [32, 32]
    c.data["volume_shape"] = list(DUMMY_VOLUME)
    c.expansion["target_per_hospital"] = 60
    c.expansion["cap_multiplier"] = 2.0
    c.paths = dict(c.paths)
    c.paths["cache_2d"] = str(tmp_path / "cache")
    c.paths["manifests"] = str(tmp_path / "manifests")
    c.paths["plans_dir"] = str(tmp_path / "plans")
    c.paths["results_csv"] = str(tmp_path / "results" / "ablation_results.csv")
    return c


@pytest.fixture
def manifests(cfg):
    """Disjoint hospitalA / hospitalB / heldout manifests."""
    return make_dummy_manifests(Path(cfg.paths["manifests"]), counts=(4, 5, 3))


@pytest.fixture
def cache(cfg, manifests):
    """A synthetic slice cache covering every subject in the manifests."""
    hospital_of = {sid: name for name, ids in manifests.items() for sid in ids}
    subjects = [sid for ids in manifests.values() for sid in ids]
    cache_dir = Path(cfg.paths["cache_2d"])
    make_dummy_cache(
        cache_dir, subjects, cfg,
        volume_shape=DUMMY_VOLUME,
        hospital_of=lambda s: hospital_of[s],
        seed=7,
    )
    return cache_dir


@pytest.fixture
def entry_factory(manifests):
    """Build a single plan entry for hospital A's first subject."""
    def make(plane="axial", slice_index=8, is_augmented=False, aug_seed=None):
        return {
            "hospital": "hospitalA",
            "source_subject_id": manifests["hospitalA"][0],
            "plane": plane,
            "slice_index": slice_index,
            "is_augmented": is_augmented,
            "aug_seed": aug_seed,
        }
    return make
