"""Shared fixtures: a tiny synthetic cache in the real layout, CPU only.

Volume shape (24, 24, 16) reproduces the real axial/coronal asymmetry in
miniature -- axial slices are 24x24 and coronal are 24x16, the same mismatch as
240x240 vs 240x155 -- so the padding and restacking paths are genuinely
exercised without the real dataset.

The cache is **session-scoped**. The real layout is one file per slice, so a
per-test rebuild would write hundreds of files per test and dominate runtime.
Manifests are copied per test instead, since a few tests deliberately corrupt
them to prove the leakage assertions fire.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.dummy import make_dummy_cache, make_dummy_manifests  # noqa: E402

DUMMY_VOLUME = (24, 24, 16)
MANIFEST_COUNTS = (4, 5, 3)


def _base_config():
    c = load_config()
    c.data["common_size"] = [32, 32]
    c.data["volume_shape"] = list(DUMMY_VOLUME)
    return c


@pytest.fixture(scope="session")
def _shared(tmp_path_factory):
    """Build the synthetic cache and manifests once for the whole session."""
    root = tmp_path_factory.mktemp("shared")
    cfg = _base_config()

    manifests = make_dummy_manifests(root / "manifests", counts=MANIFEST_COUNTS)
    subjects = [sid for ids in manifests.values() for sid in ids]
    make_dummy_cache(root / "cache", subjects, cfg, volume_shape=DUMMY_VOLUME, seed=7)

    return {"root": root, "cache": root / "cache",
            "manifests_dir": root / "manifests", "manifests": manifests}


@pytest.fixture
def cfg(tmp_path, _shared):
    """Config pointed at the shared cache, with per-test writable paths."""
    c = _base_config()
    c.expansion = dict(c.expansion)
    c.expansion["target_per_hospital"] = 60
    c.expansion["cap_multiplier"] = 2.0
    c.eval = dict(c.eval)

    # Manifests are copied so a test can corrupt them without touching others.
    manifest_dir = tmp_path / "manifests"
    shutil.copytree(_shared["manifests_dir"], manifest_dir)

    c.paths = dict(c.paths)
    c.paths["cache_2d"] = str(_shared["cache"])
    c.paths["manifests"] = str(manifest_dir)
    c.paths["plans_dir"] = str(tmp_path / "plans")
    c.paths["results_csv"] = str(tmp_path / "results" / "ablation_results.csv")
    return c


@pytest.fixture
def manifests(_shared):
    """The subject-ID lists: disjoint hospitalA / hospitalB / heldout."""
    return {k: list(v) for k, v in _shared["manifests"].items()}


@pytest.fixture
def cache(_shared):
    """The shared synthetic slice cache, already indexed."""
    return _shared["cache"]


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
