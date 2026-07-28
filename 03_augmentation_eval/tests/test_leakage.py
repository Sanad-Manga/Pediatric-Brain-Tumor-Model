"""Leakage rules — the thing most likely to silently ruin this.

With ~57,000 slices a random slice-level split puts adjacent, nearly identical
slices of the same patient in both train and validation, producing a beautiful
Dice around 0.95 that means nothing. These tests exist to make that impossible
and to fail loudly if it ever becomes possible again.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.dataset import EvalSubjectDataset, PatientBalancedBatchSampler, SliceDataset
from src.plan import build_plans
from src.slices import load_manifest


# ----------------------------------------------------------------- Req 33
def test_the_three_real_manifests_are_disjoint_at_subject_level():
    """Runs against the real manifests, not the fixtures."""
    manifest_dir = Path(__file__).resolve().parent.parent.parent / "00_shared" / "manifests"
    a = set(load_manifest(manifest_dir, "hospitalA"))
    b = set(load_manifest(manifest_dir, "hospitalB"))
    held = set(load_manifest(manifest_dir, "heldout"))

    assert not a & b, f"hospitalA and hospitalB share subjects: {sorted(a & b)}"
    assert not a & held, f"hospitalA leaks into held-out: {sorted(a & held)}"
    assert not b & held, f"hospitalB leaks into held-out: {sorted(b & held)}"
    assert len(a) == 53 and len(b) == 92 and len(held) == 82


def test_fixture_manifests_are_disjoint(manifests):
    a, b, held = (set(manifests[k]) for k in ("hospitalA", "hospitalB", "heldout"))
    assert not (a & b) and not (a & held) and not (b & held)


def test_disjointness_failure_is_loud(cfg, cache, manifests, tmp_path):
    """Deliberately leak a held-out subject into hospital A and expect a raise."""
    import json

    leaked = manifests["heldout"][0]
    path = Path(cfg.paths["manifests"]) / "hospitalA.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifests["hospitalA"] + [leaked], fh)

    with pytest.raises(AssertionError, match="shares subjects with the held-out"):
        build_plans(cfg, cache)


# ----------------------------------------------------------------- Req 34
def test_no_subject_appears_in_more_than_one_plan(cfg, cache):
    plans = build_plans(cfg, cache)
    a = {e["source_subject_id"] for e in plans["hospitalA"]["entries"]}
    b = {e["source_subject_id"] for e in plans["hospitalB"]["entries"]}
    assert not a & b


def test_the_split_is_by_patient_not_by_slice(cfg, cache, manifests):
    """Every slice of a subject lands on one side of the split, never both."""
    plans = build_plans(cfg, cache)
    owner = {}
    for hospital, plan in plans.items():
        for entry in plan["entries"]:
            sid = entry["source_subject_id"]
            assert owner.setdefault(sid, hospital) == hospital


def test_eval_set_is_exactly_the_heldout_manifest(cfg, cache, manifests):
    ds = EvalSubjectDataset(manifests["heldout"], cfg, cache_dir=cache)
    assert {sid for sid, _plane in ds.items} == set(manifests["heldout"])


# ----------------------------------------------------------------- Req 35
def test_heldout_is_never_augmented_regardless_of_the_flag(cfg, cache, manifests):
    cfg.use_augmentation = False
    off = EvalSubjectDataset(manifests["heldout"], cfg, cache_dir=cache)[0]
    cfg.use_augmentation = True
    on = EvalSubjectDataset(manifests["heldout"], cfg, cache_dir=cache)[0]

    assert np.array_equal(off["images"], on["images"])
    assert np.array_equal(off["labels"], on["labels"])


def test_no_augmented_copy_of_a_training_subject_reaches_evaluation(cfg, cache, manifests):
    plans = build_plans(cfg, cache)
    training = {e["source_subject_id"] for p in plans.values() for e in p["entries"]}
    evaluated = {sid for sid, _ in EvalSubjectDataset(
        manifests["heldout"], cfg, cache_dir=cache).items}
    assert not training & evaluated


# ------------------------------------------------------------- Req 36, 37, 38
def test_batches_draw_distinct_patients_from_a_single_plane(cfg, cache, manifests):
    entries = []
    for p, sid in enumerate(f"P{i:03d}" for i in range(20)):
        for plane in ("axial", "coronal"):
            for k in range(6):
                entries.append({"hospital": "hospitalA", "source_subject_id": sid,
                                "plane": plane, "slice_index": k,
                                "is_augmented": False, "aug_seed": None})

    sampler = PatientBalancedBatchSampler(entries, batch_size=8, cfg=cfg, seed=0)
    batches = list(sampler)
    assert batches, "sampler produced no batches"

    for batch in batches:
        subjects = [entries[i]["source_subject_id"] for i in batch]
        planes = {entries[i]["plane"] for i in batch}
        assert len(batch) == 8
        assert len(set(subjects)) == 8, subjects   # Req 38: 8 distinct patients
        assert len(planes) == 1, planes            # Req 37: one plane per batch


def test_sampler_respects_max_slices_per_patient(cfg):
    entries = [{"hospital": "hospitalA", "source_subject_id": f"P{p:03d}",
                "plane": "axial", "slice_index": k,
                "is_augmented": False, "aug_seed": None}
               for p in range(10) for k in range(10)]

    sampler = PatientBalancedBatchSampler(entries, batch_size=6, cfg=cfg,
                                          max_per_patient=2, seed=1)
    for batch in sampler:
        counts = {}
        for i in batch:
            sid = entries[i]["source_subject_id"]
            counts[sid] = counts.get(sid, 0) + 1
        assert max(counts.values()) <= 2


def test_sampler_never_repeats_an_index_within_a_batch(cfg):
    entries = [{"hospital": "hospitalA", "source_subject_id": f"P{p:03d}",
                "plane": "axial", "slice_index": k,
                "is_augmented": False, "aug_seed": None}
               for p in range(12) for k in range(4)]
    for batch in PatientBalancedBatchSampler(entries, batch_size=8, cfg=cfg, seed=2):
        assert len(set(batch)) == len(batch)
