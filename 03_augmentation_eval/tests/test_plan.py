"""Balanced expansion: per-hospital budgets, per-patient caps, provenance."""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from src.plan import PROVENANCE_FIELDS, build_plans, format_report, read_plan, write_plan
from src.slices import index_cache


# ----------------------------------------------------------------- Req 24
def test_hospitals_are_never_pooled(cfg, cache, manifests):
    plans = build_plans(cfg, cache)
    for hospital, plan in plans.items():
        owned = set(manifests[hospital])
        assert {e["source_subject_id"] for e in plan["entries"]} <= owned
        assert {e["hospital"] for e in plan["entries"]} == {hospital}


def test_no_heldout_subject_reaches_any_plan(cfg, cache, manifests):
    plans = build_plans(cfg, cache)
    heldout = set(manifests["heldout"])
    for plan in plans.values():
        assert not {e["source_subject_id"] for e in plan["entries"]} & heldout


# ----------------------------------------------------------------- Req 25
def test_each_hospital_gets_the_same_slice_budget(cfg, cache):
    plans = build_plans(cfg, cache)
    target = cfg.expansion["target_per_hospital"]
    counts = {h: len(p["entries"]) for h, p in plans.items()}
    assert set(counts.values()) == {target}, counts


def test_shortfall_is_reported_not_borrowed(cfg, cache, manifests):
    """An unreachable target emits what it can; it never pulls from the other hospital."""
    cfg.expansion["target_per_hospital"] = 100000
    cfg.expansion["cap_multiplier"] = 0.00001     # cap collapses to 1 per patient
    plans = build_plans(cfg, cache)
    for hospital, plan in plans.items():
        stats = plan["stats"]
        assert stats["entries"] < stats["target"]
        assert stats["entries"] == len(manifests[hospital])   # exactly the cap of 1 each


# --------------------------------------------------- Reqs 25/26/27 invariants
@pytest.mark.parametrize("target", [20, 60, 200, 500])
@pytest.mark.parametrize("empty_ratio", [0.0, 0.3, 0.5])
@pytest.mark.parametrize("cap_multiplier", [0.5, 2.0, 5.0])
def test_plan_invariants_hold_across_configs(cfg, cache, manifests,
                                             target, empty_ratio, cap_multiplier):
    """The budget, cap and empty ceiling must hold together, not one at a time.

    Two bugs escaped the single-config tests and were caught by sweeping these
    combinations: phases collectively overshooting the target, and the empty
    *fraction* exceeding its ceiling whenever the per-patient cap bound before
    the target did (the plan is smaller than target, so a count-based ceiling
    is too loose).
    """
    cfg.expansion["target_per_hospital"] = target
    cfg.expansion["empty_slice_ratio"] = empty_ratio
    cfg.expansion["cap_multiplier"] = cap_multiplier

    for hospital, plan in build_plans(cfg, cache).items():
        stats, entries = plan["stats"], plan["entries"]
        counts = Counter(e["source_subject_id"] for e in entries)
        cap = max(1, math.ceil(cap_multiplier * target / stats["subjects"]))

        assert len(entries) <= target, f"{hospital}: overshot the target"
        assert max(counts.values()) <= cap, f"{hospital}: broke the per-patient cap"
        assert stats["empty_fraction"] <= empty_ratio + 1e-9, \
            f"{hospital}: empty fraction above the ceiling"
        assert set(counts) == set(manifests[hospital]), f"{hospital}: lost a subject"


# ----------------------------------------------------------------- Req 26
def test_empty_slices_stay_under_the_ratio_ceiling(cfg, cache):
    plans = build_plans(cfg, cache)
    ceiling = cfg.expansion["empty_slice_ratio"]
    for hospital, plan in plans.items():
        assert plan["stats"]["empty_fraction"] <= ceiling + 1e-9, hospital


def test_tumour_slices_are_prioritised_over_empty_ones(cfg, cache):
    plans = build_plans(cfg, cache)
    for plan in plans.values():
        stats = plan["stats"]
        assert stats["tumor_entries"] >= stats["empty_entries"]


# ------------------------------------------------------------ Req 54 (ET floor)
def _et_fraction(plan):
    return plan["stats"]["et_fraction"]


def test_et_floor_is_met_when_supply_allows(cfg, cache):
    """Enhancing tumour is the smallest scored region -- it gets a floor."""
    plans = build_plans(cfg, cache)
    floor = cfg.expansion["et_slice_ratio"]
    for hospital, plan in plans.items():
        stats = plan["stats"]
        if stats["et_available"] >= stats["et_floor"]:
            assert _et_fraction(plan) >= floor - 1e-9, (hospital, stats)


def test_et_floor_beats_a_tumour_blind_selection(cfg, cache):
    """Turning the floor off should not produce more ET than turning it on."""
    with_floor = build_plans(cfg, cache)
    cfg.expansion["et_slice_ratio"] = 0.0
    without = build_plans(cfg, cache)
    for hospital in with_floor:
        assert (with_floor[hospital]["stats"]["et_entries"]
                >= without[hospital]["stats"]["et_entries"]), hospital


def test_a_zero_et_floor_disables_the_reservation(cfg, cache):
    cfg.expansion["et_slice_ratio"] = 0.0
    plans = build_plans(cfg, cache)
    for plan in plans.values():
        assert plan["stats"]["et_floor"] == 0
        assert plan["stats"]["entries"] == plan["stats"]["target"]


def test_et_shortage_degrades_gracefully(cfg, cache):
    """A floor the cache cannot meet emits what exists rather than raising."""
    cfg.expansion["et_slice_ratio"] = 0.99
    plans = build_plans(cfg, cache)
    for plan in plans.values():
        stats = plan["stats"]
        assert stats["entries"] == stats["target"]          # budget still met
        assert stats["et_entries"] <= stats["et_available"]  # never invented


def test_et_entries_really_contain_enhancing_tumour(cfg, cache):
    plans = build_plans(cfg, cache)
    index = index_cache(cache, cfg.planes)
    for plan in plans.values():
        counted = 0
        for entry in plan["entries"]:
            mask = index[entry["source_subject_id"]][entry["plane"]]["has_et"]
            if bool(np.asarray(mask, dtype=bool)[entry["slice_index"]]):
                counted += 1
        assert counted == plan["stats"]["et_entries"]


def test_et_floor_does_not_break_the_empty_ceiling(cfg, cache):
    plans = build_plans(cfg, cache)
    ceiling = cfg.expansion["empty_slice_ratio"]
    for plan in plans.values():
        assert plan["stats"]["empty_fraction"] <= ceiling + 1e-9


def test_et_floor_does_not_break_the_per_patient_cap(cfg, cache):
    plans = build_plans(cfg, cache)
    for plan in plans.values():
        counts = Counter(e["source_subject_id"] for e in plan["entries"])
        assert max(counts.values()) <= plan["stats"]["per_patient_cap"]


def test_report_shows_the_et_columns(cfg, cache):
    report = format_report(build_plans(cfg, cache))
    assert "ET" in report and "ET%" in report


# ----------------------------------------------------------------- Req 27
def test_no_patient_exceeds_the_cap(cfg, cache, manifests):
    plans = build_plans(cfg, cache)
    target = cfg.expansion["target_per_hospital"]
    mult = cfg.expansion["cap_multiplier"]

    for hospital, plan in plans.items():
        n_patients = len(manifests[hospital])
        cap = max(1, math.ceil(mult * target / n_patients))
        counts = Counter(e["source_subject_id"] for e in plan["entries"])
        assert plan["stats"]["per_patient_cap"] == cap
        assert max(counts.values()) <= cap, (hospital, counts)


def test_no_patient_dominates_the_hospital(cfg, cache, manifests):
    """Max share stays within cap_multiplier / n_patients of the total."""
    plans = build_plans(cfg, cache)
    mult = cfg.expansion["cap_multiplier"]
    for hospital, plan in plans.items():
        counts = Counter(e["source_subject_id"] for e in plan["entries"])
        n_patients = len(manifests[hospital])
        max_share = max(counts.values()) / len(plan["entries"])
        assert max_share <= mult / n_patients + 1e-6, (hospital, max_share)


# ----------------------------------------------------------------- Req 28
@pytest.fixture
def upsampled_plans(cfg, cache):
    """A target the cache cannot fill from originals, so Phase C must run."""
    cfg.expansion["target_per_hospital"] = 200
    plans = build_plans(cfg, cache)
    assert any(e["is_augmented"] for p in plans.values() for e in p["entries"]), \
        "fixture did not force any up-sampling"
    return plans


def test_upsampling_uses_augmented_copies_with_distinct_seeds(upsampled_plans):
    for plan in upsampled_plans.values():
        augmented = [e for e in plan["entries"] if e["is_augmented"]]
        assert augmented
        seeds = [e["aug_seed"] for e in augmented]
        assert len(seeds) == len(set(seeds))
        assert all(isinstance(s, int) for s in seeds)


def test_upsampling_targets_under_represented_patients(upsampled_plans):
    """Augmented copies go to the patients with the fewest original slices."""
    for plan in upsampled_plans.values():
        originals = Counter(e["source_subject_id"]
                            for e in plan["entries"] if not e["is_augmented"])
        augmented = Counter(e["source_subject_id"]
                            for e in plan["entries"] if e["is_augmented"])
        totals = Counter(e["source_subject_id"] for e in plan["entries"])
        # Up-sampling levels patients up: the spread after it is no wider
        # than the spread of originals alone.
        if len(originals) > 1 and augmented:
            spread_before = max(originals.values()) - min(originals.values())
            spread_after = max(totals.values()) - min(totals.values())
            assert spread_after <= spread_before


def test_upsampling_never_breaks_the_empty_ceiling(cfg, upsampled_plans):
    ceiling = cfg.expansion["empty_slice_ratio"]
    for plan in upsampled_plans.values():
        assert plan["stats"]["empty_fraction"] <= ceiling + 1e-9


def test_upsampling_respects_the_per_patient_cap(upsampled_plans):
    for plan in upsampled_plans.values():
        counts = Counter(e["source_subject_id"] for e in plan["entries"])
        assert max(counts.values()) <= plan["stats"]["per_patient_cap"]


def test_augmented_slices_follow_the_same_order_as_the_originals(cfg, cache, upsampled_plans):
    """An augmented copy sits in the same sequence position as its original.

    Phase C walks each patient's already-selected slices in emission order and
    cycles, so the augmented run mirrors the original run rather than being a
    re-sorted or re-shuffled set. This keeps an augmented slice positionally
    equivalent to the slice it came from, so the model sees the two the same way.
    """
    index = index_cache(cache, cfg.planes)

    def kind(entry):
        planes = index[entry["source_subject_id"]][entry["plane"]]
        i = entry["slice_index"]
        if bool(np.asarray(planes["has_et"], dtype=bool)[i]):
            return "et"
        if bool(np.asarray(planes["has_tumor"], dtype=bool)[i]):
            return "tumor"
        return "empty"

    for plan in upsampled_plans.values():
        by_subject = {}
        for entry in plan["entries"]:
            by_subject.setdefault(entry["source_subject_id"], []).append(entry)

        for subject_id, entries in by_subject.items():
            if not any(e["is_augmented"] for e in entries):
                continue
            # Each pool is drawn in its own original order, cycling. Checking
            # per pool is what makes the property hold however ET, tumour and
            # empty copies interleave.
            for pool in ("et", "tumor", "empty"):
                originals = [(e["plane"], e["slice_index"]) for e in entries
                             if not e["is_augmented"] and kind(e) == pool]
                augmented = [(e["plane"], e["slice_index"]) for e in entries
                             if e["is_augmented"] and kind(e) == pool]
                if not augmented:
                    continue
                assert originals, f"{subject_id}/{pool}: copies with no source"
                expected = [originals[i % len(originals)] for i in range(len(augmented))]
                assert augmented == expected, f"{subject_id}/{pool}"


def test_augmented_entries_reference_a_real_source_slice(cfg, cache, upsampled_plans):
    """An augmented copy must point at a slice that exists in the cache."""
    index = index_cache(cache, cfg.planes)
    for plan in upsampled_plans.values():
        for entry in plan["entries"]:
            if not entry["is_augmented"]:
                continue
            summary = index[entry["source_subject_id"]][entry["plane"]]
            assert 0 <= entry["slice_index"] < summary["n_slices"]


def test_originals_are_emitted_before_augmented_copies(cfg, cache):
    plans = build_plans(cfg, cache)
    for plan in plans.values():
        flags = [e["is_augmented"] for e in plan["entries"]]
        first_aug = next((i for i, f in enumerate(flags) if f), len(flags))
        assert not any(flags[:first_aug])
        assert all(flags[first_aug:]) or first_aug == len(flags)


# ----------------------------------------------------------------- Req 29
def test_every_cached_subject_appears_at_least_once(cfg, cache, manifests):
    plans = build_plans(cfg, cache)
    for hospital, plan in plans.items():
        present = {e["source_subject_id"] for e in plan["entries"]}
        assert present == set(manifests[hospital]), hospital


def test_hospitals_are_not_forced_to_match_each_other(cfg, cache, manifests):
    """A and B keep their own patient counts; only the slice budget is equal."""
    plans = build_plans(cfg, cache)
    a = len({e["source_subject_id"] for e in plans["hospitalA"]["entries"]})
    b = len({e["source_subject_id"] for e in plans["hospitalB"]["entries"]})
    assert a == len(manifests["hospitalA"])
    assert b == len(manifests["hospitalB"])
    assert a != b     # the fixture deliberately gives them different sizes


# ----------------------------------------------------------------- Req 30
def test_every_entry_carries_full_provenance(cfg, cache):
    plans = build_plans(cfg, cache)
    for plan in plans.values():
        for entry in plan["entries"]:
            assert set(entry) == set(PROVENANCE_FIELDS)
            assert entry["plane"] in cfg.planes
            assert isinstance(entry["slice_index"], int)
            # aug_seed is null exactly when the entry is an original.
            assert (entry["aug_seed"] is None) is (not entry["is_augmented"])


# ----------------------------------------------------------------- Req 31
def test_report_table_lists_every_required_column(cfg, cache):
    report = format_report(build_plans(cfg, cache))
    for column in ("hospital", "subjects", "entries", "tumor", "empty",
                   "augment", "cap", "max/pt", "med/pt"):
        assert column in report
    assert "hospitalA" in report and "hospitalB" in report


def test_plan_is_written_as_json_per_hospital(cfg, cache, tmp_path):
    plans = build_plans(cfg, cache)
    path = write_plan(plans["hospitalA"], tmp_path / "hospitalA_plan.json")
    assert path.exists()
    loaded = read_plan(path)
    assert loaded["entries"] == plans["hospitalA"]["entries"]
    assert loaded["stats"]["hospital"] == "hospitalA"


# ----------------------------------------------------------------- Req 32
def test_plan_is_byte_identical_across_runs_for_a_fixed_seed(cfg, cache, tmp_path):
    first = write_plan(build_plans(cfg, cache)["hospitalA"], tmp_path / "a.json")
    second = write_plan(build_plans(cfg, cache)["hospitalA"], tmp_path / "b.json")
    assert first.read_bytes() == second.read_bytes()


def test_a_different_seed_changes_the_plan(cfg, cache, tmp_path):
    first = write_plan(build_plans(cfg, cache)["hospitalA"], tmp_path / "a.json")
    cfg.expansion["seed"] = 4242
    second = write_plan(build_plans(cfg, cache)["hospitalA"], tmp_path / "b.json")
    assert first.read_bytes() != second.read_bytes()
