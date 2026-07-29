"""Per-hospital balanced slice plan.

The 3D rule was "150 subjects per hospital with tumour-type ratios preserved".
In 2D that becomes a slice budget that still balances at the *patient* level:

* Hospitals stay completely separate -- A and B are never pooled.
* Each hospital gets the same slice budget.
* No single patient may dominate: a per-patient cap stops a patient with a large
  tumour from supplying thousands of slices.
* Under-represented patients are up-sampled via augmentation instead.
* Each hospital keeps its own patient-level composition; A and B are not forced
  to match each other.

Selection runs in four phases, all round-robin over patients so the allocation
is equal by construction rather than by a post-hoc correction:

    Phase 0  one entry per subject          -- guarantees nobody is dropped
    Phase A  tumour originals               -- priority; empty slices never win
    Phase B  empty originals                -- capped at `empty_slice_ratio`
    Phase C  augmented copies               -- only to reach the target

Tumour slices are kept in full whenever the budget allows. When a hospital has
more tumour slices than the whole budget (hospital B does: ~92 patients x ~127
tumour slices vs a budget of 8000) the empty quota is reserved first, so the
plan keeps a usable negative set instead of degenerating to 100% tumour.

Every entry carries full provenance: hospital, source_subject_id, plane,
slice_index, is_augmented, aug_seed.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import numpy as np

from .config import Config
from .slices import index_cache, load_manifest

LOGGER = logging.getLogger(__name__)

HOSPITALS = ("hospitalA", "hospitalB")

#: every plan entry has exactly these keys, in this order
PROVENANCE_FIELDS = (
    "hospital",
    "source_subject_id",
    "plane",
    "slice_index",
    "is_augmented",
    "aug_seed",
)


def _entry(hospital, subject_id, plane, slice_index, is_augmented=False, aug_seed=None) -> dict:
    return {
        "hospital": hospital,
        "source_subject_id": subject_id,
        "plane": plane,
        "slice_index": int(slice_index),
        "is_augmented": bool(is_augmented),
        "aug_seed": None if aug_seed is None else int(aug_seed),
    }


def _round_robin(patients, pools, budget, cap, counts, emit):
    """Take one item per patient per round until ``budget`` or supply runs out.

    ``pools`` maps a patient to a list of remaining candidates; ``counts`` tracks
    each patient's running total against ``cap``. Returns the number emitted.
    """
    taken = 0
    while taken < budget:
        progressed = False
        for pid in patients:
            if taken >= budget:
                break
            pool = pools.get(pid)
            if not pool or counts[pid] >= cap:
                continue
            emit(pid, pool.pop())
            counts[pid] += 1
            taken += 1
            progressed = True
        if not progressed:
            break
    return taken


def build_hospital_plan(hospital: str, subject_ids, cache_index, cfg: Config) -> dict:
    """Build one hospital's slice plan. Deterministic for a fixed seed."""
    exp = cfg.expansion
    target = int(exp.get("target_per_hospital", 8000))
    empty_ratio = float(exp.get("empty_slice_ratio", 0.30))
    et_ratio = float(exp.get("et_slice_ratio", 0.25))
    cap_multiplier = float(exp.get("cap_multiplier", 2.0))
    seed = int(exp.get("seed", 1337))

    present = [sid for sid in sorted(subject_ids) if sid in cache_index]
    missing = sorted(set(subject_ids) - set(present))
    if not present:
        raise ValueError(
            f"{hospital}: none of its {len(subject_ids)} subjects are in the slice cache"
        )
    for sid in missing:
        LOGGER.warning("%s: subject %s is not in the slice cache; skipping", hospital, sid)

    n_patients = len(present)
    cap = max(1, math.ceil(cap_multiplier * target / n_patients))

    # Candidate pools per patient. Shuffled with a per-hospital seeded RNG so a
    # patient contributes slices spread through the volume rather than one
    # contiguous run of near-duplicates -- and so the plan is reproducible.
    rng = np.random.default_rng([seed, sum(ord(c) for c in hospital)])
    # Three pools, not two: enhancing-tumour slices are separated from the rest
    # so they can be given first claim on the tumour budget.
    et_pool, tumor_pool, empty_pool = {}, {}, {}
    for sid in present:
        et, tumor, empty = [], [], []
        for plane in sorted(cache_index[sid]):
            summary = cache_index[sid][plane]
            et_set = set(summary["et"])
            et.extend((plane, i) for i in summary["et"])
            tumor.extend((plane, i) for i in summary["tumor"] if i not in et_set)
            empty.extend((plane, i) for i in summary["empty"])
        # pop() takes from the end, so shuffle then reverse-consume
        et_pool[sid] = [et[k] for k in rng.permutation(len(et))]
        tumor_pool[sid] = [tumor[k] for k in rng.permutation(len(tumor))]
        empty_pool[sid] = [empty[k] for k in rng.permutation(len(empty))]

    avail_et = sum(len(v) for v in et_pool.values())
    avail_tumor = sum(min(len(et_pool[s]) + len(tumor_pool[s]), cap) for s in present)
    avail_empty = sum(len(v) for v in empty_pool.values())

    # What this hospital can actually deliver. For a small hospital the
    # per-patient cap binds before the target does, and every ratio below has
    # to be taken against the plan we will really emit -- budgeting against the
    # nominal target would let the empty *fraction* exceed its ceiling even
    # while the empty *count* stayed under it.
    with_slices = [s for s in present
                   if et_pool[s] or tumor_pool[s] or empty_pool[s]]
    achievable = min(target, len(with_slices) * cap)

    # `empty_slice_ratio` is a ceiling on the *whole* plan, augmented copies
    # included -- not just on the originals.
    max_empty = int(empty_ratio * achievable)

    # Reserve the empty quota only when tumour supply would otherwise fill the
    # entire budget; when tumour slices fit, keep all of them.
    if avail_tumor <= achievable:
        tumor_budget = avail_tumor
        empty_budget = min(avail_empty, achievable - tumor_budget, max_empty)
    else:
        empty_budget = min(avail_empty, max_empty)
        tumor_budget = achievable - empty_budget

    # ET floor: reserve this much of the budget for enhancing-tumour slices
    # before any other tumour slice competes for it.
    et_floor = int(et_ratio * achievable)
    et_target = min(avail_et, et_floor, max(0, tumor_budget))

    entries: list[dict] = []
    counts = {sid: 0 for sid in present}
    empty_count = 0
    et_count = 0

    def emit(pid, candidate):
        plane, idx = candidate
        entries.append(_entry(hospital, pid, plane, idx))

    def room(budget: int) -> int:
        """Clamp a phase budget to what is left in the plan.

        Each phase tracks its own budget, and those budgets are set before
        Phase 0 spends from them. Without this guard the phases can
        collectively overshoot -- so the global remaining count is the
        authority, not the per-phase arithmetic.
        """
        return max(0, min(budget, achievable - len(entries)))

    # --- Phase 0: one entry per subject, ET-bearing where possible, then any
    # tumour, then empty. This guarantees no cached subject is dropped.
    for sid in present:
        pool = et_pool[sid] or tumor_pool[sid] or empty_pool[sid]
        if not pool:
            LOGGER.warning("%s: subject %s has no cached slices at all", hospital, sid)
            continue
        if len(entries) >= achievable:
            break
        is_et = bool(et_pool[sid])
        is_tumor = is_et or bool(tumor_pool[sid])
        emit(sid, pool.pop())
        counts[sid] += 1
        if is_tumor:
            tumor_budget -= 1
        else:
            empty_budget -= 1
            empty_count += 1
        if is_et:
            et_count += 1

    # --- Phase A1: the ET floor, taken before other tumour slices.
    taken = _round_robin(
        present, et_pool, room(et_target - et_count), cap, counts, emit
    )
    et_count += taken
    tumor_budget -= taken

    # --- Phase A2: the remaining tumour budget, non-ET slices.
    tumor_budget -= _round_robin(
        present, tumor_pool, room(tumor_budget), cap, counts, emit
    )

    # --- Phase A3: non-ET supply exhausted but tumour budget left -> more ET.
    taken = _round_robin(present, et_pool, room(tumor_budget), cap, counts, emit)
    et_count += taken

    # --- Phase B: empty originals.
    empty_count += _round_robin(
        present, empty_pool, room(empty_budget), cap, counts, emit
    )

    n_original = len(entries)

    # --- Phase C: augmented up-sampling for patients below their fair share.
    # Only reached when a hospital cannot fill the target from originals.
    # Preference order ET > other tumour > empty: duplicating empty slices
    # would push the plan past the empty ceiling and undo exactly the balance
    # phases A and B just established, and ET is the scarcest signal of the
    # three scored regions.
    aug_seed_counter = seed
    shortfall = achievable - len(entries)
    if shortfall > 0:
        selected_et = {sid: [] for sid in present}
        selected_tumor = {sid: [] for sid in present}
        selected_empty = {sid: [] for sid in present}
        tumor_slices = {sid: _tumor_set(cache_index, sid) for sid in present}
        et_slices = {sid: _et_set(cache_index, sid) for sid in present}
        for e in entries:
            sid = e["source_subject_id"]
            key = (e["plane"], e["slice_index"])
            if key in et_slices[sid]:
                bucket = selected_et
            elif key in tumor_slices[sid]:
                bucket = selected_tumor
            else:
                bucket = selected_empty
            if key not in bucket[sid]:
                bucket[sid].append(key)

        # One cursor per (subject, pool) so each pool is consumed in its own
        # original emission order, cycling, however the pools interleave.
        cursors = {(sid, name): 0 for sid in present
                   for name in ("et", "tumor", "empty")}
        emitted = 0
        while emitted < shortfall:
            progressed = False
            for sid in present:
                if emitted >= shortfall:
                    break
                if counts[sid] >= cap:
                    continue
                # Draw ET copies only while the plan is below the ET floor;
                # past it, fall back to general tumour slices rather than
                # duplicating a handful of ET slices dozens of times.
                if et_count < et_floor and selected_et[sid]:
                    pool, name = selected_et[sid], "et"
                else:
                    pool, name = selected_tumor[sid], "tumor"
                    if not pool:
                        pool, name = selected_et[sid], "et"
                if not pool:
                    if empty_count >= max_empty:
                        continue
                    pool, name = selected_empty[sid], "empty"
                if not pool:
                    continue
                is_et = name == "et"
                is_empty = name == "empty"

                plane, idx = pool[cursors[(sid, name)] % len(pool)]
                cursors[(sid, name)] += 1
                aug_seed_counter += 1
                entries.append(_entry(hospital, sid, plane, idx, True, aug_seed_counter))
                counts[sid] += 1
                empty_count += is_empty
                et_count += is_et
                emitted += 1
                progressed = True
            if not progressed:
                break

    if len(entries) < target:
        LOGGER.warning(
            "%s: emitted %d of the %d target entries (per-patient cap %d over %d patients)",
            hospital, len(entries), target, cap, n_patients,
        )

    tumor_sets = {sid: _tumor_set(cache_index, sid) for sid in present}
    et_sets = {sid: _et_set(cache_index, sid) for sid in present}
    n_empty = sum(
        1 for e in entries
        if (e["plane"], e["slice_index"]) not in tumor_sets[e["source_subject_id"]]
    )
    n_et = sum(
        1 for e in entries
        if (e["plane"], e["slice_index"]) in et_sets[e["source_subject_id"]]
    )
    if avail_et and n_et < et_floor:
        LOGGER.warning(
            "%s: only %d ET slices reached the plan against a floor of %d "
            "(%d available in the cache)",
            hospital, n_et, et_floor, avail_et,
        )
    per_patient = [counts[sid] for sid in present]
    stats = {
        "hospital": hospital,
        "subjects": n_patients,
        "subjects_missing_from_cache": missing,
        "entries": len(entries),
        "target": target,
        "tumor_entries": len(entries) - n_empty,
        "empty_entries": n_empty,
        "empty_fraction": (n_empty / len(entries)) if entries else 0.0,
        "et_entries": n_et,
        "et_fraction": (n_et / len(entries)) if entries else 0.0,
        "et_available": avail_et,
        "et_floor": et_floor,
        "achievable": achievable,
        "augmented_entries": len(entries) - n_original,
        "per_patient_cap": cap,
        "max_per_patient": max(per_patient) if per_patient else 0,
        "median_per_patient": int(np.median(per_patient)) if per_patient else 0,
        "min_per_patient": min(per_patient) if per_patient else 0,
        "seed": seed,
    }
    return {"entries": entries, "stats": stats}


def _mask_set(cache_index, subject_id, key) -> set:
    """``{(plane, slice_index)}`` for one of the index's slice-index lists."""
    return {
        (plane, int(i))
        for plane, summary in cache_index.get(subject_id, {}).items()
        for i in summary[key]
    }


def _tumor_set(cache_index, subject_id) -> set:
    """``{(plane, slice_index)}`` of tumour-bearing slices for one subject."""
    return _mask_set(cache_index, subject_id, "tumor")


def _et_set(cache_index, subject_id) -> set:
    """``{(plane, slice_index)}`` of enhancing-tumour slices for one subject."""
    return _mask_set(cache_index, subject_id, "et")


def build_plans(cfg: Config, cache_dir: Path | None = None) -> dict[str, dict]:
    """Build a plan for every hospital. Held-out subjects are never included."""
    cache_dir = Path(cache_dir) if cache_dir else cfg.resolve("cache_2d")
    cache_index = index_cache(cache_dir, cfg.planes)
    manifest_dir = cfg.resolve("manifests")

    heldout = set(load_manifest(manifest_dir, "heldout"))
    plans = {}
    for hospital in HOSPITALS:
        subject_ids = load_manifest(manifest_dir, hospital)
        leaked = sorted(set(subject_ids) & heldout)
        if leaked:
            raise AssertionError(
                f"{hospital} shares subjects with the held-out manifest: {leaked}"
            )
        plans[hospital] = build_hospital_plan(hospital, subject_ids, cache_index, cfg)
    return plans


def write_plan(plan: dict, path: Path) -> Path:
    """Write one hospital's plan as JSON. Byte-identical for a fixed seed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"stats": plan["stats"], "entries": plan["entries"]}
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return path


def read_plan(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def format_report(plans: dict[str, dict]) -> str:
    """The per-hospital table printed by ``run.py plan --report``."""
    cols = [
        ("hospital", "hospital", 10),
        ("subjects", "subjects", 9),
        ("entries", "entries", 8),
        ("tumor", "tumor_entries", 8),
        ("ET", "et_entries", 7),
        ("ET%", "et_fraction", 6),
        ("empty", "empty_entries", 8),
        ("empty%", "empty_fraction", 8),
        ("augment", "augmented_entries", 8),
        ("cap", "per_patient_cap", 6),
        ("max/pt", "max_per_patient", 7),
        ("med/pt", "median_per_patient", 7),
    ]
    header = "  ".join(f"{title:>{w}}" for title, _, w in cols)
    lines = [header, "-" * len(header)]
    for hospital in sorted(plans):
        stats = plans[hospital]["stats"]
        cells = []
        for _, key, w in cols:
            value = stats[key]
            if key in ("empty_fraction", "et_fraction"):
                cells.append(f"{value * 100:>{w}.1f}")
            else:
                cells.append(f"{value:>{w}}")
        lines.append("  ".join(cells))
    return "\n".join(lines)
