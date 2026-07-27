"""Ratio-preserving expansion of each hospital to exactly 150 training subjects.

The rules, in order of precedence:

1. Hospitals are expanded **independently**. Hospital A's plan never sees
   Hospital B's ratios and vice versa — the goal is to preserve each hospital's
   own real distribution, not to harmonize them.
2. Every original subject is kept. Real data is never dropped.
3. The per-stratum budget is `150 * n_s / N`, rounded by largest remainder so
   the targets sum to exactly 150.
4. The shortfall in a stratum is filled by augmented copies of source subjects
   drawn from *that same stratum in that same hospital*.
5. Held-out subjects never enter a plan.

Augmented entries are **virtual**: the plan stores `(source_subject_id, aug_seed)`
and the Dataset materializes the volume on the fly. The plan JSON is the durable
provenance record.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

from .config import Config, load_config
from .data import load_all_manifests
from .strata import read_strata_csv, strata_by_hospital

LOGGER = logging.getLogger(__name__)

HOSPITALS = ("A", "B")


def largest_remainder(counts: dict[str, int], total: int) -> dict[str, int]:
    """Allocate ``total`` across strata proportionally, summing to exactly ``total``.

    Strata with zero subjects receive zero (no divide-by-zero, and we cannot
    augment a stratum with no source subjects anyway).
    """
    n = sum(counts.values())
    if n == 0:
        raise ValueError("cannot allocate a budget across zero subjects")

    exact = {k: (total * v / n) for k, v in counts.items()}
    floors = {k: int(v) for k, v in exact.items()}
    remainder = total - sum(floors.values())

    # Distribute the remainder to the largest fractional parts; ties broken by
    # stratum name so the allocation is deterministic.
    order = sorted(exact, key=lambda k: (-(exact[k] - floors[k]), k))
    for k in order[:remainder]:
        floors[k] += 1
    return floors


def _rebalance_no_shrink(targets: dict[str, int], counts: dict[str, int], total: int) -> dict[str, int]:
    """Raise any target below its original count, taking the difference elsewhere.

    Real data is never dropped, so a stratum can only grow. If proportional
    allocation would shrink one, we clamp it to its original count and reclaim
    the excess from the strata that still have augmented headroom, keeping the
    grand total at exactly ``total``.
    """
    adjusted = dict(targets)
    debt = 0
    for stratum, target in adjusted.items():
        original = counts.get(stratum, 0)
        if target < original:
            LOGGER.warning(
                "stratum %r target %d is below its %d originals; clamping (real data is never dropped)",
                stratum,
                target,
                original,
            )
            debt += original - target
            adjusted[stratum] = original

    while debt > 0:
        # Take from whichever stratum currently has the most augmented headroom.
        donors = [s for s in adjusted if adjusted[s] > counts.get(s, 0)]
        if not donors:
            raise ValueError(
                f"cannot reach a total of {total}: originals already exceed the budget"
            )
        donor = max(donors, key=lambda s: (adjusted[s] - counts.get(s, 0), s))
        adjusted[donor] -= 1
        debt -= 1

    if sum(adjusted.values()) != total:
        raise AssertionError(f"rebalance produced {sum(adjusted.values())}, expected {total}")
    return adjusted


def plan_expansion(
    hospital: str,
    strata_rows: list[dict],
    target: int = 150,
    seed: int = 1337,
) -> dict:
    """Build one hospital's expansion plan.

    Returns a dict with ``hospital``, ``target``, ``seed``, ``strata`` (the
    per-stratum accounting) and ``entries`` (the 150 plan rows).
    """
    grouped = strata_by_hospital(strata_rows, hospital)
    if not grouped:
        raise ValueError(f"no subjects found for hospital {hospital!r} in the strata table")

    counts = {stratum: len(ids) for stratum, ids in grouped.items()}
    n_original = sum(counts.values())

    targets = largest_remainder(counts, target)
    targets = _rebalance_no_shrink(targets, counts, target)

    rng = random.Random(f"{seed}:{hospital}")
    entries: list[dict] = []
    accounting: list[dict] = []

    for stratum in sorted(grouped):
        sources = sorted(grouped[stratum])
        n_orig = len(sources)
        n_target = targets[stratum]
        n_aug = n_target - n_orig

        for sid in sources:
            entries.append(
                {
                    "hospital": hospital,
                    "source_subject_id": sid,
                    "stratum": stratum,
                    "is_augmented": False,
                    "aug_seed": None,
                    "sample_id": sid,
                }
            )

        for i in range(n_aug):
            src = rng.choice(sources)
            aug_seed = rng.randrange(2**31 - 1)
            entries.append(
                {
                    "hospital": hospital,
                    "source_subject_id": src,
                    "stratum": stratum,
                    "is_augmented": True,
                    "aug_seed": aug_seed,
                    "sample_id": f"{src}__aug{i:03d}",
                }
            )

        accounting.append(
            {
                "stratum": stratum,
                "n_original": n_orig,
                "original_prop": n_orig / n_original,
                "n_expanded": n_target,
                "expanded_prop": n_target / target,
                "n_augmented": n_aug,
            }
        )

    if len(entries) != target:
        raise AssertionError(
            f"hospital {hospital} plan has {len(entries)} entries, expected exactly {target}"
        )

    return {
        "hospital": hospital,
        "target": target,
        "seed": seed,
        "n_original": n_original,
        "strata": accounting,
        "entries": entries,
    }


def assert_no_leakage(plans: dict[str, dict], manifest_dir: str | Path) -> None:
    """Assert three-way disjointness of A / B / held-out at source-subject level."""
    manifests = load_all_manifests(manifest_dir)
    heldout = set(manifests["heldout"])

    sources = {
        h: {e["source_subject_id"] for e in plan["entries"]} for h, plan in plans.items()
    }

    for hospital, ids in sources.items():
        overlap = ids & heldout
        if overlap:
            raise AssertionError(
                f"LEAKAGE: hospital {hospital} plan contains held-out subject(s) {sorted(overlap)}"
            )

    names = sorted(sources)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            overlap = sources[a] & sources[b]
            if overlap:
                raise AssertionError(
                    f"LEAKAGE: hospitals {a} and {b} share subject(s) {sorted(overlap)}"
                )


def write_plan(plan: dict, plans_dir: str | Path) -> Path:
    plans_dir = Path(plans_dir)
    plans_dir.mkdir(parents=True, exist_ok=True)
    path = plans_dir / f"hospital{plan['hospital']}_plan.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(plan, fh, indent=2)
    return path


def read_plan(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"expansion plan not found: {path}. Run `python -m src.expansion` first."
        )
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def build_all_plans(cfg: Config, strata_rows: list[dict] | None = None) -> dict[str, dict]:
    """Build (and leakage-check) plans for both hospitals."""
    rows = strata_rows if strata_rows is not None else read_strata_csv(cfg.resolve("strata_csv"))
    target = int(cfg.expansion.get("target_per_hospital", 150))
    seed = int(cfg.expansion.get("seed", 1337))
    plans = {h: plan_expansion(h, rows, target=target, seed=seed) for h in HOSPITALS}
    assert_no_leakage(plans, cfg.resolve("manifests"))
    return plans


def format_report(plans: dict[str, dict]) -> str:
    lines = []
    for hospital in sorted(plans):
        plan = plans[hospital]
        lines.append(f"\nHospital {hospital}: {plan['n_original']} originals -> {plan['target']} expanded")
        lines.append(f"  {'stratum':<20} {'orig':>5} {'orig%':>8} {'exp':>5} {'exp%':>8} {'aug':>5} {'drift(pp)':>10}")
        for row in plan["strata"]:
            drift = (row["expanded_prop"] - row["original_prop"]) * 100
            lines.append(
                f"  {row['stratum']:<20} {row['n_original']:>5} {row['original_prop']:>7.2%} "
                f"{row['n_expanded']:>5} {row['expanded_prop']:>7.2%} {row['n_augmented']:>5} {drift:>+10.3f}"
            )
        total = sum(r["n_expanded"] for r in plan["strata"])
        lines.append(f"  {'TOTAL':<20} {plan['n_original']:>5} {'':>8} {total:>5}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument("--report", action="store_true", help="print the per-hospital ratio table")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config(args.config)
    plans = build_all_plans(cfg)

    plans_dir = cfg.resolve("plans_dir")
    for plan in plans.values():
        path = write_plan(plan, plans_dir)
        print(f"wrote {len(plan['entries'])} entries -> {path}")

    if args.report:
        print(format_report(plans))
    return 0


if __name__ == "__main__":
    sys.exit(main())
