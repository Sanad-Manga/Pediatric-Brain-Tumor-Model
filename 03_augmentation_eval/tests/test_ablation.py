"""Ablation protocol invariants.

The runs themselves need a GPU and hours, so what is testable here is the part
that decides whether the table means anything: the conditions must differ in
exactly one thing at a time, must respect the config's gating contract, and
already-measured conditions must not be silently re-measured into duplicate rows.
"""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "run_ablation", ROOT / "tools" / "run_ablation.py")
run_ablation = importlib.util.module_from_spec(_spec)
sys.modules["run_ablation"] = run_ablation
_spec.loader.exec_module(run_ablation)

from src.evaluate import CSV_COLUMNS, append_result_row  # noqa: E402


def test_condition_names_are_unique():
    names = [n for n, _, _ in run_ablation.CONDITIONS]
    assert len(names) == len(set(names))


def test_mixup_is_never_enabled_without_augmentation():
    """`use_augmentation` masters `use_mixup` (SPEC Req 23).

    A condition asking for mixup with augmentation off would silently get no
    mixup, and the row would claim to measure something it did not.
    """
    for name, use_aug, use_mixup in run_ablation.CONDITIONS:
        if use_mixup:
            assert use_aug, f"{name} enables mixup with augmentation off"


def test_conditions_change_one_variable_at_a_time():
    """Consecutive conditions must differ in exactly one flag, or the table
    cannot attribute a Dice difference to any single cause."""
    flags = [(a, m) for _, a, m in run_ablation.CONDITIONS]
    for before, after in zip(flags, flags[1:]):
        changed = sum(1 for x, y in zip(before, after) if x != y)
        assert changed == 1, f"{before} -> {after} changes {changed} flags"


def test_completed_experiments_reads_existing_rows(tmp_path):
    csv_path = tmp_path / "ablation.csv"
    row = {c: 0.0 for c in CSV_COLUMNS}
    row.update(experiment_name="no_augmentation", use_augmentation=False,
               use_federation=False, use_domain_adaptation=False)
    append_result_row(row, csv_path)

    assert run_ablation.completed_experiments(csv_path) == {"no_augmentation"}


def test_completed_experiments_on_a_missing_or_empty_file(tmp_path):
    assert run_ablation.completed_experiments(tmp_path / "nope.csv") == set()
    empty = tmp_path / "empty.csv"
    empty.touch()
    assert run_ablation.completed_experiments(empty) == set()


def test_results_csv_rejects_a_foreign_header(tmp_path):
    """The schema is fixed by CONTRACTS.md; appending misaligned columns would
    corrupt the table rather than fail loudly."""
    csv_path = tmp_path / "wrong.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        csv.writer(fh).writerow(["something", "else"])

    row = {c: 0.0 for c in CSV_COLUMNS}
    row["experiment_name"] = "x"
    with pytest.raises(ValueError, match="unexpected header"):
        append_result_row(row, csv_path)
