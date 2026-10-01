#!/usr/bin/env python
"""Create deterministic, stratified five-fold manifests from the labelled cohort."""
from __future__ import annotations

import argparse
import itertools
import json
import random
from pathlib import Path

import numpy as np

SEC01 = Path(__file__).resolve().parent.parent
REPO_ROOT = SEC01.parent
DEFAULT_TRAIN_MANIFEST = REPO_ROOT / "00_shared" / "manifests" / "hospitalA_plus_B.json"
DEFAULT_HELDOUT_MANIFEST = REPO_ROOT / "00_shared" / "manifests" / "heldout.json"
DEFAULT_OUT_DIR = REPO_ROOT / "00_shared" / "manifests" / "cv5"
N_FOLDS = 5
STRATA = ("no_et", "et_small", "et_medium", "et_large")


def _read_ids(path: str | Path) -> list[str]:
    with Path(path).open("r", encoding="utf-8") as file:
        ids = json.load(file)
    if not isinstance(ids, list) or not all(isinstance(value, str) for value in ids):
        raise ValueError(f"manifest must be a flat JSON list of patient IDs: {path}")
    if len(ids) != len(set(ids)):
        raise ValueError(f"manifest contains duplicate patient IDs: {path}")
    return ids


def _et_counts(patient_ids: list[str], cache_path: str | Path) -> dict[str, int]:
    cache = Path(cache_path)
    result = {}
    for patient_id in patient_ids:
        path = cache / f"{patient_id}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"ET-stratification cache not found: {path}")
        with np.load(path) as data:
            if "seg" not in data:
                raise ValueError(f"cache file has no 'seg' array: {path}")
            result[patient_id] = int(np.count_nonzero(data["seg"] == 1))
    return result


def _stratify_et_sizes(et_counts: dict[str, int]) -> dict[str, str]:
    positive = sorted(
        ((count, patient_id) for patient_id, count in et_counts.items() if count > 0),
        key=lambda item: (item[0], item[1]),
    )
    strata = {patient_id: "no_et" for patient_id, count in et_counts.items() if count == 0}
    n_positive = len(positive)
    for rank, (_count, patient_id) in enumerate(positive):
        group = min(rank * 3 // n_positive, 2)
        strata[patient_id] = ("et_small", "et_medium", "et_large")[group]
    return strata


def _fold_stratum_counts(stratum_sizes: list[int], seed: int) -> list[list[int]]:
    """Allocate stratum remainders while balancing total and ET-positive counts."""
    base = [size // N_FOLDS for size in stratum_sizes]
    remainders = [size % N_FOLDS for size in stratum_sizes]
    remainder_places = [tuple(itertools.combinations(range(N_FOLDS), r)) for r in remainders]
    valid = []

    for selected in itertools.product(*remainder_places):
        counts = [[base[group] for group in range(len(STRATA))] for _ in range(N_FOLDS)]
        for group, folds in enumerate(selected):
            for fold in folds:
                counts[fold][group] += 1
        fold_sizes = [sum(row) for row in counts]
        et_present = [sum(row[1:]) for row in counts]
        if max(et_present) - min(et_present) > 1:
            continue
        if max(fold_sizes) - min(fold_sizes) > 1:
            continue
        size_variance = sum((size * N_FOLDS - sum(stratum_sizes)) ** 2 for size in fold_sizes)
        et_variance = sum((count * N_FOLDS - sum(stratum_sizes[1:])) ** 2 for count in et_present)
        valid.append((size_variance + et_variance, counts))

    if not valid:
        raise ValueError(
            "could not distribute strata with per-fold patient and ET-present counts differing by at most one"
        )
    best_score = min(score for score, _counts in valid)
    tied = [counts for score, counts in valid if score == best_score]
    return random.Random(seed).choice(tied)


def make_folds(
    train_ids: list[str],
    heldout_ids: list[str],
    cache_path: str | Path,
    seed: int = 42,
) -> list[tuple[list[str], list[str]]]:
    """Return (train, validation) ID lists for five folds over exactly the supplied cohort."""
    if not train_ids or not heldout_ids:
        raise ValueError("both input manifests must contain patients")
    if len(train_ids) != len(set(train_ids)) or len(heldout_ids) != len(set(heldout_ids)):
        raise ValueError("input manifests must not contain duplicate patient IDs")
    overlap = set(train_ids) & set(heldout_ids)
    if overlap:
        raise ValueError(f"input manifests overlap: {sorted(overlap)[:5]}")

    all_ids = train_ids + heldout_ids
    et_counts = _et_counts(all_ids, cache_path)
    strata_by_id = _stratify_et_sizes(et_counts)
    stratum_ids = {
        stratum: [patient_id for patient_id in all_ids if strata_by_id[patient_id] == stratum]
        for stratum in STRATA
    }
    allocations = _fold_stratum_counts([len(stratum_ids[s]) for s in STRATA], seed)

    rng = random.Random(seed)
    fold_validation: list[list[str]] = [[] for _ in range(N_FOLDS)]
    for group_index, stratum in enumerate(STRATA):
        subjects = list(stratum_ids[stratum])
        rng.shuffle(subjects)
        offset = 0
        for fold in range(N_FOLDS):
            count = allocations[fold][group_index]
            fold_validation[fold].extend(subjects[offset:offset + count])
            offset += count

    folds = []
    for val_ids in fold_validation:
        rng.shuffle(val_ids)
        val_set = set(val_ids)
        folds.append(([patient_id for patient_id in all_ids if patient_id not in val_set], val_ids))
    return folds


def write_folds(
    folds: list[tuple[list[str], list[str]]], out_dir: str | Path
) -> list[tuple[Path, Path]]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for fold, (train_ids, val_ids) in enumerate(folds):
        train_path = out / f"fold{fold}_train.json"
        val_path = out / f"fold{fold}_val.json"
        train_path.write_text(json.dumps(train_ids, indent=2) + "\n", encoding="utf-8")
        val_path.write_text(json.dumps(val_ids, indent=2) + "\n", encoding="utf-8")
        paths.append((train_path, val_path))
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-path", required=True, help="directory containing <patient-id>.npz files with key 'seg'")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args(argv)

    train_ids = _read_ids(DEFAULT_TRAIN_MANIFEST)
    heldout_ids = _read_ids(DEFAULT_HELDOUT_MANIFEST)
    folds = make_folds(train_ids, heldout_ids, args.cache_path, seed=args.seed)
    for train_path, val_path in write_folds(folds, args.out_dir):
        print(f"wrote {train_path} and {val_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
