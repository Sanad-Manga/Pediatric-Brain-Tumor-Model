import json
from pathlib import Path

import numpy as np

from tools.make_cv_folds import make_folds, write_folds
from tools.run_cv import run_cv


def _fake_seg_cache(tmp_path: Path, patient_ids: list[str], counts: list[int]) -> Path:
    cache = tmp_path / "cache"
    cache.mkdir()
    for patient_id, et_count in zip(patient_ids, counts):
        seg = np.zeros((8, 8, 8), dtype=np.uint8)
        seg.flat[:et_count] = 1
        np.savez_compressed(cache / f"{patient_id}.npz", seg=seg)
    return cache


def test_make_folds_are_stratified_complete_and_repeatable(tmp_path):
    patients = [f"PAT-{index:02d}" for index in range(20)]
    train_ids, heldout_ids = patients[:14], patients[14:]
    et_counts = [0 if index < 5 else index for index in range(20)]
    cache = _fake_seg_cache(tmp_path, patients, et_counts)

    folds = make_folds(train_ids, heldout_ids, cache, seed=23)
    repeated = make_folds(train_ids, heldout_ids, cache, seed=23)
    assert folds == repeated

    all_seen = []
    et_present_counts = []
    size_group_counts = {name: [] for name in ("et_small", "et_medium", "et_large")}
    from tools.make_cv_folds import _stratify_et_sizes, _et_counts

    groups = _stratify_et_sizes(_et_counts(patients, cache))
    for train_fold, val_fold in folds:
        assert len(train_fold) + len(val_fold) == len(patients)
        assert set(train_fold).isdisjoint(val_fold)
        assert set(train_fold) == set(patients) - set(val_fold)
        all_seen.extend(val_fold)
        et_present_counts.append(sum(et_counts[int(subject[-2:])] > 0 for subject in val_fold))
        for group in size_group_counts:
            size_group_counts[group].append(sum(groups[subject] == group for subject in val_fold))

    assert sorted(all_seen) == sorted(patients)
    assert max(et_present_counts) - min(et_present_counts) <= 1
    for counts in size_group_counts.values():
        assert max(counts) - min(counts) <= 1

    out_a, out_b = tmp_path / "folds_a", tmp_path / "folds_b"
    paths_a = write_folds(folds, out_a)
    paths_b = write_folds(repeated, out_b)
    for (train_a, val_a), (train_b, val_b) in zip(paths_a, paths_b):
        assert train_a.read_bytes() == train_b.read_bytes()
        assert val_a.read_bytes() == val_b.read_bytes()


def test_fold_runner_dry_run_prints_five_train_and_score_commands(tmp_path, capsys):
    folds_dir = tmp_path / "folds"
    folds_dir.mkdir()
    patients = [f"P-{index}" for index in range(10)]
    for fold in range(5):
        val_ids = patients[fold * 2:fold * 2 + 2]
        train_ids = [patient for patient in patients if patient not in val_ids]
        (folds_dir / f"fold{fold}_train.json").write_text(json.dumps(train_ids))
        (folds_dir / f"fold{fold}_val.json").write_text(json.dumps(val_ids))

    assert run_cv([
        "--folds-dir", str(folds_dir), "--out-dir", str(tmp_path / "out"), "--dry-run",
        "--data-mode", "dummy", "--epochs", "1",
    ]) == 0
    output = capsys.readouterr().out
    assert output.count("TRAIN fold") == 5
    assert output.count("SCORE fold") == 5
    for fold in range(5):
        assert f"fold{fold}_train.json" in output
        assert f"fold{fold}_val.json" in output


def test_fold_runner_skips_folds_with_checkpoint_and_score(tmp_path, capsys):
    folds_dir = tmp_path / "folds"
    out_dir = tmp_path / "out"
    folds_dir.mkdir()
    for fold in range(5):
        (folds_dir / f"fold{fold}_train.json").write_text(json.dumps([f"train-{fold}"]))
        (folds_dir / f"fold{fold}_val.json").write_text(json.dumps([f"val-{fold}"]))
        checkpoint_dir = out_dir / f"fold{fold}" / "checkpoints" / f"cv_fold{fold}"
        checkpoint_dir.mkdir(parents=True)
        checkpoint = checkpoint_dir / "epoch_0.pt"
        checkpoint.write_bytes(b"complete")
        score = {
            "checkpoint": str(checkpoint),
            "n_subjects_scored": 1,
            "dice": {"dice_ET": 0.1, "dice_NC": 0.2, "dice_WT": 0.3},
        }
        score_path = out_dir / f"fold{fold}" / "score.json"
        score_path.write_text(json.dumps(score))

    assert run_cv(["--folds-dir", str(folds_dir), "--out-dir", str(out_dir)]) == 0
    output = capsys.readouterr().out
    assert output.count("completed checkpoint and score found; skipping") == 5
    summary = json.loads((out_dir / "summary.json").read_text())
    assert len(summary["folds"]) == 5
    assert summary["summary"]["ET"]["mean"] == 0.1
    assert summary["summary"]["ET"]["std"] == 0.0


def test_dummy_cv_run_writes_summary_and_skips_on_rerun(tmp_path, capsys):
    folds_dir = tmp_path / "folds"
    folds_dir.mkdir()
    patients = [f"DUMMY-{index}" for index in range(10)]
    for fold in range(5):
        val_ids = patients[fold * 2:fold * 2 + 2]
        train_ids = [patient for patient in patients if patient not in val_ids]
        (folds_dir / f"fold{fold}_train.json").write_text(json.dumps(train_ids))
        (folds_dir / f"fold{fold}_val.json").write_text(json.dumps(val_ids))

    config_path = tmp_path / "tiny.yaml"
    config_path.write_text(
        "model:\n  width: 1\n  depth: 2\n"
        "loss:\n  kind: dice_ce\n  class_weights: null\n"
        "schedule:\n  kind: none\n  min_lr: 1.0e-5\n"
    )
    args = [
        "--folds-dir", str(folds_dir), "--out-dir", str(tmp_path / "out"),
        "--config", str(config_path), "--data-mode", "dummy", "--epochs", "1",
    ]

    assert run_cv(args) == 0
    first_output = capsys.readouterr().out
    summary_path = tmp_path / "out" / "summary.json"
    summary = json.loads(summary_path.read_text())
    assert len(summary["folds"]) == 5
    assert all(fold["n_subjects_scored"] == 2 for fold in summary["folds"])
    assert set(summary["summary"]) == {"ET", "NC", "WT"}
    assert all((tmp_path / "out" / f"fold{fold}" / "score.json").is_file() for fold in range(5))
    assert all(
        (tmp_path / "out" / f"fold{fold}" / "checkpoints" / f"cv_fold{fold}" / "epoch_0.pt").is_file()
        for fold in range(5)
    )

    assert run_cv(args) == 0
    second_output = capsys.readouterr().out
    assert second_output.count("completed checkpoint and score found; skipping") == 5


# ---- follow-up to #46: a manifest patient with no cache file is excluded with a warning, not a crash ----
def test_patients_missing_from_the_cache_are_excluded_and_reported(tmp_path):
    import warnings
    patients = [f"PAT-{index:02d}" for index in range(21)]
    train_ids, heldout_ids = patients[:14], patients[14:]
    present = [p for p in patients if p != "PAT-20"]                      # PAT-20 has no cache file
    cache = _fake_seg_cache(tmp_path, present, [0 if i < 5 else i for i in range(20)])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        folds = make_folds(train_ids, heldout_ids, cache, seed=1)
    assert any("PAT-20" in str(w.message) for w in caught)
    seen = sorted(p for _train, val in folds for p in val)
    assert seen == sorted(present)
    assert all("PAT-20" not in train for train, _val in folds)


def test_cli_writes_the_excluded_list(tmp_path, monkeypatch):
    import tools.make_cv_folds as m
    patients = [f"PAT-{index:02d}" for index in range(21)]
    (tmp_path / "train.json").write_text(json.dumps(patients[:14]))
    (tmp_path / "heldout.json").write_text(json.dumps(patients[14:]))
    cache = _fake_seg_cache(tmp_path, patients[:20], [0 if i < 5 else i for i in range(20)])
    monkeypatch.setattr(m, "DEFAULT_TRAIN_MANIFEST", tmp_path / "train.json")
    monkeypatch.setattr(m, "DEFAULT_HELDOUT_MANIFEST", tmp_path / "heldout.json")
    assert m.main(["--cache-path", str(cache), "--out-dir", str(tmp_path / "out")]) == 0
    assert json.loads((tmp_path / "out" / "excluded.json").read_text()) == ["PAT-20"]


def test_a_cache_file_without_seg_still_fails_loudly(tmp_path):
    import pytest
    patients = [f"PAT-{index:02d}" for index in range(20)]
    cache = _fake_seg_cache(tmp_path, patients, [0 if i < 5 else i for i in range(20)])
    np.savez_compressed(cache / "PAT-03.npz", image=np.zeros(3))         # corrupt: present but no 'seg'
    with pytest.raises(ValueError, match="seg"):
        make_folds(patients[:14], patients[14:], cache, seed=1)
