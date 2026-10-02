"""Unit tests for src/best_checkpoint.py. Pure logic: no training, no real data, no torch."""
import pytest

from src.best_checkpoint import (
    BestCheckpointTracker,
    copy_to_best,
    evaluate_and_track,
    is_new_best,
    worst_region,
)


# --- the three cases the handoff doc requires -------------------------------------------

def test_first_epoch_becomes_best():
    assert is_new_best(None, 0.10)
    t = BestCheckpointTracker()
    assert t.update(2, {"WT": 0.9, "NC": 0.8, "ET": 0.5}) is True
    assert (t.best_epoch, t.best_score, t.best_region) == (2, 0.5, "ET")


def test_worse_score_does_not_overwrite_best():
    t = BestCheckpointTracker()
    t.update(2, {"WT": 0.9, "ET": 0.70})
    assert t.update(4, {"WT": 0.95, "ET": 0.65}) is False
    assert (t.best_epoch, t.best_score) == (2, 0.70)
    assert t.last_message is None


def test_tie_does_not_flip_flop():
    assert not is_new_best(0.70, 0.70)
    t = BestCheckpointTracker()
    t.update(2, {"WT": 0.9, "ET": 0.70})
    assert t.update(4, {"WT": 0.8, "ET": 0.70}) is False
    assert t.best_epoch == 2


# --- the rest of the selection behaviour -------------------------------------------------

def test_better_score_becomes_best():
    t = BestCheckpointTracker()
    t.update(2, {"WT": 0.9, "ET": 0.60})
    assert t.update(4, {"WT": 0.8, "ET": 0.75}) is True
    assert (t.best_epoch, t.best_score) == (4, 0.75)


def test_uses_worst_region_not_mean():
    # Epoch 2 has the higher mean but the worse worst-region; it must lose.
    t = BestCheckpointTracker()
    t.update(2, {"WT": 0.99, "NC": 0.99, "ET": 0.40})
    assert t.update(4, {"WT": 0.70, "NC": 0.70, "ET": 0.60}) is True
    assert t.best_epoch == 4


def test_worst_region_returns_name_and_score():
    assert worst_region({"WT": 0.9, "NC": 0.71, "ET": 0.8}) == ("NC", 0.71)


def test_worst_region_empty_raises():
    with pytest.raises(ValueError):
        worst_region({})


def test_message_format():
    t = BestCheckpointTracker()
    t.update(86, {"WT": 0.9, "NC": 0.68})
    t.update(87, {"WT": 0.9, "NC": 0.71})
    assert t.last_message == "epoch 87: new best (worst region NC 0.71 > previous 0.68)"


# --- persistence (resume) ----------------------------------------------------------------

def test_tracker_roundtrips_through_disk(tmp_path):
    t = BestCheckpointTracker()
    t.update(7, {"WT": 0.9, "ET": 0.55})
    t.save(tmp_path / "best.json")
    t2 = BestCheckpointTracker.load(tmp_path / "best.json")
    assert (t2.best_score, t2.best_region, t2.best_epoch) == (0.55, "ET", 7)
    # a resumed run must not overwrite a better pre-resume best with a worse score
    assert t2.update(9, {"WT": 0.9, "ET": 0.50}) is False


def test_load_missing_file_gives_fresh_tracker(tmp_path):
    t = BestCheckpointTracker.load(tmp_path / "nope.json")
    assert t.best_score is None


def test_load_corrupt_file_warns_and_starts_fresh(tmp_path):
    (tmp_path / "best.json").write_text("{not json")
    with pytest.warns(UserWarning):
        t = BestCheckpointTracker.load(tmp_path / "best.json")
    assert t.best_score is None


# --- copying + the evaluate step ---------------------------------------------------------

def test_copy_to_best_copies_and_keeps_original(tmp_path):
    src = tmp_path / "epoch_4.pt"
    src.write_bytes(b"fake-checkpoint")
    dest = copy_to_best(src, tmp_path)
    assert dest == tmp_path / "best.pt"
    assert dest.read_bytes() == b"fake-checkpoint"
    assert src.exists()


class _FakeModel:
    def __init__(self):
        self.modes = []

    def eval(self):
        self.modes.append("eval")

    def train(self):
        self.modes.append("train")


def test_evaluate_and_track_writes_best_only_on_improvement(tmp_path, capsys):
    model, tracker = _FakeModel(), BestCheckpointTracker()
    scores = iter([{"ET": 0.5, "NC": 0.7}, {"ET": 0.4, "NC": 0.9}, {"ET": 0.6, "NC": 0.7}])
    ckpts = {}
    for epoch in (1, 3, 5):
        p = tmp_path / f"epoch_{epoch}.pt"
        p.write_bytes(f"ckpt-{epoch}".encode())
        ckpts[epoch] = p

    assert evaluate_and_track(model, lambda m: next(scores), tracker, 1, ckpts[1], tmp_path) is True
    assert (tmp_path / "best.pt").read_bytes() == b"ckpt-1"
    assert evaluate_and_track(model, lambda m: next(scores), tracker, 3, ckpts[3], tmp_path) is False
    assert (tmp_path / "best.pt").read_bytes() == b"ckpt-1"          # worse: untouched
    assert evaluate_and_track(model, lambda m: next(scores), tracker, 5, ckpts[5], tmp_path) is True
    assert (tmp_path / "best.pt").read_bytes() == b"ckpt-5"
    assert BestCheckpointTracker.load(tmp_path / "best.json").best_epoch == 5
    assert all(c.exists() for c in ckpts.values())                    # per-epoch files kept
    assert model.modes == ["eval", "train"] * 3                       # always back in train mode
    assert "epoch 5: new best (worst region ET 0.60 > previous 0.50)" in capsys.readouterr().out


def test_evaluate_and_track_restores_train_mode_if_scorer_raises(tmp_path):
    model = _FakeModel()

    def boom(m):
        raise RuntimeError("scorer failed")

    with pytest.raises(RuntimeError):
        evaluate_and_track(model, boom, BestCheckpointTracker(), 1, tmp_path / "x.pt", tmp_path)
    assert model.modes == ["eval", "train"]
    assert not (tmp_path / "best.pt").exists()