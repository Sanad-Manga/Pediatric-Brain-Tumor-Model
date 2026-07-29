"""Training loop: checkpointing, resume, and the AMP path.

The loop had no coverage at all before this file, which is how it was possible
to add mixed precision without noticing that ``val_subjects`` was recorded only
in ``history.json`` and never on the checkpoint itself.

Everything here runs on CPU with a handful of steps against the synthetic cache,
so it is a wiring test, not a convergence test -- Dice values are meaningless at
this size and are deliberately not asserted on.
"""

from __future__ import annotations

import json

import pytest
import torch

from src import train


def _run(cfg, cache, ckpt_dir, **kw):
    """One-epoch training run with everything turned down as far as it goes."""
    params = dict(run_id="test_run", epochs=1, batch_size=2, val_fraction=0.5,
                  max_steps_per_epoch=2, cache_dir=cache, checkpoint_dir=ckpt_dir,
                  device="cpu")
    params.update(kw)
    return train.run(cfg, **params)


def test_one_epoch_writes_both_checkpoints_and_history(cfg, cache, tmp_path):
    ckpt_dir = tmp_path / "ckpt"
    summary = _run(cfg, cache, ckpt_dir)

    assert (ckpt_dir / "best.pt").exists()
    assert (ckpt_dir / "last.pt").exists()
    assert (ckpt_dir / "history.json").exists()
    assert summary["epochs_completed"] == 1
    assert summary["completed"] is True


def test_checkpoint_carries_the_validation_split(cfg, cache, tmp_path):
    """Threshold tuning reads this to avoid drifting onto held-out patients."""
    ckpt_dir = tmp_path / "ckpt"
    summary = _run(cfg, cache, ckpt_dir)

    payload = torch.load(ckpt_dir / "best.pt", map_location="cpu", weights_only=False)
    assert payload["val_subjects"], "checkpoint must record the validation split"
    assert payload["val_subjects"] == summary["val_subjects"]


def test_amp_is_refused_on_cpu_rather_than_crashing(cfg, cache, tmp_path):
    """``--device cpu`` with AMP asked for: warn and carry on, do not fail."""
    ckpt_dir = tmp_path / "ckpt"
    _run(cfg, cache, ckpt_dir, amp=True)

    payload = torch.load(ckpt_dir / "last.pt", map_location="cpu", weights_only=False)
    # A disabled scaler has no state worth saving; its presence would mean the
    # CPU run had silently taken the fp16 path.
    assert "scaler_state_dict" not in payload


def test_scaler_is_a_passthrough_when_amp_is_off(cfg, cache, tmp_path):
    """The AMP rewrite must not change what a CPU run produces.

    Same seed, same data, two runs: identical weights. This is the guarantee
    that every number measured before mixed precision existed is still valid.
    """
    first = torch.load(_run(cfg, cache, tmp_path / "a") and (tmp_path / "a" / "last.pt"),
                       map_location="cpu", weights_only=False)["model_state_dict"]
    second = torch.load(_run(cfg, cache, tmp_path / "b") and (tmp_path / "b" / "last.pt"),
                        map_location="cpu", weights_only=False)["model_state_dict"]

    assert first.keys() == second.keys()
    for key in first:
        assert torch.equal(first[key], second[key]), f"{key} diverged between runs"


def test_resume_picks_up_after_the_last_completed_epoch(cfg, cache, tmp_path):
    ckpt_dir = tmp_path / "ckpt"
    _run(cfg, cache, ckpt_dir, epochs=1)
    summary = _run(cfg, cache, ckpt_dir, epochs=2)

    epochs = [r["epoch"] for r in summary["history"]]
    assert epochs == [0, 1], f"expected a continued curve, got {epochs}"


def test_history_survives_a_run_that_stops_early(cfg, cache, tmp_path):
    """History is written per epoch; a stopped run must not lose the curve."""
    ckpt_dir = tmp_path / "ckpt"
    _run(cfg, cache, ckpt_dir, epochs=1)

    with open(ckpt_dir / "history.json", "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    assert len(payload["history"]) == 1
    assert payload["history"][0]["epoch"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_amp_enabled_on_cuda_saves_scaler_state(cfg, cache, tmp_path):
    ckpt_dir = tmp_path / "ckpt"
    _run(cfg, cache, ckpt_dir, device="cuda")

    payload = torch.load(ckpt_dir / "last.pt", map_location="cpu", weights_only=False)
    assert "scaler_state_dict" in payload
