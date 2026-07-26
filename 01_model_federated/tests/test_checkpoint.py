import pytest
import torch

from src.checkpoint import latest_checkpoint, load_checkpoint, save_checkpoint
from src.config import TrainConfig
from src.model import build_model
from src.train_single import train_single_client


def test_save_and_load_roundtrip(tmp_path):
    model = build_model()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    save_checkpoint(str(tmp_path), "run1", epoch=0, model_state=model.state_dict(),
                     optimizer_state=opt.state_dict())
    save_checkpoint(str(tmp_path), "run1", epoch=1, model_state=model.state_dict(),
                     optimizer_state=opt.state_dict())

    latest = latest_checkpoint(str(tmp_path), "run1")
    assert latest.name == "epoch_1.pt"

    ckpt = load_checkpoint(str(tmp_path), "run1")
    assert ckpt["epoch"] == 1


def test_load_checkpoint_missing_returns_none(tmp_path):
    assert load_checkpoint(str(tmp_path), "no_such_run") is None


def test_corrupted_checkpoint_raises(tmp_path):
    run_dir = tmp_path / "bad_run"
    run_dir.mkdir()
    (run_dir / "epoch_0.pt").write_text("not a real checkpoint")
    with pytest.raises(RuntimeError):
        load_checkpoint(str(tmp_path), "bad_run")


def test_resume_continues_from_next_epoch(tmp_path, small_manifest):
    manifest_path = small_manifest("hospA", 2)
    config = TrainConfig(run_id="resume_run", checkpoint_dir=str(tmp_path / "ckpt"))

    model, losses_1 = train_single_client(config, manifest_path, num_epochs=2)
    latest = latest_checkpoint(config.checkpoint_dir, config.run_id)
    assert latest.name == "epoch_1.pt"

    _model, losses_2 = train_single_client(config, manifest_path, num_epochs=1, resume=True)
    latest_after_resume = latest_checkpoint(config.checkpoint_dir, config.run_id)
    assert latest_after_resume.name == "epoch_2.pt"
    assert len(losses_2) == 1


def test_resume_without_checkpoint_warns_and_starts_fresh(tmp_path, small_manifest):
    manifest_path = small_manifest("hospA", 2)
    config = TrainConfig(run_id="fresh_resume", checkpoint_dir=str(tmp_path / "ckpt"))
    with pytest.warns(UserWarning):
        _model, losses = train_single_client(config, manifest_path, num_epochs=1, resume=True)
    assert len(losses) == 1
    latest = latest_checkpoint(config.checkpoint_dir, config.run_id)
    assert latest.name == "epoch_0.pt"


def test_interrupted_run_matches_uninterrupted_run_loss(tmp_path, small_manifest):
    manifest_path = small_manifest("hospA", 2)

    torch.manual_seed(0)
    config_uninterrupted = TrainConfig(run_id="uninterrupted", checkpoint_dir=str(tmp_path / "u"))
    _model_u, losses_uninterrupted = train_single_client(config_uninterrupted, manifest_path, num_epochs=2)

    torch.manual_seed(0)
    config_interrupted = TrainConfig(run_id="interrupted", checkpoint_dir=str(tmp_path / "i"))
    train_single_client(config_interrupted, manifest_path, num_epochs=1)
    _model_i, losses_interrupted_part2 = train_single_client(
        config_interrupted, manifest_path, num_epochs=1, resume=True
    )

    assert abs(losses_uninterrupted[1] - losses_interrupted_part2[0]) < 1e-3


def test_resume_restores_model_and_optimizer_state(tmp_path, small_manifest):
    manifest_path = small_manifest("hospA", 2)
    config = TrainConfig(run_id="state_check", checkpoint_dir=str(tmp_path / "ckpt"))

    model, _ = train_single_client(config, manifest_path, num_epochs=1)
    saved_state = {k: v.clone() for k, v in model.state_dict().items()}

    ckpt = load_checkpoint(config.checkpoint_dir, config.run_id)
    for key, saved_val in saved_state.items():
        assert torch.equal(ckpt["model_state"][key], saved_val)
