import pytest

from src.config import TrainConfig
from src.train_single import train_single_client


def test_single_client_runs_and_logs_loss(tmp_path, small_manifest):
    manifest_path = small_manifest("hospA", 2)
    config = TrainConfig(run_id="t1", checkpoint_dir=str(tmp_path / "ckpt"))
    _model, losses = train_single_client(config, manifest_path, num_epochs=2)
    assert len(losses) == 2
    assert all(isinstance(l, float) for l in losses)


def test_augmentation_flag_gates_transform_calls(tmp_path, small_manifest):
    manifest_path = small_manifest("hospA", 2)
    call_count = {"n": 0}

    def transform(x, y):
        call_count["n"] += 1
        return x, y

    config_on = TrainConfig(
        run_id="aug_on", checkpoint_dir=str(tmp_path / "ckpt_on"), use_augmentation=True
    )
    train_single_client(config_on, manifest_path, num_epochs=1, augmentation_transform=transform)
    assert call_count["n"] == 2  # one call per sample, one epoch

    call_count["n"] = 0
    config_off = TrainConfig(
        run_id="aug_off", checkpoint_dir=str(tmp_path / "ckpt_off"), use_augmentation=False
    )
    train_single_client(config_off, manifest_path, num_epochs=1, augmentation_transform=transform)
    assert call_count["n"] == 0


def test_non_unit_batch_size_rejected():
    with pytest.raises(ValueError):
        TrainConfig(batch_size=2)


def test_domain_adaptation_flag_is_noop(tmp_path, small_manifest):
    manifest_path = small_manifest("hospA", 2)
    config_true = TrainConfig(
        run_id="da_true", checkpoint_dir=str(tmp_path / "ckpt_true"), use_domain_adaptation=True
    )
    config_false = TrainConfig(
        run_id="da_false", checkpoint_dir=str(tmp_path / "ckpt_false"), use_domain_adaptation=False
    )
    # Both must run without error, regardless of flag value.
    train_single_client(config_true, manifest_path, num_epochs=1)
    train_single_client(config_false, manifest_path, num_epochs=1)
