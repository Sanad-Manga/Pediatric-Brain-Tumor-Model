import torch
import pytest

from src.config import TrainConfig
from src.federated import train_federated, weighted_average_state_dicts


def test_fedavg_weighted_aggregation_matches_analytic_average():
    sd_a = {"w": torch.tensor([1.0, 2.0])}
    sd_b = {"w": torch.tensor([3.0, 4.0])}
    counts = [53, 92]  # Hospital A, Hospital B subject counts

    result = weighted_average_state_dicts([sd_a, sd_b], counts)

    total = sum(counts)
    expected = (sd_a["w"] * counts[0] + sd_b["w"] * counts[1]) / total
    assert torch.allclose(result["w"], expected, atol=1e-5)


def test_federated_loop_runs_on_two_clients(tmp_path, small_manifest):
    manifest_a = small_manifest("hospA", 2)
    manifest_b = small_manifest("hospB", 3)
    config = TrainConfig(
        use_federation=True,
        run_id="fed1",
        checkpoint_dir=str(tmp_path / "ckpt"),
    )
    _model, round_losses = train_federated(
        config, [manifest_a, manifest_b], num_rounds=1, local_epochs=1
    )
    assert len(round_losses) == 1


def test_federated_requires_at_least_two_clients(tmp_path, small_manifest):
    manifest_a = small_manifest("hospA", 2)
    config = TrainConfig(use_federation=True, run_id="fed_single", checkpoint_dir=str(tmp_path / "ckpt"))
    with pytest.raises(ValueError):
        train_federated(config, [manifest_a], num_rounds=1, local_epochs=1)
