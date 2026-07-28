import pytest
import torch

from src.config import TrainConfig
from src.domain_adaptation import FeatureQueue, coral_loss, covariance
from src.model import build_model
import src.federated as federated
import src.domain_adaptation as da


def test_covariance_matches_torch_reference():
    x = torch.randn(7, 5, dtype=torch.float64)
    assert torch.allclose(covariance(x), torch.cov(x.T))


def test_coral_is_zero_for_identical_features():
    x = torch.randn(8, 12)
    assert coral_loss(x, x).item() == pytest.approx(0.0, abs=1e-8)


def test_coral_is_symmetric_and_nonnegative():
    a, b = torch.randn(6, 9), torch.randn(5, 9) * 2
    assert coral_loss(a, b).item() >= 0
    assert torch.allclose(coral_loss(a, b), coral_loss(b, a))


def test_coral_backpropagates_to_both_domains():
    a = torch.randn(4, 6, requires_grad=True)
    b = torch.randn(4, 6, requires_grad=True)
    coral_loss(a, b).backward()
    assert a.grad is not None and torch.isfinite(a.grad).all()
    assert b.grad is not None and torch.isfinite(b.grad).all()


def test_single_subject_is_rejected():
    with pytest.raises(ValueError, match="at least two"):
        coral_loss(torch.randn(1, 4), torch.randn(2, 4))


def test_queue_preserves_only_current_gradient():
    queue = FeatureQueue(3)
    old = torch.randn(1, 4, requires_grad=True)
    queue.append(old)
    current = torch.randn(1, 4, requires_grad=True)
    combined = queue.with_current(current)
    combined.sum().backward()
    assert old.grad is None
    assert current.grad is not None


def test_federated_flag_runs_coral_phase(monkeypatch, tmp_path, small_manifest):
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([1.0]))

    calls = []
    monkeypatch.setattr(federated, "build_model", TinyModel)
    monkeypatch.setattr(
        federated,
        "train_single_client",
        lambda config, manifest_path, num_epochs, model, **kwargs: (model, [0.5]),
    )
    monkeypatch.setattr(
        federated,
        "train_coral_alignment",
        lambda model, config, source, target: calls.append((source, target)) or 0.25,
    )
    monkeypatch.setattr(federated, "save_checkpoint", lambda *args, **kwargs: None)

    manifests = [small_manifest("A", 2), small_manifest("B", 3)]
    config = TrainConfig(
        use_federation=True,
        use_domain_adaptation=True,
        checkpoint_dir=str(tmp_path),
    )
    federated.train_federated(config, manifests, num_rounds=1, local_epochs=1)
    assert calls == [(manifests[0], manifests[1])]


def test_alignment_phase_runs_end_to_end_with_unit_batches(monkeypatch):
    class TinyFeatureModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = torch.nn.Linear(4, 3)

        def forward(self, x):
            features = self.projection(x)
            return torch.empty(x.shape[0], 1), features

    source_x = torch.tensor(
        [[1.0, 0.0, 2.0, 1.0], [2.0, 1.0, 0.0, 1.0], [1.0, 3.0, 1.0, 0.0]]
    )
    target_x = torch.tensor(
        [[4.0, 0.0, 1.0, 2.0], [0.0, 2.0, 4.0, 1.0], [3.0, 1.0, 2.0, 4.0]]
    )
    datasets = {
        "source.json": torch.utils.data.TensorDataset(source_x, torch.zeros(3)),
        "target.json": torch.utils.data.TensorDataset(target_x, torch.zeros(3)),
    }
    monkeypatch.setattr(
        da,
        "build_dataset",
        lambda manifest, *args, **kwargs: datasets[manifest],
    )
    config = TrainConfig(coral_queue_size=3, coral_steps_per_round=3)
    loss = da.train_coral_alignment(
        TinyFeatureModel(), config, "source.json", "target.json"
    )
    assert loss >= 0.0
    assert torch.isfinite(torch.tensor(loss))


def test_bottleneck_embeddings_do_not_collapse_across_inputs():
    model = build_model().eval()
    inputs = torch.stack(
        [
            torch.zeros(4, 32, 32, 32),
            torch.randn(4, 32, 32, 32),
        ]
    )
    with torch.inference_mode():
        _, features = model(inputs)
    distance = torch.linalg.vector_norm(features[0] - features[1])
    assert distance > 1e-5
