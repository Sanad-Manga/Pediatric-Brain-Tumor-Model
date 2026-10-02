"""SPEC.md Addendum 6 (Req 68-74): hybrid loss = region loss + the existing dice_ce. CPU only."""
import math
import sys
import time

import pytest
import torch
from monai.losses import DiceCELoss, DiceFocalLoss

import run
from src.checkpoint import latest_checkpoint, load_checkpoint
from src.config import TrainConfig
from src.model import FederatedUNet3D, build_model
from src.region_loss import RegionDiceBCELoss, RegionHybridLoss
from src.train_single import _build_loss, train_single_client


def _volume(size=16):
    """Every label 0-4, nested like a real tumour, with two cystic voxels."""
    y = torch.zeros(1, size, size, size, dtype=torch.long)
    y[:, 2:14, 2:14, 2:14] = 4
    y[:, 5:11, 5:11, 5:11] = 2
    y[:, 7:9, 7:9, 7:9] = 1
    y[:, 10, 10, 10] = 3
    y[:, 5, 5, 5] = 3
    return y


def _dice_ce():
    return DiceCELoss(to_onehot_y=True, softmax=True, include_background=True)


# ------------------------------------------------------------------------------------------ Req 68
@pytest.mark.parametrize("with_channel", [False, True])
def test_hybrid_equals_the_two_terms_computed_independently(with_channel):
    torch.manual_seed(0)
    logits = torch.randn(1, 5, 16, 16, 16)
    y = _volume()
    y_in = y[:, None] if with_channel else y
    expected = RegionDiceBCELoss()(logits, y) + _dice_ce()(logits.float(), y[:, None])
    assert RegionHybridLoss()(logits, y_in).item() == pytest.approx(expected.item(), abs=1e-6)


def test_hybrid_is_a_float32_scalar_with_a_gradient():
    logits = torch.randn(1, 5, 16, 16, 16, requires_grad=True)
    loss = RegionHybridLoss()(logits, _volume())
    assert loss.ndim == 0 and loss.dtype == torch.float32 and loss.requires_grad
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


# ------------------------------------------------------------------------------------------ Req 69
def test_swapping_classes_2_and_3_changes_the_hybrid_but_not_the_region_loss():
    torch.manual_seed(0)
    logits = torch.randn(1, 5, 16, 16, 16)
    swapped = logits.clone()
    swapped[:, [2, 3]] = logits[:, [3, 2]]
    y = _volume()
    assert abs(RegionDiceBCELoss()(logits, y).item() - RegionDiceBCELoss()(swapped, y).item()) < 1e-6
    assert abs(RegionHybridLoss()(logits, y).item() - RegionHybridLoss()(swapped, y).item()) > 1e-3


# ------------------------------------------------------------------------------------------ Req 70
def test_hybrid_pushes_harder_toward_et_on_under_called_et_voxels():
    size = 32
    y = torch.zeros(1, size, size, size, dtype=torch.long)
    y[:, 4:28, 4:28, 4:28] = 4
    y[:, 8:24, 8:24, 8:24] = 2
    block = (slice(12, 20),) * 3          # 512 ET voxels inside the core
    y[0][block] = 1

    def under_called():
        lg = torch.zeros(1, 5, size, size, size)
        lg.scatter_(1, y[:, None], 5.0)
        lg[0, 1][block] = -5.0           # ET predicted as label 2
        lg[0, 2][block] = 5.0
        return lg.requires_grad_(True)

    sums = {}
    for name, loss_fn in (("region", RegionDiceBCELoss()), ("hybrid", RegionHybridLoss())):
        lg = under_called()
        loss_fn(lg, y).backward()
        g = lg.grad[0, 1][block]
        if name == "hybrid":
            assert bool((g < 0).all())   # descent raises the ET logit on every under-called voxel
        sums[name] = g.abs().sum().item()
    assert sums["hybrid"] > sums["region"]


# ------------------------------------------------------------------------------------------ Req 71
def test_finite_with_no_tumour_at_all():
    y = torch.zeros(1, 32, 32, 32, dtype=torch.long)
    lg = torch.randn(1, 5, 32, 32, 32, requires_grad=True)
    loss = RegionHybridLoss()(lg, y)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(lg.grad).all()


def test_finite_with_logits_of_magnitude_1000():
    torch.manual_seed(0)
    lg = (torch.randn(1, 5, 16, 16, 16).sign() * 1000).requires_grad_(True)
    loss = RegionHybridLoss()(lg, _volume())
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(lg.grad).all()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_half_precision_gives_a_float32_loss_close_to_float32(dtype):
    torch.manual_seed(0)
    lg = torch.randn(1, 5, 16, 16, 16)
    ref = RegionHybridLoss()(lg, _volume())
    half = RegionHybridLoss()(lg.to(dtype), _volume())
    assert half.dtype == torch.float32 and torch.isfinite(half)
    assert abs(half.item() - ref.item()) < 1e-2


def test_runs_inside_cpu_autocast():
    torch.manual_seed(0)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = RegionHybridLoss()(torch.randn(1, 5, 16, 16, 16), _volume())
    assert torch.isfinite(loss)


@pytest.mark.parametrize("bad", [5, -1])
def test_labels_outside_0_to_4_raise(bad):
    y = _volume()
    y[0, 0, 0, 0] = bad
    with pytest.raises(ValueError, match=str(bad)):
        RegionHybridLoss()(torch.zeros(1, 5, 16, 16, 16), y)


# ------------------------------------------------------------------------------------------ Req 72
def test_dispatch_adds_the_hybrid_and_leaves_the_other_kinds_unchanged():
    torch.manual_seed(0)
    lg = torch.randn(1, 5, 8, 8, 8)
    y = torch.randint(0, 5, (1, 1, 8, 8, 8))
    direct = {
        "dice_ce": _dice_ce(),
        "dice_focal": DiceFocalLoss(to_onehot_y=True, softmax=True, include_background=True, gamma=2.0),
        "region_dice_bce": RegionDiceBCELoss(),
    }
    for kind, ref in direct.items():
        built = _build_loss(kind)
        assert type(built) is type(ref)
        assert abs(built(lg, y).item() - ref(lg, y).item()) < 1e-7
    assert isinstance(_build_loss("region_hybrid"), RegionHybridLoss)


def test_unknown_loss_kind_names_all_four_valid_kinds():
    with pytest.raises(ValueError) as exc:
        _build_loss("nope")
    for kind in ("dice_ce", "dice_focal", "region_dice_bce", "region_hybrid"):
        assert kind in str(exc.value)


def test_training_with_the_hybrid_saves_a_checkpoint_every_existing_tool_accepts(tmp_path, small_manifest):
    manifest_path = small_manifest("hospA", 2)
    config = TrainConfig(run_id="hybrid1", checkpoint_dir=str(tmp_path / "ckpt"))
    _model, losses = train_single_client(config, manifest_path, num_epochs=2, loss_kind="region_hybrid")
    assert len(losses) == 2 and all(math.isfinite(l) for l in losses)

    payload = load_checkpoint(config.checkpoint_dir, config.run_id)
    assert payload["loss_kind"] == "region_hybrid"
    build_model().load_state_dict(payload["model_state"], strict=True)

    from tools.eval_heldout_3d import load_model
    assert isinstance(load_model(latest_checkpoint(config.checkpoint_dir, config.run_id), "cpu"), FederatedUNet3D)


# ------------------------------------------------------------------------------------------ Req 73
def _run_main(monkeypatch, argv):
    calls = {}

    def record(name):
        def _fn(**kw):
            calls[name] = kw
            return None, []
        return _fn

    monkeypatch.setattr(run, "train_single_client", record("single"))
    monkeypatch.setattr(run, "train_federated", record("federated"))
    monkeypatch.setattr(sys, "argv", ["run.py", *argv])
    run.main()
    return calls


def test_cli_default_is_unchanged_and_the_hybrid_is_forwarded(monkeypatch):
    assert run.build_arg_parser().parse_args([]).loss == "dice_ce"
    assert _run_main(monkeypatch, ["--loss", "region_hybrid"])["single"]["loss_kind"] == "region_hybrid"


@pytest.mark.parametrize("kind", ["dice_ce", "dice_focal", "region_dice_bce"])
def test_cli_still_forwards_the_other_kinds(monkeypatch, kind):
    assert _run_main(monkeypatch, ["--loss", kind])["single"]["loss_kind"] == kind


def test_cli_rejects_the_hybrid_with_federation(monkeypatch, capsys):
    calls = {}

    def boom(**kw):
        calls["ran"] = True
        return None, []

    monkeypatch.setattr(run, "train_single_client", boom)
    monkeypatch.setattr(run, "train_federated", boom)
    monkeypatch.setattr(sys, "argv", ["run.py", "--loss", "region_hybrid", "--use-federation"])
    with pytest.raises(SystemExit) as exc:
        run.main()
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "--loss" in err and "region_hybrid" in err and "--use-federation" in err
    assert "ran" not in calls


def test_cli_loss_help_mentions_the_hybrid():
    action = next(a for a in run.build_arg_parser()._actions if a.dest == "loss")
    assert "region_hybrid" in action.help and "region_hybrid" in action.choices


# ------------------------------------------------------------------------------------------ Req 74
def _region_dice(pred, truth, labels):
    p, t = torch.isin(pred, torch.tensor(labels)), torch.isin(truth, torch.tensor(labels))
    return (2.0 * (p & t).sum() / (p.sum() + t.sum()).clamp(min=1)).item()


def test_the_unchanged_model_learns_the_regions_with_the_hybrid_on_cpu():
    prev_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        torch.manual_seed(0)
        size = 32
        y = torch.zeros(1, size, size, size, dtype=torch.long)
        y[:, 6:26, 6:26, 6:26] = 4
        y[:, 10:22, 10:22, 10:22] = 2
        y[:, 13:19, 13:19, 13:19] = 1
        gen = torch.Generator().manual_seed(0)
        x = torch.randn(1, 4, size, size, size, generator=gen) * 0.3
        for channel, label in enumerate([1, 2, 4, 4]):
            x[:, channel][y == label] += 2.0

        model = build_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = RegionHybridLoss()
        start = time.time()
        reached, dice = False, None
        for step in range(1, 151):
            model.train()
            optimizer.zero_grad()
            logits, _features = model(x)
            loss_fn(logits, y).backward()
            optimizer.step()
            if step % 10 == 0:
                model.eval()
                with torch.no_grad():
                    pred = model(x)[0].argmax(1)
                dice = (_region_dice(pred, y, [1]), _region_dice(pred, y, [1, 2, 3]), _region_dice(pred, y, [1, 2, 3, 4]))
                if min(dice) >= 0.8:
                    reached = True
                    break
        elapsed = time.time() - start
        assert reached, f"ET/NC/WT Dice {dice} after 150 steps"
        assert elapsed < 60, f"took {elapsed:.0f}s"
    finally:
        torch.set_num_threads(prev_threads)
