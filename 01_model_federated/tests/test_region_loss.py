"""SPEC.md Addendum 5 (Req 61-66): region-based loss on the existing 5-class head. CPU only."""
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
from src.region_loss import RegionDiceBCELoss, labels_to_region_targets, region_probs_from_logits
from src.train_single import _build_loss, train_single_client


def _volume(size=16):
    """One volume containing every label 0-4, nested like a real tumour."""
    y = torch.zeros(1, size, size, size, dtype=torch.long)
    y[:, 2:14, 2:14, 2:14] = 4   # oedema
    y[:, 5:11, 5:11, 5:11] = 2   # non-enhancing core
    y[:, 7:9, 7:9, 7:9] = 1      # enhancing
    y[:, 6, 6, 6] = 3            # cystic
    return y


def _logits_for(y, hot=20.0):
    """Logit `hot` on the true class of every voxel, 0 elsewhere."""
    lg = torch.zeros(y.shape[0], 5, *y.shape[1:])
    lg.scatter_(1, y[:, None], hot)
    return lg


# ------------------------------------------------------------------------------------------ Req 61
def test_targets_are_exact_float32_and_nested():
    y = _volume()
    t = labels_to_region_targets(y)
    assert t.shape == (1, 3, 16, 16, 16) and t.dtype == torch.float32
    assert torch.equal(t[:, 0], (y > 0).float())
    assert torch.equal(t[:, 1], ((y >= 1) & (y <= 3)).float())
    assert torch.equal(t[:, 2], (y == 1).float())
    assert bool((t[:, 2] <= t[:, 1]).all() and (t[:, 1] <= t[:, 0]).all())


def test_targets_give_identical_output_for_both_input_forms():
    y = _volume()
    assert torch.equal(labels_to_region_targets(y), labels_to_region_targets(y[:, None]))


@pytest.mark.parametrize("bad", [5, -1])
def test_targets_reject_labels_outside_0_to_4(bad):
    y = _volume()
    y[0, 0, 0, 0] = bad
    with pytest.raises(ValueError, match=str(bad)):
        labels_to_region_targets(y)


def test_region_probs_match_an_independent_softmax_and_are_nested():
    gen = torch.Generator().manual_seed(0)
    for _ in range(100):
        lg = torch.randn(2, 5, 8, 8, 8, generator=gen) * 3
        rp = region_probs_from_logits(lg)
        p = torch.softmax(lg, dim=1)
        assert rp.shape == (2, 3, 8, 8, 8) and rp.dtype == torch.float32
        assert rp.min() >= 0 and rp.max() <= 1
        assert bool((rp[:, 2] <= rp[:, 1]).all() and (rp[:, 1] <= rp[:, 0]).all())  # exact, not approximate
        expected = torch.stack([1 - p[:, 0], p[:, 1] + p[:, 2] + p[:, 3], p[:, 1]], dim=1)
        assert torch.allclose(rp, expected, atol=1e-6)


def test_region_probs_are_invariant_to_swapping_classes_2_and_3():
    """Pins the documented limitation: the loss cannot tell label 2 from label 3."""
    lg = torch.randn(2, 5, 8, 8, 8)
    swapped = lg.clone()
    swapped[:, [2, 3]] = lg[:, [3, 2]]
    assert torch.allclose(region_probs_from_logits(lg), region_probs_from_logits(swapped), atol=1e-6)


# ------------------------------------------------------------------------------------------ Req 62
def test_loss_matches_the_analytic_value_at_zero_logits():
    y = _volume()
    n = y.numel()
    t_count = labels_to_region_targets(y).sum((0, 2, 3, 4))
    p = torch.tensor([0.8, 0.6, 0.2])   # softmax 0.2 per class -> WT 0.8, TC 0.6, ET 0.2 everywhere
    s = 1.0
    f = t_count / n
    expected = ((1 - (2 * p * t_count + s) / (p * n + t_count + s)) + (-f * p.log() - (1 - f) * (1 - p).log())).mean()
    got = RegionDiceBCELoss()(torch.zeros(1, 5, 16, 16, 16), y)
    assert got.item() == pytest.approx(expected.item(), abs=1e-5)


def test_loss_is_tiny_for_perfect_logits_and_large_for_all_background():
    y = _volume()
    loss_fn = RegionDiceBCELoss()
    assert loss_fn(_logits_for(y), y).item() < 1e-3
    background = torch.zeros(1, 5, 16, 16, 16)
    background[:, 0] = 20.0
    assert loss_fn(background, y).item() > 1.0


def test_loss_is_a_float32_scalar_with_a_gradient():
    y = _volume()
    lg = torch.randn(1, 5, 16, 16, 16, requires_grad=True)
    loss = RegionDiceBCELoss()(lg, y)
    assert loss.ndim == 0 and loss.dtype == torch.float32 and loss.requires_grad
    loss.backward()
    assert lg.grad is not None and torch.isfinite(lg.grad).all()


@pytest.mark.parametrize("bad", [0, -1.0, float("nan"), float("inf")])
def test_smooth_must_be_positive_and_finite(bad):
    with pytest.raises(ValueError, match=str(bad)):
        RegionDiceBCELoss(smooth=bad)


# ------------------------------------------------------------------------------------------ Req 63
def _empty_et_volume(size=32):
    y = torch.zeros(1, size, size, size, dtype=torch.long)
    y[:, 4:28, 4:28, 4:28] = 4
    y[:, 8:24, 8:24, 8:24] = 2   # tumour, core, but no enhancing label anywhere
    return y


def _et_dice_term(logits, y, smooth=1.0):
    p = region_probs_from_logits(logits)[:, 2]
    t = labels_to_region_targets(y)[:, 2]
    return 1.0 - (2.0 * (p * t).sum() + smooth) / (p.sum() + t.sum() + smooth)


def test_correct_no_et_is_nearly_free_and_a_500_voxel_false_blob_is_not():
    y = _empty_et_volume()
    quiet = torch.zeros(1, 5, 32, 32, 32)
    quiet[:, 1] = -20.0
    assert _et_dice_term(quiet, y).item() < 0.01
    blob = (slice(10, 18),) * 3   # 512 voxels
    noisy = quiet.clone()
    noisy[0, 1][blob] = 20.0
    assert _et_dice_term(noisy, y).item() > 0.99


def test_false_et_blob_gradient_pushes_the_et_logit_down():
    y = _empty_et_volume()
    blob = (slice(10, 18),) * 3
    lg = torch.zeros(1, 5, 32, 32, 32)
    lg[:, 1] = -20.0
    lg[0, 1][blob] = 3.0
    lg.requires_grad_(True)
    RegionDiceBCELoss()(lg, y).backward()
    assert bool((lg.grad[0, 1][blob] > 0).all())   # positive gradient => descent lowers the ET logit


def test_loss_and_gradients_are_finite_with_no_tumour_at_all():
    y = torch.zeros(1, 32, 32, 32, dtype=torch.long)
    lg = torch.randn(1, 5, 32, 32, 32, requires_grad=True)
    loss = RegionDiceBCELoss()(lg, y)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(lg.grad).all()


def test_loss_and_gradients_are_finite_with_logits_of_magnitude_1000():
    torch.manual_seed(0)
    lg = (torch.randn(1, 5, 16, 16, 16).sign() * 1000).requires_grad_(True)
    loss = RegionDiceBCELoss()(lg, _volume())
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(lg.grad).all()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_half_precision_logits_give_a_float32_loss_close_to_float32(dtype):
    torch.manual_seed(0)
    lg = torch.randn(1, 5, 16, 16, 16)
    y = _volume()
    ref = RegionDiceBCELoss()(lg, y)
    half = RegionDiceBCELoss()(lg.to(dtype), y)
    assert half.dtype == torch.float32 and torch.isfinite(half)
    assert abs(half.item() - ref.item()) < 1e-2


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_the_loss_is_computed_in_float32_not_in_the_logit_dtype(dtype):
    """Large logits make float16/bfloat16 arithmetic visibly lossy. A loss that really runs in float32
    gives the same value as the float32 loss of the very same (already rounded) numbers."""
    torch.manual_seed(0)
    half = (torch.randn(1, 5, 16, 16, 16) * 20).to(dtype)
    y = _volume()
    assert RegionDiceBCELoss()(half, y).item() == pytest.approx(RegionDiceBCELoss()(half.float(), y).item(), abs=1e-6)


def test_loss_runs_inside_cpu_autocast():
    torch.manual_seed(0)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = RegionDiceBCELoss()(torch.randn(1, 5, 16, 16, 16), _volume())
    assert torch.isfinite(loss)


# ------------------------------------------------------------------------------------------ Req 64
def test_existing_loss_kinds_are_unchanged_and_the_new_one_is_dispatched():
    torch.manual_seed(0)
    lg = torch.randn(1, 5, 8, 8, 8)
    y = torch.randint(0, 5, (1, 1, 8, 8, 8))
    direct = {
        "dice_ce": DiceCELoss(to_onehot_y=True, softmax=True, include_background=True),
        "dice_focal": DiceFocalLoss(to_onehot_y=True, softmax=True, include_background=True, gamma=2.0),
    }
    for kind, ref in direct.items():
        built = _build_loss(kind)
        assert type(built) is type(ref)
        assert abs(built(lg, y).item() - ref(lg, y).item()) < 1e-7
    assert isinstance(_build_loss("region_dice_bce"), RegionDiceBCELoss)


def test_unknown_loss_kind_names_all_three_valid_kinds():
    with pytest.raises(ValueError) as exc:
        _build_loss("nope")
    for kind in ("dice_ce", "dice_focal", "region_dice_bce"):
        assert kind in str(exc.value)


def test_training_with_the_region_loss_saves_a_checkpoint_every_existing_tool_accepts(tmp_path, small_manifest):
    manifest_path = small_manifest("hospA", 2)
    config = TrainConfig(run_id="region1", checkpoint_dir=str(tmp_path / "ckpt"))
    _model, losses = train_single_client(config, manifest_path, num_epochs=2, loss_kind="region_dice_bce")
    assert len(losses) == 2 and all(math.isfinite(l) for l in losses)

    payload = load_checkpoint(config.checkpoint_dir, config.run_id)
    assert payload["loss_kind"] == "region_dice_bce"
    build_model().load_state_dict(payload["model_state"], strict=True)   # same 5-channel architecture as always

    from tools.eval_heldout_3d import load_model   # the existing evaluation tool, unmodified
    loaded = load_model(latest_checkpoint(config.checkpoint_dir, config.run_id), "cpu")
    assert isinstance(loaded, FederatedUNet3D)


# ------------------------------------------------------------------------------------------ Req 65
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


def test_cli_default_loss_is_unchanged():
    assert run.parse_args([]).loss == "dice_ce"


@pytest.mark.parametrize("kind", ["dice_ce", "dice_focal", "region_dice_bce"])
def test_cli_forwards_the_loss_kind(monkeypatch, kind):
    calls = _run_main(monkeypatch, ["--loss", kind])
    assert calls["single"]["loss_kind"] == kind


def test_cli_rejects_the_region_loss_with_federation(monkeypatch, capsys):
    calls = {}

    def boom(**kw):
        calls["ran"] = True
        return None, []

    monkeypatch.setattr(run, "train_single_client", boom)
    monkeypatch.setattr(run, "train_federated", boom)
    monkeypatch.setattr(sys, "argv", ["run.py", "--loss", "region_dice_bce", "--use-federation"])
    with pytest.raises(SystemExit) as exc:
        run.main()
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "--loss" in err and "--use-federation" in err
    assert "ran" not in calls


def test_cli_still_accepts_the_other_losses_with_federation(monkeypatch):
    calls = _run_main(monkeypatch, ["--loss", "dice_focal", "--use-federation"])
    assert "federated" in calls


def test_cli_loss_help_mentions_the_region_option():
    action = next(a for a in run.build_arg_parser()._actions if a.dest == "loss")
    assert "region" in action.help.lower()
    assert "region_dice_bce" in action.choices


# ------------------------------------------------------------------------------------------ Req 66
def _region_dice(pred, truth, labels):
    p, t = torch.isin(pred, torch.tensor(labels)), torch.isin(truth, torch.tensor(labels))
    return (2.0 * (p & t).sum() / (p.sum() + t.sum()).clamp(min=1)).item()


def test_the_unchanged_model_learns_the_regions_on_a_synthetic_volume_on_cpu():
    """Throwaway-probe setup from the spec: nested WT/TC/ET boxes, class-dependent intensities,
    at most 150 Adam steps. Failing to reach Dice 0.8 on all three regions is a failure, not a skip."""
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
        for channel, label in enumerate([1, 2, 4, 4]):   # class-dependent intensities
            x[:, channel][y == label] += 2.0

        model = build_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = RegionDiceBCELoss()
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
