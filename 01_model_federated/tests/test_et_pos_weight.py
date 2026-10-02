"""SPEC.md Addendum 8 (Req 83-88): positive weight on the enhancing-tumour BCE term. CPU only."""
import math
import sys
import time

import pytest
import torch
from monai.losses import DiceCELoss, DiceFocalLoss

import run
from src.checkpoint import load_checkpoint
from src.config import TrainConfig
from src.model import build_model
from src.region_loss import RegionDiceBCELoss, RegionHybridLoss, labels_to_region_targets
from src.train_single import _build_loss, train_single_client


def _volume(size=16):
    y = torch.zeros(1, size, size, size, dtype=torch.long)
    y[:, 2:14, 2:14, 2:14] = 4
    y[:, 5:11, 5:11, 5:11] = 2
    y[:, 7:9, 7:9, 7:9] = 1
    y[:, 10, 10, 10] = 3
    return y


def _reference(logits, y, w=1.0, terms="both", s=1.0):
    """Independent re-implementation: Dice + BCE per region, positive ET voxels weighted by w."""
    lg = logits.float()
    t = labels_to_region_targets(y)
    P = torch.softmax(lg, 1)
    p = torch.stack([P[:, 1] + P[:, 2] + P[:, 3] + P[:, 4], P[:, 1] + P[:, 2] + P[:, 3], P[:, 1]], 1)
    lse = torch.logsumexp(lg, 1)
    l0, l1, l2, l3, l4 = lg.unbind(1)
    log_p = torch.stack([torch.logsumexp(lg[:, 1:], 1), torch.logsumexp(lg[:, 1:4], 1), l1], 1) - lse[:, None]
    log_q = torch.stack([l0, torch.logsumexp(torch.stack([l0, l4], 1), 1),
                         torch.logsumexp(torch.stack([l0, l2, l3, l4], 1), 1)], 1) - lse[:, None]
    d = (0, 2, 3, 4)
    dice = 1 - (2 * (p * t).sum(d) + s) / (p.sum(d) + t.sum(d) + s)
    bce_wt = -(t[:, 0] * log_p[:, 0] + (1 - t[:, 0]) * log_q[:, 0]).mean()
    bce_tc = -(t[:, 1] * log_p[:, 1] + (1 - t[:, 1]) * log_q[:, 1]).mean()
    bce_et = -(w * t[:, 2] * log_p[:, 2] + (1 - t[:, 2]) * log_q[:, 2]).mean()
    bce = torch.stack([bce_wt, bce_tc, bce_et])
    return {"both": (dice + bce).mean(), "dice": dice.mean(), "bce": bce.mean()}[terms]


# ------------------------------------------------------------------------------------------ Req 83
@pytest.mark.parametrize("terms", ["both", "dice", "bce"])
def test_weight_one_is_exactly_the_old_loss(terms):
    torch.manual_seed(0)
    lg = torch.randn(1, 5, 16, 16, 16)
    y = _volume()
    old = RegionDiceBCELoss(terms=terms)(lg, y)
    assert abs(RegionDiceBCELoss(terms=terms, et_pos_weight=1.0)(lg, y).item() - old.item()) < 1e-7
    assert abs(old.item() - _reference(lg, y, 1.0, terms).item()) < 1e-6


@pytest.mark.parametrize("w", [6.0, 36.0])
@pytest.mark.parametrize("terms", ["both", "bce"])
def test_weighted_loss_matches_the_reference(w, terms):
    torch.manual_seed(1)
    lg = torch.randn(1, 5, 16, 16, 16)
    y = _volume()
    got = RegionDiceBCELoss(terms=terms, et_pos_weight=w)(lg, y)
    assert got.item() == pytest.approx(_reference(lg, y, w, terms).item(), abs=1e-5)
    assert got.item() > RegionDiceBCELoss(terms=terms)(lg, y).item()   # positives weighted up -> larger loss


def test_weight_has_no_effect_on_the_dice_half():
    torch.manual_seed(2)
    lg = torch.randn(1, 5, 16, 16, 16)
    y = _volume()
    a = RegionDiceBCELoss(terms="dice")(lg, y)
    b = RegionDiceBCELoss(terms="dice", et_pos_weight=36.0)(lg, y)
    assert abs(a.item() - b.item()) < 1e-7


@pytest.mark.parametrize("bad", [0, -1.0, float("nan"), float("inf"), True])
def test_invalid_weights_are_rejected(bad):
    with pytest.raises(ValueError, match="et_pos_weight"):
        RegionDiceBCELoss(et_pos_weight=bad)


# ------------------------------------------------------------------------------------------ Req 84
def test_pressure_on_under_called_et_grows_with_the_weight():
    size = 32
    y = torch.zeros(1, size, size, size, dtype=torch.long)
    y[:, 4:28, 4:28, 4:28] = 4
    y[:, 8:24, 8:24, 8:24] = 2
    block = (slice(12, 20),) * 3
    y[0][block] = 1
    sums = []
    for w in (1.0, 6.0, 36.0):
        lg = torch.zeros(1, 5, size, size, size)
        lg.scatter_(1, y[:, None], 5.0)
        lg[0, 1][block] = -5.0
        lg[0, 2][block] = 5.0
        lg.requires_grad_(True)
        RegionDiceBCELoss(et_pos_weight=w)(lg, y).backward()
        g = lg.grad[0, 1][block]
        assert bool((g < 0).all())
        sums.append(g.abs().sum().item())
    assert sums[0] < sums[1] < sums[2]


def test_weight_changes_nothing_on_a_patient_without_et():
    torch.manual_seed(3)
    y = torch.zeros(1, 32, 32, 32, dtype=torch.long)
    y[:, 4:28, 4:28, 4:28] = 4
    y[:, 8:24, 8:24, 8:24] = 2
    lg = torch.randn(1, 5, 32, 32, 32)
    vals = [RegionDiceBCELoss(et_pos_weight=w)(lg, y).item() for w in (1.0, 6.0, 36.0)]
    assert max(vals) - min(vals) < 1e-7


# ------------------------------------------------------------------------------------------ Req 85
@pytest.mark.parametrize("w", [36.0, 1000.0])
def test_finite_and_float32_in_hard_cases(w):
    loss_fn = RegionDiceBCELoss(et_pos_weight=w)
    torch.manual_seed(4)
    for y, lg in ((torch.zeros(1, 32, 32, 32, dtype=torch.long), torch.randn(1, 5, 32, 32, 32)),
                  (_volume(), torch.randn(1, 5, 16, 16, 16).sign() * 1000)):
        lg = lg.clone().requires_grad_(True)
        loss = loss_fn(lg, y)
        loss.backward()
        assert loss.dtype == torch.float32 and torch.isfinite(loss) and torch.isfinite(lg.grad).all()
    lg = torch.randn(1, 5, 16, 16, 16)
    ref = loss_fn(lg, _volume())
    for dtype in (torch.float16, torch.bfloat16):
        half = loss_fn(lg.to(dtype), _volume())
        assert half.dtype == torch.float32 and torch.isfinite(half)
        assert abs(half.item() - ref.item()) < 1e-2 * max(1.0, abs(ref.item()))


# ------------------------------------------------------------------------------------------ Req 86
def test_default_dispatch_is_unchanged_for_all_four_kinds():
    torch.manual_seed(5)
    lg = torch.randn(1, 5, 8, 8, 8)
    y = torch.randint(0, 5, (1, 1, 8, 8, 8))
    direct = {
        "dice_ce": DiceCELoss(to_onehot_y=True, softmax=True, include_background=True),
        "dice_focal": DiceFocalLoss(to_onehot_y=True, softmax=True, include_background=True, gamma=2.0),
        "region_dice_bce": RegionDiceBCELoss(),
        "region_hybrid": RegionHybridLoss(),
    }
    for kind, ref in direct.items():
        built = _build_loss(kind)
        assert type(built) is type(ref)
        assert abs(built(lg, y).item() - ref(lg, y).item()) < 1e-7


def test_weight_reaches_the_region_loss():
    built = _build_loss("region_dice_bce", et_pos_weight=6.0)
    assert isinstance(built, RegionDiceBCELoss) and built.et_pos_weight == 6.0


@pytest.mark.parametrize("kind, terms", [("dice_ce", "both"), ("dice_focal", "both"), ("region_hybrid", "both"),
                                         ("region_dice_bce", "dice")])
def test_weight_with_a_loss_it_cannot_affect_is_rejected(kind, terms):
    with pytest.raises(ValueError) as exc:
        _build_loss(kind, region_terms=terms, et_pos_weight=6.0) if kind == "region_dice_bce" else _build_loss(kind, et_pos_weight=6.0)
    assert "et_pos_weight" in str(exc.value) and kind in str(exc.value)


def test_training_builds_exactly_the_weighted_loss_and_records_it(tmp_path, small_manifest, monkeypatch):
    import src.train_single as ts
    built = []
    real_build = ts._build_loss

    def spy(*args, **kwargs):
        loss = real_build(*args, **kwargs)
        built.append(loss)
        return loss

    monkeypatch.setattr(ts, "_build_loss", spy)
    config = TrainConfig(run_id="w6", checkpoint_dir=str(tmp_path / "ckpt"))
    _model, losses = train_single_client(config, small_manifest("hospA", 1), num_epochs=1,
                                         loss_kind="region_dice_bce", et_pos_weight=6.0)
    assert all(math.isfinite(l) for l in losses)
    assert len(built) == 1 and isinstance(built[0], RegionDiceBCELoss) and built[0].et_pos_weight == 6.0
    payload = load_checkpoint(config.checkpoint_dir, config.run_id)
    assert payload["et_pos_weight"] == 6.0
    build_model().load_state_dict(payload["model_state"], strict=True)


def test_default_path_checkpoints_get_no_weight_key(tmp_path, small_manifest):
    config = TrainConfig(run_id="ce", checkpoint_dir=str(tmp_path / "ckpt"))
    train_single_client(config, small_manifest("hospA", 1), num_epochs=1)
    assert "et_pos_weight" not in load_checkpoint(config.checkpoint_dir, config.run_id)


# ------------------------------------------------------------------------------------------ Req 87
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


def test_cli_default_and_forwarding(monkeypatch):
    assert run.build_arg_parser().parse_args([]).et_pos_weight == 1.0
    assert _run_main(monkeypatch, [])["single"]["et_pos_weight"] == 1.0
    calls = _run_main(monkeypatch, ["--loss", "region_dice_bce", "--et-pos-weight", "36"])
    assert calls["single"]["et_pos_weight"] == 36.0
    calls = _run_main(monkeypatch, ["--loss", "region_dice_bce", "--region-terms", "bce", "--et-pos-weight", "6"])
    assert calls["single"]["et_pos_weight"] == 6.0 and calls["single"]["region_terms"] == "bce"


@pytest.mark.parametrize("argv, names", [
    (["--loss", "region_dice_bce", "--et-pos-weight", "0"], ["--et-pos-weight"]),
    (["--loss", "region_dice_bce", "--et-pos-weight", "-3"], ["--et-pos-weight"]),
    (["--loss", "region_dice_bce", "--et-pos-weight", "nan"], ["--et-pos-weight"]),
    (["--loss", "region_dice_bce", "--et-pos-weight", "inf"], ["--et-pos-weight"]),
    (["--et-pos-weight", "6"], ["--et-pos-weight", "--loss"]),
    (["--loss", "region_hybrid", "--et-pos-weight", "6"], ["--et-pos-weight", "--loss"]),
    (["--loss", "region_dice_bce", "--region-terms", "dice", "--et-pos-weight", "6"], ["--et-pos-weight", "--region-terms"]),
])
def test_cli_rejects_bad_or_pointless_weights(monkeypatch, capsys, argv, names):
    calls = {}
    monkeypatch.setattr(run, "train_single_client", lambda **kw: calls.setdefault("ran", True) and (None, []))
    monkeypatch.setattr(run, "train_federated", lambda **kw: calls.setdefault("ran", True) and (None, []))
    monkeypatch.setattr(sys, "argv", ["run.py", *argv])
    with pytest.raises(SystemExit) as exc:
        run.main()
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert all(n in err for n in names)
    assert "ran" not in calls


def test_cli_help_explains_the_weight():
    action = next(a for a in run.build_arg_parser()._actions if a.dest == "et_pos_weight")
    assert "positive enhancing-tumour voxels" in action.help and "BCE" in action.help


# ------------------------------------------------------------------------------------------ Req 88
def _region_dice(pred, truth, labels):
    p, t = torch.isin(pred, torch.tensor(labels)), torch.isin(truth, torch.tensor(labels))
    return (2.0 * (p & t).sum() / (p.sum() + t.sum()).clamp(min=1)).item()


def test_weighted_loss_learns_the_regions_on_cpu():
    prev = torch.get_num_threads()
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
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = RegionDiceBCELoss(et_pos_weight=36.0)
        start, reached, dice = time.time(), False, None
        for step in range(1, 201):
            model.train()
            opt.zero_grad()
            loss_fn(model(x)[0], y).backward()
            opt.step()
            if step % 10 == 0:
                model.eval()
                with torch.no_grad():
                    pred = model(x)[0].argmax(1)
                dice = (_region_dice(pred, y, [1]), _region_dice(pred, y, [1, 2, 3]), _region_dice(pred, y, [1, 2, 3, 4]))
                if min(dice) >= 0.8:
                    reached = True
                    break
        assert reached, f"ET/NC/WT Dice {dice} after 200 steps"
        assert time.time() - start < 60
    finally:
        torch.set_num_threads(prev)
