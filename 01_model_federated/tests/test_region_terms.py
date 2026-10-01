"""SPEC.md Addendum 7 (Req 76-81): region-loss component switch (--region-terms). CPU only."""
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


def _halves(logits, y, s=1.0):
    """Independent re-implementation of the two halves (per region), as SPEC Addendum 5 defines them."""
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
    bce = -(t * log_p + (1 - t) * log_q).mean(d)
    return dice.mean(), bce.mean()


# ------------------------------------------------------------------------------------------ Req 76
def test_both_and_default_are_unchanged_and_each_half_matches_its_formula():
    torch.manual_seed(0)
    lg = torch.randn(1, 5, 16, 16, 16)
    y = _volume()
    dice_ref, bce_ref = _halves(lg, y)
    assert RegionDiceBCELoss()(lg, y).item() == pytest.approx((dice_ref + bce_ref).item(), abs=1e-6)
    assert RegionDiceBCELoss(terms="both")(lg, y).item() == pytest.approx((dice_ref + bce_ref).item(), abs=1e-6)
    dice = RegionDiceBCELoss(terms="dice")(lg, y)
    bce = RegionDiceBCELoss(terms="bce")(lg, y)
    assert dice.item() == pytest.approx(dice_ref.item(), abs=1e-6)
    assert bce.item() == pytest.approx(bce_ref.item(), abs=1e-6)
    assert (dice + bce).item() == pytest.approx(RegionDiceBCELoss()(lg, y).item(), abs=1e-6)


@pytest.mark.parametrize("bad", ["Dice", "none", "", "both "])
def test_unknown_terms_value_is_rejected_at_construction(bad):
    with pytest.raises(ValueError, match=repr(bad).strip("'") if bad else "terms"):
        RegionDiceBCELoss(terms=bad)


# ------------------------------------------------------------------------------------------ Req 77
@pytest.mark.parametrize("terms", ["dice", "bce"])
def test_each_half_is_finite_and_float32_in_hard_cases(terms):
    loss_fn = RegionDiceBCELoss(terms=terms)
    for y, lg in ((torch.zeros(1, 32, 32, 32, dtype=torch.long), torch.randn(1, 5, 32, 32, 32)),
                  (_volume(), torch.randn(1, 5, 16, 16, 16).sign() * 1000)):
        lg = lg.clone().requires_grad_(True)
        loss = loss_fn(lg, y)
        loss.backward()
        assert loss.dtype == torch.float32 and torch.isfinite(loss) and torch.isfinite(lg.grad).all()
    torch.manual_seed(0)
    lg = torch.randn(1, 5, 16, 16, 16)
    ref = loss_fn(lg, _volume())
    for dtype in (torch.float16, torch.bfloat16):
        half = loss_fn(lg.to(dtype), _volume())
        assert half.dtype == torch.float32 and abs(half.item() - ref.item()) < 1e-2


def test_dice_half_keeps_the_empty_region_false_blob_penalty():
    y = torch.zeros(1, 32, 32, 32, dtype=torch.long)
    y[:, 4:28, 4:28, 4:28] = 4
    y[:, 8:24, 8:24, 8:24] = 2   # no ET
    lg = torch.zeros(1, 5, 32, 32, 32)
    lg[:, 1] = -20.0
    lg[0, 1][(slice(10, 18),) * 3] = 20.0    # 512-voxel false ET blob
    p_et = torch.softmax(lg, 1)[:, 1]
    et_dice_term = 1 - (2 * 0 + 1.0) / (p_et.sum() + 0 + 1.0)
    assert et_dice_term.item() > 0.99
    # and the dice-only loss carries it: at least a third of the term (it is one of three region terms)
    assert RegionDiceBCELoss(terms="dice")(lg, y).item() > 0.99 / 3


# ------------------------------------------------------------------------------------------ Req 78
def test_default_dispatch_is_unchanged_for_all_four_kinds():
    torch.manual_seed(0)
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


@pytest.mark.parametrize("terms", ["both", "dice", "bce"])
def test_region_terms_reaches_the_region_loss(terms):
    built = _build_loss("region_dice_bce", region_terms=terms)
    assert isinstance(built, RegionDiceBCELoss) and built.terms == terms


@pytest.mark.parametrize("kind", ["dice_ce", "dice_focal", "region_hybrid"])
def test_non_default_region_terms_with_another_loss_is_rejected(kind):
    with pytest.raises(ValueError) as exc:
        _build_loss(kind, region_terms="dice")
    assert kind in str(exc.value) and "dice" in str(exc.value)


# ------------------------------------------------------------------------------------------ Req 79
@pytest.mark.parametrize("terms", ["dice", "bce"])
def test_training_with_one_half_records_it_in_the_checkpoint(tmp_path, small_manifest, terms):
    config = TrainConfig(run_id=f"t_{terms}", checkpoint_dir=str(tmp_path / "ckpt"))
    _model, losses = train_single_client(config, small_manifest("hospA", 2), num_epochs=1,
                                         loss_kind="region_dice_bce", region_terms=terms)
    assert all(math.isfinite(l) for l in losses)
    payload = load_checkpoint(config.checkpoint_dir, config.run_id)
    assert payload["region_terms"] == terms and payload["loss_kind"] == "region_dice_bce"
    build_model().load_state_dict(payload["model_state"], strict=True)


def test_default_path_checkpoints_get_no_new_key(tmp_path, small_manifest):
    config = TrainConfig(run_id="t_ce", checkpoint_dir=str(tmp_path / "ckpt"))
    train_single_client(config, small_manifest("hospA", 2), num_epochs=1)   # old-style call, no new argument
    assert "region_terms" not in load_checkpoint(config.checkpoint_dir, config.run_id)


# ------------------------------------------------------------------------------------------ Req 80
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


def test_cli_default_is_both_and_values_are_forwarded(monkeypatch):
    assert run.build_arg_parser().parse_args([]).region_terms == "both"
    assert _run_main(monkeypatch, [])["single"]["region_terms"] == "both"
    for t in ("dice", "bce"):
        calls = _run_main(monkeypatch, ["--loss", "region_dice_bce", "--region-terms", t])
        assert calls["single"]["region_terms"] == t and calls["single"]["loss_kind"] == "region_dice_bce"


@pytest.mark.parametrize("argv, must_name", [
    (["--region-terms", "everything"], "--region-terms"),
    (["--region-terms", "dice"], "--loss"),                                   # default loss is dice_ce
    (["--loss", "region_hybrid", "--region-terms", "bce"], "--region-terms"),
    (["--loss", "dice_focal", "--region-terms", "dice"], "--loss"),
])
def test_cli_rejects_bad_or_pointless_region_terms(monkeypatch, capsys, argv, must_name):
    calls = {}
    monkeypatch.setattr(run, "train_single_client", lambda **kw: calls.setdefault("ran", True) and (None, []))
    monkeypatch.setattr(run, "train_federated", lambda **kw: calls.setdefault("ran", True) and (None, []))
    monkeypatch.setattr(sys, "argv", ["run.py", *argv])
    with pytest.raises(SystemExit) as exc:
        run.main()
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert must_name in err and "--region-terms" in err
    assert "ran" not in calls


def test_cli_help_names_the_three_values():
    action = next(a for a in run.build_arg_parser()._actions if a.dest == "region_terms")
    assert all(v in action.help for v in ("both", "dice", "bce"))


# ------------------------------------------------------------------------------------------ Req 81
def _region_dice(pred, truth, labels):
    p, t = torch.isin(pred, torch.tensor(labels)), torch.isin(truth, torch.tensor(labels))
    return (2.0 * (p & t).sum() / (p.sum() + t.sum()).clamp(min=1)).item()


@pytest.mark.parametrize("terms", ["dice", "bce"])
def test_each_half_alone_learns_the_regions_on_cpu(terms):
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
        loss_fn = RegionDiceBCELoss(terms=terms)
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
        assert reached, f"{terms}: ET/NC/WT Dice {dice} after 200 steps"
        assert time.time() - start < 60
    finally:
        torch.set_num_threads(prev)
