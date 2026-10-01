"""SPEC.md Addendum 9 (Req 90-95): small-ET oversampling sampler. CPU only."""
import json
import math
import sys

import numpy as np
import pytest
import torch
from torch.utils.data import WeightedRandomSampler

import run
import src.train_single as ts
from src.checkpoint import load_checkpoint
from src.config import TrainConfig
from src.data import MODALITIES, BraTSPedsDataset
from src.sampling import build_small_et_sampler, count_et_voxels, small_et_sampling_weights
from src.train_single import train_single_client


# ------------------------------------------------------------------------------------------ helpers
def _fake_cache(tmp_path, et_counts, size=32):
    """A real-mode cache of tiny volumes whose seg has exactly the requested number of label-1 voxels."""
    cache = tmp_path / "cache"
    cache.mkdir()
    ids = []
    rng = np.random.default_rng(0)
    for i, n_et in enumerate(et_counts):
        sid = f"P{i:03d}"
        seg = np.zeros((size, size, size), dtype=np.int64)
        seg[4:28, 4:28, 4:28] = 4
        flat = seg.reshape(-1)
        flat[:n_et] = 1
        arrays = {m: (rng.random((size, size, size)) + 0.5).astype(np.float32) for m in MODALITIES}
        np.savez(cache / f"{sid}.npz", seg=seg, **arrays)
        ids.append(sid)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(ids))
    return str(manifest), str(cache)


# ------------------------------------------------------------------------------------------ Req 90
def test_weights_keep_et_free_share_and_ratio():
    counts = [0, 0, 0, 0, 1, 300, 600, 601, 5000, 90000]   # 4 ET-free, 3 small (1, 300, 600), 3 large
    p = small_et_sampling_weights(counts, 3.0, 600)
    assert p.dtype == torch.float64 and abs(p.sum().item() - 1.0) < 1e-12
    free, small, large = p[:4], p[4:7], p[7:]
    assert abs(free.sum().item() - 4 / 10) < 1e-12 and torch.allclose(free, free[0].expand(4))
    assert torch.allclose(small, (3.0 * large[0]).expand(3), atol=1e-15)          # boundary 600 is small, 601 large
    assert torch.allclose(large, large[0].expand(3))


def test_weight_one_is_uniform():
    counts = [0, 0, 5, 700, 1, 20000]
    p = small_et_sampling_weights(counts, 1.0, 600)
    assert torch.allclose(p, torch.full((6,), 1 / 6, dtype=torch.float64), atol=1e-15)


@pytest.mark.parametrize("counts", [[0, 0, 900, 4000],        # no small
                                    [0, 0, 10, 20],            # no large
                                    [5, 900, 10],              # no ET-free
                                    [0, 0, 0]])                # all ET-free
def test_edge_groups_still_sum_to_one(counts):
    p = small_et_sampling_weights(counts, 3.0, 600)
    assert abs(p.sum().item() - 1.0) < 1e-12 and torch.isfinite(p).all() and (p > 0).all()
    if all(c == 0 for c in counts):
        assert torch.allclose(p, torch.full((len(counts),), 1 / len(counts), dtype=torch.float64))


@pytest.mark.parametrize("bad", [0, -1.0, float("nan"), float("inf"), True])
def test_invalid_weight_is_rejected(bad):
    with pytest.raises(ValueError, match="weight"):
        small_et_sampling_weights([0, 5, 900], bad, 600)


@pytest.mark.parametrize("bad", [0, -5, 2.5, True])
def test_invalid_max_voxels_is_rejected(bad):
    with pytest.raises(ValueError, match="max_voxels"):
        small_et_sampling_weights([0, 5, 900], 3.0, bad)


# ------------------------------------------------------------------------------------------ Req 91
def test_counts_come_from_the_dataset_itself(tmp_path):
    wanted = [0, 7, 600, 1500, 0]
    manifest, cache = _fake_cache(tmp_path, wanted)
    ds = BraTSPedsDataset(manifest, mode="real", cache_path=cache)
    assert count_et_voxels(ds) == wanted


# ------------------------------------------------------------------------------------------ Req 92
@pytest.mark.parametrize("weight", [1.0, 3.0])
def test_sampler_matches_its_probabilities(tmp_path, weight):
    counts = [0] * 4 + [50] * 3 + [1200] * 3
    manifest, cache = _fake_cache(tmp_path, counts)
    ds = BraTSPedsDataset(manifest, mode="real", cache_path=cache)
    sampler, got_counts = build_small_et_sampler(ds, weight, 600)
    assert isinstance(sampler, WeightedRandomSampler) and sampler.num_samples == len(ds) and sampler.replacement
    probs = small_et_sampling_weights(got_counts, weight, 600)
    assert torch.allclose(sampler.weights.double(), probs)
    gen = torch.Generator().manual_seed(0)
    draws = torch.multinomial(sampler.weights, 20000, replacement=True, generator=gen)
    for lo, hi in ((0, 4), (4, 7), (7, 10)):
        share = ((draws >= lo) & (draws < hi)).double().mean().item()
        assert abs(share - probs[lo:hi].sum().item()) < 0.02


# ------------------------------------------------------------------------------------------ Req 93
def _spy_loaders(monkeypatch):
    made = []
    real = ts.DataLoader

    def spy(*args, **kwargs):
        made.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(ts, "DataLoader", spy)
    return made


def test_default_loader_is_exactly_todays(tmp_path, small_manifest, monkeypatch):
    made = _spy_loaders(monkeypatch)
    config = TrainConfig(run_id="plain", checkpoint_dir=str(tmp_path / "ckpt"))
    train_single_client(config, small_manifest("hospA", 2), num_epochs=1)
    assert len(made) == 1 and made[0].get("shuffle") is False and "sampler" not in made[0]
    payload = load_checkpoint(config.checkpoint_dir, config.run_id)
    assert not any(k.startswith("small_et") for k in payload)


def test_weighted_run_trains_records_and_logs(tmp_path, monkeypatch, capsys):
    made = _spy_loaders(monkeypatch)
    manifest, cache = _fake_cache(tmp_path, [0, 0, 30, 30, 900, 900])
    config = TrainConfig(run_id="w3", checkpoint_dir=str(tmp_path / "ckpt"), data_mode="real", cache_path=cache)
    _model, losses = train_single_client(config, manifest, num_epochs=1, small_et_weight=3.0)
    assert all(math.isfinite(l) for l in losses)
    assert len(made) == 1 and isinstance(made[0]["sampler"], WeightedRandomSampler)
    payload = load_checkpoint(config.checkpoint_dir, config.run_id)
    assert payload["small_et_weight"] == 3.0 and payload["small_et_max_voxels"] == 600
    out = capsys.readouterr().out
    assert out.count("small-ET sampler") == 1 and "2 ET-free, 2 small-ET, 2 large-ET" in out


def test_patch_mode_is_rejected_before_training(tmp_path):
    config = TrainConfig(run_id="p", checkpoint_dir=str(tmp_path / "ckpt"), data_mode="patch", cache_path=str(tmp_path))
    with pytest.raises(Exception):
        train_single_client(config, str(tmp_path / "none.json"), num_epochs=1, small_et_weight=3.0)


# ------------------------------------------------------------------------------------------ Req 94
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


def test_cli_defaults_and_forwarding(monkeypatch):
    args = run.build_arg_parser().parse_args([])
    assert args.small_et_weight is None and args.small_et_max_voxels is None   # unset; 600 is forwarded
    calls = _run_main(monkeypatch, [])
    assert calls["single"]["small_et_weight"] is None and calls["single"]["small_et_max_voxels"] == 600
    calls = _run_main(monkeypatch, ["--small-et-weight", "3"])
    assert calls["single"]["small_et_weight"] == 3.0 and calls["single"]["small_et_max_voxels"] == 600
    calls = _run_main(monkeypatch, ["--small-et-weight", "1", "--small-et-max-voxels", "300"])
    assert calls["single"]["small_et_weight"] == 1.0 and calls["single"]["small_et_max_voxels"] == 300


@pytest.mark.parametrize("argv, names", [
    (["--small-et-weight", "0"], ["--small-et-weight"]),
    (["--small-et-weight", "-2"], ["--small-et-weight"]),
    (["--small-et-weight", "nan"], ["--small-et-weight"]),
    (["--small-et-weight", "3", "--small-et-max-voxels", "0"], ["--small-et-max-voxels"]),
    (["--small-et-max-voxels", "300"], ["--small-et-max-voxels", "--small-et-weight"]),
    (["--small-et-weight", "3", "--use-federation"], ["--small-et-weight", "--use-federation"]),
    (["--small-et-weight", "3", "--data-mode", "patch", "--cache-path", "x"], ["--small-et-weight", "--data-mode patch"]),
])
def test_cli_rejects_bad_or_pointless_values(monkeypatch, capsys, argv, names):
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


def test_cli_help_explains_the_tumour_free_share():
    action = next(a for a in run.build_arg_parser()._actions if a.dest == "small_et_weight")
    assert "tumour-free patients keep their share" in action.help


# ------------------------------------------------------------------------------------------ Req 95
def test_realised_draw_shares_over_200_epochs():
    counts = [0] * 10 + [50] * 10 + [1500] * 10
    probs = small_et_sampling_weights(counts, 3.0, 600)
    sampler = WeightedRandomSampler(weights=probs, num_samples=30, replacement=True,
                                    generator=torch.Generator().manual_seed(1))
    draws = torch.tensor([i for _ in range(200) for i in iter(sampler)])
    shares = [((draws >= lo) & (draws < hi)).double().mean().item() for lo, hi in ((0, 10), (10, 20), (20, 30))]
    for share, expected in zip(shares, (1 / 3, 1 / 2, 1 / 6)):
        assert abs(share - expected) < 0.025
    ratio = shares[1] / shares[2]                     # equal group sizes -> per-patient ratio = group ratio
    assert 2.6 <= ratio <= 3.4
