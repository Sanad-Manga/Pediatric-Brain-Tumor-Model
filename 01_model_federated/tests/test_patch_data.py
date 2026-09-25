import sys
from collections import Counter

import numpy as np
import pytest
import torch

import run
from src.augment3d import Augment3D
from src.config import TrainConfig
from src.patch_data import CATEGORIES, PatchDataset
from src.train_single import train_single_client

from .patch_helpers import make_cohort

P16 = (16, 16, 16)


@pytest.fixture
def cohort(tmp_path):
    return make_cohort(tmp_path)


def _ds(cohort, **kw):
    kw.setdefault("patch_size", P16)
    kw.setdefault("patches_per_epoch", 8)
    return PatchDataset(cohort.manifest, cohort.cache, **kw)


def _window(start, patch=P16):
    return tuple(slice(s, s + p) for s, p in zip(start, patch))


def test_length_shapes_dtypes_and_label_range(cohort):  # Req 37
    ds = _ds(cohort, patches_per_epoch=11)
    assert len(ds) == 11
    for i in range(11):
        x, y = ds[i]
        assert x.dtype == torch.float32 and tuple(x.shape) == (4, *P16)
        assert y.dtype == torch.int64 and tuple(y.shape) == P16
        assert set(torch.unique(y).tolist()) <= {0, 1, 2, 3, 4}


def test_normalisation_matches_independent_computation_and_x_y_share_one_crop(tmp_path):  # Req 38
    c = make_cohort(tmp_path, positional=True)
    ds = _ds(c, patches_per_epoch=1)
    for _ in range(25):
        x, y = ds[0]
        d = ds.last_draw
        sid, win = d["sid"], _window(d["start"])
        raw = c.img[sid][(slice(None),) + win]
        mean, std = c.stats[sid][:, 0], c.stats[sid][:, 1]
        expected = np.zeros_like(raw)
        for ch in range(4):
            brain = raw[ch] > 0
            expected[ch][brain] = (raw[ch][brain] - mean[ch]) / std[ch]
        assert np.allclose(x.numpy(), expected, atol=1e-5)
        assert (x.numpy()[raw <= 0] == 0.0).all(), "background must be exactly 0"
        assert np.array_equal(y.numpy(), c.seg[sid][win].astype(np.int64))
        # the positional channels prove the crop really starts where the draw says it does
        origin = x.numpy()[:3, 0, 0, 0] * std[:3] + mean[:3]
        assert np.allclose(origin, np.array(d["start"]) + 1.0, atol=0.05)


def test_category_rates_match_the_fractions(cohort):  # Req 39
    ds = _ds(cohort)
    n = 3000
    counts = Counter(ds.draw()["category"] for _ in range(n))
    for category, fraction in zip(CATEGORIES, (0.35, 0.45, 0.20)):
        assert abs(counts[category] / n - fraction) <= 0.05, (category, counts)


def test_sampling_guarantees_hold_on_every_draw(cohort):  # Req 40
    ds = _ds(cohort)
    shape = cohort.seg[cohort.sids[0]].shape
    for _ in range(2500):
        d = ds.draw()
        sid, start = d["sid"], np.array(d["start"])
        assert (start >= 0).all() and (start + np.array(P16) <= np.array(shape)).all()
        seg = cohort.seg[sid][_window(d["start"])]
        if d["category"] == "et":
            assert sid in cohort.et_patients
            assert (seg == 1).any()
        elif d["category"] == "tumor":
            assert sid in cohort.tumor_patients
            assert (seg > 0).any()


def test_patches_are_jittered_around_the_centre_voxel(cohort):  # Req 40 (sampling design)
    """The ET patients all carry the same 4-voxel-wide ET block at x = 14..17. Without jitter the block's
    position inside 'et' patches could only vary by the 3 voxels between candidate centres; with the
    specified +-patch//4 jitter it swings across about 11."""
    ds = _ds(cohort)
    offsets = []
    while len(offsets) < 400:
        d = ds.draw()
        if d["category"] == "et":
            offsets.append(14 - d["start"][0])
    assert max(offsets) - min(offsets) >= 7, (min(offsets), max(offsets))


def test_same_seed_reproducible_different_seed_differs_and_calls_differ(cohort):  # Req 41
    a, b = _ds(cohort, seed=7), _ds(cohort, seed=7)
    for i in range(20):
        (xa, ya), (xb, yb) = a[i], b[i]
        assert torch.equal(xa, xb) and torch.equal(ya, yb)
    c = _ds(cohort, seed=8)
    assert any(not torch.equal(_ds(cohort, seed=7)[i][0], c[i][0]) for i in range(20))
    d = _ds(cohort, seed=9)
    first, second = d[0][0], d[0][0]
    assert not torch.equal(first, second), "the same index returned the same patch twice"


def test_patch_size_must_be_a_multiple_of_16(cohort):  # Req 42
    with pytest.raises(ValueError, match="multiple of 16"):
        _ds(cohort, patch_size=(20, 16, 16))


def test_patch_larger_than_the_volume_is_rejected(cohort):  # Req 42
    with pytest.raises(ValueError, match="larger than the volume"):
        _ds(cohort, patch_size=(64, 16, 16))


def test_missing_cache_file_names_the_path(cohort):  # Req 42
    from pathlib import Path

    (Path(cohort.cache) / "P01.seg.npy").unlink()
    with pytest.raises(FileNotFoundError, match=r"P01\.seg\.npy"):
        _ds(cohort)


def test_missing_image_file_is_caught_at_construction_not_at_first_sample(cohort):  # Req 42
    from pathlib import Path

    (Path(cohort.cache) / "P02.img.npy").unlink()
    with pytest.raises(FileNotFoundError, match=r"P02\.img\.npy"):
        _ds(cohort)


def test_category_with_no_eligible_patient_is_an_error(tmp_path):  # Req 42
    no_et = make_cohort(tmp_path / "a", n_et=0, n_tumor_only=2, n_empty=1)
    with pytest.raises(ValueError, match="'et'"):
        _ds(no_et)
    _ds(no_et, fractions=(0.0, 0.5, 0.5))  # zero fraction: nothing needs to be eligible
    empty = make_cohort(tmp_path / "b", n_et=0, n_tumor_only=0, n_empty=3)
    with pytest.raises(ValueError, match="'tumor'"):
        _ds(empty, fractions=(0.0, 0.5, 0.5))


@pytest.mark.parametrize("fractions", [(-0.1, 0.6, 0.5), (0.5, 0.5, 0.5), (0.35, 0.45, 0.2001)])
def test_invalid_fractions_are_rejected(cohort, fractions):  # Req 42
    with pytest.raises(ValueError, match="fractions"):
        _ds(cohort, fractions=fractions)


def test_fractions_within_tolerance_are_accepted(cohort):  # Req 42
    _ds(cohort, fractions=(0.35, 0.45, 0.2000005))


def test_train_config_patch_fields_and_defaults():  # Req 45
    cfg = TrainConfig()
    assert cfg.patch_size == (128, 128, 128) and cfg.patches_per_epoch == 580
    assert cfg.patch_fractions == (0.35, 0.45, 0.20)
    ok = TrainConfig(data_mode="patch", cache_path="x", patch_size=[32, 32, 32])
    assert ok.patch_size == (32, 32, 32)
    with pytest.raises(ValueError, match="cache_path"):
        TrainConfig(data_mode="patch")
    with pytest.raises(ValueError, match="cache_path"):
        TrainConfig(data_mode="real")  # existing behaviour unchanged
    for bad in (dict(patch_size=(32, 32)), dict(patch_size=(0, 16, 16)), dict(patch_size=(16.5, 16, 16)),
                dict(patches_per_epoch=0), dict(patch_fractions=(0.5, 0.5, 0.5)),
                dict(patch_fractions=(-0.2, 0.6, 0.6))):
        with pytest.raises(ValueError):
            TrainConfig(**bad)


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


def test_cli_patch_flags_reach_the_config(monkeypatch):  # Req 46
    calls = _run_main(monkeypatch, [
        "--data-mode", "patch", "--cache-path", "somewhere", "--patch-size", "32", "48", "64",
        "--patches-per-epoch", "9", "--patch-fractions", "0.5", "0.3", "0.2"])
    cfg = calls["single"]["config"]
    assert cfg.data_mode == "patch" and cfg.cache_path == "somewhere"
    assert cfg.patch_size == (32, 48, 64) and cfg.patches_per_epoch == 9
    assert cfg.patch_fractions == (0.5, 0.3, 0.2)
    plain = _run_main(monkeypatch, [])["single"]["config"]  # existing invocation unchanged
    assert plain.data_mode == "dummy" and plain.patch_size == (128, 128, 128)


def test_cli_patch_with_federation_is_an_argparse_error(monkeypatch, capsys):  # Req 46
    ran = {}
    monkeypatch.setattr(run, "train_single_client", lambda **kw: ran.setdefault("ran", True))
    monkeypatch.setattr(run, "train_federated", lambda **kw: ran.setdefault("ran", True))
    monkeypatch.setattr(sys, "argv", ["run.py", "--data-mode", "patch", "--cache-path", "x", "--use-federation"])
    with pytest.raises(SystemExit) as exc:
        run.main()
    assert exc.value.code == 2
    assert "--use-federation" in capsys.readouterr().err
    assert "ran" not in ran


def test_patch_mode_trains_end_to_end_with_augmentation_and_dropout(tmp_path):  # Req 47
    c = make_cohort(tmp_path, n_et=2, n_tumor_only=1, n_empty=1)
    config = TrainConfig(run_id="patch_e2e", checkpoint_dir=str(tmp_path / "ckpt"), use_augmentation=True,
                         data_mode="patch", cache_path=c.cache, patch_size=(32, 32, 32), patches_per_epoch=2)
    _model, losses = train_single_client(
        config, c.manifest, num_epochs=1, augmentation_transform=Augment3D(modality_dropout_prob=0.3))
    assert len(losses) == 1 and np.isfinite(losses[0])
    assert (tmp_path / "ckpt" / "patch_e2e" / "epoch_0.pt").is_file()
