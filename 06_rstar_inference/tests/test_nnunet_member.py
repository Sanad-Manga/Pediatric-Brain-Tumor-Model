"""SPEC Addendum 4 (Req 44-52): nnU-Net as an optional third member, off by default."""
import json
import os
from pathlib import Path

import numpy as np
import pytest
from _helpers import TINY, make_volume, tiny_config

from rstar import RStarConfig, RStarSegmenter
from rstar.cli import main
from rstar.config import load_manifest, validate_manifest
from rstar.fusion import apply_small_et_rule, cyst_override, fuse, fuse_weighted, remove_fragments
from rstar.guards import IdentityStub
from rstar.models import ModelIntegrityError
from rstar.nnunet_member import load_nnunet, nnunet_probs

MODELS_ROOT, TEST_DATA = os.environ.get("RSTAR_MODELS_ROOT"), os.environ.get("RSTAR_TEST_DATA")
BLOB = (slice(10, 30), slice(12, 34), slice(8, 28))    # 8800-voxel tumour
ET = (slice(14, 24), slice(16, 26), slice(12, 20))      # 800 ET voxels inside it


def _softmax(z, axis=0):
    e = np.exp(z - z.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def _pointwise_logits(x):
    """Five logits per voxel from the four channels: any fixed per-voxel function will do."""
    return np.stack([x[0] * 0.01, x[1] * 0.02, x[2] * 0.03 - x[0] * 0.01, x[3] * 0.01, (x[0] - x[3]) * 0.02]).astype(np.float32)


class StubPredictor:
    """nnU-Net's predict_single_npy_array interface; returns the softmax of _pointwise_logits on the (C, Z, Y, X) it gets,
    or fixed probabilities given in the (5, X, Y, Z) frame."""

    def __init__(self, fixed=None, fail=False):
        self.calls, self.last_image, self.last_props, self.fixed, self.fail = 0, None, None, fixed, fail

    def predict_single_npy_array(self, input_image, image_properties, segmentation_previous_stage=None,
                                 output_file_truncated=None, save_or_return_probabilities=False):
        if self.fail:
            raise AssertionError("the predictor must not be called")
        self.calls += 1
        self.last_image, self.last_props = input_image.copy(), dict(image_properties)
        assert save_or_return_probabilities and output_file_truncated is None and segmentation_previous_stage is None
        probs = self.fixed.transpose(0, 3, 2, 1) if self.fixed is not None else _softmax(_pointwise_logits(input_image))
        return probs.argmax(0), probs.copy()


def _probs(et_prob=0.9):
    p = np.zeros((5, *TINY), np.float32)
    p[0] = 1.0
    for region, cls, prob in ((BLOB, 4, 0.9), (ET, 1, et_prob)):
        p[(slice(None), *region)] = 0.0
        p[(cls, *region)] = prob
        p[(0, *region)] = 1.0 - prob
    return p


def _seg(p2=None, p3=None, pn=None, predictor=None, **cfg):
    """Segmenter whose 2D / 3D / nnU-Net probabilities are fixed volumes; counts calls per member."""
    seg = RStarSegmenter(tiny_config(**cfg), models_2d=[IdentityStub()], models_3d=[IdentityStub()],
                         nnunet_predictor=predictor if predictor is not None else StubPredictor(fixed=pn if pn is not None else _probs()))
    seg.calls = {"2d": 0, "3d": 0}
    p2 = _probs() if p2 is None else p2
    p3 = _probs() if p3 is None else p3

    def probs_2d(vol, models=None):
        seg.calls["2d"] += 1
        return p2.copy()

    def probs_3d(vol, present, models=None):
        seg.calls["3d"] += 1
        return p3.copy()

    seg._probs_2d, seg._probs_3d = probs_2d, probs_3d
    return seg


# ------------------------------------------------------------------------------------------------ Req 44
def test_nnunet_is_off_by_default_and_validated():
    assert RStarConfig().nnunet is False
    for bad in (1, "yes", None):
        with pytest.raises(ValueError, match="nnunet"):
            RStarConfig(nnunet=bad)


def test_manifest_nnunet_entry_is_optional_and_validated():
    manifest = load_manifest()
    assert "nnunet" in manifest and manifest["nnunet"]["path"].endswith("fold_all/checkpoint_final.pth")
    without = {k: v for k, v in manifest.items() if k != "nnunet"}
    assert validate_manifest(dict(without)) == without
    for bad in ({"name": "n", "path": "C:/abs/checkpoint_final.pth", "sha256": "a" * 64},
                {"name": "n", "path": "rel/checkpoint_final.pth", "sha256": "xyz"}, {"name": "n", "path": "rel"}):
        with pytest.raises(ValueError, match="nnunet"):
            validate_manifest(dict(without, nnunet=bad))


def test_missing_entry_and_hash_mismatch_fail_before_nnunet_opens_anything(tmp_path):
    manifest = load_manifest()
    without = {k: v for k, v in manifest.items() if k != "nnunet"}
    path = tmp_path / "m.json"
    path.write_text(json.dumps(without))
    with pytest.raises(ValueError, match="nnunet"):
        load_nnunet(RStarConfig(manifest_path=path, nnunet=True))
    ck = tmp_path / "model" / "fold_all" / "checkpoint_final.pth"
    ck.parent.mkdir(parents=True)
    ck.write_bytes(b"not the pinned file")
    path.write_text(json.dumps(dict(without, nnunet={"name": "n", "path": "model/fold_all/checkpoint_final.pth", "sha256": "0" * 64})))
    with pytest.raises(ModelIntegrityError, match="checkpoint_final"):
        load_nnunet(RStarConfig(manifest_path=path, nnunet=True, models_root=tmp_path))


# ------------------------------------------------------------------------------------------------ Req 45
def test_fuse_weighted_equals_an_independent_computation():
    rng = np.random.default_rng(1)
    ps = [_softmax(rng.normal(size=(5, 6, 7, 8)).astype(np.float32)) for _ in range(3)]
    w = [0.2, 0.3, 0.5]
    acc = sum(np.float32(wi) * p for wi, p in zip(w, ps))
    acc[0] *= np.float32(0.5)
    assert np.array_equal(fuse_weighted(ps, w, 0.5), np.argmax(acc, axis=0).astype(np.uint8))


@pytest.mark.parametrize("w3d", [0.0, 0.3, 0.5, 1.0])
def test_fuse_weighted_with_two_members_equals_fuse(w3d):
    rng = np.random.default_rng(2)
    p2, p3 = (_softmax(rng.normal(size=(5, 9, 8, 7)).astype(np.float32)) for _ in range(2))
    assert np.array_equal(fuse_weighted([p2, p3], [1.0 - w3d, w3d], 0.5), fuse(p2, p3, w3d, 0.5))


def test_fuse_weighted_ties_and_invalid_inputs():
    tie = np.full((5, 2, 2, 2), 0.2, np.float32)
    assert (fuse_weighted([tie, tie], [0.5, 0.5], 1.0) == 0).all()
    p = np.zeros((5, 2, 2, 2), np.float32)
    for args in (([p, p], [1.0]), ([p, p[:, :1]], [0.5, 0.5]), ([p, p], [-0.5, 1.5]), ([p, p], [0.5, 0.6]), ([], [])):
        with pytest.raises(ValueError):
            fuse_weighted(*args, 0.5)


# ------------------------------------------------------------------------------------------------ Req 46
@pytest.mark.parametrize("present", [(True, True, True, True), (True, False, True, True)])
@pytest.mark.parametrize("extra", [{}, {"review_flags": True, "review_min_voxels": 1}, {"fragment_cleanup": True}])
def test_off_by_default_the_predictor_is_never_touched_and_outputs_are_unchanged(present, extra):
    vol, _ = make_volume()
    with_stub = _seg(predictor=StubPredictor(fail=True), **extra).segment(vol, present)
    plain = RStarSegmenter(tiny_config(**extra), models_2d=[IdentityStub()], models_3d=[IdentityStub()])
    plain._probs_2d = lambda v, models=None: _probs()
    plain._probs_3d = lambda v, pr, models=None: _probs()
    ref = plain.segment(vol, present)
    assert np.array_equal(with_stub.labels, ref.labels)
    assert (with_stub.mode, with_stub.status, with_stub.warnings) == (ref.mode, ref.status, ref.warnings)
    a = {k: v for k, v in with_stub.diagnostics.items() if k != "elapsed_s"}
    b = {k: v for k, v in ref.diagnostics.items() if k != "elapsed_s"}
    assert a == b and with_stub.mode in ("R*", "3D-only")
    if extra.get("review_flags"):
        assert np.array_equal(with_stub.review_mask, ref.review_mask) and with_stub.review_spots == ref.review_spots


# ------------------------------------------------------------------------------------------------ Req 47
def test_nnunet_probs_axis_reversal_is_consistent_in_and_out():
    vol, _ = make_volume()
    before = vol.copy()
    stub = StubPredictor()
    out = nnunet_probs(stub, vol, (True, True, True, True))
    assert stub.last_image.shape == (4, TINY[2], TINY[1], TINY[0])
    assert stub.last_props == {"spacing": [1.0, 1.0, 1.0]}
    assert out.shape == (5, *TINY) and out.dtype == np.float32
    np.testing.assert_allclose(out, _softmax(_pointwise_logits(vol)), atol=1e-6)
    assert np.array_equal(vol, before)


def test_absent_channels_reach_nnunet_as_exact_zeros():
    vol, _ = make_volume()
    vol[1] = 123.0                                             # garbage in an absent channel is ignored
    stub = StubPredictor()
    nnunet_probs(stub, vol, (True, False, True, True))
    assert (stub.last_image[1] == 0).all() and (stub.last_image[0] != 0).any()
    assert (vol[1] == 123.0).all()


# ------------------------------------------------------------------------------------------------ Req 48
def test_all_present_gives_the_three_way_fusion():
    vol, _ = make_volume()
    p2, p3, pn = _probs(0.9), _probs(0.4), _probs(0.6)
    seg = _seg(p2, p3, pn, nnunet=True)
    result = seg.segment(vol)
    assert result.mode == "R*+nnU-Net" and seg.calls == {"2d": 1, "3d": 1} and seg.nnunet_predictor.calls == 1
    expected = apply_small_et_rule(fuse_weighted([p2, p3, pn], [1 / 3, 1 / 3, 1 / 3], 0.5), 500.0, 1.0)[0]
    assert np.array_equal(result.labels, expected)
    assert result.status == "ok" and result.warnings == []


def test_one_absent_gives_3d_plus_nnunet_without_2d():
    vol, _ = make_volume()
    p3, pn = _probs(0.4), _probs(0.7)
    seg = _seg(p3=p3, pn=pn, nnunet=True)
    result = seg.segment(vol, (True, True, False, True))
    assert result.mode == "3D+nnU-Net" and seg.calls["2d"] == 0 and seg.calls["3d"] == 1 and seg.nnunet_predictor.calls == 1
    assert np.array_equal(result.labels, apply_small_et_rule(fuse_weighted([p3, pn], [0.5, 0.5], 0.5), 500.0, 1.0)[0])
    assert result.status == "ok" and len(result.warnings) == 1
    assert "t2f" in result.warnings[0] and "3D+nnU-Net" in result.warnings[0]
    assert result.diagnostics["agreement"] is None


def test_two_absent_is_review():
    vol, _ = make_volume()
    result = _seg(nnunet=True).segment(vol, (True, False, False, True))
    assert result.mode == "3D+nnU-Net" and result.status == "review"


def test_the_2d_vs_3d_agreement_guard_still_applies():
    vol, _ = make_volume()
    empty = np.zeros((5, *TINY), np.float32)
    empty[0] = 1.0                                             # 2D predicts nothing: agreement 0.0
    result = _seg(p2=empty, nnunet=True).segment(vol)
    assert result.mode == "R*+nnU-Net" and result.status == "review" and result.diagnostics["agreement"] == 0.0
    assert any("collapse" in w and "orientation" in w for w in result.warnings)
    assert result.labels is not None and (result.labels > 0).any()   # 3D + nnU-Net still outvote the collapsed 2D


def test_the_nnunet_checkpoint_is_listed_when_loaded(monkeypatch):
    import rstar.pipeline as pipeline_mod

    monkeypatch.setattr(pipeline_mod.nnunet_member, "load_nnunet",
                        lambda cfg: (StubPredictor(fixed=_probs()), [{"name": "nn", "sha256": "f" * 64}]))
    seg = RStarSegmenter(tiny_config(nnunet=True), models_2d=[IdentityStub()], models_3d=[IdentityStub()])
    seg._probs_2d = lambda v, models=None: _probs()
    seg._probs_3d = lambda v, pr, models=None: _probs()
    vol, _ = make_volume()
    assert {"name": "nn", "sha256": "f" * 64} in seg.segment(vol).diagnostics["checkpoints"]


# ------------------------------------------------------------------------------------------------ Req 49
@pytest.mark.parametrize("present, members, weights", [
    ((True, True, True, True), ("p2", "p3", "pn"), (1 / 3, 1 / 3, 1 / 3)),
    ((True, True, True, False), ("p3", "pn"), (0.5, 0.5)),
])
def test_review_flags_use_the_weighted_et_probability(present, members, weights):
    vol, _ = make_volume()
    probs = {"p2": _probs(0.5), "p3": _probs(0.6), "pn": _probs(0.4)}
    result = _seg(probs["p2"], probs["p3"], probs["pn"], nnunet=True, review_flags=True, review_prob_cut=0.99).segment(vol, present)
    et_prob = sum(np.float32(w) * probs[m][1] for m, w in zip(members, weights))
    assert len(result.review_spots) == 1
    assert result.review_spots[0]["mean_et_prob"] == pytest.approx(float(et_prob[ET].mean()), abs=1e-6)


def test_cleanup_composes_with_nnunet():
    vol, _ = make_volume()
    p2, p3, pn = _probs(0.9), _probs(0.4), _probs(0.6)
    stray = (slice(36, 39), slice(44, 47), slice(30, 33))
    for p in (p2, p3, pn):
        p[(slice(None), *stray)] = 0.0
        p[(4, *stray)] = 1.0
    result = _seg(p2, p3, pn, nnunet=True, fragment_cleanup=True).segment(vol)
    pre = fuse_weighted([p2, p3, pn], [1 / 3, 1 / 3, 1 / 3], 0.5)
    cleaned, removed, _ = remove_fragments(pre, 200)
    assert np.array_equal(result.labels, apply_small_et_rule(cleaned, 500.0, 1.0)[0])
    assert result.diagnostics["fragment_voxels_removed"] == removed == 27


# ------------------------------------------------------------------------------------------------ Req 50
def _write_case(tmp_path):
    import nibabel as nib

    vol, _ = make_volume()
    paths = {}
    for c, name in enumerate(("t1c", "t1n", "t2f", "t2w")):
        paths[name] = tmp_path / f"{name}.nii.gz"
        nib.save(nib.Nifti1Image(vol[c], np.eye(4)), str(paths[name]))
    return paths


def test_cli_nnunet_flag_sets_the_option_and_reports_the_mode(tmp_path):
    paths = _write_case(tmp_path)
    seg = _seg()
    args = [f"--{k}={v}" for k, v in paths.items()] + [f"--out={tmp_path / 'seg.nii.gz'}", f"--json={tmp_path / 'r.json'}"]
    assert main(args + ["--nnunet"], segmenter=seg) == 0
    assert seg.cfg.nnunet is True
    assert json.loads((tmp_path / "r.json").read_text())["mode"] == "R*+nnU-Net"
    seg2 = _seg()
    assert main(args, segmenter=seg2) == 0 and seg2.cfg.nnunet is False
    assert json.loads((tmp_path / "r.json").read_text())["mode"] == "R*"


def test_cli_builds_the_segmenter_with_the_option(tmp_path, monkeypatch):
    import rstar.pipeline as pipeline_mod

    built = {}

    class Fake:
        def __init__(self, cfg):
            built["cfg"] = cfg
            raise SystemExit(0)

    monkeypatch.setattr(pipeline_mod, "RStarSegmenter", Fake)
    paths = _write_case(tmp_path)
    for flag, expected in (([], False), (["--nnunet"], True)):
        with pytest.raises(SystemExit):
            main([f"--{k}={v}" for k, v in paths.items()] + [f"--out={tmp_path / 's.nii.gz'}"] + flag)
        assert built["cfg"].nnunet is expected


# ------------------------------------------------------------------------------------------------ Req 51 (slow)
needs_data = pytest.mark.skipif(not (MODELS_ROOT and TEST_DATA), reason="set RSTAR_MODELS_ROOT and RSTAR_TEST_DATA")
REFERENCE = json.loads((Path(__file__).parent / "fixtures" / "rstar_nnunet_reference_scores.json").read_text(encoding="utf-8"))["patients"]


def _dice(a, b):
    sa, sb = int(a.sum()), int(b.sum())
    return 1.0 if sa == sb == 0 else 2.0 * int((a & b).sum()) / (sa + sb)


@needs_data
def test_the_nnunet_hash_matches_the_real_file():
    from rstar.models import verify_sha256

    entry = load_manifest()["nnunet"]
    verify_sha256(Path(MODELS_ROOT) / entry["path"], entry["sha256"])


@needs_data
def test_three_fresh_patients_reproduce_the_fusion_scores():
    import nibabel as nib

    seg = RStarSegmenter(RStarConfig(models_root=Path(MODELS_ROOT), nnunet=True))
    for sid, ref in REFERENCE.items():
        base = Path(TEST_DATA) / sid
        result, _ = seg.segment_paths({n: base / f"{sid}-{n}.nii.gz" for n in ("t1c", "t1n", "t2f", "t2w")})
        truth = np.rint(np.asarray(nib.load(str(base / f"{sid}-seg.nii.gz")).dataobj, dtype=np.float32)).astype(np.int64)
        lab = result.labels.astype(np.int64)
        assert result.mode == "R*+nnU-Net"
        got = (_dice(lab == 1, truth == 1) + _dice(np.isin(lab, (1, 2, 3)), np.isin(truth, (1, 2, 3))) + _dice(lab > 0, truth > 0)) / 3
        assert got == pytest.approx(ref["mean"], abs=0.005), sid       # F + cyst override (Req 55)


# ------------------------------------------------------------------------------------------------ Req 53
def test_cyst_override_exact_on_a_hand_built_volume():
    lab = np.zeros((6, 7, 8), np.int16)
    lab[1:4, 1:4, 1:4] = 4
    lab[2, 2, 2] = 1
    lab[5, 6, 7] = 3                                    # already cyst: stays 3, not counted
    pn = np.zeros((5, 6, 7, 8), np.float32)
    pn[0] = 1.0
    cyst = np.zeros((6, 7, 8), bool)
    cyst[2:5, 2:4, 2:6] = True                          # covers tumour, ET, background
    cyst[5, 6, 7] = True
    pn[3][cyst] = 2.0
    before = lab.copy()
    out, changed = cyst_override(lab, pn)
    assert np.array_equal(lab, before) and out.dtype == lab.dtype
    assert (out[cyst] == 3).all() and np.array_equal(out[~cyst], lab[~cyst])
    assert changed == int((cyst & (lab != 3)).sum()) == int((out != lab).sum())
    with pytest.raises(ValueError):
        cyst_override(lab, pn[:, :5])


# ------------------------------------------------------------------------------------------------ Req 54
def _with_cyst(p, region, prob=0.9):
    p = p.copy()
    p[(slice(None), *region)] = 0.0
    p[(3, *region)] = prob
    p[(0, *region)] = 1.0 - prob
    return p


CYST = (slice(12, 16), slice(28, 32), slice(20, 24))     # 64 voxels inside BLOB, outside ET


@pytest.mark.parametrize("present", [(True, True, True, True), (True, True, False, True)])
def test_pipeline_applies_the_cyst_override_after_fusion(present):
    vol, _ = make_volume()
    p2, p3, pn = _probs(0.9), _probs(0.8), _with_cyst(_probs(0.7), CYST, prob=0.6)
    result = _seg(p2, p3, pn, nnunet=True).segment(vol, present)
    members, w = ([p2, p3, pn], [1 / 3] * 3) if all(present) else ([p3, pn], [0.5, 0.5])
    fused = fuse_weighted(members, w, 0.5)
    assert not (fused[CYST] == 3).all()                 # the fusion alone would not call the cyst
    over, n = cyst_override(fused, pn)
    assert np.array_equal(result.labels, apply_small_et_rule(over, 500.0, 1.0)[0])
    assert (result.labels[CYST] == 3).all() and result.diagnostics["cyst_override_voxels"] == n == 64


def test_flags_and_cleanup_see_the_overridden_labels():
    vol, _ = make_volume()
    cyst_in_et = (slice(14, 18), slice(16, 20), slice(12, 16))   # 64 voxels inside the ET block
    pn = _with_cyst(_probs(0.6), cyst_in_et)
    result = _seg(_probs(0.5), _probs(0.6), pn, nnunet=True, review_flags=True, review_prob_cut=0.99).segment(vol)
    assert result.review_spots[0]["voxels"] == 800 - 64           # the cyst voxels are no longer part of the ET spot
    assert (result.review_mask[cyst_in_et] == 0).all()


def test_no_cyst_override_key_when_off():
    vol, _ = make_volume()
    result = _seg(pn=_with_cyst(_probs(), CYST)).segment(vol)
    assert "cyst_override_voxels" not in result.diagnostics and not (result.labels[CYST] == 3).any()


# ------------------------------------------------------------------------------------------------ Req 56 (Addendum 4c)
def _multi(paths, shas=None):
    return {"name": "n5", "checkpoints": [{"path": p, "sha256": s} for p, s in zip(paths, shas or ["a" * 64] * len(paths))]}


def test_multi_fold_entry_is_validated():
    base = {k: v for k, v in load_manifest().items() if k != "nnunet"}
    ok = _multi([f"m/fold_{k}/checkpoint_final.pth" for k in range(5)])
    assert validate_manifest(dict(base, nnunet=ok))["nnunet"] == ok
    bad = [
        {"name": "n5", "checkpoints": []},
        dict(_multi(["m/fold_0/c.pth"]), path="m/fold_0/c.pth", sha256="a" * 64),
        {"checkpoints": [{"path": "m/fold_0/c.pth", "sha256": "a" * 64}]},
        _multi(["m/fold_0/c.pth", "C:/abs/fold_1/c.pth"]),
        _multi(["m/fold_0/c.pth"], ["xyz"]),
        _multi(["m/fold_0/c.pth", "other/fold_1/c.pth"]),
    ]
    for entry in bad:
        with pytest.raises(ValueError, match="nnunet"):
            validate_manifest(dict(base, nnunet=entry))


def test_the_226_patient_manifest_lists_five_folds_of_one_model():
    m = load_manifest(Path(__file__).parents[1] / "config" / "models.nnunet226.json")
    paths = [c["path"] for c in m["nnunet"]["checkpoints"]]
    assert len(paths) == 5 and all(p.endswith(f"fold_{k}/checkpoint_final.pth") for k, p in enumerate(paths))
    assert m["ensemble_2d"] == load_manifest()["ensemble_2d"] and m["family_3d"] == load_manifest()["family_3d"]


# ------------------------------------------------------------------------------------------------ Req 57
def _fold_files(tmp_path, n=3):
    import hashlib

    entries = []
    for k in range(n):
        f = tmp_path / "model" / f"fold_{k}" / "checkpoint_final.pth"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(f"fold {k}".encode())
        entries.append({"path": f"model/fold_{k}/checkpoint_final.pth", "sha256": hashlib.sha256(f.read_bytes()).hexdigest()})
    return entries


def _manifest_with(tmp_path, entry):
    base = {k: v for k, v in load_manifest().items() if k != "nnunet"}
    path = tmp_path / "m.json"
    path.write_text(json.dumps(dict(base, nnunet=entry)))
    return path


def test_every_fold_hash_is_checked_before_any_file_is_opened(tmp_path, monkeypatch):
    import torch

    opened = []
    monkeypatch.setattr(torch, "load", lambda *a, **k: opened.append(a[0]))
    entries = _fold_files(tmp_path)
    entries[-1]["sha256"] = "0" * 64                    # the LAST fold is wrong
    path = _manifest_with(tmp_path, {"name": "n", "checkpoints": entries})
    with pytest.raises(ModelIntegrityError, match="fold_2"):
        load_nnunet(RStarConfig(manifest_path=path, nnunet=True, models_root=tmp_path))
    assert opened == []


def test_folds_reach_nnunet_in_manifest_order_with_one_info_item_each(tmp_path, monkeypatch):
    pytest.importorskip("nnunetv2")
    import torch
    import nnunetv2.inference.predict_from_raw_data as pr
    import nnunetv2.training.nnUNetTrainer.nnUNetTrainer as tr
    import nnunetv2.utilities.plans_handling.plans_handler as ph
    import batchgenerators.utilities.file_and_folder_operations as fo
    import nnunetv2.utilities.label_handling.label_handling as lh

    entries = _fold_files(tmp_path)
    path = _manifest_with(tmp_path, {"name": "n", "checkpoints": entries})
    monkeypatch.setattr(torch, "load", lambda p, **k: {"network_weights": f"weights of {Path(p).parent.name}",
                                                       "init_args": {"configuration": "3d_fullres"}, "trainer_name": "t"})
    monkeypatch.setattr(fo, "load_json", lambda p: {})

    class FakePlans:
        def __init__(self, plans):
            pass

        def get_configuration(self, name):
            return name

        def get_label_manager(self, ds):
            return type("LM", (), {"num_segmentation_heads": 5})()

    monkeypatch.setattr(ph, "PlansManager", FakePlans)
    monkeypatch.setattr(lh, "determine_num_input_channels", lambda *a: 4)
    monkeypatch.setattr(tr.nnUNetTrainer, "build_network_architecture", staticmethod(lambda *a, **k: "net"))
    seen = {}

    class FakePredictor:
        def __init__(self, **kwargs):
            pass

        def manual_initialization(self, network, plans, conf, params, *rest):
            seen["params"] = params

    monkeypatch.setattr(pr, "nnUNetPredictor", FakePredictor)
    _, info = load_nnunet(RStarConfig(manifest_path=path, nnunet=True, models_root=tmp_path, device="cpu"))
    assert seen["params"] == ["weights of fold_0", "weights of fold_1", "weights of fold_2"]
    assert [i["sha256"] for i in info] == [e["sha256"] for e in entries] and len({i["name"] for i in info}) == 3


# ------------------------------------------------------------------------------------------------ Req 58 (slow)
REF226 = json.loads((Path(__file__).parent / "fixtures" / "rstar_nnunet226_reference_scores.json").read_text(encoding="utf-8"))["patients"]


@needs_data
def test_the_226_patient_ensemble_reproduces_its_reference():
    import nibabel as nib

    manifest = Path(__file__).parents[1] / "config" / "models.nnunet226.json"
    folds = [Path(MODELS_ROOT) / c["path"] for c in load_manifest(manifest)["nnunet"]["checkpoints"]]
    if not all(f.is_file() for f in folds):
        pytest.skip("the 226-patient fold checkpoints are not under RSTAR_MODELS_ROOT")
    seg = RStarSegmenter(RStarConfig(models_root=Path(MODELS_ROOT), manifest_path=manifest, nnunet=True))
    assert len([c for c in seg._checkpoints if "nnunet" in c["name"]]) == 5
    for sid, ref in REF226.items():
        base = Path(TEST_DATA) / sid
        result, _ = seg.segment_paths({n: base / f"{sid}-{n}.nii.gz" for n in ("t1c", "t1n", "t2f", "t2w")})
        truth = np.rint(np.asarray(nib.load(str(base / f"{sid}-seg.nii.gz")).dataobj, dtype=np.float32)).astype(np.int64)
        lab = result.labels.astype(np.int64)
        got = (_dice(lab == 1, truth == 1) + _dice(np.isin(lab, (1, 2, 3)), np.isin(truth, (1, 2, 3))) + _dice(lab > 0, truth > 0)) / 3
        assert got == pytest.approx(ref["mean"], abs=0.005), sid
