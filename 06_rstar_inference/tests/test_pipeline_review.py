"""SPEC Addendum 2 (Req 29-36): review flags wired into RStarSegmenter, off by default, labels unchanged."""
import json
import math
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from _helpers import TINY, make_volume, tiny_config

import rstar.pipeline as pipeline_mod
from rstar import RStarConfig, RStarSegmenter
from rstar.cli import main
from rstar.guards import IdentityStub

REQ14_KEYS = {"agreement", "et_voxels_before_rule", "et_relabelled", "present", "checkpoints", "elapsed_s", "mode"}
A = (slice(4, 10), slice(4, 10), slice(4, 10))      # 216 voxels, confident enhancing tumour
B = (slice(20, 26), slice(30, 36), slice(20, 26))   # 216 voxels, low-probability enhancing tumour
C = (slice(36, 39), slice(8, 11), slice(30, 33))    # 27 voxels, low-probability, below review_min_voxels


def _probs(a_et=0.95, b_et=0.5, c_et=0.5):
    """(5, *TINY) probabilities: background everywhere except the three blobs, where ET wins the fused argmax."""
    p = np.zeros((5, *TINY), dtype=np.float32)
    p[0] = 1.0
    for region, et in ((A, a_et), (B, b_et), (C, c_et)):
        p[(slice(None), *region)] = 0.0
        p[(1, *region)] = et
        p[(0, *region)] = (1.0 - et) * 0.6
        p[(2, *region)] = (1.0 - et) * 0.4
    return p


def _segmenter(monkeypatch=None, probs=None, **cfg):
    seg = RStarSegmenter(tiny_config(**cfg), models_2d=[IdentityStub()], models_3d=[IdentityStub()])
    p = _probs() if probs is None else probs
    seg._probs_2d = lambda vol, models=None: p.copy()
    seg._probs_3d = lambda vol, present, models=None: p.copy()
    return seg


def _mask(region):
    m = np.zeros(TINY, dtype=bool)
    m[region] = True
    return m


# ------------------------------------------------------------------------------------------------ Req 29
def test_review_flag_config_defaults():
    cfg = RStarConfig()
    assert cfg.review_flags is False and cfg.review_prob_cut == 0.7 and cfg.review_min_voxels == 50


@pytest.mark.parametrize("field, value", [
    ("review_prob_cut", -0.1), ("review_prob_cut", 1.5), ("review_prob_cut", math.nan),
    ("review_min_voxels", -1), ("review_min_voxels", 2.5), ("review_min_voxels", True),
    ("review_flags", 1), ("review_flags", "yes"),
])
def test_invalid_review_flag_settings_raise_naming_the_field(field, value):
    with pytest.raises(ValueError, match=field):
        RStarConfig(**{field: value})


# ------------------------------------------------------------------------------------------------ Req 30
def test_flags_are_off_by_default_and_add_nothing():
    vol, _ = make_volume()
    result = _segmenter().segment(vol)
    assert result.review_mask is None and result.review_spots is None
    assert set(result.diagnostics) == REQ14_KEYS


# ------------------------------------------------------------------------------------------------ Req 31
@pytest.mark.parametrize("present, et_min", [
    ((True, True, True, True), 500.0),
    ((True, True, False, True), 500.0),               # 3D-only mode
    ((True, True, True, True), 1e9),                  # the 500 mm^3 rule (here: any total) relabels all ET
])
def test_labels_status_and_diagnostics_are_identical_with_flags_on_and_off(present, et_min):
    vol, _ = make_volume()
    off = _segmenter(et_min_mm3=et_min).segment(vol, present)
    on = _segmenter(et_min_mm3=et_min, review_flags=True).segment(vol, present)
    assert np.array_equal(off.labels, on.labels)
    assert (off.mode, off.status, off.warnings) == (on.mode, on.status, on.warnings)
    for key in REQ14_KEYS - {"elapsed_s"}:
        assert off.diagnostics[key] == on.diagnostics[key], key
    assert set(on.diagnostics) == REQ14_KEYS | {"review_spot_count"}


# ------------------------------------------------------------------------------------------------ Req 32
def test_exactly_the_large_low_confidence_spot_is_flagged():
    vol, _ = make_volume()
    result = _segmenter(review_flags=True).segment(vol)
    assert len(result.review_spots) == 1
    spot = result.review_spots[0]
    assert spot["spot_id"] == 1 and spot["voxels"] == int(_mask(B).sum())
    assert spot["mean_et_prob"] == pytest.approx(0.5, abs=1e-6)          # 0.5 * 0.5 (3D) + 0.5 * 0.5 (2D)
    assert spot["reason"] == "low_confidence" and spot["models_agree"] is None
    assert result.review_mask.dtype == np.uint8 and result.review_mask.shape == TINY
    assert np.array_equal(result.review_mask == 1, _mask(B)) and set(np.unique(result.review_mask)) == {0, 1}
    assert result.diagnostics["review_spot_count"] == 1


def test_the_mean_probability_uses_the_fused_branches_in_r_star_mode_and_p3_alone_in_3d_only_mode():
    vol, _ = make_volume()
    p2, p3 = _probs(b_et=0.6), _probs(b_et=0.5)
    seg = _segmenter(review_flags=True, w3d=0.25)
    seg._probs_2d = lambda v, models=None: p2.copy()
    seg._probs_3d = lambda v, present, models=None: p3.copy()
    fused = seg.segment(vol).review_spots[0]["mean_et_prob"]
    assert fused == pytest.approx(0.25 * 0.5 + 0.75 * 0.6, abs=1e-6)
    alone = seg.segment(vol, (True, False, True, True)).review_spots[0]["mean_et_prob"]
    assert alone == pytest.approx(0.5, abs=1e-6)


def test_no_enhancing_tumour_gives_empty_flags():
    vol, _ = make_volume()
    p = np.zeros((5, *TINY), dtype=np.float32)
    p[0] = 1.0
    result = _segmenter(probs=p, review_flags=True).segment(vol)
    assert not result.review_mask.any() and result.review_spots == [] and result.diagnostics["review_spot_count"] == 0


# ------------------------------------------------------------------------------------------------ Req 33
def test_spots_erased_by_the_small_et_rule_are_still_flagged():
    vol, _ = make_volume()
    result = _segmenter(review_flags=True, et_min_mm3=1e9).segment(vol)
    assert result.diagnostics["et_relabelled"] and not (result.labels == 1).any()
    assert len(result.review_spots) == 1 and np.array_equal(result.review_mask == 1, _mask(B))


# ------------------------------------------------------------------------------------------------ Req 34
def test_a_higher_probability_cut_also_flags_the_confident_spot():
    vol, _ = make_volume()
    result = _segmenter(review_flags=True, review_prob_cut=0.96).segment(vol)
    assert len(result.review_spots) == 2
    assert np.array_equal(result.review_mask > 0, _mask(A) | _mask(B))


def test_a_smaller_size_floor_also_flags_the_small_spot():
    vol, _ = make_volume()
    result = _segmenter(review_flags=True, review_min_voxels=1).segment(vol)
    assert sorted(s["voxels"] for s in result.review_spots) == [27, 216]
    assert np.array_equal(result.review_mask > 0, _mask(B) | _mask(C))


def test_a_spot_exactly_at_the_probability_cut_is_not_flagged():
    vol, _ = make_volume()
    result = _segmenter(review_flags=True, review_prob_cut=0.5).segment(vol)
    assert result.review_spots == [] and not result.review_mask.any()


# ------------------------------------------------------------------------------------------------ Req 35
AFFINE = np.array([[-1.0, 0, 0, 120.0], [0, -1.0, 0, 130.0], [0, 0, 1.0, -70.0], [0, 0, 0, 1.0]])


def _paths(tmp_path):
    vol, _ = make_volume()
    paths = {}
    for c, name in enumerate(("t1c", "t1n", "t2f", "t2w")):
        img = nib.Nifti1Image(vol[c], AFFINE)
        img.header.set_zooms((1.0, 1.0, 1.0))
        nib.save(img, tmp_path / f"{name}.nii.gz")
        paths[name] = str(tmp_path / f"{name}.nii.gz")
    return paths


def _argv(paths, out, *extra):
    return [a for n in ("t1c", "t1n", "t2f", "t2w") for a in (f"--{n}", paths[n])] + ["--out", str(out), *extra]


def test_the_cli_writes_the_review_mask_and_the_spots_in_the_report(tmp_path):
    paths = _paths(tmp_path)
    flags, report = tmp_path / "flags.nii.gz", tmp_path / "report.json"
    seg = _segmenter()
    code = main(_argv(paths, tmp_path / "seg.nii.gz", "--review-mask", str(flags), "--json", str(report)), segmenter=seg)
    assert code == 0 and seg.cfg.review_flags is True
    expected = _segmenter(review_flags=True).segment(make_volume()[0])
    img = nib.load(str(flags))
    assert img.get_data_dtype() == np.uint8 and np.allclose(img.affine, AFFINE)
    assert np.array_equal(np.asarray(img.dataobj), expected.review_mask)
    assert json.loads(report.read_text(encoding="utf-8"))["review_spots"] == json.loads(json.dumps(expected.review_spots))


def test_the_cli_writes_nothing_extra_without_the_option(tmp_path, monkeypatch):
    paths = _paths(tmp_path)
    built = []

    class Recorder(RStarSegmenter):
        def __init__(self, config=None, **kw):
            built.append(config)
            super().__init__(tiny_config(review_flags=config.review_flags), models_2d=[IdentityStub()], models_3d=[IdentityStub()])

    monkeypatch.setattr(pipeline_mod, "RStarSegmenter", Recorder)
    report = tmp_path / "report.json"
    assert main(_argv(paths, tmp_path / "seg.nii.gz", "--json", str(report))) == 0
    assert built and built[0].review_flags is False
    assert "review_spots" not in json.loads(report.read_text(encoding="utf-8"))
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(
        ["t1c.nii.gz", "t1n.nii.gz", "t2f.nii.gz", "t2w.nii.gz", "seg.nii.gz", "report.json"])


def test_the_cli_builds_a_flagging_segmenter_when_asked(tmp_path, monkeypatch):
    paths = _paths(tmp_path)
    built = []

    class Recorder(RStarSegmenter):
        def __init__(self, config=None, **kw):
            built.append(config)
            super().__init__(tiny_config(review_flags=config.review_flags), models_2d=[IdentityStub()], models_3d=[IdentityStub()])

    monkeypatch.setattr(pipeline_mod, "RStarSegmenter", Recorder)
    assert main(_argv(paths, tmp_path / "seg.nii.gz", "--review-mask", str(tmp_path / "f.nii.gz"))) == 0
    assert built[0].review_flags is True and (tmp_path / "f.nii.gz").is_file()


# ------------------------------------------------------------------------------------------------ Req 36
README = Path(__file__).resolve().parents[1] / "README.md"
DEMO_CACHE = Path(__file__).resolve().parents[2] / "05_frontend_demo" / "comparison_cache"
MODELS_ROOT, TEST_DATA = os.environ.get("RSTAR_MODELS_ROOT"), os.environ.get("RSTAR_TEST_DATA")


def test_the_readme_documents_review_flags():
    text = README.read_text(encoding="utf-8")
    for phrase in ("review_flags=True", "--review-mask", "0.7", "50 voxels", "2026-10-02", "labels, mode, status and warnings are identical"):
        assert phrase in text, phrase


@pytest.mark.skipif(not (MODELS_ROOT and TEST_DATA), reason="set RSTAR_MODELS_ROOT and RSTAR_TEST_DATA to run")
@pytest.mark.parametrize("sid", ["BraTS-PED-00004-000", "BraTS-PED-00188-000", "BraTS-PED-00021-000"])
def test_real_patients_match_the_demo_package_flags(sid):
    seg = RStarSegmenter(RStarConfig(models_root=Path(MODELS_ROOT), review_flags=True))
    paths = {m: Path(TEST_DATA) / sid / f"{sid}-{m}.nii.gz" for m in ("t1c", "t1n", "t2f", "t2w")}
    result, _ = seg.segment_paths(paths)
    demo = json.loads((DEMO_CACHE / sid / "review_spots.json").read_text(encoding="utf-8"))
    assert [s["reason"] for s in result.review_spots] == [s["reason"] for s in demo]
    for got, want in zip(result.review_spots, demo):         # CPU vs GPU arithmetic differs at a few boundary voxels
        assert got["voxels"] == pytest.approx(want["voxels"], rel=0.01)
        assert got["mean_et_prob"] == pytest.approx(want["mean_et_prob"], abs=0.01)
