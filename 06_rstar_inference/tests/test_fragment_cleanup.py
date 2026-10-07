"""SPEC Addendum 3 (Req 37-43): optional fragment cleanup before the 500 mm^3 rule, off by default."""
import os
from pathlib import Path

import numpy as np
import pytest
from _helpers import TINY, make_volume, tiny_config
from scipy import ndimage

import rstar.pipeline as pipeline_mod
from rstar import RStarConfig, RStarSegmenter
from rstar.cli import main
from rstar.fusion import apply_small_et_rule, fuse, remove_fragments
from rstar.guards import IdentityStub

BIG = (slice(4, 20), slice(4, 20), slice(4, 20))       # 4096-voxel tumour
ET_IN_BIG = (slice(8, 18), slice(8, 18), slice(8, 13))  # 500 ET voxels inside it
ET_SPECK = (slice(5, 7), slice(5, 7), slice(5, 7))      # 8 ET voxels inside BIG, apart from ET_IN_BIG
STRAY = (slice(30, 33), slice(40, 43), slice(30, 33))   # 27-voxel stray tumour fragment


# ------------------------------------------------------------------------------------------------ Req 37
def test_cleanup_config_defaults():
    cfg = RStarConfig()
    assert cfg.fragment_cleanup is False and cfg.cleanup_min_voxels == 200


@pytest.mark.parametrize("field, value", [
    ("fragment_cleanup", 1), ("fragment_cleanup", "on"),
    ("cleanup_min_voxels", -1), ("cleanup_min_voxels", 2.5), ("cleanup_min_voxels", True),
])
def test_invalid_cleanup_settings_raise_naming_the_field(field, value):
    with pytest.raises(ValueError, match=field):
        RStarConfig(**{field: value})


# ------------------------------------------------------------------------------------------------ Req 38
def test_whole_tumour_components_below_the_size_are_removed_and_at_the_size_kept():
    lab = np.zeros((20, 20, 20), np.uint8)
    lab[1:3, 1:3, 1:3] = 4                     # 8 voxels  -> removed at min 9
    lab[10:13, 10:13, 10] = 2                  # 9 voxels  -> kept at min 9
    out, removed, relabelled = remove_fragments(lab, 9)
    assert not out[1:3, 1:3, 1:3].any() and (out[10:13, 10:13, 10] == 2).all()
    assert removed == 8 and relabelled == 0


def test_a_small_et_component_inside_a_kept_tumour_becomes_non_enhancing():
    lab = np.zeros(TINY, np.uint8)
    lab[BIG] = 4
    lab[ET_IN_BIG] = 1
    lab[ET_SPECK] = 1
    out, removed, relabelled = remove_fragments(lab, 200)
    assert (out[ET_SPECK] == 2).all() and (out[ET_IN_BIG] == 1).all()
    assert removed == 0 and relabelled == 8
    rest = np.ones(TINY, bool)
    rest[ET_SPECK] = False
    assert np.array_equal(out[rest], lab[rest])


def test_blocks_touching_only_at_a_corner_are_one_component():
    lab = np.zeros((10, 10, 10), np.uint8)
    lab[0:2, 0:2, 0:2] = 4
    lab[2:4, 2:4, 2:4] = 4                     # touches the first block only at a corner: 16 voxels together
    assert remove_fragments(lab, 16)[0].sum() == lab.sum()
    assert remove_fragments(lab, 17)[0].sum() == 0


def test_the_input_is_untouched_dtype_kept_and_zero_size_is_identity():
    lab = np.zeros((10, 10, 10), np.int16)
    lab[2:4, 2:4, 2:4] = 1
    before = lab.copy()
    out, _, _ = remove_fragments(lab, 50)
    assert np.array_equal(lab, before) and out.dtype == lab.dtype
    same, removed, relabelled = remove_fragments(lab, 0)
    assert np.array_equal(same, lab) and same is not lab and removed == relabelled == 0


# ------------------------------------------------------------------------------------------------ helpers for 39-41
def _probs():
    """Fused argmax: BIG tumour (edema) with ET_IN_BIG (confident) and ET_SPECK (low confidence) + a STRAY fragment."""
    p = np.zeros((5, *TINY), np.float32)
    p[0] = 1.0
    for region, cls, prob in ((BIG, 4, 0.9), (ET_IN_BIG, 1, 0.9), (ET_SPECK, 1, 0.55), (STRAY, 4, 0.9)):
        p[(slice(None), *region)] = 0.0
        p[(cls, *region)] = prob
        p[(0, *region)] = 1.0 - prob
    return p


def _seg(probs=None, **cfg):
    seg = RStarSegmenter(tiny_config(**cfg), models_2d=[IdentityStub()], models_3d=[IdentityStub()])
    p = _probs() if probs is None else probs
    seg._probs_2d = lambda vol, models=None: p.copy()
    seg._probs_3d = lambda vol, present, models=None: p.copy()
    return seg


# ------------------------------------------------------------------------------------------------ Req 39
def test_cleanup_is_off_by_default_and_changes_nothing():
    vol, _ = make_volume()
    result = _seg().segment(vol)
    expected = apply_small_et_rule(fuse(_probs(), _probs(), 0.5, 0.5), 500.0, 1.0)[0]
    assert np.array_equal(result.labels, expected)
    assert "fragment_voxels_removed" not in result.diagnostics and "et_fragment_voxels_relabelled" not in result.diagnostics


# ------------------------------------------------------------------------------------------------ Req 40
@pytest.mark.parametrize("present", [(True, True, True, True), (True, False, True, True)])
def test_cleanup_on_equals_rule_of_cleaned_labels_with_exact_counts(present):
    vol, _ = make_volume()
    result = _seg(fragment_cleanup=True).segment(vol, present)
    p = _probs()
    pre = fuse(p, p, 0.5, 0.5) if all(present) else np.argmax(p, axis=0).astype(np.uint8)
    cleaned, removed, relabelled = remove_fragments(pre, 200)
    assert np.array_equal(result.labels, apply_small_et_rule(cleaned, 500.0, 1.0)[0])
    assert result.diagnostics["fragment_voxels_removed"] == removed == 27
    assert result.diagnostics["et_fragment_voxels_relabelled"] == relabelled == 8


def test_the_configured_size_is_the_one_used():
    vol, _ = make_volume()
    result = _seg(fragment_cleanup=True, cleanup_min_voxels=10).segment(vol)     # stray (27) kept, speck (8) relabelled
    assert (result.labels[STRAY] == 4).all() and (result.labels[ET_SPECK] == 2).all()
    assert result.diagnostics["fragment_voxels_removed"] == 0 and result.diagnostics["et_fragment_voxels_relabelled"] == 8
    larger = _seg(fragment_cleanup=True, cleanup_min_voxels=600).segment(vol)    # the 500-voxel ET block is now too small
    assert not (larger.labels == 1).any() and larger.diagnostics["et_fragment_voxels_relabelled"] == 508


def test_review_flags_are_identical_with_cleanup_on_and_off():
    vol, _ = make_volume()
    off = _seg(review_flags=True, review_min_voxels=1).segment(vol)
    on = _seg(review_flags=True, review_min_voxels=1, fragment_cleanup=True).segment(vol)
    assert np.array_equal(off.review_mask, on.review_mask) and off.review_spots == on.review_spots
    assert len(on.review_spots) == 1 and on.review_spots[0]["voxels"] == 8     # the low-confidence speck, seen before cleanup


# ------------------------------------------------------------------------------------------------ Req 41
def test_cleanup_runs_before_the_small_et_rule():
    vol, _ = make_volume()
    # 500 ET voxels + 8-voxel speck = 508 >= 500 without cleanup; after cleanup 500 remain, so a 501 threshold relabels
    kept = _seg(et_min_mm3=505.0).segment(vol)
    assert not kept.diagnostics["et_relabelled"] and (kept.labels == 1).sum() == 508
    cleaned = _seg(et_min_mm3=505.0, fragment_cleanup=True).segment(vol)
    assert cleaned.diagnostics["et_relabelled"] and not (cleaned.labels == 1).any()
    assert cleaned.diagnostics["et_voxels_before_rule"] == 500


# ------------------------------------------------------------------------------------------------ Req 42
def _cli_paths(tmp_path):
    import nibabel as nib
    vol, _ = make_volume()
    affine = np.eye(4)
    paths = []
    for c, name in enumerate(("t1c", "t1n", "t2f", "t2w")):
        img = nib.Nifti1Image(vol[c], affine)
        img.header.set_zooms((1.0, 1.0, 1.0))
        nib.save(img, tmp_path / f"{name}.nii.gz")
        paths += [f"--{name}", str(tmp_path / f"{name}.nii.gz")]
    return paths + ["--out", str(tmp_path / "seg.nii.gz")]


@pytest.mark.parametrize("flag, expected", [(["--fragment-cleanup"], True), ([], False)])
def test_the_cli_option_controls_cleanup(tmp_path, monkeypatch, flag, expected):
    built = []

    class Recorder(RStarSegmenter):
        def __init__(self, config=None, **kw):
            built.append(config)
            super().__init__(tiny_config(fragment_cleanup=config.fragment_cleanup), models_2d=[IdentityStub()],
                             models_3d=[IdentityStub()])

    monkeypatch.setattr(pipeline_mod, "RStarSegmenter", Recorder)
    assert main(_cli_paths(tmp_path) + flag) == 0
    assert built[0].fragment_cleanup is expected


def test_the_cli_option_switches_cleanup_on_for_an_injected_segmenter(tmp_path):
    seg = RStarSegmenter(tiny_config(), models_2d=[IdentityStub()], models_3d=[IdentityStub()])
    assert main(_cli_paths(tmp_path) + ["--fragment-cleanup"], segmenter=seg) == 0
    assert seg.cfg.fragment_cleanup is True


# ------------------------------------------------------------------------------------------------ Req 43
README = Path(__file__).resolve().parents[1] / "README.md"
MODELS_ROOT, TEST_DATA = os.environ.get("RSTAR_MODELS_ROOT"), os.environ.get("RSTAR_TEST_DATA")


def test_the_readme_documents_fragment_cleanup():
    text = README.read_text(encoding="utf-8")
    for phrase in ("fragment_cleanup=True", "--fragment-cleanup", "200 voxels", "2026-10-04", "off by default", "cross-validation"):
        assert phrase in text, phrase


@pytest.mark.skipif(not (MODELS_ROOT and TEST_DATA), reason="set RSTAR_MODELS_ROOT and RSTAR_TEST_DATA to run")
@pytest.mark.parametrize("sid", ["BraTS-PED-00077-000", "BraTS-PED-00103-000", "BraTS-PED-00127-000"])
def test_real_patients_change_little_and_keep_no_small_components(sid):
    import nibabel as nib
    seg = RStarSegmenter(RStarConfig(models_root=Path(MODELS_ROOT)))
    paths = {m: Path(TEST_DATA) / sid / f"{sid}-{m}.nii.gz" for m in ("t1c", "t1n", "t2f", "t2w")}
    off, _ = seg.segment_paths(paths)
    seg.cfg.fragment_cleanup = True
    on, _ = seg.segment_paths(paths)
    gt = np.rint(np.asarray(nib.load(str(Path(TEST_DATA) / sid / f"{sid}-seg.nii.gz")).dataobj)).astype(np.uint8)

    def mean_dice(lab):
        out = []
        for c in ((1,), (1, 2, 3), (1, 2, 3, 4)):
            p, g = np.isin(lab, c), np.isin(gt, c)
            out.append(1.0 if p.sum() + g.sum() == 0 else 2 * (p & g).sum() / (p.sum() + g.sum()))
        return float(np.mean(out))

    assert abs(mean_dice(on.labels) - mean_dice(off.labels)) <= 0.01
    comp, n = ndimage.label(on.labels > 0, np.ones((3, 3, 3), bool))
    assert n == 0 or np.bincount(comp.ravel())[1:].min() >= 200
