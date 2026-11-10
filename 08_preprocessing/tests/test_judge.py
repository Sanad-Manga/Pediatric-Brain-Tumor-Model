"""Tests for prep.judge and tools/visual_check.py (issue #65).

The judge has to get both directions right on a phantom whose truth we set: when the sequence really is out of place
our step wins, when it is already in place leaving it alone wins. A judge that only ever says "aligned better" would
rubber-stamp the step, which is the whole thing the issue is asking about.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import phantoms  # noqa: E402
from prep import align, judge  # noqa: E402


@pytest.fixture(scope="module")
def pair():
    return phantoms.brain_pair()


def test_brain_mask_drops_background_and_erodes(pair):
    t1c, _ = pair
    arr = sitk.GetArrayFromImage(t1c)
    plain = sitk.GetArrayFromImage(judge.brain_mask(t1c, erode_mm=0.0)) > 0
    eroded = sitk.GetArrayFromImage(judge.brain_mask(t1c, erode_mm=3.0)) > 0
    assert not plain[arr == 0].any()          # background is never in the mask
    assert eroded.sum() < plain.sum()         # erosion removes the outline
    assert eroded[~plain].sum() == 0          # erosion only ever removes


def test_half_voxel_grid_shifts_by_half_a_voxel(pair):
    t1c, _ = pair
    grid = judge.half_voxel_grid(t1c)
    assert grid.GetSize() == t1c.GetSize()
    assert grid.GetSpacing() == t1c.GetSpacing()
    assert np.allclose(np.array(grid.GetOrigin()) - np.array(t1c.GetOrigin()), [s / 2 for s in t1c.GetSpacing()])
    # and resampling onto it really interpolates, so neither candidate is the sharper one for free
    resampled = sitk.Resample(t1c, grid, sitk.Euler3DTransform(), sitk.sitkLinear, 0.0, sitk.sitkFloat32)
    assert not np.allclose(sitk.GetArrayFromImage(resampled), sitk.GetArrayFromImage(t1c))


def test_a_naive_comparison_would_punish_resampling_on_its_own(pair):
    """Why both candidates are resampled once: an extra interpolation costs edge agreement even with no movement, so
    comparing a sharp original against a resampled aligned scan is rigged in the original's favour."""
    t1c, t2w = pair
    mask = sitk.GetArrayFromImage(judge.brain_mask(t1c)) > 0
    grid = judge.half_voxel_grid(t2w)
    there = sitk.Resample(t2w, grid, sitk.Euler3DTransform(), sitk.sitkLinear, 0.0, sitk.sitkFloat32)
    back = sitk.Resample(there, t2w, sitk.Euler3DTransform(), sitk.sitkLinear, 0.0, sitk.sitkFloat32)
    assert judge.gradient_correlation(t1c, back, mask) < judge.gradient_correlation(t1c, t2w, mask)


def test_rejects_a_sequence_that_is_not_on_the_reference_grid(pair):
    t1c, t2w = pair
    cropped = sitk.RegionOfInterest(t2w, [40, 40, 40], [4, 4, 4])
    with pytest.raises(ValueError, match="T1c grid"):
        judge.judge(cropped, t1c, sitk.Euler3DTransform())


def test_a_re_registration_that_lands_near_neither_candidate_does_not_vote():
    assert judge.opinion_vote(judge.MAX_OPINION_MM + 1, judge.MAX_OPINION_MM + 1) == "failed"
    # far from ours but sitting on the scan's own position is a vote, not a failure
    assert judge.opinion_vote(0.1, judge.MAX_OPINION_MM * 20) == judge.IDENTITY
    assert judge.opinion_vote(2.0, 0.1) == judge.ALIGNED
    assert judge.opinion_vote(0.1, 2.0) == judge.IDENTITY
    assert judge.opinion_vote(1.0, 0.9) == ""          # does not separate the two candidates
    assert judge.combine(["failed", "tie", "tie", "tie", "tie"]) == "can't tell"


def test_one_criterion_alone_is_not_a_verdict():
    assert judge.combine([judge.ALIGNED, "tie", "tie", "tie"]) == "can't tell"
    assert judge.combine([judge.ALIGNED, judge.ALIGNED, "tie", "failed"]) == "aligned better"


def test_any_criterion_pointing_the_other_way_blocks_a_verdict():
    assert judge.combine([judge.ALIGNED, judge.ALIGNED, judge.ALIGNED, judge.IDENTITY]) == "can't tell"


def test_criterion_votes_tie_inside_the_tolerance():
    base = dict.fromkeys(judge.CRITERIA, 1.0)
    nudged = {k: 1.0 + 2 * judge.REL_TOL for k in judge.CRITERIA}
    assert set(judge.criterion_votes({judge.IDENTITY: base, judge.ALIGNED: dict(base)}).values()) == {"tie"}
    assert set(judge.criterion_votes({judge.IDENTITY: base, judge.ALIGNED: nudged}).values()) == {judge.ALIGNED}
    assert set(judge.criterion_votes({judge.IDENTITY: nudged, judge.ALIGNED: base}).values()) == {judge.IDENTITY}


def test_says_no_disagreement_when_the_step_leaves_the_sequence_where_it_is(pair):
    t1c, t2w = pair
    _, ours, _ = align.align_to_reference(t2w, t1c)
    result = judge.judge(t2w, t1c, ours)
    assert result["disagreement_mm"] < judge.MIN_GAP_MM
    assert result["verdict"] == "no disagreement"
    assert set(result["criterion_votes"].values()) == {"tie"}  # nothing moved, so nothing to prefer


def test_says_aligned_better_when_the_sequence_really_is_out_of_place(pair):
    t1c, t2w = pair
    moved, _ = phantoms.displaced(t2w)
    _, ours, _ = align.align_to_reference(moved, t1c)
    result = judge.judge(moved, t1c, ours)
    assert result["verdict"] == "aligned better"
    assert set(result["criterion_votes"].values()) == {judge.ALIGNED}  # all four, not just the one it optimised
    assert (result["scores"]["aligned"]["gradient_correlation"]
            > result["scores"]["original"]["gradient_correlation"])


def test_says_original_better_when_the_step_proposes_a_move_the_scan_does_not_need(pair):
    t1c, t2w = pair
    bogus = phantoms.euler(t2w, (0.6, -0.4, 0.5), (1.1, -0.8, 0.4))  # a 1.5 mm move of an already-aligned scan
    result = judge.judge(t2w, t1c, bogus)
    assert result["verdict"] == "original better"
    assert set(result["criterion_votes"].values()) == {judge.IDENTITY}
    assert (result["scores"]["original"]["gradient_correlation"]
            > result["scores"]["aligned"]["gradient_correlation"])


def test_same_inputs_give_the_same_verdict(pair):
    t1c, t2w = pair
    bogus = phantoms.euler(t2w, (0.6, -0.4, 0.5), (1.1, -0.8, 0.4))
    assert judge.judge(t2w, t1c, bogus) == judge.judge(t2w, t1c, bogus)


def test_visual_check_writes_a_figure_and_a_verdict_per_sequence(pair, tmp_path):
    t1c, t2w = pair
    moved, _ = phantoms.displaced(t2w)
    sitk.WriteImage(t1c, str(tmp_path / "P-t1c.nii.gz"))
    sitk.WriteImage(t2w, str(tmp_path / "P-t2f.nii.gz"))    # already in place
    sitk.WriteImage(moved, str(tmp_path / "P-t2w.nii.gz"))  # 1.5 mm out of place
    out = tmp_path / "out"
    code = subprocess.call([sys.executable, str(HERE / "tools" / "visual_check.py"),
                            "--t1c", str(tmp_path / "P-t1c.nii.gz"), "--t2f", str(tmp_path / "P-t2f.nii.gz"),
                            "--t2w", str(tmp_path / "P-t2w.nii.gz"), "--out-dir", str(out), "--label", "P"])
    assert code == 0
    verdicts = json.loads((out / "verdicts.json").read_text(encoding="utf-8"))
    assert verdicts["sequences"]["t2w"]["verdict"] == "aligned better"
    assert verdicts["sequences"]["t2f"]["verdict"] == "no disagreement"
    for seq in ("t2f", "t2w"):
        assert (out / f"P-{seq}.png").stat().st_size > 10_000


def test_visual_check_exits_with_a_message_when_an_input_is_missing(tmp_path):
    proc = subprocess.run([sys.executable, str(HERE / "tools" / "visual_check.py"),
                           "--t1c", str(tmp_path / "nope-t1c.nii.gz"), "--t2f", str(tmp_path / "nope-t2f.nii.gz"),
                           "--out-dir", str(tmp_path / "out"), "--sequences", "t2f"],
                          capture_output=True, text=True)
    assert proc.returncode == 1
    assert "missing input" in proc.stderr


def test_visual_check_rejects_a_sequence_it_cannot_check(tmp_path):
    proc = subprocess.run([sys.executable, str(HERE / "tools" / "visual_check.py"),
                           "--t1c", str(tmp_path / "a.nii.gz"), "--out-dir", str(tmp_path / "out"),
                           "--sequences", "t1c"], capture_output=True, text=True)
    assert proc.returncode == 1
    assert "cannot check" in proc.stderr
