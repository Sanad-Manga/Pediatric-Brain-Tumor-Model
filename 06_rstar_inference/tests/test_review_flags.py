"""SPEC.md Addendum 1 (Req 22-28): spot-level review flags and the per-spot error scorecard."""
import json
from pathlib import Path

import numpy as np
import pytest

from rstar import fusion
from rstar.review_flags import Decision, Spot, decide, find_et_spots, review_outputs, score_patient, t500_decisions


def _vol(shape=(20, 20, 20)):
    return np.zeros(shape, dtype=np.uint8)


# ------------------------------------------------------------------------------------------ Req 22
def test_corner_touching_blocks_are_one_spot_and_gapped_blocks_are_two():
    lab = _vol()
    lab[2:4, 2:4, 2:4] = 1
    lab[4:6, 4:6, 4:6] = 1          # touches the first only at a corner -> same 26-connected spot
    lab[10:12, 10:12, 10:12] = 1
    lab[13:15, 10:12, 10:12] = 1    # one voxel gap -> separate
    ids, spots = find_et_spots(lab, np.zeros(lab.shape))
    assert [s.voxels for s in spots] == [16, 8, 8]
    assert ids.dtype == np.uint16 and set(np.unique(ids)) == {0, 1, 2, 3}


def test_features_and_agreement_are_exact():
    lab = _vol()
    lab[5:7, 5:7, 5:7] = 1           # 8 voxels
    p = np.zeros(lab.shape)
    p[5:7, 5:7, 5:7] = 0.5
    p[5, 5, 5] = 0.9
    full = lab == 1
    half = np.zeros(lab.shape, bool)
    half[5, 5:7, 5:7] = True        # 4 of the 8 voxels
    _, spots = find_et_spots(lab, p, [full, half])
    s = spots[0]
    assert s.voxels == 8 and abs(s.mean_et_prob - (0.5 * 7 + 0.9) / 8) < 1e-6 and abs(s.max_et_prob - 0.9) < 1e-6
    assert s.agreement == pytest.approx(0.75, abs=1e-12)
    assert find_et_spots(lab, p)[1][0].agreement is None


def test_no_et_gives_nothing_and_shape_mismatch_raises():
    ids, spots = find_et_spots(_vol(), np.zeros((20, 20, 20)))
    assert spots == [] and not ids.any()
    with pytest.raises(ValueError, match=r"\(20, 20, 20\).*\(20, 20, 19\)"):
        find_et_spots(_vol(), np.zeros((20, 20, 19)))


# ------------------------------------------------------------------------------------------ Req 23
def _spot(i, voxels, prob=0.9, agree=1.0):
    return Spot(spot_id=i, voxels=voxels, mean_et_prob=prob, max_et_prob=prob, agreement=agree)


def test_decisions_reasons_and_boundaries():
    spots = [_spot(1, 1000), _spot(2, 999), _spot(3, 500), _spot(4, 499), _spot(5, 2000, prob=0.59),
             _spot(6, 2000, prob=0.6), _spot(7, 2000, agree=0.49), _spot(8, 2000, agree=0.5),
             _spot(9, 100, prob=0.1, agree=0.0), _spot(10, 2000, prob=0.1, agree=0.0), _spot(11, 2000, agree=None)]
    d = {x.spot_id: (x.action, x.reason) for x in decide(spots, size_cut=1000, prob_cut=0.6, agree_cut=0.5)}
    assert d[1] == ("keep", None)                       # == size_cut keeps
    assert d[2] == ("review", "borderline")
    assert d[3] == ("review", "borderline")             # == small_cut is borderline, not small
    assert d[4] == ("review", "small")
    assert d[5] == ("review", "low_confidence")
    assert d[6] == ("keep", None)                       # == prob_cut keeps
    assert d[7] == ("review", "models_disagree")
    assert d[8] == ("keep", None)                       # == agree_cut keeps
    assert d[9] == ("review", "small")                  # priority: size first
    assert d[10] == ("review", "low_confidence")        # confidence before agreement
    assert d[11] == ("keep", None)                      # None agreement never disagrees


@pytest.mark.parametrize("kwargs", [dict(size_cut=-1), dict(small_cut=-1), dict(size_cut=400, small_cut=500),
                                    dict(prob_cut=1.5), dict(prob_cut=float("nan")), dict(agree_cut=-0.1)])
def test_invalid_thresholds_raise(kwargs):
    base = dict(size_cut=1000, prob_cut=0.6, agree_cut=0.5)
    base.update(kwargs)
    with pytest.raises(ValueError):
        decide([_spot(1, 10)], **base)


# ------------------------------------------------------------------------------------------ Req 24
def test_review_outputs_format():
    lab = _vol()
    lab[1:3, 1:3, 1:3] = 1          # spot 1, 8 vox -> review (small)
    lab[8:16, 8:16, 8:16] = 1       # spot 2, 512 vox -> keep
    lab[18:20, 18:20, 18:20] = 1    # spot 3, 8 vox -> review
    ids, spots = find_et_spots(lab, np.where(lab == 1, 0.8, 0.0), [lab == 1])
    decisions = decide(spots, size_cut=100, prob_cut=0.5, agree_cut=0.5, small_cut=50)
    mask, listed = review_outputs(ids, spots, decisions)
    assert mask.dtype == np.uint8 and mask.shape == lab.shape
    assert set(np.unique(mask)) == {0, 1, 2} and not mask[8:16, 8:16, 8:16].any()
    assert (mask[1:3, 1:3, 1:3] == 1).all() and (mask[18:20, 18:20, 18:20] == 2).all()
    assert [e["spot_id"] for e in listed] == [1, 2]
    assert all(set(e) == {"spot_id", "voxels", "mean_et_prob", "models_agree", "reason"} for e in listed)
    assert listed[0]["models_agree"] == 1.0 and listed[0]["reason"] == "small"
    json.dumps(listed)
    assert all(type(e["spot_id"]) is int and type(e["voxels"]) is int and type(e["mean_et_prob"]) is float for e in listed)


# ------------------------------------------------------------------------------------------ Req 25
def test_scorecard_counts_and_dice_on_a_known_patient():
    pred = _vol((30, 30, 30))
    gt = _vol((30, 30, 30))
    pred[1:5, 1:5, 1:5] = 1;      gt[1:5, 1:5, 1:5] = 1          # A kept real (64)
    pred[10:14, 1:5, 1:5] = 1                                     # B kept false (64)
    pred[20:22, 1:3, 1:3] = 1;    gt[20:22, 1:3, 1:3] = 1        # C review real (8)
    pred[1:3, 20:22, 1:3] = 1                                     # D review false (8)
    gt[25:27, 25:27, 25:27] = 1                                   # E lesion covered by nothing (8)
    ids, spots = find_et_spots(pred, np.where(pred == 1, 0.9, 0.0))
    want = {1: "keep", 2: "review", 3: "keep", 4: "review"}   # scan order: A(1), D(2), B(3), C(4)
    assert [s.voxels for s in spots] == [64, 8, 64, 8]
    decisions = [Decision(s.spot_id, want[s.spot_id], None) for s in spots]
    r = score_patient(ids, spots, decisions, gt)
    assert (r["kept_real"], r["kept_false"], r["review_real"], r["review_false"]) == (1, 1, 1, 1)
    assert (r["lesions"], r["lesions_caught_keep"], r["lesions_caught_review"], r["lesions_missed"]) == (3, 1, 1, 1)
    assert r["silent_false"] == 1 and r["silent_missed"] == 1
    gt_n = 64 + 8 + 8
    assert r["et_dice_kept"] == pytest.approx(2 * 64 / (128 + gt_n), abs=1e-9)
    assert r["et_dice_with_review"] == pytest.approx(2 * 72 / (144 + gt_n), abs=1e-9)


def test_empty_patient_is_perfect():
    ids, spots = find_et_spots(_vol(), np.zeros((20, 20, 20)))
    r = score_patient(ids, spots, [], _vol())
    assert r["et_dice_kept"] == 1.0 and r["et_dice_with_review"] == 1.0
    assert all(r[k] == 0 for k in ("kept_real", "kept_false", "review_real", "review_false", "lesions", "lesions_missed"))


# ------------------------------------------------------------------------------------------ Req 26
@pytest.mark.parametrize("n_et", [300, 500, 800])
def test_todays_rule_as_decisions_matches_the_existing_rule(n_et):
    pred = _vol((40, 40, 40))
    pred.reshape(-1)[:n_et] = 1
    gt = _vol((40, 40, 40))
    gt.reshape(-1)[:400] = 1
    ids, spots = find_et_spots(pred, np.where(pred == 1, 0.9, 0.0))
    d = t500_decisions(spots, int((pred == 1).sum()), voxel_mm3=1.0, min_mm3=500.0)
    assert all(x.action == ("keep" if n_et >= 500 else "drop") for x in d)
    ours = score_patient(ids, spots, d, gt)["et_dice_kept"]
    relabelled, _, _ = fusion.apply_small_et_rule(pred, 500.0, 1.0)
    p, t = relabelled == 1, gt == 1
    ref = 1.0 if not p.any() and not t.any() else 2 * (p & t).sum() / (p.sum() + t.sum())
    assert ours == pytest.approx(ref, abs=1e-9)


def test_dropped_spots_are_neither_kept_nor_reviewed_in_the_counts():
    pred = _vol((30, 30, 30))
    gt = _vol((30, 30, 30))
    pred[1:5, 1:5, 1:5] = 1; gt[1:5, 1:5, 1:5] = 1      # a real spot
    pred[10:12, 10:12, 10:12] = 1                        # a false spot
    ids, spots = find_et_spots(pred, np.where(pred == 1, 0.9, 0.0))
    r = score_patient(ids, spots, t500_decisions(spots, 72), gt)       # 72 voxels < 500 -> all dropped
    assert (r["kept_real"], r["kept_false"], r["review_real"], r["review_false"]) == (0, 0, 0, 0)
    assert r["lesions_missed"] == 1 and r["silent_missed"] == 1 and r["lesions_caught_review"] == 0


def test_todays_rule_respects_voxel_volume():
    pred = _vol()
    pred[0:5, 0:5, 0:4] = 1                                     # 100 voxels
    _, spots = find_et_spots(pred, np.zeros(pred.shape))
    assert t500_decisions(spots, 100, voxel_mm3=5.0)[0].action == "keep"     # 500 mm^3
    assert t500_decisions(spots, 100, voxel_mm3=4.0)[0].action == "drop"     # 400 mm^3


# ------------------------------------------------------------------------------------------ Req 27
def test_dtypes_validation_edges_and_determinism():
    lab = _vol()
    lab[0:2, 0:2, 0:2] = 1                                      # touches the volume edge
    for arr in (lab.astype(np.int64), lab.astype(np.float32), lab == 1):
        ids, spots = find_et_spots(arr if arr.dtype != bool else arr.astype(np.uint8), np.zeros(lab.shape))
        assert len(spots) == 1 and spots[0].voxels == 8
    with pytest.raises(ValueError):
        find_et_spots(lab, np.full(lab.shape, 1.5))
    with pytest.raises(ValueError):
        find_et_spots(lab, np.full(lab.shape, np.nan))
    with pytest.raises(ValueError):
        find_et_spots(lab, np.zeros(lab.shape), [np.zeros((5, 5, 5), bool)])
    a = find_et_spots(lab, np.zeros(lab.shape))
    b = find_et_spots(lab, np.zeros(lab.shape))
    assert (a[0] == b[0]).all() and a[1] == b[1]


# ------------------------------------------------------------------------------------------ Req 28
def test_pipeline_uses_review_flags_only_when_enabled():
    # Addendum 2 supersedes Addendum 1's "pipeline does not import review_flags": the pipeline calls the module only
    # behind cfg.review_flags (off by default); behaviour with flags on and off is tested in test_pipeline_review.py.
    src = (Path(__file__).resolve().parents[1] / "rstar" / "pipeline.py").read_text(encoding="utf-8")
    assert "if self.cfg.review_flags:" in src
    assert src.count("review_flags.") == 3            # find_et_spots, decide, review_outputs inside _review_flags
