"""Cyst (label 3) Dice: conventions, aggregation, and that ET/NC/WT are untouched."""

from __future__ import annotations

import numpy as np

from src.metrics import (aggregate, aggregate_cyst, cyst_record, dice_regions,
                         format_cyst)


def _vol(label=None, n=4):
    v = np.zeros((6, 6, 6), dtype=np.int64)
    if label is not None:
        v[:1, :1, :n] = label
    return v


def test_no_cyst_in_either_scores_one():
    r = cyst_record(_vol(), _vol())
    assert r == {"dice_cyst": 1.0, "gt_has_cyst": False, "pred_has_cyst": False}


def test_cyst_in_only_one_scores_zero():
    assert cyst_record(_vol(), _vol(3))["dice_cyst"] == 0.0      # missed
    assert cyst_record(_vol(3), _vol())["dice_cyst"] == 0.0      # hallucinated
    assert cyst_record(_vol(3), _vol())["pred_has_cyst"] is True


def test_partial_overlap_dice_value():
    pred, true = _vol(3, 2), _vol(3, 4)       # 2 vs 4 voxels, 2 shared
    assert cyst_record(pred, true)["dice_cyst"] == 2 * 2 / (2 + 4)


def test_other_labels_do_not_count_as_cyst():
    assert cyst_record(_vol(2), _vol(1))["dice_cyst"] == 1.0     # no label 3 anywhere


def test_aggregate_reports_cyst_prevalence_and_positive_only_mean():
    recs = [cyst_record(_vol(3), _vol(3)),    # 1.0, has cyst
            cyst_record(_vol(), _vol(3)),     # 0.0, has cyst
            cyst_record(_vol(), _vol()),      # 1.0, no cyst
            cyst_record(_vol(), _vol())]      # 1.0, no cyst
    agg = aggregate_cyst(recs)
    assert agg["n_subjects"] == 4 and agg["n_gt_cyst"] == 2 and agg["n_pred_cyst"] == 1
    assert agg["dice_cyst"] == 0.75
    assert agg["dice_cyst_gt_cyst"] == 0.5
    assert "2 with an expert cyst" in format_cyst(agg)


def test_aggregate_with_no_cyst_patients_gives_none_for_positive_mean():
    agg = aggregate_cyst([cyst_record(_vol(), _vol())])
    assert agg["dice_cyst_gt_cyst"] is None and "n/a" in format_cyst(agg)


def test_et_nc_wt_numbers_are_unchanged_by_the_cyst_additions():
    pred, true = _vol(3, 2), _vol(3, 4)
    base = dice_regions(pred, true)
    assert set(base) == {"dice_ET", "dice_NC", "dice_WT"}
    merged = {**base, **cyst_record(pred, true)}
    assert aggregate([merged]) == aggregate([base])


def test_tool_modules_expose_cyst_in_their_output():
    # The 3D tools load this same metrics file by path; guard the wiring.
    import importlib.util
    from pathlib import Path
    root = Path(__file__).resolve().parents[2] / "01_model_federated" / "tools"
    for name in ("eval_heldout_3d", "ensemble_heldout_3d"):
        src = (root / f"{name}.py").read_text(encoding="utf-8")
        assert "cyst_record" in src and "aggregate_cyst" in src and '"cyst": cyst' in src
