import json

import numpy as np
import pytest
import torch

from src.model import FederatedUNet3D
from tools.eval_patch3d import apply_et_min_voxels, evaluate, main

from .patch_helpers import make_cohort


def _pred(et_voxels):
    pred = np.zeros((10, 10, 10), dtype=np.int8)
    pred[5:8, 5:8, 5:8] = 4
    pred.reshape(-1)[:et_voxels] = 1  # exactly `et_voxels` ET voxels (a single row would cap at 10)
    assert int((pred == 1).sum()) == et_voxels
    pred[9, 9, 0:3] = 2
    pred[9, 8, 0:3] = 3
    return pred


def test_apply_et_min_voxels_relabels_small_et_and_nothing_else():  # Req 48
    pred = _pred(5)
    original = pred.copy()
    out = apply_et_min_voxels(pred, 10)
    assert not (out == 1).any()
    assert (out[original == 1] == 2).all()
    assert np.array_equal(out[original != 1], original[original != 1])
    assert np.array_equal(pred, original), "the input array was modified in place"


def test_apply_et_min_voxels_leaves_large_et_and_t0_and_no_et_alone():  # Req 48
    large = _pred(12)
    assert np.array_equal(apply_et_min_voxels(large, 10), large)
    assert np.array_equal(apply_et_min_voxels(large, 12), large)  # "at" the threshold is not below it
    small = _pred(5)
    assert np.array_equal(apply_et_min_voxels(small, 0), small)
    none = _pred(0)
    assert np.array_equal(apply_et_min_voxels(none, 500), none)


@pytest.fixture(scope="module")
def eval_setup(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("eval")
    cohort = make_cohort(tmp, n_et=1, n_tumor_only=1, n_empty=0)
    torch.manual_seed(0)
    ckpt = tmp / "tiny.pt"
    torch.save({"model_state": FederatedUNet3D().state_dict()}, ckpt)
    return cohort, ckpt


def test_evaluate_returns_valid_dice_and_nc_wt_are_unchanged_by_the_et_rule(eval_setup):  # Req 49
    cohort, ckpt = eval_setup
    result = evaluate(ckpt, cohort.cache, cohort.manifest, roi_size=(32, 32, 32),
                      et_min_voxels_list=(0, 500), device="cpu")
    assert result["n_subjects_scored"] == 2 and result["n_subjects_missing"] == 0
    assert set(result["results"]) == {"0", "500"}
    for row in result["results"].values():
        for key in ("mean", "ET", "NC", "WT"):
            assert 0.0 <= row[key] <= 1.0
    assert result["results"]["0"]["NC"] == result["results"]["500"]["NC"]
    assert result["results"]["0"]["WT"] == result["results"]["500"]["WT"]


def test_a_manifest_patient_missing_from_the_cache_is_reported_not_fatal(eval_setup, tmp_path):
    cohort, ckpt = eval_setup
    manifest = tmp_path / "with_ghost.json"
    manifest.write_text(json.dumps(cohort.sids[:1] + ["GHOST"]))
    result = evaluate(ckpt, cohort.cache, manifest, roi_size=(32, 32, 32), et_min_voxels_list=(0,), device="cpu")
    assert result["n_subjects_scored"] == 1 and result["missing_subjects"] == ["GHOST"]


def test_cli_prints_a_parseable_result_json_line(eval_setup, capsys):  # Req 49
    cohort, ckpt = eval_setup
    code = main(["--checkpoint", str(ckpt), "--cache-path", cohort.cache, "--manifest", cohort.manifest,
                 "--roi-size", "32", "32", "32", "--et-min-voxels", "0", "500", "--device", "cpu", "--limit", "1"])
    assert code == 0
    line = next(l for l in capsys.readouterr().out.splitlines() if l.startswith("RESULT_JSON:"))
    parsed = json.loads(line[len("RESULT_JSON:"):])
    assert set(parsed["results"]) == {"0", "500"} and parsed["n_subjects_scored"] == 1


def test_evaluate_applies_the_et_rule_for_each_threshold(eval_setup, tmp_path, monkeypatch):  # Req 49
    """A random-weight network can't show that the rule is wired into evaluate(), so feed it a crafted
    prediction: a tiny false-positive ET blob on a patient whose truth has NO ET. Without the rule the ET Dice
    is 0; with T=500 the blob is dropped and 'no ET' is correct, so it is 1."""
    import tools.eval_patch3d as ev

    cohort, ckpt = eval_setup
    manifest = tmp_path / "tumour_only.json"
    manifest.write_text(json.dumps([cohort.sids[1]]))  # sids[1] has tumour but no ET

    def crafted(model, image, roi_size, **kwargs):
        probs = torch.zeros(5, *image.shape[1:])
        probs[0] = 1.0
        probs[0, :2, :2, :2] = 0.0
        probs[1, :2, :2, :2] = 1.0  # an 8-voxel ET blob
        return probs

    monkeypatch.setattr(ev, "predict_volume", crafted)
    result = evaluate(ckpt, cohort.cache, manifest, roi_size=(32, 32, 32), et_min_voxels_list=(0, 500), device="cpu")
    assert result["results"]["0"]["ET"] == 0.0
    assert result["results"]["500"]["ET"] == 1.0
    assert result["results"]["0"]["NC"] == result["results"]["500"]["NC"]
    assert result["results"]["0"]["WT"] == result["results"]["500"]["WT"]
