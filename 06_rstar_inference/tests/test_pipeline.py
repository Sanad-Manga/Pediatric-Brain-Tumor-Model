"""Req 13-17: modes, outputs, the ET diagnostic, agreement statuses and the geometry self-check."""
import numpy as np
import pytest
import torch
from _helpers import TINY, ConstantStub, CountingStub, RecordingStub, make_volume, tiny_config

from rstar import ContractError, RStarSegmenter, SelfCheckError
from rstar import fusion, preprocess
from rstar import pipeline as pipeline_module
from rstar.guards import IdentityStub

ALL = (True, True, True, True)


def _segmenter(models_2d=None, models_3d=None, **cfg):
    return RStarSegmenter(tiny_config(**cfg), models_2d=models_2d or [IdentityStub()], models_3d=models_3d or [IdentityStub()])


# ------------------------------------------------------------------------------------------------ Req 13
def test_all_four_sequences_run_both_branches_in_r_star_mode():
    s2, s3 = CountingStub(), CountingStub()
    result = _segmenter([s2], [s3]).segment(make_volume()[0])
    assert result.mode == "R*" and result.status in ("ok", "review")
    assert s2.calls > 0 and s3.calls > 0
    assert result.diagnostics["agreement"] is not None


def test_one_absent_sequence_gives_3d_only_status_ok_and_a_warning_naming_it():
    s2, s3 = CountingStub(), RecordingStub()
    vol, _ = make_volume()
    vol[2] = np.nan                                              # garbage in the absent channel must never reach a model
    result = _segmenter([s2], [s3]).segment(vol, present=(True, True, False, True))
    assert result.mode == "3D-only" and result.status == "ok"
    assert s2.calls == 0, "the 2D branch must not run when a sequence is absent"
    assert s3.last.shape == (1, 4, 96, 96, 96) and (s3.last[0, 2] == 0).all() and torch.isfinite(s3.last).all()
    assert any("t2f" in w for w in result.warnings)
    assert result.diagnostics["agreement"] is None and result.diagnostics["present"] == (True, True, False, True)


@pytest.mark.parametrize("present", [(False, True, False, True), (True, False, False, False)])
def test_two_or_more_absent_sequences_need_review(present):
    result = _segmenter().segment(make_volume()[0], present=present)
    assert result.mode == "3D-only" and result.status == "review"
    assert any("not validated" in w for w in result.warnings)


def test_no_sequence_present_is_a_contract_error():
    with pytest.raises(ContractError):
        _segmenter().segment(make_volume()[0], present=(False, False, False, False))


# ------------------------------------------------------------------------------------------------ Req 14
def test_the_output_has_the_right_shape_dtype_values_and_keys_and_is_deterministic():
    vol, _ = make_volume()
    before = vol.copy()
    seg = _segmenter()
    a, b = seg.segment(vol), seg.segment(vol)
    assert np.array_equal(vol, before), "the input volume was modified"
    assert a.labels.dtype == np.uint8 and a.labels.shape == TINY and set(np.unique(a.labels)) <= {0, 1, 2, 3, 4}
    assert np.array_equal(a.labels, b.labels) and a.status == b.status
    assert set(a.diagnostics) >= {"agreement", "et_voxels_before_rule", "et_relabelled", "present", "checkpoints", "elapsed_s", "mode"}
    assert isinstance(a.warnings, list) and a.diagnostics["elapsed_s"] >= 0.0


def test_the_labels_follow_the_fused_probabilities():
    vol, blob = make_volume()
    labels = _segmenter().segment(vol).labels
    assert labels.shape == blob.shape
    assert (labels[blob] > 0).mean() > 0.9, "the bright blob should be segmented as tumour"
    assert (labels[vol[0] == 0] == 0).all(), "nothing outside the brain may be labelled"


# ------------------------------------------------------------------------------------------------ Req 15
def test_et_relabelled_is_true_exactly_when_the_volume_is_between_zero_and_the_threshold():
    vol, _ = make_volume()
    stubs = dict(models_2d=[ConstantStub(1)], models_3d=[ConstantStub(1)])
    base = _segmenter(**stubs, et_min_mm3=500.0).segment(vol)
    n_et = base.diagnostics["et_voxels_before_rule"]
    assert n_et > 1000 and not base.diagnostics["et_relabelled"] and (base.labels == 1).sum() == n_et
    at = _segmenter(**stubs, et_min_mm3=float(n_et)).segment(vol)                     # exactly at the threshold: kept
    assert not at.diagnostics["et_relabelled"] and (at.labels == 1).sum() == n_et
    below = _segmenter(**stubs, et_min_mm3=n_et + 0.5).segment(vol)
    assert below.diagnostics["et_relabelled"] and not (below.labels == 1).any() and (below.labels == 2).sum() == n_et
    tiny_voxel = _segmenter(**stubs, voxel_mm3=1e-3, et_min_mm3=500.0).segment(vol)   # in cubic millimetres, not voxels
    assert tiny_voxel.diagnostics["et_relabelled"] and not (tiny_voxel.labels == 1).any()


def test_no_et_is_never_reported_as_relabelled():
    vol, _ = make_volume()
    result = _segmenter([ConstantStub(2)], [ConstantStub(2)]).segment(vol)
    assert result.diagnostics["et_voxels_before_rule"] == 0 and not result.diagnostics["et_relabelled"]
    assert (result.labels == 2).any()


# ------------------------------------------------------------------------------------------------ Req 16
@pytest.mark.parametrize("value, status, must_have", [
    (0.95, "ok", []), (0.70, "ok", []),
    (0.50, "review", ["disagree"]),
    (0.05, "review", ["collapse", "orientation"]),
])
def test_agreement_thresholds_set_the_status_and_labels_are_always_returned(monkeypatch, value, status, must_have):
    monkeypatch.setattr(fusion, "agreement", lambda p2, p3: value)
    result = _segmenter().segment(make_volume()[0])
    assert result.status == status and result.labels is not None
    text = " ".join(result.warnings)
    for word in must_have:
        assert word in text
    if status == "ok":
        assert result.warnings == []
    assert abs(result.diagnostics["agreement"] - value) < 1e-12


def test_a_2d_model_that_predicts_nothing_gives_the_strong_warning_but_still_returns_labels():
    vol, blob = make_volume()
    result = _segmenter([ConstantStub(0)], [IdentityStub()]).segment(vol)
    assert result.diagnostics["agreement"] == 0.0 and result.status == "review"
    assert "collapse" in " ".join(result.warnings) and "orientation" in " ".join(result.warnings)
    assert result.labels is not None and (result.labels[blob] > 0).any()          # the 3D branch still speaks


def test_agreeing_identity_models_give_status_ok():
    result = _segmenter().segment(make_volume()[0])
    assert result.diagnostics["agreement"] > 0.7 and result.status == "ok" and result.warnings == []


# ------------------------------------------------------------------------------------------------ Req 17
def test_self_check_passes_on_the_correct_code():
    assert _segmenter().self_check() is True


def test_a_wrong_frame_constant_makes_the_self_check_fail(monkeypatch):
    seg = _segmenter()
    monkeypatch.setattr(preprocess, "FRAME_FLIPS", (0,))
    with pytest.raises(SelfCheckError, match="2D path"):
        seg.self_check()


def test_a_displaced_3d_path_makes_the_self_check_fail(monkeypatch):
    seg = _segmenter()
    original = preprocess.prepare_3d_input
    monkeypatch.setattr(preprocess, "prepare_3d_input", lambda vol, present: np.roll(original(vol, present), 12, axis=2))
    with pytest.raises(SelfCheckError, match="3D path"):
        seg.self_check()


def test_self_check_runs_only_when_asked_for_injected_models_and_once_per_process_for_loaded_ones(monkeypatch):
    calls = []
    monkeypatch.setattr(RStarSegmenter, "self_check", lambda self: calls.append(1) or True)
    RStarSegmenter(tiny_config(), models_2d=[IdentityStub()], models_3d=[IdentityStub()])
    assert calls == []                                                                 # injected models: never automatic
    RStarSegmenter(tiny_config(), models_2d=[IdentityStub()], models_3d=[IdentityStub()], run_self_check=True)
    assert calls == [1]
    calls.clear()
    monkeypatch.setattr(pipeline_module.model_loading, "load_2d_ensemble", lambda cfg: ([IdentityStub()], [{"name": "a", "sha256": "0" * 64}]))
    monkeypatch.setattr(pipeline_module.model_loading, "load_3d_family", lambda cfg: ([IdentityStub()], [{"name": "b", "sha256": "1" * 64}]))
    monkeypatch.setitem(pipeline_module._SELF_CHECK_DONE, "done", False)
    first = RStarSegmenter(tiny_config())
    RStarSegmenter(tiny_config())
    assert calls == [1], "loaded models must self-check once per process, not once per instance"
    assert [c["name"] for c in first.segment(make_volume()[0]).diagnostics["checkpoints"]] == ["a", "b"]
    off = RStarSegmenter(tiny_config(), run_self_check=False)
    assert calls == [1] and off is not None


def test_the_strong_warning_starts_below_the_strong_threshold_not_at_it(monkeypatch):
    monkeypatch.setattr(fusion, "agreement", lambda p2, p3: 0.10)
    text = " ".join(_segmenter().segment(make_volume()[0]).warnings)
    assert "disagree" in text and "collapse" not in text


class _TwoClass(torch.nn.Module):
    """Background 0.55 vs class 2 at 0.45 everywhere: plain argmax says background, a halved background says class 2."""

    def forward(self, x):
        n = x.shape[0]
        p = torch.tensor([0.55, 0.0, 0.45, 0.0, 0.0]).clamp_min(1e-9).log()
        return p.reshape(1, 5, *([1] * (x.ndim - 2))).expand(n, 5, *x.shape[2:]).clone(), None


def test_three_d_only_mode_uses_the_plain_argmax_not_the_background_scale():
    vol, _ = make_volume()
    plain = _segmenter(models_3d=[_TwoClass()]).segment(vol, present=(True, True, False, True))
    assert plain.mode == "3D-only" and (plain.labels == 0).all()
    fused = _segmenter(models_2d=[_TwoClass()], models_3d=[_TwoClass()]).segment(vol)
    assert fused.mode == "R*" and (fused.labels == 2).all()
