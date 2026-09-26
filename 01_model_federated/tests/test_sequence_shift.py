"""Addendum 4 (Req 50-60): sequence-shift augmentation. CPU only, no real data.

Every behavioural claim is checked against an independent computation (np.roll, analytic ramps) rather
than against the implementation itself, and randomness is observed through the transform's own
`randomize` plan, replayed with a same-seed twin so the plan and the applied output describe one draw.
"""
import sys

import numpy as np
import pytest
import torch
from monai.transforms import RandGaussianNoised

import run
from src.augment3d import (
    IMAGE_KEY,
    LABEL_KEY,
    AssertLabelValuesd,
    Augment3D,
    RandModalityDropoutd,
    RandSequenceShiftd,
    build_transforms3d,
)
from src.config import TrainConfig
from src.train_single import train_single_client

ALL_OFF = dict(flip_prob=0.0, rotate_prob=0.0, zoom_prob=0.0, scale_intensity_prob=0.0,
               shift_intensity_prob=0.0, gaussian_noise_prob=0.0)


def _make(prob=1.0, max_shift=1.2, seed=0):
    t = RandSequenceShiftd(keys=[IMAGE_KEY], prob=prob, max_shift_voxels=max_shift)
    t.set_random_state(seed=seed)
    return t


def _twins(seed, prob=1.0, max_shift=1.2):
    """Two identically seeded transforms: one reveals the plan, the other is applied."""
    return _make(prob, max_shift, seed), _make(prob, max_shift, seed)


def _volume(c=4, shape=(9, 10, 11), seed=0, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(c, *shape, generator=g).to(dtype)


# ------------------------------------------------------------------------------------------ Req 50
@pytest.mark.parametrize("bad", [-0.1, 1.5, float("nan")])
def test_bad_prob_raises_value_error(bad):
    with pytest.raises(ValueError, match="sequence_shift_prob"):
        RandSequenceShiftd(keys=[IMAGE_KEY], prob=bad, max_shift_voxels=1.0)


@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf"), float("-inf")])
def test_bad_max_shift_raises_value_error(bad):
    with pytest.raises(ValueError, match="sequence_shift_max_voxels"):
        RandSequenceShiftd(keys=[IMAGE_KEY], prob=0.5, max_shift_voxels=bad)


def test_boundary_values_are_accepted():
    RandSequenceShiftd(keys=[IMAGE_KEY], prob=0.0, max_shift_voxels=0.0)
    RandSequenceShiftd(keys=[IMAGE_KEY], prob=1.0, max_shift_voxels=5.0)


@pytest.mark.parametrize("prob", [0.0, 1.0])
def test_single_channel_input_raises_at_call_time_even_when_not_applied(prob):
    t = _make(prob=prob)
    with pytest.raises(ValueError, match="at least 2 channels"):
        t({IMAGE_KEY: torch.randn(1, 6, 6, 6)})
    with pytest.raises(ValueError, match="at least 2 channels"):
        t.randomize(1)


# ------------------------------------------------------------------------------------------ Req 51
@pytest.mark.parametrize("kind", ["tensor", "ndarray"])
def test_not_applied_is_a_bitwise_copy(kind):
    x = _volume()
    x = x if kind == "tensor" else x.numpy()
    before = x.clone() if kind == "tensor" else x.copy()
    out = _make(prob=0.0)({IMAGE_KEY: x})[IMAGE_KEY]
    assert out is not x, "output aliases the input"
    assert (torch.equal(out, before) if kind == "tensor" else np.array_equal(out, before))
    out[0, 0, 0, 0] = 123.0            # writing to the output must not reach the input
    assert (torch.equal(x, before) if kind == "tensor" else np.array_equal(x, before))


@pytest.mark.parametrize("kind", ["tensor", "ndarray"])
def test_applied_never_modifies_the_input_in_place(kind):
    x = _volume()
    x = x if kind == "tensor" else x.numpy()
    before = x.clone() if kind == "tensor" else x.copy()
    t = _make(prob=1.0, max_shift=2.0, seed=3)
    for _ in range(5):
        out = t({IMAGE_KEY: x})[IMAGE_KEY]
        assert out is not x
    assert (torch.equal(x, before) if kind == "tensor" else np.array_equal(x, before))


# ------------------------------------------------------------------------------------------ Req 52
def test_reference_channel_is_uniform_and_never_moved():
    n_draws = 480
    probe, applied = _twins(seed=7)
    refs = []
    x = _volume()
    for _ in range(n_draws):
        ref, shifts = probe.randomize(4)
        out = applied({IMAGE_KEY: x})[IMAGE_KEY]
        assert ref in (0, 1, 2, 3)
        assert np.all(shifts[ref] == 0.0), "the reference has a non-zero shift vector"
        assert torch.equal(out[ref], x[ref]), "the reference channel changed"
        refs.append(ref)
    for c in range(4):
        rate = refs.count(c) / n_draws
        assert 0.15 <= rate <= 0.35, f"channel {c} is the reference in {rate:.2f} of the draws"


# ------------------------------------------------------------------------------------------ Req 53
def test_every_applied_call_moves_at_least_one_other_channel():
    probe, applied = _twins(seed=11, max_shift=1.5)
    x = _volume(seed=1)
    for _ in range(250):
        ref, shifts = probe.randomize(4)
        others = [c for c in range(4) if c != ref]
        assert np.any(np.any(shifts[others] != 0.0, axis=1)), "an applied draw moved nothing"
        out = applied({IMAGE_KEY: x})[IMAGE_KEY]
        assert not torch.equal(out, x)
        changed = [c for c in range(4) if not torch.equal(out[c], x[c])]
        assert changed and ref not in changed
        assert set(changed) == {c for c in range(4) if np.any(shifts[c] != 0.0)}


def test_selection_rate_of_non_reference_channels_is_about_one_half():
    t = _make(seed=5)
    moved = total = 0
    for _ in range(600):
        ref, shifts = t.randomize(4)
        for c in range(4):
            if c != ref:
                total += 1
                moved += bool(np.any(shifts[c] != 0.0))
    # 0.5 chosen independently, plus the forced pick when none was chosen
    assert 0.45 <= moved / total <= 0.62


def test_probability_controls_how_often_the_shift_is_applied():
    t = _make(prob=0.3, seed=2)
    applied = sum(t.randomize(4)[0] != -1 for _ in range(1000))
    assert 240 <= applied <= 360


# ------------------------------------------------------------------------------------------ Req 54
class _Fixed(RandSequenceShiftd):
    def __init__(self, plan):
        super().__init__(keys=[IMAGE_KEY], prob=1.0, max_shift_voxels=10.0)
        self._plan = np.asarray(plan, dtype=np.float64)

    def randomize(self, n_channels):
        return 1, self._plan.copy()


def _interior(shape, shift):
    """Slices of the output where the source location lies inside the volume."""
    sl = []
    for n, s in zip(shape, shift):
        s = int(s)
        sl.append(slice(s, n) if s >= 0 else slice(0, n + s))
    return tuple(sl)


def test_integer_shift_equals_np_roll_on_the_interior_and_vacated_voxels_are_zero():
    plan = [[0, 0, 0], [2, -1, 3], [0, 0, 0], [-3, 2, 1]]
    x = _volume(shape=(9, 10, 11), seed=4)
    out = _Fixed(plan)({IMAGE_KEY: x})[IMAGE_KEY]
    for c, shift in enumerate(plan):
        if not any(shift):
            assert torch.equal(out[c], x[c]), f"channel {c} has no shift and must be untouched"
            continue
        expected = np.roll(x[c].numpy(), shift=tuple(int(v) for v in shift), axis=(0, 1, 2))
        inner = _interior(x.shape[1:], shift)
        np.testing.assert_allclose(out[c].numpy()[inner], expected[inner], atol=1e-4)
        mask = np.ones(x.shape[1:], dtype=bool)
        mask[inner] = False
        assert np.all(out[c].numpy()[mask] == 0.0), "vacated voxels are not exactly 0"


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("shift", [0.4, -0.7, 1.25])
def test_subvoxel_shift_of_a_ramp_matches_the_analytic_ramp(axis, shift):
    shape = (8, 9, 10)
    idx = torch.arange(shape[axis], dtype=torch.float32)
    view = [1, 1, 1]
    view[axis] = shape[axis]
    ramp = (idx.reshape(view) + 5.0).expand(*shape).contiguous()
    vec = [0.0, 0.0, 0.0]
    vec[axis] = shift
    plan = [[0, 0, 0], vec, [0, 0, 0], [0, 0, 0]]
    x = torch.stack([ramp] * 4)
    out = _Fixed(plan)({IMAGE_KEY: x})[IMAGE_KEY][1]
    src = idx - shift                                    # out[i] = in[i - s]
    valid = (src >= 0) & (src <= shape[axis] - 1)
    expected = (src + 5.0).reshape(view).expand(*shape)
    valid_full = valid.reshape(view).expand(*shape)
    np.testing.assert_allclose(out.numpy()[valid_full.numpy()], expected.numpy()[valid_full.numpy()], atol=1e-4)
    assert np.all(out.numpy()[~valid_full.numpy()] == 0.0)


# ------------------------------------------------------------------------------------------ Req 55
def _ramp_input(axis, shape=(12, 13, 14)):
    view = [1, 1, 1]
    view[axis] = shape[axis]
    ramp = (torch.arange(shape[axis], dtype=torch.float32).reshape(view) + 3.0).expand(*shape).contiguous()
    return torch.stack([ramp] * 4)


def _recover(out_c, axis, shape):
    """Recover the shift along `axis` from a ramp channel, using voxels whose source is inside."""
    idx = torch.arange(shape[axis], dtype=torch.float32)
    view = [1, 1, 1]
    view[axis] = shape[axis]
    base = (idx.reshape(view) + 3.0).expand(*shape)
    nonzero = out_c != 0.0
    return float((base - out_c)[nonzero].mean())


def test_shifts_never_exceed_the_maximum_and_are_what_the_plan_says():
    shape = (12, 13, 14)
    max_shift = 1.7
    for seed in range(220):
        probe = _make(max_shift=max_shift, seed=seed)
        ref, plan = probe.randomize(4)
        assert np.all(np.abs(plan) <= max_shift + 1e-9)
        for axis in range(3):
            out = _make(max_shift=max_shift, seed=seed)({IMAGE_KEY: _ramp_input(axis, shape)})[IMAGE_KEY]
            for c in range(4):
                if c == ref or not np.any(plan[c] != 0.0):
                    continue
                measured = _recover(out[c], axis, shape)
                assert abs(measured) <= max_shift + 1e-3
                assert measured == pytest.approx(plan[c][axis], abs=1e-3)


def test_shift_components_are_symmetric_and_fill_the_whole_range():
    t = _make(max_shift=2.0, seed=17)
    vectors = []
    for _ in range(500):
        _ref, shifts = t.randomize(4)
        vectors.extend(v for v in shifts if np.any(v != 0.0))
    arr = np.array(vectors)
    assert len(arr) > 500
    for axis in range(3):
        assert abs(arr[:, axis].mean()) < 0.2, "shifts are biased to one direction"
        assert arr[:, axis].min() < -1.8 and arr[:, axis].max() > 1.8, "the range is not fully used"


def test_each_shifted_channel_gets_its_own_vector():
    t = _make(seed=13)
    multi = distinct = 0
    for _ in range(300):
        ref, shifts = t.randomize(4)
        moved = [shifts[c] for c in range(4) if np.any(shifts[c] != 0.0)]
        if len(moved) >= 2:
            multi += 1
            distinct += any(not np.allclose(moved[i], moved[j]) for i in range(len(moved)) for j in range(i))
    assert multi > 100
    assert distinct / multi > 0.5


# ------------------------------------------------------------------------------------------ Req 56
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("kind", ["tensor", "ndarray"])
def test_shape_dtype_and_type_are_preserved(dtype, kind):
    x = _volume(dtype=dtype)
    x = x if kind == "tensor" else x.numpy()
    out = _make(seed=1, max_shift=2.0)({IMAGE_KEY: x})[IMAGE_KEY]
    assert type(out) is type(x)
    assert out.shape == x.shape and out.dtype == x.dtype


def test_the_label_entry_is_never_touched():
    x = _volume()
    y = torch.randint(0, 5, (1, 9, 10, 11))
    y_before = y.clone()
    out = _make(seed=4, max_shift=2.0)({IMAGE_KEY: x, LABEL_KEY: y})
    assert out[LABEL_KEY] is y
    assert torch.equal(y, y_before)


# ------------------------------------------------------------------------------------------ Req 57
def test_same_seed_is_bitwise_reproducible_and_a_different_seed_differs():
    x = _volume(shape=(12, 12, 12), seed=2)
    a = _make(seed=21, max_shift=2.0)({IMAGE_KEY: x})[IMAGE_KEY]
    b = _make(seed=21, max_shift=2.0)({IMAGE_KEY: x})[IMAGE_KEY]
    c = _make(seed=22, max_shift=2.0)({IMAGE_KEY: x})[IMAGE_KEY]
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


# ------------------------------------------------------------------------------------------ Req 58
BASELINE = ["RandFlipd", "RandFlipd", "RandFlipd", "RandRotated", "RandZoomd", "RandScaleIntensityd",
            "RandShiftIntensityd", "RandGaussianNoised", "AssertLabelValuesd"]


def _names(compose):
    return [type(t).__name__ for t in compose.transforms]


def test_default_chain_is_exactly_the_pre_addendum_chain():
    assert _names(build_transforms3d()) == BASELINE
    assert _names(build_transforms3d(sequence_shift_prob=0.0, sequence_shift_max_voxels=9.0)) == BASELINE
    assert _names(build_transforms3d(modality_dropout_prob=0.2)) == BASELINE[:-1] + ["RandModalityDropoutd", "AssertLabelValuesd"]


def test_shift_sits_after_the_noise_and_before_dropout_and_the_label_assert():
    names = _names(build_transforms3d(sequence_shift_prob=0.5))
    assert names == BASELINE[:-1] + ["RandSequenceShiftd", "AssertLabelValuesd"]
    names = _names(build_transforms3d(sequence_shift_prob=0.5, modality_dropout_prob=0.2))
    assert names == BASELINE[:-1] + ["RandSequenceShiftd", "RandModalityDropoutd", "AssertLabelValuesd"]
    chain = build_transforms3d(sequence_shift_prob=0.5).transforms
    assert isinstance(chain[7], RandGaussianNoised) and isinstance(chain[8], RandSequenceShiftd)
    assert isinstance(chain[9], AssertLabelValuesd)


def test_dropped_channels_stay_exactly_zero_with_shift_and_dropout_both_on():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(8, 4, 20, 20, 20, generator=g)
    y = torch.randint(0, 5, (8, 20, 20, 20), generator=g)
    ax, ay = Augment3D(seed=3, sequence_shift_prob=1.0, sequence_shift_max_voxels=2.0,
                       modality_dropout_prob=0.6, **ALL_OFF)(x, y)
    assert ax.shape == x.shape and set(torch.unique(ay).tolist()) <= {0, 1, 2, 3, 4}
    n_zero = 0
    for i in range(8):
        zero = [bool((ax[i, c] == 0).all()) for c in range(4)]
        assert not all(zero), "the model must never receive an all-zero input"
        n_zero += sum(zero)
    assert n_zero > 0


def test_augment3d_forwards_the_kwargs_to_the_transform():
    aug = Augment3D(sequence_shift_prob=0.7, sequence_shift_max_voxels=2.5)
    (t,) = [t for t in aug._compose.transforms if isinstance(t, RandSequenceShiftd)]
    assert t.prob == pytest.approx(0.7) and t.max_shift_voxels == pytest.approx(2.5)


@pytest.mark.parametrize("kwargs", [
    dict(sequence_shift_prob=2.0), dict(sequence_shift_prob=-0.1), dict(sequence_shift_prob=float("nan")),
    dict(sequence_shift_prob=0.0, sequence_shift_max_voxels=-1.0),
    dict(sequence_shift_prob=0.5, sequence_shift_max_voxels=float("nan")),
    dict(sequence_shift_prob=0.5, sequence_shift_max_voxels=float("inf")),
])
def test_invalid_values_raise_at_construction(kwargs):
    with pytest.raises(ValueError):
        build_transforms3d(**kwargs)
    with pytest.raises(ValueError):
        Augment3D(**kwargs)


# ------------------------------------------------------------------------------------------ Req 59
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


def _shift_transforms(aug):
    return [t for t in aug._compose.transforms if isinstance(t, RandSequenceShiftd)]


def test_cli_defaults_and_no_shift_transform_without_the_flag(monkeypatch):
    args = run.build_arg_parser().parse_args([])
    assert args.sequence_shift == 0.0 and args.sequence_shift_max_voxels == 1.2
    calls = _run_main(monkeypatch, ["--use-augmentation"])
    assert _shift_transforms(calls["single"]["augmentation_transform"]) == []


def test_cli_forwards_shift_in_real_mode(monkeypatch):
    calls = _run_main(monkeypatch, ["--use-augmentation", "--sequence-shift", "0.4",
                                    "--sequence-shift-max-voxels", "2.0", "--data-mode", "real",
                                    "--cache-path", "somewhere"])
    (t,) = _shift_transforms(calls["single"]["augmentation_transform"])
    assert t.prob == pytest.approx(0.4) and t.max_shift_voxels == pytest.approx(2.0)


def test_cli_forwards_shift_in_patch_mode(monkeypatch):
    calls = _run_main(monkeypatch, ["--use-augmentation", "--sequence-shift", "0.3",
                                    "--sequence-shift-max-voxels", "1.5", "--data-mode", "patch",
                                    "--cache-path", "somewhere"])
    assert calls["single"]["config"].data_mode == "patch"
    (t,) = _shift_transforms(calls["single"]["augmentation_transform"])
    assert t.prob == pytest.approx(0.3) and t.max_shift_voxels == pytest.approx(1.5)


def test_cli_forwards_shift_to_the_federated_path_too(monkeypatch):
    calls = _run_main(monkeypatch, ["--use-augmentation", "--sequence-shift", "0.2", "--use-federation"])
    (t,) = _shift_transforms(calls["federated"]["augmentation_transform"])
    assert t.prob == pytest.approx(0.2)


@pytest.mark.parametrize("argv, must_name", [
    (["--sequence-shift", "0.2"], "--use-augmentation"),                                     # would silently do nothing
    (["--sequence-shift", "0.2"], "--sequence-shift"),
    (["--use-augmentation", "--sequence-shift", "1.0"], "--sequence-shift"),
    (["--use-augmentation", "--sequence-shift", "-0.1"], "--sequence-shift"),
    (["--use-augmentation", "--sequence-shift", "nan"], "--sequence-shift"),
    (["--use-augmentation", "--sequence-shift-max-voxels", "0"], "--sequence-shift-max-voxels"),
    (["--use-augmentation", "--sequence-shift-max-voxels", "-2"], "--sequence-shift-max-voxels"),
    (["--use-augmentation", "--sequence-shift-max-voxels", "nan"], "--sequence-shift-max-voxels"),
    (["--use-augmentation", "--sequence-shift-max-voxels", "inf"], "--sequence-shift-max-voxels"),
])
def test_cli_rejects_bad_arguments_before_anything_runs(monkeypatch, capsys, argv, must_name):
    calls = {}

    def boom(**kw):
        calls["ran"] = True
        return None, []

    monkeypatch.setattr(run, "train_single_client", boom)
    monkeypatch.setattr(run, "train_federated", boom)
    monkeypatch.setattr(sys, "argv", ["run.py", *argv])
    with pytest.raises(SystemExit) as exc:
        run.main()
    assert exc.value.code == 2
    assert must_name in capsys.readouterr().err
    assert "ran" not in calls


# ------------------------------------------------------------------------------------------ Req 60
def test_shift_with_dropout_runs_end_to_end_through_train_single_client(tmp_path, small_manifest):
    """Real Augment3D (not a mock) through the real training loop. Says nothing about robustness to
    misregistration; only that the path runs and the loss stays finite."""
    manifest_path = small_manifest("hospA", 2)
    config = TrainConfig(run_id="aug_shift", checkpoint_dir=str(tmp_path / "ckpt"), use_augmentation=True)
    _model, losses = train_single_client(
        config, manifest_path, num_epochs=1,
        augmentation_transform=Augment3D(sequence_shift_prob=1.0, modality_dropout_prob=0.15),
    )
    assert len(losses) == 1
    assert torch.isfinite(torch.tensor(losses[0]))
