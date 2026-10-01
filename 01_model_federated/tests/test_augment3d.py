import sys

import pytest
import torch

import run
from src.augment3d import AssertLabelValuesd, Augment3D, RandModalityDropoutd, build_transforms3d
from src.config import TrainConfig
from src.train_single import train_single_client

VALID_LABELS = {0, 1, 2, 3, 4}


def _dummy_volume(seed: int = 0):
    rng = torch.Generator().manual_seed(seed)
    x = torch.randn(1, 4, 32, 32, 32, generator=rng)
    y = torch.randint(0, 5, (1, 32, 32, 32), generator=rng)
    return x, y


def test_shape_and_dtype_preserved():
    x, y = _dummy_volume()
    aug = Augment3D(seed=0)
    ax, ay = aug(x, y)
    assert ax.shape == x.shape
    assert ay.shape == y.shape
    assert ax.dtype == x.dtype
    assert ay.dtype == y.dtype


def test_labels_stay_in_valid_set_even_when_everything_fires():
    x, y = _dummy_volume()
    aug = Augment3D(flip_prob=1.0, rotate_prob=1.0, zoom_prob=1.0,
                    scale_intensity_prob=1.0, shift_intensity_prob=1.0,
                    gaussian_noise_prob=1.0, seed=1)
    _ax, ay = aug(x, y)
    assert set(torch.unique(ay).tolist()) <= VALID_LABELS


def test_forced_augmentation_actually_changes_the_image():
    """Guards against a silent no-op (e.g. probabilities wired to the wrong
    transform, or the dict keys not matching what Compose expects)."""
    x, y = _dummy_volume()
    aug = Augment3D(flip_prob=1.0, rotate_prob=1.0, zoom_prob=1.0,
                    scale_intensity_prob=1.0, shift_intensity_prob=1.0,
                    gaussian_noise_prob=1.0, seed=2)
    ax, _ay = aug(x, y)
    assert not torch.equal(ax, x.float())


def test_all_probabilities_zero_is_a_true_noop():
    """Every Rand*d branch has an independent prob; setting them all to 0.0
    must be a hard guarantee of no change, not just likely."""
    x, y = _dummy_volume()
    aug = Augment3D(flip_prob=0.0, rotate_prob=0.0, zoom_prob=0.0,
                    scale_intensity_prob=0.0, shift_intensity_prob=0.0,
                    gaussian_noise_prob=0.0, seed=3)
    ax, ay = aug(x, y)
    assert torch.equal(ax, x.float())
    assert torch.equal(ay, y)


def test_reproducible_with_the_same_seed():
    x, y = _dummy_volume()
    aug_a = Augment3D(flip_prob=1.0, rotate_prob=1.0, seed=42)
    aug_b = Augment3D(flip_prob=1.0, rotate_prob=1.0, seed=42)
    xa, ya = aug_a(x, y)
    xb, yb = aug_b(x, y)
    assert torch.equal(xa, xb)
    assert torch.equal(ya, yb)


def test_assert_label_values_raises_on_out_of_range_label():
    guard = AssertLabelValuesd(keys=["label"], valid_labels=VALID_LABELS)
    try:
        guard({"label": torch.tensor([0.0, 1.0, 2.0, 7.0])})
    except AssertionError:
        return
    raise AssertionError("expected AssertLabelValuesd to raise on label value 7")


def test_multi_sample_batch_processed_independently():
    """batch_size is fixed to 1 everywhere else in this section, but the
    callable itself must not silently assume that -- covers B>1 directly.

    Shape parity alone doesn't prove "independently": a forced RandFlipd
    always looks the same regardless of per-call randomness (prob=1.0 just
    means "always fire", not "draw something new each time"), so two
    identical inputs processed by a flip-only stack would pass a shape-only
    check even if the loop secretly reused one draw for the whole batch.
    RandRotated at prob=1.0 draws a fresh random angle on every call, so
    duplicating one volume across the batch and forcing rotation on is a
    real test: equal outputs afterward would mean the batch dimension isn't
    actually being iterated independently.
    """
    rng = torch.Generator().manual_seed(5)
    x = torch.randn(3, 4, 16, 16, 16, generator=rng)
    y = torch.randint(0, 5, (3, 16, 16, 16), generator=rng)
    aug = Augment3D(flip_prob=1.0, seed=5)
    ax, ay = aug(x, y)
    assert ax.shape == x.shape
    assert ay.shape == y.shape

    single_x, single_y = _dummy_volume(seed=7)
    duped_x = single_x[0:1].repeat(2, 1, 1, 1, 1)
    duped_y = single_y[0:1].repeat(2, 1, 1, 1)
    aug_rotate = Augment3D(rotate_prob=1.0, rotate_range_deg=25.0, seed=8)
    rax, _ray = aug_rotate(duped_x, duped_y)
    assert not torch.equal(rax[0], rax[1]), (
        "two identical inputs in one batch produced identical output -- "
        "samples are not being drawn independently"
    )


def test_wired_into_train_single_client_end_to_end(tmp_path, small_manifest):
    """The real Augment3D (not a mock) running through the actual training
    loop on dummy data -- the same code path run.py now uses when
    --use-augmentation is passed."""
    manifest_path = small_manifest("hospA", 2)
    config = TrainConfig(
        run_id="aug_real", checkpoint_dir=str(tmp_path / "ckpt"), use_augmentation=True
    )
    _model, losses = train_single_client(
        config, manifest_path, num_epochs=1, augmentation_transform=Augment3D()
    )
    assert len(losses) == 1
    assert torch.isfinite(torch.tensor(losses[0]))


# --- input-sequence dropout (SPEC Addendum 2, Req 25-34) -------------------

ALL_OFF = dict(flip_prob=0.0, rotate_prob=0.0, zoom_prob=0.0, scale_intensity_prob=0.0,
               shift_intensity_prob=0.0, gaussian_noise_prob=0.0)
ALL_ON = dict(flip_prob=1.0, rotate_prob=1.0, zoom_prob=1.0, scale_intensity_prob=1.0,
              shift_intensity_prob=1.0, gaussian_noise_prob=1.0)


def _nonzero_volume(d: int = 6):
    """Channel c holds the constant c+1 everywhere, so a channel is 'kept' iff
    it is still non-zero and 'dropped' iff it is exactly all-zero."""
    x = torch.arange(1, 5, dtype=torch.float32).reshape(1, 4, 1, 1, 1).expand(1, 4, d, d, d).contiguous()
    y = torch.randint(0, 5, (1, d, d, d), generator=torch.Generator().manual_seed(0))
    return x, y


def _dropped(out: torch.Tensor):
    """Per-channel bool for one sample (4, D, H, W): True where exactly all-zero."""
    return [bool((out[c] == 0.0).all()) for c in range(out.shape[0])]


def test_default_adds_no_dropout_transform_and_is_bitwise_identical():  # Req 25
    assert not any(isinstance(t, RandModalityDropoutd) for t in build_transforms3d().transforms)
    x, y = _dummy_volume()
    ax, ay = Augment3D(seed=11, **ALL_ON)(x, y)
    bx, by = Augment3D(seed=11, modality_dropout_prob=0.0, **ALL_ON)(x, y)
    assert torch.equal(ax, bx) and torch.equal(ay, by)


def test_dropout_transform_sits_after_intensity_transforms_before_label_guard():  # Req 25/28
    names = [type(t).__name__ for t in build_transforms3d(modality_dropout_prob=0.2).transforms]
    assert names.count("RandModalityDropoutd") == 1
    assert names.index("RandGaussianNoised") < names.index("RandModalityDropoutd") < names.index("AssertLabelValuesd")


@pytest.mark.parametrize("bad", [-0.1, 1.0, 1.5, float("nan")])
def test_invalid_dropout_probability_raises_value_error(bad):  # Req 26
    with pytest.raises(ValueError, match="modality_dropout_prob"):
        build_transforms3d(modality_dropout_prob=bad)
    with pytest.raises(ValueError, match="modality_dropout_prob"):
        Augment3D(modality_dropout_prob=bad)


def test_each_channel_is_unchanged_or_exactly_zero_and_label_untouched():  # Req 27
    x, y = _dummy_volume(seed=3)
    aug = Augment3D(seed=3, modality_dropout_prob=0.5, **ALL_OFF)
    saw_drop = False
    for _ in range(40):
        ax, ay = aug(x, y)
        assert torch.equal(ay, y)
        for c in range(4):
            same = torch.equal(ax[0, c], x[0, c])
            zero = bool((ax[0, c] == 0.0).all())
            assert same or zero, f"channel {c} was partially modified"
            saw_drop = saw_drop or zero
    assert saw_drop, "dropout never fired in 40 draws at p=0.5"


def test_dropout_does_not_mutate_the_callers_input():
    """Earlier no-op transforms hand back the caller's own tensor; zeroing it in
    place would silently corrupt the dataset's cached sample."""
    x, y = _dummy_volume(seed=4)
    before = x.clone()
    aug = Augment3D(seed=4, modality_dropout_prob=0.9, **ALL_OFF)
    for _ in range(10):
        aug(x, y)
    assert torch.equal(x, before)


def test_dropped_channels_stay_exactly_zero_even_when_every_other_transform_fires():  # Req 28
    x, y = _nonzero_volume(d=12)
    aug = Augment3D(seed=5, modality_dropout_prob=0.99, **ALL_ON)
    for _ in range(15):
        ax, ay = aug(x, y)
        # p=0.99 => almost always 3 dropped + 1 kept; if scale/shift/noise ran AFTER
        # dropout they would make dropped channels non-zero and this count would fall.
        assert sum(_dropped(ax[0])) >= 2, "dropped channels were re-populated by a later transform"
        assert not all(_dropped(ax[0]))
        assert set(torch.unique(ay).tolist()) <= VALID_LABELS


def test_never_all_channels_dropped_and_kept_channel_is_not_biased():  # Req 29
    x, y = _nonzero_volume()
    aug = Augment3D(seed=6, modality_dropout_prob=0.99, **ALL_OFF)
    kept_counts = [0, 0, 0, 0]
    for _ in range(400):
        ax, _ay = aug(x, y)
        dropped = _dropped(ax[0])
        assert not all(dropped), "an all-zero input was produced"
        for c, is_dropped in enumerate(dropped):
            kept_counts[c] += 0 if is_dropped else 1
    assert all(n > 0 for n in kept_counts), f"a channel was never kept: {kept_counts}"


def test_observed_drop_rate_matches_the_requested_probability():  # Req 30
    x, y = _nonzero_volume()
    aug = Augment3D(seed=7, modality_dropout_prob=0.3, **ALL_OFF)
    drops = [0, 0, 0, 0]
    n = 400
    for _ in range(n):
        ax, _ay = aug(x, y)
        for c, is_dropped in enumerate(_dropped(ax[0])):
            drops[c] += int(is_dropped)
    for c, count in enumerate(drops):
        assert 0.2 <= count / n <= 0.4, f"channel {c} drop rate {count / n:.3f} outside [0.2, 0.4]"


def test_dropout_reproducible_with_the_same_seed():  # Req 31
    x, y = _dummy_volume(seed=8)
    a = Augment3D(seed=99, modality_dropout_prob=0.5)
    b = Augment3D(seed=99, modality_dropout_prob=0.5)
    for _ in range(5):
        xa, ya = a(x, y)
        xb, yb = b(x, y)
        assert torch.equal(xa, xb) and torch.equal(ya, yb)


def test_dropout_patterns_are_independent_across_a_batch():  # Req 32
    x, y = _nonzero_volume()
    xb, yb = x.repeat(16, 1, 1, 1, 1), y.repeat(16, 1, 1, 1)
    ax, _ay = Augment3D(seed=9, modality_dropout_prob=0.5, **ALL_OFF)(xb, yb)
    patterns = {tuple(_dropped(ax[i])) for i in range(16)}
    assert len(patterns) > 1, "all 16 identical samples got the same drop pattern"


def _run_main(monkeypatch, argv):
    """Run run.main() with both training entry points replaced by recorders, so
    the CLI path is real but nothing trains and no data is loaded."""
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


def _dropout_transforms(aug):
    return [t for t in aug._compose.transforms if isinstance(t, RandModalityDropoutd)]


def test_cli_default_is_zero_and_no_dropout_without_the_flag(monkeypatch):  # Req 33
    assert run.build_arg_parser().parse_args([]).modality_dropout == 0.0
    calls = _run_main(monkeypatch, ["--use-augmentation"])
    assert _dropout_transforms(calls["single"]["augmentation_transform"]) == []


def test_cli_passes_dropout_to_both_training_paths(monkeypatch):  # Req 33
    single = _run_main(monkeypatch, ["--use-augmentation", "--modality-dropout", "0.2"])
    (t,) = _dropout_transforms(single["single"]["augmentation_transform"])
    assert t.drop_prob == pytest.approx(0.2)
    fed = _run_main(monkeypatch, ["--use-augmentation", "--modality-dropout", "0.2", "--use-federation"])
    (t,) = _dropout_transforms(fed["federated"]["augmentation_transform"])
    assert t.drop_prob == pytest.approx(0.2)


@pytest.mark.parametrize("argv, must_name", [
    (["--modality-dropout", "0.2"], "--use-augmentation"),  # would silently do nothing
    (["--use-augmentation", "--modality-dropout", "1.0"], "--modality-dropout"),
    (["--use-augmentation", "--modality-dropout", "-0.1"], "--modality-dropout"),
    (["--use-augmentation", "--modality-dropout", "nan"], "--modality-dropout"),
])
def test_cli_rejects_bad_dropout_arguments_before_anything_runs(monkeypatch, capsys, argv, must_name):  # Req 33
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


def test_dropout_wired_into_train_single_client_end_to_end(tmp_path, small_manifest):  # Req 34
    """Real Augment3D with dropout on (not a mock) through the real training
    loop. Says nothing about held-out Dice -- only that the path runs and the
    loss stays finite."""
    manifest_path = small_manifest("hospA", 2)
    config = TrainConfig(
        run_id="aug_dropout", checkpoint_dir=str(tmp_path / "ckpt"), use_augmentation=True
    )
    _model, losses = train_single_client(
        config, manifest_path, num_epochs=1,
        augmentation_transform=Augment3D(modality_dropout_prob=0.3),
    )
    assert len(losses) == 1
    assert torch.isfinite(torch.tensor(losses[0]))
