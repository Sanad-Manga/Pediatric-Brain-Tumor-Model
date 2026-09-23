import torch

from src.augment3d import AssertLabelValuesd, Augment3D
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
