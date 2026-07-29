"""Per-patient scoring, empty-region conventions, and the CSV contract."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from src.dummy import DummySegNet2D, save_dummy_checkpoint
from src.evaluate import CSV_COLUMNS, append_result_row, restack_volume
from src.evaluate import run as run_eval
from src.metrics import dice_regions, dice_score, region_mask

CONTRACT_HEADER = (
    "experiment_name,use_augmentation,use_federation,use_domain_adaptation,"
    "dice_ET,dice_NC,dice_WT,mean_dice"
)


# ----------------------------------------------------------------- Req 39
def test_restacking_rebuilds_the_volume_frame_from_either_plane(cfg):
    rng = np.random.default_rng(0)
    volume = rng.integers(0, 5, size=(24, 24, 16)).astype(np.int64)

    for plane in ("axial", "coronal"):
        axis = cfg.plane_axis(plane)
        slices = np.moveaxis(volume, axis, 0)
        assert np.array_equal(restack_volume(slices, plane, cfg), volume)


def test_restacking_places_slices_at_their_true_index(cfg):
    """Real subjects have blank edge slices dropped: indices run e.g. 8..134.

    Stacking those densely would shorten the volume and shift every slice, and
    would give axial and coronal different shapes so `both` could not average
    them. This is the case that silently corrupts per-patient Dice.
    """
    vs = cfg.volume_shape
    axis = cfg.plane_axis("axial")
    n_full = vs[axis]

    full = np.random.default_rng(7).integers(0, 5, vs).astype(np.int64)
    kept = list(range(3, n_full - 2))                      # blank edges dropped
    slices = np.moveaxis(full, axis, 0)[kept]

    out = restack_volume(slices, "axial", cfg, indices=kept)
    assert out.shape == tuple(vs)                          # full size, not len(kept)
    assert np.array_equal(np.moveaxis(out, axis, 0)[kept], slices)
    dropped = [i for i in range(n_full) if i not in kept]
    assert not np.moveaxis(out, axis, 0)[dropped].any()    # missing stay background


def test_trimmed_planes_restack_to_a_common_shape(cfg):
    """Axial and coronal trimmed differently must still be averageable."""
    vs = cfg.volume_shape
    volumes = {}
    for plane in ("axial", "coronal"):
        axis = cfg.plane_axis(plane)
        kept = list(range(2, vs[axis] - 3))
        arr = np.zeros((5, len(kept), *[s for i, s in enumerate(vs) if i != axis]),
                       dtype=np.float32)
        volumes[plane] = restack_volume(arr, plane, cfg, channelled=True, indices=kept)

    assert volumes["axial"].shape == volumes["coronal"].shape == (5, *vs)
    assert (volumes["axial"] + volumes["coronal"]).shape == (5, *vs)


def test_restacking_handles_the_channel_axis(cfg):
    rng = np.random.default_rng(1)
    probs = rng.random((5, 16, 24, 24)).astype(np.float32)   # (C, N, H, W) axial
    out = restack_volume(probs, "axial", cfg, channelled=True)
    assert out.shape == (5, 24, 24, 16)


def test_per_patient_scoring_differs_from_per_slice_averaging(cfg):
    """The failure mode this requirement exists to prevent.

    A volume where the tumour occupies one slice, predicted entirely empty:
    per-patient Dice is 0.0, while averaging per slice rewards every correctly
    empty slice with 1.0 and reports ~0.94.
    """
    true_volume = np.zeros((24, 24, 16), dtype=np.int64)
    true_volume[8:16, 8:16, 8] = 1
    pred_volume = np.zeros_like(true_volume)

    per_patient = dice_regions(pred_volume, true_volume)["dice_WT"]

    axis = cfg.plane_axis("axial")
    per_slice = np.mean([
        dice_score(region_mask(p, "WT"), region_mask(t, "WT"))
        for p, t in zip(np.moveaxis(pred_volume, axis, 0), np.moveaxis(true_volume, axis, 0))
    ])

    assert per_patient == 0.0
    assert per_slice > 0.9          # the overstatement, quantified
    assert per_slice != per_patient


# ----------------------------------------------------------------- Req 42
def test_region_empty_in_both_scores_one():
    empty = np.zeros((8, 8, 8), dtype=np.int64)
    assert dice_regions(empty, empty) == {"dice_ET": 1.0, "dice_NC": 1.0, "dice_WT": 1.0}


def test_region_empty_in_exactly_one_scores_zero():
    empty = np.zeros((8, 8, 8), dtype=np.int64)
    present = np.zeros((8, 8, 8), dtype=np.int64)
    present[2:5, 2:5, 2:5] = 1

    assert dice_regions(empty, present)["dice_ET"] == 0.0    # missed a real tumour
    assert dice_regions(present, empty)["dice_ET"] == 0.0    # hallucinated one


def test_a_perfect_prediction_scores_one():
    volume = np.zeros((8, 8, 8), dtype=np.int64)
    volume[2:5, 2:5, 2:5] = 1
    volume[5:6, 2:5, 2:5] = 4
    scores = dice_regions(volume, volume)
    assert scores == {"dice_ET": 1.0, "dice_NC": 1.0, "dice_WT": 1.0}


# ------------------------------------------------------------- Req 44, 45, 46
def test_csv_header_matches_the_contract_byte_for_byte(cfg, tmp_path):
    path = tmp_path / "results.csv"
    append_result_row({c: 0 for c in CSV_COLUMNS}, path)
    assert path.read_text(encoding="utf-8").splitlines()[0] == CONTRACT_HEADER


def test_exactly_one_row_is_appended_per_call(cfg, tmp_path):
    path = tmp_path / "results.csv"
    for name in ("baseline", "aug", "aug_fed"):
        append_result_row({**{c: 0 for c in CSV_COLUMNS}, "experiment_name": name}, path)

    rows = list(csv.DictReader(open(path, encoding="utf-8", newline="")))
    assert len(rows) == 3
    assert [r["experiment_name"] for r in rows] == ["baseline", "aug", "aug_fed"]


def test_a_mismatched_existing_header_raises_and_writes_nothing(tmp_path):
    path = tmp_path / "results.csv"
    path.write_text("wrong,header\n1,2\n", encoding="utf-8")
    before = path.read_bytes()

    with pytest.raises(ValueError, match="unexpected header"):
        append_result_row({c: 0 for c in CSV_COLUMNS}, path)
    assert path.read_bytes() == before


def test_a_missing_column_raises(tmp_path):
    with pytest.raises(ValueError, match="missing column"):
        append_result_row({"experiment_name": "x"}, tmp_path / "results.csv")


def test_results_parent_directory_is_created(tmp_path):
    path = tmp_path / "nested" / "deeper" / "results.csv"
    append_result_row({c: 0 for c in CSV_COLUMNS}, path)
    assert path.exists()


# ----------------------------------------------------------------- Req 47
@pytest.mark.parametrize("plane", ["axial", "coronal", "both"])
def test_eval_runs_with_no_cache_and_no_trained_model(cfg, tmp_path, plane):
    cfg.eval["plane"] = plane
    out_csv = tmp_path / "results" / "ablation_results.csv"

    row = run_eval(
        cfg, experiment_name=f"dummy_{plane}",
        dummy_checkpoint=True, dummy_data=True, dummy_n=3,
        csv_path=out_csv, tmp_dir=tmp_path / "tmpcache",
    )

    assert out_csv.exists()
    assert row["experiment_name"] == f"dummy_{plane}"
    assert row["_n_subjects"] == 3
    for region in ("ET", "NC", "WT"):
        assert 0.0 <= row[f"dice_{region}"] <= 1.0


def test_mean_dice_is_the_mean_of_the_three_regions(cfg, tmp_path):
    row = run_eval(
        cfg, experiment_name="mean_check", dummy_checkpoint=True, dummy_data=True,
        dummy_n=2, csv_path=tmp_path / "r.csv", tmp_dir=tmp_path / "c",
    )
    expected = round((row["dice_ET"] + row["dice_NC"] + row["dice_WT"]) / 3.0, 6)
    assert row["mean_dice"] == expected


# ----------------------------------------------------------------- Req 43
def test_empty_ground_truth_counts_are_reported(cfg, tmp_path):
    row = run_eval(
        cfg, experiment_name="empty_counts", dummy_checkpoint=True, dummy_data=True,
        dummy_n=3, csv_path=tmp_path / "r.csv", tmp_dir=tmp_path / "c",
    )
    counts = row["_empty_ground_truth"]
    assert set(counts) == {"ET", "NC", "WT"}
    assert all(0 <= v <= 3 for v in counts.values())


# ----------------------------------------------------------------- Req 48
def test_eval_without_a_checkpoint_option_raises(cfg, tmp_path):
    with pytest.raises(ValueError, match="--checkpoint"):
        run_eval(cfg, experiment_name="no_ckpt", csv_path=tmp_path / "r.csv",
                 tmp_dir=tmp_path / "c")


def test_a_missing_checkpoint_file_raises(cfg, tmp_path):
    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        run_eval(cfg, experiment_name="bad_ckpt", checkpoint=tmp_path / "nope.pt",
                 dummy_data=True, csv_path=tmp_path / "r.csv", tmp_dir=tmp_path / "c")


def test_a_real_checkpoint_round_trips(cfg, tmp_path):
    ckpt = save_dummy_checkpoint(
        tmp_path / "ckpt.pt",
        DummySegNet2D(in_channels=len(cfg.modalities), num_classes=cfg.num_classes),
    )
    row = run_eval(
        cfg, experiment_name="from_ckpt", checkpoint=ckpt, dummy_data=True, dummy_n=2,
        csv_path=tmp_path / "r.csv", tmp_dir=tmp_path / "c",
    )
    assert row["experiment_name"] == "from_ckpt"


# ----------------------------------------------------------------- Req 50
def test_federation_and_da_flags_pass_through_unchanged(cfg, tmp_path):
    cfg.use_federation = True
    cfg.use_domain_adaptation = True
    cfg.use_augmentation = False

    row = run_eval(
        cfg, experiment_name="flags", dummy_checkpoint=True, dummy_data=True, dummy_n=2,
        csv_path=tmp_path / "r.csv", tmp_dir=tmp_path / "c",
    )
    assert row["use_federation"] is True
    assert row["use_domain_adaptation"] is True
    assert row["use_augmentation"] is False


# --------------------------------------------------------- model interface
def test_dummy_model_returns_logits_and_features(cfg):
    """CONTRACTS.md: model(x) -> (seg_logits, features)."""
    import torch

    model = DummySegNet2D(in_channels=4, num_classes=5)
    seg_logits, features = model(torch.zeros(2, 4, 32, 32))
    assert seg_logits.shape == (2, 5, 32, 32)
    assert features.ndim == 2 and features.shape[0] == 2
