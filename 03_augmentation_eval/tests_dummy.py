"""Dummy-tensor test suite for the augmentation + ablation pipeline.

Runs entirely on random tensors and a randomly-initialized DummySegNet — no
trained checkpoint, no reads from the real 96-cube cache. This is what lets this
section be built and verified before `01_model_federated` delivers a model.

    python tests_dummy.py          # or: pytest tests_dummy.py -q
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.augmentation import apply_transforms, build_transforms
from src.config import load_config
from src.data import znorm_channel
from src.dataset import DummyDataset, EvalDataset, ExpansionDataset
from src.dummy_model import DummySegNet, load_checkpoint, save_dummy_checkpoint
from src.expansion import (
    assert_no_leakage,
    largest_remainder,
    plan_expansion,
)
from src.metrics import dice_regions, dice_score, logits_to_classes, mean_dice, region_mask
from src.mixup import MixupBuffer, build_mixup, mixup_3d, one_hot_seg, sample_lambda
from src.runner import CSV_COLUMNS, append_result_row, run
from src.strata import classify, load_tumor_type_csv, seg_features

SHAPE = (16, 16, 16)  # small volumes keep the suite fast; shape logic is identical
CHANNELS = 4
CLASSES = 5


# --------------------------------------------------------------------- fixtures
@pytest.fixture
def cfg(tmp_path):
    c = load_config()
    c.data["spatial_size"] = list(SHAPE)
    c.paths["results_csv"] = str(tmp_path / "results.csv")
    c.paths["plans_dir"] = str(tmp_path / "plans")
    c.root = tmp_path
    return c


@pytest.fixture
def fake_cache(tmp_path):
    """A tiny on-disk cache in the real .npz format, for Dataset tests."""
    cache = tmp_path / "cache"
    cache.mkdir()
    rng = np.random.default_rng(0)
    ids = [f"SUBJ-{i:03d}" for i in range(6)]
    for i, sid in enumerate(ids):
        arrays = {m: (rng.random(SHAPE) * 1000).astype(np.float16)
                  for m in ("t1c", "t1n", "t2f", "t2w")}
        seg = np.zeros(SHAPE, dtype=np.uint8)
        seg[2:8, 2:8, 2:8] = (i % 4) + 1
        seg[3:5, 3:5, 3:5] = 1
        arrays["seg"] = seg
        np.savez(cache / f"{sid}.npz", **arrays)
    return cache, ids


def _strata_rows(hospital="A", n_a=38, n_b=15):
    rows = [{"subject_id": f"{hospital}-astro-{i:03d}", "hospital": hospital,
             "stratum": "astrocytoma_like", "source": "proxy_v1"} for i in range(n_a)]
    rows += [{"subject_id": f"{hospital}-dmg-{i:03d}", "hospital": hospital,
              "stratum": "dmg_like", "source": "proxy_v1"} for i in range(n_b)]
    return rows


# ---------------------------------------------------------------- Req 2: z-score
def test_znorm_zero_mean_unit_std_background_preserved():
    rng = np.random.default_rng(1)
    ch = (rng.random(SHAPE) * 1000).astype(np.float32)
    ch[:4] = 0.0  # background region

    out = znorm_channel(ch)

    fg = out[out != 0]
    assert abs(float(fg.mean())) < 1e-3
    assert abs(float(fg.std()) - 1.0) < 1e-3
    assert np.all(out[:4] == 0.0), "background must stay exactly zero"


def test_znorm_all_background_channel_is_not_nan():
    ch = np.zeros(SHAPE, dtype=np.float32)
    out = znorm_channel(ch)
    assert not np.isnan(out).any()
    assert np.all(out == 0.0)


# ------------------------------------------------------------ Req 3/4: strata
def test_seg_features_on_midline_inferior_mask():
    seg = np.zeros((32, 32, 32), dtype=np.uint8)
    seg[14:18, 10:20, 2:10] = 2  # centred L-R, low on the I-S axis, no ET
    feats = seg_features(seg, lr_axis=0, is_axis=2)
    assert feats["midline_offset"] < 0.15
    assert feats["inferior_frac"] == 1.0
    assert feats["et_frac"] == 0.0


def test_seg_features_empty_mask_is_neutral():
    feats = seg_features(np.zeros((16, 16, 16), dtype=np.uint8))
    assert feats["n_tumor"] == 0
    assert not np.isnan(list(feats.values())).any()


def test_classify_applies_all_three_conditions(cfg):
    dmg = {"midline_offset": 0.05, "inferior_frac": 0.9, "et_frac": 0.01}
    assert classify(dmg, cfg.strata) == "dmg_like"
    # lateral -> not DMG, even with the other two conditions satisfied
    assert classify({**dmg, "midline_offset": 0.5}, cfg.strata) == "astrocytoma_like"
    # superior -> not DMG
    assert classify({**dmg, "inferior_frac": 0.1}, cfg.strata) == "astrocytoma_like"
    # strongly enhancing -> not DMG
    assert classify({**dmg, "et_frac": 0.8}, cfg.strata) == "astrocytoma_like"


def test_tumor_type_csv_override_loads(tmp_path):
    p = tmp_path / "types.csv"
    p.write_text("subject_id,tumor_type\nS-1,dmg_like\nS-2,astrocytoma_like\n", encoding="utf-8")
    mapping = load_tumor_type_csv(p)
    assert mapping == {"S-1": "dmg_like", "S-2": "astrocytoma_like"}


def test_tumor_type_csv_missing_column_raises(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("subject_id,label\nS-1,dmg\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing column"):
        load_tumor_type_csv(p)


# ------------------------------------------------- Req 5/6/7/8: expansion plans
def test_largest_remainder_sums_exactly():
    for counts in ({"a": 38, "b": 15}, {"a": 54, "b": 38}, {"a": 7, "b": 7, "c": 7}):
        alloc = largest_remainder(counts, 150)
        assert sum(alloc.values()) == 150


def test_largest_remainder_handles_empty_stratum():
    alloc = largest_remainder({"a": 10, "b": 0}, 150)
    assert alloc["b"] == 0
    assert sum(alloc.values()) == 150


def test_plan_totals_exactly_150():
    for rows, n in ((_strata_rows("A", 38, 15), 53), (_strata_rows("B", 54, 38), 92)):
        plan = plan_expansion(rows[0]["hospital"], rows, target=150)
        assert len(plan["entries"]) == 150
        assert plan["n_original"] == n


def test_plan_preserves_ratio_within_tolerance():
    tol = 1.0 / 150 + 1e-9
    for rows in (_strata_rows("A", 38, 15), _strata_rows("B", 54, 38)):
        plan = plan_expansion(rows[0]["hospital"], rows, target=150)
        for s in plan["strata"]:
            assert abs(s["expanded_prop"] - s["original_prop"]) <= tol


def test_hospitals_are_planned_independently():
    """B's ratio must not leak into A's plan."""
    plan_a = plan_expansion("A", _strata_rows("A", 38, 15), target=150)
    combined = _strata_rows("A", 38, 15) + _strata_rows("B", 54, 38)
    plan_a_again = plan_expansion("A", combined, target=150)
    props = lambda p: {s["stratum"]: s["n_expanded"] for s in p["strata"]}
    assert props(plan_a) == props(plan_a_again)


def test_every_original_appears_unaugmented():
    rows = _strata_rows("A", 38, 15)
    plan = plan_expansion("A", rows, target=150)
    originals = {e["source_subject_id"] for e in plan["entries"] if not e["is_augmented"]}
    assert originals == {r["subject_id"] for r in rows}
    assert len(originals) == 53


def test_provenance_present_on_every_entry():
    plan = plan_expansion("A", _strata_rows("A", 38, 15), target=150)
    for e in plan["entries"]:
        assert set(e) >= {"hospital", "source_subject_id", "stratum", "is_augmented", "aug_seed"}
        assert e["hospital"] == "A"
        if e["is_augmented"]:
            assert isinstance(e["aug_seed"], int)


def test_augmented_copies_stay_within_their_stratum():
    rows = _strata_rows("A", 38, 15)
    lookup = {r["subject_id"]: r["stratum"] for r in rows}
    plan = plan_expansion("A", rows, target=150)
    for e in plan["entries"]:
        assert lookup[e["source_subject_id"]] == e["stratum"]


def test_no_leakage_between_hospitals_and_heldout(tmp_path):
    md = tmp_path / "manifests"
    md.mkdir()
    import json
    json.dump(["A-astro-000"], open(md / "hospitalA.json", "w"))
    json.dump(["B-astro-000"], open(md / "hospitalB.json", "w"))
    json.dump(["HELD-000"], open(md / "heldout.json", "w"))

    plans = {
        "A": plan_expansion("A", _strata_rows("A", 38, 15), target=150),
        "B": plan_expansion("B", _strata_rows("B", 54, 38), target=150),
    }
    assert_no_leakage(plans, md)  # must not raise

    plans["A"]["entries"].append({
        "hospital": "A", "source_subject_id": "HELD-000", "stratum": "dmg_like",
        "is_augmented": True, "aug_seed": 1, "sample_id": "x",
    })
    with pytest.raises(AssertionError, match="LEAKAGE"):
        assert_no_leakage(plans, md)


# ------------------------------------------------- Req 10/11: augmentation flag
def _sample():
    rng = np.random.default_rng(3)
    image = rng.standard_normal((CHANNELS, *SHAPE)).astype(np.float32)
    label = rng.integers(0, CLASSES, size=(1, *SHAPE)).astype(np.int64)
    return image, label


def test_augmentation_off_returns_none_and_is_deterministic(cfg):
    cfg.use_augmentation = False
    tfm = build_transforms(cfg, SHAPE)
    assert tfm is None

    image, label = _sample()
    i1, l1 = apply_transforms(tfm, image, label)
    i2, l2 = apply_transforms(tfm, image, label)
    assert torch.equal(i1, i2) and torch.equal(l1, l2)


def test_augmentation_on_changes_data_but_keeps_shapes_and_labels(cfg):
    cfg.use_augmentation = True
    tfm = build_transforms(cfg, SHAPE)
    image, label = _sample()

    i1, l1 = apply_transforms(tfm, image, label, seed=1)
    i2, l2 = apply_transforms(tfm, image, label, seed=99)

    assert i1.shape == (CHANNELS, *SHAPE)
    assert l1.shape == (1, *SHAPE)
    assert not torch.equal(i1, i2), "augmentation should vary across seeds"
    for lab in (l1, l2):
        assert set(torch.unique(lab).tolist()) <= {0, 1, 2, 3, 4}, "no fractional labels"


def test_augmentation_is_reproducible_for_a_fixed_seed(cfg):
    cfg.use_augmentation = True
    tfm = build_transforms(cfg, SHAPE)
    image, label = _sample()
    i1, l1 = apply_transforms(tfm, image, label, seed=42)
    i2, l2 = apply_transforms(tfm, image, label, seed=42)
    assert torch.equal(i1, i2) and torch.equal(l1, l2)


# ------------------------------------------------------ Req 12/13/14: mixup
def test_one_hot_seg_shapes_and_sum():
    label = torch.randint(0, CLASSES, (2, 1, *SHAPE))
    onehot = one_hot_seg(label, CLASSES)
    assert onehot.shape == (2, CLASSES, *SHAPE)
    assert torch.allclose(onehot.sum(dim=1), torch.ones(2, *SHAPE))


def test_mixup_3d_preserves_shapes_and_onehot_sum():
    images = torch.randn(2, CHANNELS, *SHAPE)
    onehot = one_hot_seg(torch.randint(0, CLASSES, (2, 1, *SHAPE)), CLASSES)

    mi, ml, lam = mixup_3d(images, onehot, alpha=0.4, rng=np.random.default_rng(0))

    assert mi.shape == images.shape
    assert ml.shape == onehot.shape
    assert 0.0 <= lam <= 1.0
    assert torch.allclose(ml.sum(dim=1), torch.ones(2, *SHAPE), atol=1e-4)


def test_mixup_rejects_non_positive_alpha():
    images = torch.randn(2, CHANNELS, *SHAPE)
    onehot = one_hot_seg(torch.randint(0, CLASSES, (2, 1, *SHAPE)), CLASSES)
    with pytest.raises(ValueError, match="alpha must be > 0"):
        mixup_3d(images, onehot, alpha=0.0)
    with pytest.raises(ValueError, match="alpha must be > 0"):
        sample_lambda(-1.0)


def test_mixup_buffer_works_at_batch_size_one():
    buf = MixupBuffer(alpha=0.4, enabled=True, seed=7)
    first_lams = []
    for step in range(10):
        images = torch.randn(1, CHANNELS, *SHAPE)
        onehot = one_hot_seg(torch.randint(0, CLASSES, (1, 1, *SHAPE)), CLASSES)
        mi, ml, lam = buf.step(images, onehot)

        assert mi.shape == (1, CHANNELS, *SHAPE)
        assert ml.shape == (1, CLASSES, *SHAPE)
        assert torch.allclose(ml.sum(dim=1), torch.ones(1, *SHAPE), atol=1e-4)
        first_lams.append(lam)

    assert first_lams[0] == 1.0, "first step has an empty buffer -> pass through"
    assert any(l != 1.0 for l in first_lams[1:]), "later steps should actually mix"


def test_mixup_disabled_by_master_flag(cfg):
    cfg.use_augmentation = False
    cfg.use_mixup = True                      # sub-flag on, master flag off
    assert cfg.mixup_enabled is False

    buf = build_mixup(cfg)
    images = torch.randn(1, CHANNELS, *SHAPE)
    onehot = one_hot_seg(torch.randint(0, CLASSES, (1, 1, *SHAPE)), CLASSES)

    buf.step(images, onehot)                  # prime the buffer
    mi, ml, lam = buf.step(images, onehot)
    assert lam == 1.0
    assert torch.equal(mi, images), "mixup must be off when use_augmentation is false"
    assert torch.equal(ml, onehot)


def test_mixup_enabled_when_both_flags_on(cfg):
    cfg.use_augmentation = True
    cfg.use_mixup = True
    assert cfg.mixup_enabled is True
    assert build_mixup(cfg).enabled is True


# ------------------------------------------------------- Req 15/16: metrics
def test_region_composition():
    seg = np.array([0, 1, 2, 3, 4])
    assert region_mask(seg, "ET").tolist() == [False, True, False, False, False]
    assert region_mask(seg, "NC").tolist() == [False, True, True, True, False]
    assert region_mask(seg, "WT").tolist() == [False, True, True, True, True]


def test_dice_identical_masks_is_one():
    rng = np.random.default_rng(5)
    mask = rng.integers(0, CLASSES, size=SHAPE)
    scores = dice_regions(mask, mask)
    assert all(abs(v - 1.0) < 1e-9 for v in scores.values())


def test_dice_disjoint_masks_is_zero():
    pred = np.zeros(SHAPE, dtype=np.int64)
    true = np.zeros(SHAPE, dtype=np.int64)
    pred[:4] = 1
    true[8:12] = 1
    assert dice_regions(pred, true)["dice_ET"] == 0.0


def test_dice_empty_in_both_is_one():
    empty = np.zeros(SHAPE, dtype=np.int64)
    assert dice_score(region_mask(empty, "ET"), region_mask(empty, "ET")) == 1.0
    assert dice_regions(empty, empty)["dice_ET"] == 1.0


def test_dice_empty_in_only_one_is_zero():
    empty = np.zeros(SHAPE, dtype=np.int64)
    present = np.zeros(SHAPE, dtype=np.int64)
    present[:4] = 1
    assert dice_regions(present, empty)["dice_ET"] == 0.0
    assert dice_regions(empty, present)["dice_ET"] == 0.0


def test_mean_dice_is_the_average_of_three_regions():
    scores = {"dice_ET": 0.2, "dice_NC": 0.5, "dice_WT": 0.8}
    assert abs(mean_dice(scores) - 0.5) < 1e-12


def test_dice_shape_mismatch_raises():
    with pytest.raises(ValueError, match="does not match"):
        dice_regions(np.zeros((4, 4, 4)), np.zeros((8, 8, 8)))


def test_logits_to_classes():
    logits = torch.randn(2, CLASSES, *SHAPE)
    classes = logits_to_classes(logits)
    assert classes.shape == (2, *SHAPE)
    assert classes.max() < CLASSES


# ------------------------------------------------------- Req 9: no eval leakage
def test_eval_dataset_never_augments(cfg, fake_cache):
    cache, ids = fake_cache
    cfg.use_augmentation = True                    # deliberately on
    ds = EvalDataset(ids, cfg, cache_dir=cache)
    a, b = ds[0], ds[0]
    assert torch.equal(a["image"], b["image"])
    assert torch.equal(a["label"], b["label"])


def test_expansion_dataset_carries_hospital_provenance(cfg, fake_cache):
    cache, ids = fake_cache
    cfg.use_augmentation = False
    rows = [{"subject_id": s, "hospital": "A", "stratum": "dmg_like", "source": "proxy_v1"}
            for s in ids]
    plan = plan_expansion("A", rows, target=10)
    ds = ExpansionDataset(plan, cfg, cache_dir=cache)

    assert len(ds) == 10
    for i in range(len(ds)):
        item = ds[i]
        assert item["hospital"] == "A"
        assert item["source_subject_id"] in ids
        assert item["image"].shape == (CHANNELS, *SHAPE)
        assert item["label"].shape == (1, *SHAPE)


def test_expansion_dataset_deterministic_when_augmentation_off(cfg, fake_cache):
    cache, ids = fake_cache
    cfg.use_augmentation = False
    rows = [{"subject_id": s, "hospital": "A", "stratum": "dmg_like", "source": "proxy_v1"}
            for s in ids]
    ds = ExpansionDataset(plan_expansion("A", rows, target=8), cfg, cache_dir=cache)
    assert torch.equal(ds[0]["image"], ds[0]["image"])


def test_expansion_dataset_augmented_entry_is_reproducible(cfg, fake_cache):
    cache, ids = fake_cache
    cfg.use_augmentation = True
    rows = [{"subject_id": s, "hospital": "A", "stratum": "dmg_like", "source": "proxy_v1"}
            for s in ids]
    ds = ExpansionDataset(plan_expansion("A", rows, target=12), cfg, cache_dir=cache)
    aug_idx = next(i for i, e in enumerate(ds.entries) if e["is_augmented"])
    assert torch.equal(ds[aug_idx]["image"], ds[aug_idx]["image"])


# ---------------------------------------------------- data-layer edge cases
def test_missing_subject_raises_with_path(cfg, fake_cache):
    from src.data import load_subject
    cache, _ = fake_cache
    with pytest.raises(FileNotFoundError, match="NOPE-999"):
        load_subject(cache, "NOPE-999")


def test_missing_modality_key_raises_named(cfg, tmp_path):
    from src.data import load_subject
    cache = tmp_path / "broken"
    cache.mkdir()
    np.savez(cache / "S.npz", t1c=np.zeros(SHAPE, np.float16), seg=np.zeros(SHAPE, np.uint8))
    with pytest.raises(KeyError, match="t1n"):
        load_subject(cache, "S")


def test_out_of_range_label_raises_named(cfg, tmp_path):
    from src.data import load_subject
    cache = tmp_path / "badlabel"
    cache.mkdir()
    seg = np.zeros(SHAPE, np.uint8)
    seg[0, 0, 0] = 7
    np.savez(cache / "S.npz", seg=seg,
             **{m: np.ones(SHAPE, np.float16) for m in ("t1c", "t1n", "t2f", "t2w")})
    with pytest.raises(ValueError, match=r"\[7\]"):
        load_subject(cache, "S")


# --------------------------------------------------- Req 18/19/20: CSV writing
def test_csv_header_is_exactly_the_contract_schema(tmp_path):
    path = tmp_path / "r.csv"
    row = {c: 0.5 for c in CSV_COLUMNS}
    row["experiment_name"] = "x"
    append_result_row(row, path)
    with open(path, newline="", encoding="utf-8") as fh:
        header = next(csv.reader(fh))
    assert header == [
        "experiment_name", "use_augmentation", "use_federation",
        "use_domain_adaptation", "dice_ET", "dice_NC", "dice_WT", "mean_dice",
    ]
    assert path.read_text(encoding="utf-8").splitlines()[0] == \
        "experiment_name,use_augmentation,use_federation,use_domain_adaptation,dice_ET,dice_NC,dice_WT,mean_dice"


def test_two_runs_produce_two_rows_and_one_header(tmp_path):
    path = tmp_path / "r.csv"
    row = {c: 0.5 for c in CSV_COLUMNS}
    row["experiment_name"] = "x"
    append_result_row(row, path)
    append_result_row({**row, "experiment_name": "y"}, path)

    lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 3, "one header + two data rows"
    assert lines[0].startswith("experiment_name")
    assert sum(l.startswith("experiment_name") for l in lines) == 1


def test_mismatched_existing_header_raises(tmp_path):
    path = tmp_path / "r.csv"
    path.write_text("a,b,c\n1,2,3\n", encoding="utf-8")
    row = {c: 0.5 for c in CSV_COLUMNS}
    row["experiment_name"] = "x"
    with pytest.raises(ValueError, match="unexpected header"):
        append_result_row(row, path)


def test_missing_results_directory_is_created(tmp_path):
    path = tmp_path / "deep" / "nested" / "r.csv"
    row = {c: 0.5 for c in CSV_COLUMNS}
    row["experiment_name"] = "x"
    append_result_row(row, path)
    assert path.exists()


# ------------------------------------------- Req 21/22: runner without a model
def test_runner_end_to_end_with_dummy_checkpoint_and_dummy_data(cfg, tmp_path):
    out = tmp_path / "ablation.csv"
    row = run(cfg, experiment_name="dummy_baseline", dummy_checkpoint=True,
              dummy_data=True, dummy_n=3, csv_path=out)

    assert out.exists()
    assert abs(row["mean_dice"] - (row["dice_ET"] + row["dice_NC"] + row["dice_WT"]) / 3) < 1e-6
    with open(out, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1
    assert rows[0]["experiment_name"] == "dummy_baseline"


def test_runner_appends_exactly_one_row_per_invocation(cfg, tmp_path):
    out = tmp_path / "ablation.csv"
    run(cfg, "run_a", dummy_checkpoint=True, dummy_data=True, dummy_n=2, csv_path=out)
    run(cfg, "run_b", dummy_checkpoint=True, dummy_data=True, dummy_n=2, csv_path=out)
    with open(out, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert [r["experiment_name"] for r in rows] == ["run_a", "run_b"]


def test_runner_records_the_ablation_flags(cfg, tmp_path):
    out = tmp_path / "ablation.csv"
    cfg.use_augmentation = True
    cfg.use_federation = False
    cfg.use_domain_adaptation = True
    run(cfg, "flagged", dummy_checkpoint=True, dummy_data=True, dummy_n=2, csv_path=out)
    with open(out, newline="", encoding="utf-8") as fh:
        row = next(csv.DictReader(fh))
    assert row["use_augmentation"] == "True"
    assert row["use_federation"] == "False"
    assert row["use_domain_adaptation"] == "True"


def test_runner_requires_a_checkpoint(cfg, tmp_path):
    with pytest.raises(ValueError, match="--dummy-checkpoint"):
        run(cfg, "no_ckpt", dummy_data=True, csv_path=tmp_path / "r.csv")


def test_missing_checkpoint_raises_with_hint(cfg, tmp_path):
    model = DummySegNet()
    with pytest.raises(FileNotFoundError, match="dummy-checkpoint"):
        load_checkpoint(tmp_path / "nope.pt", model)


def test_mismatched_checkpoint_raises(tmp_path):
    ckpt = save_dummy_checkpoint(tmp_path / "c.pt", in_channels=4, num_classes=5)
    wrong = DummySegNet(in_channels=2, num_classes=3)
    with pytest.raises(RuntimeError, match="does not match"):
        load_checkpoint(ckpt, wrong)


def test_saved_dummy_checkpoint_round_trips(tmp_path):
    ckpt = save_dummy_checkpoint(tmp_path / "c.pt")
    model = load_checkpoint(ckpt, DummySegNet())
    logits, feats = model(torch.randn(1, CHANNELS, *SHAPE))
    assert logits.shape == (1, CLASSES, *SHAPE)
    assert feats.dim() == 2, "model must return (seg_logits, features) per CONTRACTS.md"


def test_dummy_dataset_reads_nothing_from_disk(cfg):
    ds = DummyDataset(n=3, cfg=cfg)
    item = ds[0]
    assert item["image"].shape == (CHANNELS, *SHAPE)
    assert item["label"].shape == (1, *SHAPE)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
