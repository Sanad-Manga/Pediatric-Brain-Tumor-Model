"""Tumour-type proxy (dmg_like / astrocytoma_like) and its auxiliary head.

No real tumour-type labels exist anywhere in this dataset. This is a geometric
proxy ported from the 3D version's `strata.py`, used to train an auxiliary
classification head off the model's bottleneck `features` -- not to hand the
model a label. The model interface `model(x) -> (seg_logits, features)` is
untouched; the head is a separate module bolted onto `features`.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.model import TumorTypeHead, build_model
from src.tumor_type import (
    ASTROCYTOMA_LIKE,
    DMG_LIKE,
    TYPE_LABELS,
    build_type_index,
    classify,
    classify_subject,
    label_index,
    read_type_index,
    seg_features,
    subject_mask_volume,
    write_type_index,
)


# ------------------------------------------------------------ seg_features
def test_empty_mask_is_undefined_not_dmg():
    """No tumour -> falls on the astrocytoma_like side, never invents DMG."""
    feats = seg_features(np.zeros((20, 20, 16), dtype=np.uint8))
    assert feats["n_tumor"] == 0
    assert classify(feats) == ASTROCYTOMA_LIKE


def test_midline_low_enhancing_tumour_classifies_dmg_like():
    """A small, midline, low, barely-enhancing tumour -- textbook DMG proxy."""
    seg = np.zeros((40, 40, 20), dtype=np.uint8)
    cx, cy = 20, 20                       # exact midline
    seg[cx - 2:cx + 2, cy - 2:cy + 2, 0:4] = 4   # low (inferior), edema
    feats = seg_features(seg, lr_axis=0, is_axis=2)
    assert feats["midline_offset"] < 0.15
    assert feats["inferior_frac"] > 0.6
    assert feats["et_frac"] == 0.0
    assert classify(feats) == DMG_LIKE


def test_offcentre_highly_enhancing_tumour_classifies_astrocytoma_like():
    seg = np.zeros((40, 40, 20), dtype=np.uint8)
    seg[30:35, 30:35, 15:19] = 1           # off-midline, high, all enhancing
    feats = seg_features(seg, lr_axis=0, is_axis=2)
    assert feats["midline_offset"] > 0.15
    assert feats["et_frac"] > 0.10
    assert classify(feats) == ASTROCYTOMA_LIKE


def test_label_index_is_stable():
    assert TYPE_LABELS == (ASTROCYTOMA_LIKE, DMG_LIKE)
    assert label_index(ASTROCYTOMA_LIKE) == 0
    assert label_index(DMG_LIKE) == 1


# --------------------------------------------------- volume reconstruction
def test_reconstructed_volume_matches_the_stored_masks(cfg, cache, manifests):
    """Reuses the same restacking already proven exact against real data."""
    from src.slices import list_slice_indices, load_slice

    subject = manifests["hospitalA"][0]
    volume = subject_mask_volume(cache, subject, cfg, plane="axial")
    axis = cfg.plane_axis("axial")

    for i in list_slice_indices(cache, subject, "axial"):
        stored = load_slice(cache, subject, "axial", i)["mask"][0]
        assert np.array_equal(np.moveaxis(volume, axis, 0)[i], stored)


# -------------------------------------------------------------- the index
def test_classify_subject_runs_on_a_real_synthetic_subject(cfg, cache, manifests):
    result = classify_subject(cache, manifests["hospitalA"][0], cfg, plane="axial")
    assert result["stratum"] in TYPE_LABELS
    assert set(result) >= {"subject_id", "stratum", "midline_offset",
                           "inferior_frac", "et_frac", "n_tumor"}


def test_build_type_index_covers_every_subject(cfg, cache, manifests):
    subjects = manifests["hospitalA"] + manifests["hospitalB"]
    index = build_type_index(cache, subjects, cfg, progress_every=0)
    assert set(index["subjects"]) == set(subjects)
    assert set(index["counts"]) == set(TYPE_LABELS)
    assert sum(index["counts"].values()) == len(subjects)


def test_type_index_round_trips_through_json(cfg, cache, manifests, tmp_path):
    index = build_type_index(cache, manifests["hospitalA"], cfg, progress_every=0)
    write_type_index(index, tmp_path)
    assert read_type_index(tmp_path) == index


def test_missing_type_index_names_the_command(tmp_path):
    with pytest.raises(FileNotFoundError, match="run.py tumor-type"):
        read_type_index(tmp_path)


# ------------------------------------------------------ the auxiliary head
def test_head_does_not_change_the_shared_model_interface():
    """CONTRACTS.md fixes model(x) -> (seg_logits, features); this must hold."""
    model = build_model_stub = None
    from src.config import load_config
    cfg = load_config()
    model = build_model(cfg, spatial_dims=2, width=4, depth=2)
    out = model(torch.zeros(2, len(cfg.modalities), 32, 32))
    assert len(out) == 2
    seg_logits, features = out
    assert seg_logits.ndim == 4
    assert features.ndim == 2


def test_head_reads_features_and_outputs_two_classes():
    head = TumorTypeHead(in_features=16, num_types=2)
    out = head(torch.randn(5, 16))
    assert out.shape == (5, 2)


def test_head_is_trainable_end_to_end():
    """A few steps should reduce loss on a trivially separable toy problem."""
    torch.manual_seed(0)
    head = TumorTypeHead(in_features=8, num_types=2, hidden=8)
    opt = torch.optim.Adam(head.parameters(), lr=0.05)
    loss_fn = torch.nn.CrossEntropyLoss()

    x = torch.cat([torch.zeros(20, 8), torch.ones(20, 8)])
    y = torch.cat([torch.zeros(20, dtype=torch.long), torch.ones(20, dtype=torch.long)])

    first = None
    for _ in range(30):
        opt.zero_grad()
        loss = loss_fn(head(x), y)
        if first is None:
            first = float(loss)
        loss.backward()
        opt.step()
    assert float(loss) < first
