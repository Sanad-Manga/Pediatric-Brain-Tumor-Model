"""Checks for SPEC.md Req 1-4 (on the real local cache) and Req 7. Run with the neuropeds env:
    python -m pytest 07_review_prototype/test_prototype.py
Cache-dependent tests skip when the cache or the saved probability maps are not on this machine."""
import ast
import json
import os
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).parent
CACHE = Path(os.environ.get("NEUROPEDS_PROTOTYPE_CACHE", r"D:\NeuroPeds AI\review_prototype"))
needs_cache = pytest.mark.skipif(not (CACHE / "patients.json").is_file(), reason="prototype cache not on this machine")
SHAPE = (240, 240, 155)


def _patients():
    return json.loads((CACHE / "patients.json").read_text()) if (CACHE / "patients.json").is_file() else []


@needs_cache
@pytest.mark.parametrize("sid", _patients())
def test_files_masks_and_meta(sid):
    import sys
    from scipy import ndimage
    d = CACHE / sid
    arrays = {n: np.load(d / f"{n}.npz")[k] for n, k in (("t1c", "img"), ("t2f", "img"), ("gt", "labels"),
              ("today", "labels"), ("flagged", "labels"), ("review_mask", "mask"), ("specks", "mask"))}
    for n, a in arrays.items():                                                         # Req 1
        assert a.shape == SHAPE and a.dtype == np.uint8, n
    review, specks, flagged, today, gt = (arrays[k] for k in ("review_mask", "specks", "flagged", "today", "gt"))
    rs = json.loads((d / "review_spots.json").read_text()); sp = json.loads((d / "specks.json").read_text())
    meta = json.loads((d / "meta.json").read_text())
    # Req 2 (structure): today differs from flagged only where ET was relabelled to 2
    diff = today != flagged
    assert np.all(flagged[diff] == 1) and np.all(today[diff] == 2)
    # Req 3: partition of the ET voxels of `flagged` into kept / review / speck spots; thresholds; ids
    sys.path.insert(0, str(HERE.parent / "06_rstar_inference"))
    lab, n = ndimage.label(flagged == 1, structure=np.ones((3, 3, 3)))
    assert set(np.unique(review)) == {0, *range(1, len(rs) + 1)} and set(np.unique(specks)) == {0, *range(1, len(sp) + 1)}
    assert not np.any((review > 0) & (specks > 0)) and np.all(flagged[(review > 0) | (specks > 0)] == 1)
    for e in rs:
        assert e["voxels"] >= 50 and e["mean_et_prob"] < 0.7 and int((review == e["spot_id"]).sum()) == e["voxels"]
    for e in sp:
        assert e["voxels"] < 50 and int((specks == e["spot_id"]).sum()) == e["voxels"]
    sizes = np.bincount(lab.ravel())[1:]
    n_big = int((sizes >= 50).sum())
    assert meta["n_kept"] + meta["n_review"] == n_big and meta["n_specks"] == int((sizes < 50).sum())
    # Req 4: meta counts, real flags, slice ranges
    assert meta["n_review"] == len(rs) and meta["n_specks"] == len(sp)
    assert meta["today_et_voxels"] == int((today == 1).sum()) and meta["flagged_et_voxels"] == int((flagged == 1).sum())
    for mask, items in ((review, rs), (specks, sp)):
        for e in items:
            m = mask == e["spot_id"]
            assert e["real"] == bool((m & (gt == 1)).any())
            zz = np.where(m.any(axis=(0, 1)))[0]
            assert e["slices"] == [int(zz.min()), int(zz.max())]


PROBS = Path(r"D:\NeuroPeds AI\probs2d_heldout")


@needs_cache
@pytest.mark.skipif(not PROBS.is_dir(), reason="saved probability maps not on this machine")
def test_today_and_flagged_equal_the_deployed_fusion_and_rule():                       # Req 2 (exact)
    import sys
    sys.path.insert(0, str(HERE.parent / "06_rstar_inference"))
    sys.path.insert(0, str(HERE))
    from rstar import fusion
    import precompute as pc
    for sid in _patients()[:3]:
        p2 = pc._probs(pc.PROBS_2D / f"{sid}.npy"); p3 = pc._probs(pc.FAMILY_3D / f"{sid}.npy")
        fused = fusion.fuse(p2, p3, 0.5, 0.5)
        ruled, _, _ = fusion.apply_small_et_rule(fused, 500.0, 1.0)
        assert np.array_equal(np.load(CACHE / sid / "flagged.npz")["labels"], fused.astype(np.uint8))
        assert np.array_equal(np.load(CACHE / sid / "today.npz")["labels"], ruled.astype(np.uint8))


def test_app_imports_neither_torch_nor_rstar():                                        # Req 7
    tree = ast.parse((HERE / "app.py").read_text(encoding="utf-8"))
    names = {a.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for a in node.names}
    names |= {node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
    assert not names & {"torch", "rstar"}
    assert "Research prototype - not for clinical use" in (HERE / "app.py").read_text(encoding="utf-8")


# ---- Req 5-6: drive the real app script headlessly (streamlit.testing) -------------------------------------
@needs_cache
def test_every_patient_renders_without_errors():
    from streamlit.testing.v1 import AppTest
    for sid in _patients():
        at = AppTest.from_file(str(HERE / "app.py"), default_timeout=120)
        at.run()
        at.selectbox(key="patient").set_value(sid).run()
        assert not at.exception, (sid, at.exception)
        assert any("Research prototype - not for clinical use" in w.value for w in at.warning)


@needs_cache
def test_toggles_and_jump_to_spot():
    from streamlit.testing.v1 import AppTest
    sid = "BraTS-PED-00004-000"                       # has 5 review spots and 12 specks
    at = AppTest.from_file(str(HERE / "app.py"), default_timeout=120)
    at.run()
    at.selectbox(key="patient").set_value(sid).run()
    rows_before = sum("review spot" in m.value for m in at.markdown)
    at.checkbox(key="specks").check().run()
    rows_specks = sum("spot" in m.value for m in at.markdown if m.value.startswith("**"))
    assert rows_specks > rows_before                   # specks appear in the table
    at.checkbox(key="reveal").check().run()
    assert any(m.value in ("**real**", "**false**") for m in at.markdown)
    at.radio(key="seq").set_value("FLAIR").run()
    assert not at.exception
    spots = json.loads((CACHE / sid / "review_spots.json").read_text())
    lo, hi = spots[1]["slices"]
    at.button(key=f"go-review-{spots[1]['spot_id']}").click().run()
    assert at.slider(key="z").value == (lo + hi) // 2
    assert not at.exception
