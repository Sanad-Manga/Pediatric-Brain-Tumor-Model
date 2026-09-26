"""Req 21: the real checkpoints reproduce the 2026-09-26 per-patient scores. Slow; skipped unless both variables are set:

    RSTAR_MODELS_ROOT   directory holding the checkpoints (03_augmentation_eval/..., 01_model_federated/...)
    RSTAR_TEST_DATA     BraTS-PEDs 'Training' directory: <sid>/<sid>-{t1c,t1n,t2f,t2w,seg}.nii.gz
"""
import json
import os
from pathlib import Path

import numpy as np
import pytest

from rstar import RStarConfig, RStarSegmenter
from rstar.config import load_manifest
from rstar.models import verify_sha256

MODELS_ROOT, TEST_DATA = os.environ.get("RSTAR_MODELS_ROOT"), os.environ.get("RSTAR_TEST_DATA")
pytestmark = pytest.mark.skipif(not (MODELS_ROOT and TEST_DATA), reason="set RSTAR_MODELS_ROOT and RSTAR_TEST_DATA to run the regression test")
REFERENCE = json.loads((Path(__file__).parent / "fixtures" / "rstar_reference_scores.json").read_text(encoding="utf-8"))["patients"]


def _dice(a, b):
    sa, sb = int(a.sum()), int(b.sum())
    if sa == 0 and sb == 0:
        return 1.0
    if sa == 0 or sb == 0:
        return 0.0
    return 2.0 * int((a & b).sum()) / (sa + sb)


def test_the_pinned_hashes_match_the_real_files():
    manifest = load_manifest()
    for entry in [manifest["ensemble_2d"], *manifest["family_3d"]]:
        verify_sha256(Path(MODELS_ROOT) / entry["path"], entry["sha256"])


def test_three_fresh_patients_reproduce_the_reference_scores():
    import nibabel as nib

    seg = RStarSegmenter(RStarConfig(models_root=Path(MODELS_ROOT)))               # hash-verified, runs the geometry self-check
    for sid, ref in REFERENCE.items():
        base = Path(TEST_DATA) / sid
        paths = {name: base / f"{sid}-{name}.nii.gz" for name in ("t1c", "t1n", "t2f", "t2w")}
        result, _ = seg.segment_paths(paths)
        truth = np.rint(np.asarray(nib.load(str(base / f"{sid}-seg.nii.gz")).dataobj, dtype=np.float32)).astype(np.int64)
        labels = result.labels.astype(np.int64)
        assert result.mode == "R*"
        assert _dice(np.isin(labels, (1, 2, 3)), np.isin(truth, (1, 2, 3))) == pytest.approx(ref["nc"], abs=0.002)
        assert _dice(labels > 0, truth > 0) == pytest.approx(ref["wt"], abs=0.002)
        assert _dice(labels == 1, truth == 1) == pytest.approx(ref["et"], abs=0.002)
