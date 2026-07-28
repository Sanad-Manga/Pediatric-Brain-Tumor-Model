import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "visualize_features.py"
SPEC = importlib.util.spec_from_file_location("visualize_features", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_pca_and_lda_return_two_dimensions():
    rng = np.random.default_rng(42)
    features = np.concatenate(
        [
            rng.normal(-1, 1, size=(8, 10)),
            rng.normal(0, 1, size=(8, 10)),
            rng.normal(1, 1, size=(8, 10)),
        ]
    )
    labels = np.repeat(["Hospital A", "Hospital B", "Held-out"], 8)
    pca, lda = MODULE.project(features, labels)
    assert pca.shape == (24, 2)
    assert lda.shape == (24, 2)
    assert np.isfinite(pca).all()
    assert np.isfinite(lda).all()
