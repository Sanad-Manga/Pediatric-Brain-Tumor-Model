import json
import os

import pytest

REPO_MANIFEST_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "00_shared", "manifests")


@pytest.fixture
def small_manifest(tmp_path):
    def _make(name: str, n_subjects: int):
        subject_ids = [f"SUBJ-{name}-{i:03d}" for i in range(n_subjects)]
        path = tmp_path / f"{name}.json"
        with open(path, "w") as f:
            json.dump(subject_ids, f)
        return str(path)

    return _make


@pytest.fixture
def real_manifest_path():
    def _get(name: str) -> str:
        return os.path.join(REPO_MANIFEST_DIR, f"{name}.json")

    return _get
