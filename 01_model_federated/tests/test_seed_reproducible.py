"""Follow-up to #39/#45: `--seed` must make a REAL-style run reproducible, i.e. with augmentation on and across
separate processes (Python randomises str hashes per process). These run run.py in subprocesses."""
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from src.augment3d import Augment3D
from src.data import BraTSPedsDataset

SEC01 = Path(__file__).resolve().parents[1]


def _losses(tmp_path, tag, hashseed, extra=()):
    env = {**os.environ, "PYTHONHASHSEED": str(hashseed), "CUDA_VISIBLE_DEVICES": "-1"}
    cmd = [sys.executable, "run.py", "--data-mode", "dummy", "--epochs", "1", "--run-id", tag,
           "--checkpoint-dir", str(tmp_path / tag), "--manifest", str(tmp_path / "m.json"), *extra]
    out = subprocess.run(cmd, cwd=SEC01, env=env, capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-2000:]
    return re.search(r"Losses: \[(.*)\]", out.stdout).group(1)


def test_seeded_runs_with_augmentation_are_identical_across_processes(tmp_path):
    (tmp_path / "m.json").write_text('["SUBJ-a", "SUBJ-b"]')
    a = _losses(tmp_path, "a", 1, ["--seed", "1337", "--use-augmentation"])
    b = _losses(tmp_path, "b", 2, ["--seed", "1337", "--use-augmentation"])
    assert a == b


def test_different_seeds_differ(tmp_path):
    (tmp_path / "m.json").write_text('["SUBJ-a", "SUBJ-b"]')
    a = _losses(tmp_path, "a", 1, ["--seed", "1", "--use-augmentation"])
    b = _losses(tmp_path, "b", 1, ["--seed", "2", "--use-augmentation"])
    assert a != b


def test_augment3d_random_state_can_be_seeded():
    torch.manual_seed(0)
    x = torch.randn(1, 4, 32, 32, 32)
    y = torch.randint(0, 5, (1, 32, 32, 32))
    outs = []
    for seed in (7, 7, 8):
        aug = Augment3D()
        aug.set_random_state(seed)
        outs.append(aug(x, y))
    assert torch.equal(outs[0][0], outs[1][0]) and torch.equal(outs[0][1], outs[1][1])
    assert not torch.equal(outs[0][0], outs[2][0])


def test_dummy_data_does_not_depend_on_the_process_hash_seed(tmp_path):
    (tmp_path / "m.json").write_text('["SUBJ-a"]')
    code = ("import sys; sys.path.insert(0, '.'); from src.data import BraTSPedsDataset; "
            f"x, y = BraTSPedsDataset(r'{tmp_path / 'm.json'}')[0]; print(float(x.sum()), int(y.sum()))")
    outs = {subprocess.run([sys.executable, "-c", code], cwd=SEC01, capture_output=True, text=True,
                           env={**os.environ, "PYTHONHASHSEED": str(h)}).stdout for h in (1, 2, 3)}
    assert len(outs) == 1 and outs != {""}
