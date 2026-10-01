"""Builds the held-out scorer used by `run.py --eval-every N`.

Reuses tools/eval_heldout_3d.py (score_subject / aggregate / dice_regions) instead of
re-implementing Dice. That file is loaded by explicit path, the same way it loads
03_augmentation_eval/src/metrics.py, because `tools` is not a package and a bare
`import tools...` could resolve to something else on sys.path.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from .data import BraTSPedsDataset, load_manifest

SEC01 = Path(__file__).resolve().parent.parent
# --data-mode dummy only checks the plumbing (labels are random noise), so two subjects is enough.
DUMMY_EVAL_SUBJECTS = 2


def _load_eval_tool():
    path = SEC01 / "tools" / "eval_heldout_3d.py"
    spec = importlib.util.spec_from_file_location("eval_heldout_3d_tool", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_heldout_scorer(data_mode: str, cache_path: str | None, manifest_path: str) -> Callable:
    """Returns scorer(model) -> {"ET": dice, "NC": dice, "WT": dice} (mean over subjects).

    data_mode "real": scores <cache_path>/<subject>.npz exactly like eval_heldout_3d.py.
    data_mode "dummy": scores synthetic volumes (no cache exists); the numbers are
    meaningless and only exercise the selection / best.pt plumbing.
    Fails here, at startup, if a real run has nothing to score, rather than hours in.
    """
    tool = _load_eval_tool()
    subjects = load_manifest(manifest_path)

    if data_mode == "real":
        if not cache_path:
            raise ValueError("--eval-every with --data-mode real needs --cache-path")
        cache = Path(cache_path)
        if not any((cache / f"{sid}.npz").is_file() for sid in subjects):
            raise FileNotFoundError(
                f"none of the {len(subjects)} subjects in {manifest_path} found in {cache}")

        def scorer(model) -> dict[str, float]:
            device = str(next(model.parameters()).device)
            scores = [s for s in (tool.score_subject(model, cache, sid, device) for sid in subjects)
                      if s is not None]
            if not scores:
                raise RuntimeError("no held-out subjects scored (cache missing?)")
            agg = tool.aggregate(scores)
            return {r: float(agg[f"dice_{r}"]) for r in tool.REGION_ORDER}

        return scorer

    if data_mode == "dummy":
        dataset = BraTSPedsDataset(manifest_path, mode="dummy")
        n = min(DUMMY_EVAL_SUBJECTS, len(dataset))

        def scorer(model) -> dict[str, float]:
            device = next(model.parameters()).device
            scores = []
            for i in range(n):
                x, y = dataset[i]
                with torch.no_grad():
                    logits, _features = model(x[None].to(device))
                    pred = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.int64)
                scores.append(tool.dice_regions(pred, y.numpy()))
            agg = tool.aggregate(scores)
            return {r: float(agg[f"dice_{r}"]) for r in tool.REGION_ORDER}

        return scorer

    raise ValueError(f"--eval-every supports data_mode 'real' or 'dummy', got {data_mode!r}")