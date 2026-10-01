"""Best-checkpoint selection for the 3D model (HANDOFF_SEED_AND_CHECKPOINT_SELECTION.md, Fix 2).

Plain Python, no torch import: nothing in here trains or scores anything itself. The
score for an epoch is the WORST per-region Dice (not the mean), so a strong WT score
cannot hide a bad ET score -- the same idea as `selection: metric: min_region` in
03_augmentation_eval.
"""
from __future__ import annotations

import json
import shutil
import warnings
from pathlib import Path
from typing import Callable, Mapping, Optional, Tuple


def worst_region(per_region_dice: Mapping[str, float]) -> Tuple[str, float]:
    """Return (region_name, score) for the lowest-scoring region."""
    if not per_region_dice:
        raise ValueError("per_region_dice is empty")
    name = min(per_region_dice, key=per_region_dice.get)
    return name, float(per_region_dice[name])


def is_new_best(best_score: Optional[float], new_score: float) -> bool:
    """First evaluation always becomes best. Strict '>' so a tie never flip-flops."""
    return best_score is None or new_score > best_score


class BestCheckpointTracker:
    """Tracks the best worst-region score seen so far across evaluated epochs."""

    def __init__(self) -> None:
        self.best_score: Optional[float] = None
        self.best_region: Optional[str] = None
        self.best_epoch: Optional[int] = None
        self.last_message: Optional[str] = None

    def update(self, epoch: int, per_region_dice: Mapping[str, float]) -> bool:
        """Record one evaluation. Returns True if this epoch is the new best."""
        region, score = worst_region(per_region_dice)
        if not is_new_best(self.best_score, score):
            self.last_message = None
            return False

        if self.best_score is None:
            self.last_message = (
                f"epoch {epoch}: new best (worst region {region} {score:.2f}, first evaluation)"
            )
        else:
            self.last_message = (
                f"epoch {epoch}: new best (worst region {region} {score:.2f} "
                f"> previous {self.best_score:.2f})"
            )
        self.best_score = score
        self.best_region = region
        self.best_epoch = epoch
        return True

    # -- persistence: a resumed run (Colab disconnects) must not forget its best ---------
    def save(self, path) -> None:
        path = Path(path)
        payload = {"best_score": self.best_score, "best_region": self.best_region,
                   "best_epoch": self.best_epoch}
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(path)  # atomic: never leave a half-written file

    @classmethod
    def load(cls, path) -> "BestCheckpointTracker":
        """Tracker restored from `path`; a fresh one if the file is missing or unreadable."""
        tracker = cls()
        path = Path(path)
        if not path.is_file():
            return tracker
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            tracker.best_score = float(payload["best_score"])
            tracker.best_region = payload["best_region"]
            tracker.best_epoch = int(payload["best_epoch"])
        except (ValueError, KeyError, TypeError):
            warnings.warn(f"could not read {path}; best-checkpoint tracking starts fresh")
            return cls()
        return tracker


def copy_to_best(checkpoint_path, run_dir) -> Path:
    """Copy the given checkpoint to <run_dir>/best.pt.

    Copies, never moves or deletes, so prune_old_checkpoints keeps managing the
    per-epoch files.
    """
    dest = Path(run_dir) / "best.pt"
    shutil.copyfile(checkpoint_path, dest)
    return dest


def evaluate_and_track(model, scorer: Callable, tracker: BestCheckpointTracker, epoch: int,
                       checkpoint_path, run_dir) -> bool:
    """Score `model` with `scorer(model) -> {region: dice}`; if it is the new best,
    copy `checkpoint_path` to best.pt and persist the tracker. Returns True on a new best.

    `model` only needs .eval() and .train(); it is put back in train mode afterwards.
    """
    run_dir = Path(run_dir)
    model.eval()
    try:
        per_region = scorer(model)
    finally:
        model.train()

    is_best = tracker.update(epoch, per_region)
    print(f"epoch {epoch}: held-out " + "  ".join(f"{r}={v:.4f}" for r, v in per_region.items()),
          flush=True)
    if is_best:
        copy_to_best(checkpoint_path, run_dir)
        tracker.save(run_dir / "best.json")
        print(tracker.last_message, flush=True)
    return is_best