"""Pack the slice cache down to what actually gets used, for a 15GB Drive.

The full cache does not fit a free Google Drive: both planes at float32 project
to ~35GB, and float16 still leaves ~17GB. But almost none of it is needed.

Two observations do the work:

* **Training only ever reads the slices the plan selects** -- 8000 per hospital,
  not the ~57,000 available. Everything else sits on disk unused.
* **Evaluation only needs the held-out subjects**, and only the plane(s)
  ``eval.plane`` names.

So: build the index and the plan **locally**, where the whole dataset lives, then
copy across only the slices those reference. The plan and index travel with the
data, so the remote side never has to rebuild them from slices it does not have.

float16 halves it again at no cost -- the cache is already z-scored to roughly
+/-3, nowhere near float16's range or precision limits.

Result on the real cohort: ~35GB -> ~5GB, which fits with room for checkpoints.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import numpy as np

from .config import Config
from .plan import build_plans, write_plan
from .slices import (
    INDEX_FILENAME,
    list_slice_indices,
    load_slice,
    load_manifest,
    read_index,
    slice_path,
)

LOGGER = logging.getLogger(__name__)


def _copy_slice(src_cache, dst_cache, subject_id, plane, index, half: bool) -> int:
    """Copy one slice, optionally down-casting the image. Returns bytes written."""
    data = load_slice(src_cache, subject_id, plane, index)
    image = data["image"]
    if half:
        image = image.astype(np.float16)

    out = slice_path(dst_cache, subject_id, plane, index)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, image=image, mask=data["mask"])
    return out.stat().st_size


def pack(
    cfg: Config,
    out_dir: Path,
    src_cache: Path | None = None,
    half: bool = True,
    include_heldout: bool = True,
    heldout_planes=None,
) -> dict:
    """Copy the planned training slices plus the held-out set into ``out_dir``."""
    src_cache = Path(src_cache) if src_cache else cfg.resolve("cache_2d")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Plan against the full local cache -- this is why packing happens here and
    # not on the machine that receives the upload.
    plans = build_plans(cfg, src_cache)
    wanted: dict[tuple[str, str], set[int]] = {}
    for plan in plans.values():
        for e in plan["entries"]:
            wanted.setdefault((e["source_subject_id"], e["plane"]), set()).add(e["slice_index"])

    n_train = sum(len(v) for v in wanted.values())

    heldout: list[str] = []
    if include_heldout:
        planes = list(heldout_planes) if heldout_planes else (
            list(cfg.planes) if cfg.eval_plane == "both" else [cfg.eval_plane]
        )
        heldout = load_manifest(cfg.resolve("manifests"), "heldout")
        for sid in heldout:
            for plane in planes:
                idx = list_slice_indices(src_cache, sid, plane)
                if idx:
                    wanted.setdefault((sid, plane), set()).update(idx)

    total_bytes = 0
    copied = 0
    for n, ((sid, plane), indices) in enumerate(sorted(wanted.items()), 1):
        for i in sorted(indices):
            try:
                total_bytes += _copy_slice(src_cache, out_dir, sid, plane, i, half)
                copied += 1
            except FileNotFoundError:
                LOGGER.warning("missing slice %s/%s/%d; skipped", sid, plane, i)
        if n % 25 == 0:
            LOGGER.info("packed %d/%d subject-planes (%.2f GB)",
                        n, len(wanted), total_bytes / 2**30)

    # Ship the index and the plans so the remote side never rebuilds them from
    # a cache that deliberately holds only a subset.
    src_index = Path(src_cache) / INDEX_FILENAME
    if src_index.exists():
        shutil.copy2(src_index, out_dir / INDEX_FILENAME)

    plans_dir = out_dir / "plans"
    for hospital, plan in sorted(plans.items()):
        write_plan(plan, plans_dir / f"{hospital}_plan.json")

    manifest = {
        "packed_from": str(src_cache),
        "float16": half,
        "slices_copied": copied,
        "training_slices": n_train,
        "heldout_subjects": len(heldout),
        "subject_planes": len(wanted),
        "bytes": total_bytes,
        "gigabytes": round(total_bytes / 2**30, 2),
    }
    with open(out_dir / "pack_manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest
