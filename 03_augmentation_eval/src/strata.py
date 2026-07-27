"""Tumor-type stratification for ratio-preserving expansion.

    !! IMPORTANT !!
    No tumor-type ground truth exists anywhere in this dataset as delivered.
    The manifests are flat lists of subject IDs and the cached .npz files hold
    only the four modalities plus `seg`. Nothing states which subjects are
    high-grade astrocytoma and which are diffuse midline glioma / DIPG.

    This module therefore derives an IMAGING PROXY from the segmentation mask
    geometry. It is a stand-in that lets the ratio-preserving expansion stratify
    on *something* clinically motivated. It is NOT histology and must never be
    reported as a tumor-type classification result.

Proxy features, all computed from `seg` alone:

    midline_offset : |centroid along the L-R axis - image centre| / centre.
                     DMG is by definition midline, so this is small for DMG.
    inferior_frac  : fraction of tumor voxels below the axial midpoint.
                     DIPG sits in the pons/brainstem, i.e. inferior.
    et_frac        : label-1 voxels / all tumor voxels. DMG/DIPG frequently
                     enhances little or not at all.

A subject is `dmg_like` when all three conditions hold; otherwise
`astrocytoma_like`. Thresholds and the axis convention live in config.yaml.

Supplying a real label file via `--tumor-type-csv` overrides the proxy for every
subject it names; the `source` column records which path each row took.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

import numpy as np

from .config import Config, load_config
from .data import load_all_manifests, subject_path

LOGGER = logging.getLogger(__name__)

PROXY_SOURCE = "proxy_v1"
EXTERNAL_SOURCE = "external"

STRATA_COLUMNS = [
    "subject_id",
    "hospital",
    "stratum",
    "source",
    "midline_offset",
    "inferior_frac",
    "et_frac",
]

PROXY_WARNING = (
    "Tumor-type strata are an IMAGING PROXY derived from segmentation geometry "
    "(source=proxy_v1), not histology. Supply --tumor-type-csv to override with "
    "real labels."
)


def seg_features(seg: np.ndarray, lr_axis: int = 0, is_axis: int = 2) -> dict[str, float]:
    """Compute the three proxy features from a ``(D,H,W)`` class mask."""
    seg = np.asarray(seg)
    if seg.ndim == 4 and seg.shape[0] == 1:
        seg = seg[0]

    tumor = seg > 0
    n_tumor = int(tumor.sum())
    if n_tumor == 0:
        # No tumor voxels: features are undefined. Return neutral values that
        # fall on the `astrocytoma_like` side rather than inventing a DMG call.
        return {"midline_offset": 1.0, "inferior_frac": 0.0, "et_frac": 0.0, "n_tumor": 0}

    coords = np.argwhere(tumor)
    centre_lr = (seg.shape[lr_axis] - 1) / 2.0
    midline_offset = abs(float(coords[:, lr_axis].mean()) - centre_lr) / centre_lr

    midpoint_is = (seg.shape[is_axis] - 1) / 2.0
    inferior_frac = float((coords[:, is_axis] < midpoint_is).mean())

    et_frac = float((seg == 1).sum()) / float(n_tumor)

    return {
        "midline_offset": midline_offset,
        "inferior_frac": inferior_frac,
        "et_frac": et_frac,
        "n_tumor": n_tumor,
    }


def classify(features: dict[str, float], strata_cfg: dict) -> str:
    """Apply the proxy rule to one subject's features."""
    dmg_label = strata_cfg.get("dmg_label", "dmg_like")
    other_label = strata_cfg.get("other_label", "astrocytoma_like")
    is_dmg = (
        features["midline_offset"] <= float(strata_cfg.get("midline_offset_max", 0.15))
        and features["inferior_frac"] >= float(strata_cfg.get("inferior_frac_min", 0.60))
        and features["et_frac"] <= float(strata_cfg.get("et_frac_max", 0.10))
    )
    return dmg_label if is_dmg else other_label


def load_tumor_type_csv(path: str | Path) -> dict[str, str]:
    """Load an external ``subject_id,tumor_type`` mapping."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"tumor-type CSV not found: {path}")
    mapping: dict[str, str] = {}
    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        missing = {"subject_id", "tumor_type"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"tumor-type CSV {path} is missing column(s) {sorted(missing)}; "
                f"found {reader.fieldnames}"
            )
        for row in reader:
            mapping[str(row["subject_id"]).strip()] = str(row["tumor_type"]).strip()
    return mapping


def build_strata(cfg: Config, tumor_type_csv: str | Path | None = None) -> list[dict]:
    """Assign exactly one stratum to every subject across all three manifests.

    Returns a list of row dicts using :data:`STRATA_COLUMNS`.
    """
    manifests = load_all_manifests(cfg.resolve("manifests"))
    cache = cfg.resolve("cache")
    strata_cfg = cfg.strata
    lr_axis = int(strata_cfg.get("lr_axis", 0))
    is_axis = int(strata_cfg.get("is_axis", 2))

    external = load_tumor_type_csv(tumor_type_csv) if tumor_type_csv else {}
    if not external:
        LOGGER.warning(PROXY_WARNING)

    rows: list[dict] = []
    for hospital, subject_ids in manifests.items():
        for sid in subject_ids:
            path = subject_path(cache, sid)
            if not path.exists():
                raise FileNotFoundError(f"cached subject {sid!r} not found at {path}")
            with np.load(path) as npz:
                if "seg" not in set(npz.files):
                    raise KeyError(f"subject {sid!r} ({path}) is missing key 'seg'")
                seg = np.asarray(npz["seg"])
            feats = seg_features(seg, lr_axis=lr_axis, is_axis=is_axis)

            if sid in external:
                stratum, source = external[sid], EXTERNAL_SOURCE
            else:
                stratum, source = classify(feats, strata_cfg), PROXY_SOURCE

            rows.append(
                {
                    "subject_id": sid,
                    "hospital": hospital,
                    "stratum": stratum,
                    "source": source,
                    "midline_offset": round(feats["midline_offset"], 6),
                    "inferior_frac": round(feats["inferior_frac"], 6),
                    "et_frac": round(feats["et_frac"], 6),
                }
            )
    return rows


def write_strata_csv(rows: list[dict], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=STRATA_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def read_strata_csv(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"strata file not found: {path}. Run `python -m src.strata` first."
        )
    with open(path, "r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def strata_by_hospital(rows: list[dict], hospital: str) -> dict[str, list[str]]:
    """Group one hospital's subject IDs by stratum."""
    grouped: dict[str, list[str]] = {}
    for row in rows:
        if row["hospital"] != hospital:
            continue
        grouped.setdefault(row["stratum"], []).append(row["subject_id"])
    return grouped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument(
        "--tumor-type-csv",
        default=None,
        help="optional CSV (subject_id,tumor_type) that overrides the imaging proxy",
    )
    parser.add_argument("--out", default=None, help="output path (default: paths.strata_csv)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config(args.config)
    rows = build_strata(cfg, tumor_type_csv=args.tumor_type_csv)
    out = Path(args.out) if args.out else cfg.resolve("strata_csv")
    write_strata_csv(rows, out)

    counts: dict[tuple[str, str], int] = {}
    for row in rows:
        counts[(row["hospital"], row["stratum"])] = counts.get((row["hospital"], row["stratum"]), 0) + 1
    print(f"\nwrote {len(rows)} rows -> {out}")
    for hospital in ("A", "B", "heldout"):
        total = sum(v for (h, _), v in counts.items() if h == hospital)
        if not total:
            continue
        print(f"  {hospital}: n={total}")
        for (h, stratum), n in sorted(counts.items()):
            if h == hospital:
                print(f"    {stratum:<20} {n:>4}  ({n / total:6.1%})")
    if not args.tumor_type_csv:
        print(f"\nWARNING: {PROXY_WARNING}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
