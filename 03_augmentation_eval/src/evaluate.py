"""Ablation runner: checkpoint -> held-out inference -> one row in the results CSV.

Writes exactly the schema fixed in CONTRACTS.md::

    experiment_name, use_augmentation, use_federation, use_domain_adaptation,
    dice_ET, dice_NC, dice_WT, mean_dice

**Dice is scored per patient, never per slice.** Every slice of a subject is
predicted, un-padded back to its native shape, stacked into a volume, and one
ET/NC/WT Dice is computed for that subject; the means across subjects are what
land in the CSV. Averaging Dice per slice is not comparable to any BraTS number
and badly overstates results, because ~70% of slices contain no tumour at all
and an empty slice predicted empty scores 1.0.

Plane policy (``eval.plane``) is explicit rather than implicit:

* ``axial`` / ``coronal`` -- restack that plane's argmax class labels.
* ``both`` -- average per-class softmax probabilities in the shared
  (C, 240, 240, 155) voxel grid, then argmax once. This is legal because axial
  and coronal slices index the same volume, so no reorientation is needed beyond
  the inverse of the slicing axis.

Evaluated on the held-out institution only, never on hospital A or B, and never
with augmentation applied.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import numpy as np
import torch

from .config import Config
from .dataset import EvalSubjectDataset
from .dummy import DummySegNet2D, load_checkpoint
from .model import build_model, infer_geometry
from .metrics import (
    REGION_ORDER,
    aggregate,
    count_empty_ground_truth,
    dice_regions,
)
from .postproc import DEFAULT_POSTPROC, probs_to_classes
from .slices import load_manifest, unpad

LOGGER = logging.getLogger(__name__)

CSV_COLUMNS = [
    "experiment_name",
    "use_augmentation",
    "use_federation",
    "use_domain_adaptation",
    "dice_ET",
    "dice_NC",
    "dice_WT",
    "mean_dice",
]

#: slices pushed through the network at once during inference
INFER_CHUNK = 16


# ----------------------------------------------------------------- CSV writing
def append_result_row(row: dict, csv_path: str | Path) -> Path:
    """Append exactly one row, creating the file (and its parent) if needed.

    A pre-existing file with a different header is a hard error -- silently
    appending misaligned columns would corrupt the ablation table.
    """
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    missing = set(CSV_COLUMNS) - set(row)
    if missing:
        raise ValueError(f"result row is missing column(s) {sorted(missing)}")

    write_header = True
    if path.exists() and path.stat().st_size > 0:
        with open(path, "r", encoding="utf-8", newline="") as fh:
            existing = next(csv.reader(fh), None)
        if existing != CSV_COLUMNS:
            raise ValueError(
                f"results CSV {path} has an unexpected header.\n"
                f"  expected: {CSV_COLUMNS}\n"
                f"  found:    {existing}"
            )
        write_header = False

    with open(path, "a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow({k: row[k] for k in CSV_COLUMNS})
    return path


# ------------------------------------------------------------------ restacking
def restack_volume(arr: np.ndarray, plane: str, cfg: Config, channelled: bool = False,
                   indices=None, volume_shape=None) -> np.ndarray:
    """Stack per-plane slices back into the volume frame.

    ``(N, H, W)`` -> ``(X, Y, Z)``, or ``(C, N, H, W)`` -> ``(C, X, Y, Z)`` when
    ``channelled``.

    With ``indices``, each slice is placed at its **true volume position** and
    any missing slice stays background. Subjects whose blank edge slices were
    dropped from the cache run e.g. 8..134 rather than 0..154, so stacking them
    densely would shorten the volume and misalign it -- and would give axial and
    coronal different shapes, which the ``both`` policy cannot average.

    A dropped slice is background in the prediction and in the ground truth
    alike, so filling it with zeros leaves Dice unchanged.
    """
    axis = cfg.plane_axis(plane)
    if indices is None:
        return np.moveaxis(arr, 1, axis + 1) if channelled else np.moveaxis(arr, 0, axis)

    vs = tuple(volume_shape or cfg.volume_shape)
    if channelled:
        out = np.zeros((arr.shape[0], *vs), dtype=arr.dtype)
        np.moveaxis(out, axis + 1, 1)[:, list(indices)] = arr
    else:
        out = np.zeros(vs, dtype=arr.dtype)
        np.moveaxis(out, axis, 0)[list(indices)] = arr
    return out


@torch.no_grad()
def predict_slices(model, images: np.ndarray, device: str = "cpu") -> np.ndarray:
    """Run the model over ``(N, C, H, W)``, returning ``(N, C_out, H, W)`` logits."""
    outputs = []
    for start in range(0, images.shape[0], INFER_CHUNK):
        chunk = torch.as_tensor(images[start:start + INFER_CHUNK], dtype=torch.float32).to(device)
        seg_logits, _features = model(chunk)
        outputs.append(seg_logits.detach().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def _softmax(logits: np.ndarray, axis: int = 1) -> np.ndarray:
    shifted = logits - logits.max(axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=axis, keepdims=True)


@torch.no_grad()
def predict_probs(model, images: np.ndarray, device: str = "cpu",
                  tta: bool = False) -> np.ndarray:
    """Class probabilities for ``(N, C, H, W)`` slices, optionally flip-averaged.

    Averaging happens in *probability* space, not logit space: logits from two
    views are not on a common scale, so averaging them is not a defined
    operation, while averaging probabilities is a plain ensemble.

    With ``tta=False`` this is a softmax of :func:`predict_slices`, so any
    downstream argmax gives bit-identical predictions to the pre-TTA code.
    """
    probs = _softmax(predict_slices(model, images, device=device), axis=1)
    if not tta:
        return probs

    # ``.copy()`` is required: torch cannot wrap a negative-strided numpy view.
    flipped = predict_slices(model, images[..., ::-1].copy(), device=device)
    return 0.5 * (probs + _softmax(flipped, axis=1)[..., ::-1])


@torch.no_grad()
def evaluate_subject(model, items, cfg: Config, device: str = "cpu"):
    """Score one subject from its restacked prediction volume.

    ``items`` is the list of per-plane records for a single subject (one for
    ``axial``/``coronal``, two for ``both``). Returns
    ``(scores, true_volume, planes_used)``.
    """
    if not items:
        raise ValueError("no cached planes for this subject")

    prob_sum = None
    true_volume = None
    planes_used = []

    for item in items:
        plane = item["plane"]
        planes_used.append(plane)
        indices = item.get("slice_indices")

        probs = predict_probs(model, item["images"], device=device, tta=cfg.eval_tta)
        probs = unpad(probs, item["pad"])                                  # native shape
        labels = unpad(item["labels"], item["pad"])                        # (N, H, W)

        true_volume = restack_volume(labels, plane, cfg, indices=indices)

        volume_probs = restack_volume(probs.transpose(1, 0, 2, 3), plane, cfg,
                                      channelled=True, indices=indices)
        prob_sum = volume_probs if prob_sum is None else prob_sum + volume_probs

    # Mean rather than sum so a single-plane subject under the ``both`` policy is
    # still on the same probability scale -- et_boost is a threshold on that
    # scale, and would otherwise mean something different per subject.
    prob_volume = prob_sum / len(planes_used)
    pred_volume = probs_to_classes(prob_volume, **cfg.postproc)

    return dice_regions(pred_volume, true_volume), true_volume, planes_used


def build_eval_dataset(cfg: Config, dummy_data: bool, dummy_n: int, cache_dir=None,
                       tmp_dir=None, subjects: list | None = None):
    """The held-out evaluation set, or a synthetic stand-in for ``--dummy-data``."""
    if dummy_data:
        from .dummy import DUMMY_VOLUME_SHAPE, make_dummy_cache

        LOGGER.info("using a synthetic slice cache (the real cache is not read)")
        tmp_dir = Path(tmp_dir)
        subjects = [f"DUMMY-{i:04d}" for i in range(dummy_n)]
        # Restacking places slices at their true position in data.volume_shape,
        # so that has to describe the synthetic volumes, not the real dataset.
        cfg.data = dict(cfg.data)
        cfg.data["volume_shape"] = list(DUMMY_VOLUME_SHAPE)
        make_dummy_cache(tmp_dir, subjects, cfg, volume_shape=DUMMY_VOLUME_SHAPE, seed=0)
        return EvalSubjectDataset(subjects, cfg, cache_dir=tmp_dir), subjects

    if subjects is None:
        subjects = load_manifest(cfg.resolve("manifests"), "heldout")
        LOGGER.info("evaluating on %d held-out subjects", len(subjects))
    else:
        # Tuning path: these are the training-hospital validation patients, not
        # the held-out institution. Loud on purpose -- a tuning number quoted as
        # a held-out result is the single easiest way to overstate this model.
        LOGGER.warning("evaluating on %d CALLER-SUPPLIED subjects (not the "
                       "held-out manifest) -- not a generalisation estimate",
                       len(subjects))
    return EvalSubjectDataset(subjects, cfg, cache_dir=cache_dir), subjects


def run(
    cfg: Config,
    experiment_name: str,
    checkpoint: str | Path | None = None,
    dummy_checkpoint: bool = False,
    dummy_data: bool = False,
    dummy_n: int = 4,
    cache_dir: Path | None = None,
    csv_path: str | Path | None = None,
    device: str = "cpu",
    tmp_dir: Path | None = None,
    subjects: list | None = None,
    write_csv: bool = True,
) -> dict:
    """Evaluate one configuration and append its row. Returns the written row."""
    if checkpoint is None and not dummy_checkpoint:
        raise ValueError("provide --checkpoint <path> or --dummy-checkpoint")

    dataset, subjects = build_eval_dataset(cfg, dummy_data, dummy_n, cache_dir, tmp_dir,
                                           subjects=subjects)

    if dummy_checkpoint:
        LOGGER.warning(
            "running against a randomly-initialised DummySegNet2D - the Dice numbers "
            "are pipeline validation only, not model performance"
        )
        model = DummySegNet2D(in_channels=len(cfg.modalities), num_classes=cfg.num_classes)
    else:
        if not Path(checkpoint).exists():
            raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
        # The checkpoint declares what to rebuild. `spatial_dims` matters because
        # the same network trains on slices or volumes, and evaluation has to
        # match what was trained.
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        meta = payload if isinstance(payload, dict) else {}
        architecture = str(meta.get("architecture", "unet"))
        if architecture == "dummy":
            model = DummySegNet2D(in_channels=len(cfg.modalities),
                                  num_classes=cfg.num_classes)
        else:
            # Geometry comes from the checkpoint, not the config: config.model
            # may have changed since it was trained, and rebuilding at the wrong
            # width fails with a shape mismatch that looks like a corrupt file.
            geom = {k: payload[k] for k in ("width", "depth") if k in payload}
            if not geom:
                geom = infer_geometry(payload["model_state_dict"])
            model = build_model(cfg, spatial_dims=int(meta.get("spatial_dims", 2)), **geom)
        model = load_checkpoint(checkpoint, model)
    model = model.to(device).eval()

    # Group the (subject, plane) items back together per subject.
    by_subject: dict[str, list] = {}
    for i in range(len(dataset)):
        try:
            item = dataset[i]
        except FileNotFoundError as exc:
            LOGGER.warning("skipping: %s", exc)
            continue
        by_subject.setdefault(item["subject_id"], []).append(item)

    per_subject, true_masks, single_plane = [], [], []
    for subject_id in subjects:
        items = by_subject.get(subject_id)
        if not items:
            continue
        scores, true_volume, planes_used = evaluate_subject(model, items, cfg, device)
        per_subject.append(scores)
        true_masks.append(true_volume)
        if cfg.eval_plane == "both" and len(planes_used) < len(cfg.planes):
            single_plane.append(subject_id)

    if not per_subject:
        raise ValueError("evaluation set is empty; nothing to score")

    scores = aggregate(per_subject)
    empty_gt = count_empty_ground_truth(true_masks)

    row = {
        "experiment_name": experiment_name,
        "use_augmentation": cfg.use_augmentation,
        "use_federation": cfg.use_federation,
        "use_domain_adaptation": cfg.use_domain_adaptation,
        "dice_ET": round(scores["dice_ET"], 6),
        "dice_NC": round(scores["dice_NC"], 6),
        "dice_WT": round(scores["dice_WT"], 6),
    }
    # Keep the identity exact rather than a rounding artefact of the three parts.
    row["mean_dice"] = round((row["dice_ET"] + row["dice_NC"] + row["dice_WT"]) / 3.0, 6)

    out = Path(csv_path) if csv_path else cfg.resolve("results_csv")
    if write_csv:
        append_result_row(row, out)

    print(f"\n{experiment_name}: " +
          "  ".join(f"{r}={row[f'dice_{r}']:.4f}" for r in REGION_ORDER))
    print(f"  mean_dice={row['mean_dice']:.4f}")
    print(f"  scored {len(per_subject)} subjects per patient from restacked volumes")
    print(f"  eval plane policy: {cfg.eval_plane}   tta: {cfg.eval_tta}")
    pp = cfg.postproc
    if pp != DEFAULT_POSTPROC:
        print("  post-processing: " + "  ".join(f"{k}={v}" for k, v in pp.items()))
    print("  subjects with an empty ground-truth region: " +
          ", ".join(f"{r}={empty_gt[r]}" for r in REGION_ORDER))
    if single_plane:
        print(f"  single-plane subjects (expected both): {len(single_plane)}")
    print(f"  wrote: {out}" if write_csv else "  (CSV write skipped)")

    row["_empty_ground_truth"] = empty_gt
    row["_n_subjects"] = len(per_subject)
    return row
