"""Ablation runner: checkpoint -> held-out inference -> one row in the results CSV.

Writes exactly the schema fixed in CONTRACTS.md:

    experiment_name, use_augmentation, use_federation, use_domain_adaptation,
    dice_ET, dice_NC, dice_WT, mean_dice

Evaluated on the held-out 82-subject institution only, never on Hospital A or B,
and never with augmentation applied.

Usage:
    python -m src.runner --experiment-name baseline --dummy-checkpoint --dummy-data
    python -m src.runner --experiment-name aug_fed --checkpoint checkpoints/run7/last.pt
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import Config, load_config
from .data import load_heldout
from .dataset import DummyDataset, EvalDataset
from .dummy_model import DummySegNet, load_checkpoint, save_dummy_checkpoint
from .metrics import (
    REGION_ORDER,
    aggregate,
    count_empty_ground_truth,
    dice_regions,
    logits_to_classes,
)

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


# ----------------------------------------------------------------- CSV writing
def append_result_row(row: dict, csv_path: str | Path) -> Path:
    """Append exactly one row, creating the file (and its parent) if needed.

    A pre-existing file with a different header is a hard error — silently
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


# ------------------------------------------------------------------- inference
@torch.no_grad()
def evaluate(model: torch.nn.Module, dataset, device: str = "cpu") -> tuple[dict, dict]:
    """Run inference over a dataset and aggregate per-region Dice.

    Returns ``(aggregated_scores, empty_ground_truth_counts)``.
    """
    model = model.to(device).eval()
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    per_subject: list[dict[str, float]] = []
    true_masks = []

    for batch in loader:
        image = batch["image"].to(device)
        label = batch["label"]

        seg_logits, _features = model(image)
        pred_classes = logits_to_classes(seg_logits)          # (B, D, H, W)
        true_classes = label.squeeze(1).cpu().numpy()          # (B, D, H, W)

        for b in range(pred_classes.shape[0]):
            per_subject.append(dice_regions(pred_classes[b], true_classes[b]))
            true_masks.append(true_classes[b])

    if not per_subject:
        raise ValueError("evaluation set is empty; nothing to score")

    return aggregate(per_subject), count_empty_ground_truth(true_masks)


def build_eval_dataset(cfg: Config, dummy_data: bool, dummy_n: int):
    if dummy_data:
        LOGGER.info("using dummy random tensors (the real cache is not read)")
        return DummyDataset(n=dummy_n, cfg=cfg)
    subject_ids = load_heldout(cfg.resolve("manifests"))
    LOGGER.info("evaluating on %d held-out subjects", len(subject_ids))
    return EvalDataset(subject_ids, cfg)


def run(
    cfg: Config,
    experiment_name: str,
    checkpoint: str | Path | None = None,
    dummy_checkpoint: bool = False,
    dummy_data: bool = False,
    dummy_n: int = 4,
    csv_path: str | Path | None = None,
    device: str = "cpu",
) -> dict:
    """Evaluate one configuration and append its row. Returns the written row."""
    dataset = build_eval_dataset(cfg, dummy_data, dummy_n)

    model = DummySegNet(in_channels=len(cfg.modalities), num_classes=cfg.num_classes)
    if dummy_checkpoint:
        LOGGER.warning(
            "running against a randomly-initialized DummySegNet - the Dice numbers "
            "are pipeline-validation only, not model performance"
        )
    elif checkpoint is not None:
        model = load_checkpoint(checkpoint, model)
    else:
        raise ValueError("provide --checkpoint <path> or --dummy-checkpoint")

    scores, empty_gt = evaluate(model, dataset, device=device)

    row = {
        "experiment_name": experiment_name,
        "use_augmentation": cfg.use_augmentation,
        "use_federation": cfg.use_federation,
        "use_domain_adaptation": cfg.use_domain_adaptation,
        "dice_ET": round(scores["dice_ET"], 6),
        "dice_NC": round(scores["dice_NC"], 6),
        "dice_WT": round(scores["dice_WT"], 6),
        "mean_dice": round(scores["mean_dice"], 6),
    }
    # Keep the identity exact rather than a rounding artifact of the three parts.
    row["mean_dice"] = round(
        (row["dice_ET"] + row["dice_NC"] + row["dice_WT"]) / 3.0, 6
    )

    out = Path(csv_path) if csv_path else cfg.resolve("results_csv")
    append_result_row(row, out)

    print(f"\n{experiment_name}: " + "  ".join(f"{r}={row[f'dice_{r}']:.4f}" for r in REGION_ORDER))
    print(f"  mean_dice={row['mean_dice']:.4f}")
    print(f"  subjects with an empty ground-truth region: " +
          ", ".join(f"{r}={empty_gt[r]}" for r in REGION_ORDER))
    print(f"  row appended -> {out}")
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument("--experiment-name", required=True, help="label for this CSV row")
    parser.add_argument("--checkpoint", default=None, help="trained checkpoint to evaluate")
    parser.add_argument(
        "--dummy-checkpoint",
        action="store_true",
        help="evaluate a randomly-initialized DummySegNet instead of a trained model",
    )
    parser.add_argument(
        "--dummy-data",
        action="store_true",
        help="use random tensors instead of the real cache",
    )
    parser.add_argument("--dummy-n", type=int, default=4, help="number of dummy subjects")
    parser.add_argument("--out-csv", default=None, help="override paths.results_csv")
    parser.add_argument("--device", default="cpu")

    flags = parser.add_argument_group("ablation flag overrides")
    for flag in ("use-augmentation", "use-federation", "use-domain-adaptation"):
        flags.add_argument(f"--{flag}", dest=flag.replace("-", "_"),
                           action="store_true", default=None)
        flags.add_argument(f"--no-{flag}", dest=flag.replace("-", "_"),
                           action="store_false", default=None)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg = load_config(
        args.config,
        use_augmentation=args.use_augmentation,
        use_federation=args.use_federation,
        use_domain_adaptation=args.use_domain_adaptation,
    )

    checkpoint = args.checkpoint
    if not checkpoint and not args.dummy_checkpoint:
        parser.error("provide --checkpoint <path> or --dummy-checkpoint")

    run(
        cfg,
        experiment_name=args.experiment_name,
        checkpoint=checkpoint,
        dummy_checkpoint=args.dummy_checkpoint,
        dummy_data=args.dummy_data,
        dummy_n=args.dummy_n,
        csv_path=args.out_csv,
        device=args.device,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
