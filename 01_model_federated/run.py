"""CLI entry point: single-client run, federated run, and resume of either."""
from __future__ import annotations

import argparse
import math
import time

from src.augment3d import Augment3D
from src.config import TrainConfig
from src.federated import train_federated
from src.train_single import train_single_client

DEFAULT_MANIFEST_DIR = "../00_shared/manifests"


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Federated 3D U-Net training (BraTS-PEDs)")
    p.add_argument("--run-id", default="default_run")
    p.add_argument("--use-augmentation", action="store_true")
    p.add_argument("--use-federation", action="store_true")
    p.add_argument("--use-domain-adaptation", action="store_true")
    p.add_argument("--data-mode", choices=["dummy", "real", "patch"], default="dummy")
    p.add_argument("--cache-path", default=None)
    p.add_argument("--manifest", default=f"{DEFAULT_MANIFEST_DIR}/hospitalA.json",
                    help="Manifest to use for a single-client run")
    p.add_argument("--client-manifests", nargs="+", default=[
        f"{DEFAULT_MANIFEST_DIR}/hospitalA.json",
        f"{DEFAULT_MANIFEST_DIR}/hospitalB.json",
    ], help="Manifests to use for a federated run")
    p.add_argument("--epochs", type=int, default=1, help="Epochs (single-client) or local epochs per round (federated)")
    p.add_argument("--rounds", type=int, default=2, help="Federated rounds (ignored for single-client)")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--coral-weight", type=float, default=1.0)
    p.add_argument("--coral-queue-size", type=int, default=8)
    p.add_argument("--coral-steps-per-round", type=int, default=None)
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--deadline-unix", type=float, default=None,
                    help="unix timestamp; single-client loop stops between epochs once past it")
    p.add_argument("--lr-horizon", type=int, default=None,
                    help="single-client only: cosine-anneal lr to 1e-5 over this many epochs "
                         "(fixed regardless of --epochs); default: no schedule")
    p.add_argument("--loss", choices=["dice_ce", "dice_focal"], default="dice_ce",
                    help="single-client only: dice_focal down-weights easy/majority voxels, "
                         "a standard fix for the rare-class (ET) under-segmentation dice_ce shows")
    p.add_argument("--patch-size", type=int, nargs=3, default=[128, 128, 128], metavar=("D", "H", "W"),
                    help="--data-mode patch only: crop size in voxels, each a multiple of 16")
    p.add_argument("--patches-per-epoch", type=int, default=580,
                    help="--data-mode patch only: patches drawn per epoch")
    p.add_argument("--patch-fractions", type=float, nargs=3, default=[0.35, 0.45, 0.20],
                    metavar=("ET", "TUMOR", "RANDOM"),
                    help="--data-mode patch only: share of ET-centred / tumour-centred / random patches (sum 1)")
    p.add_argument("--modality-dropout", type=float, default=0.0,
                    help="per-sequence probability of zeroing a whole input channel during augmentation, "
                         "in [0.0, 1.0); requires --use-augmentation")
    p.add_argument("--sequence-shift", type=float, default=0.0,
                    help="probability per sample of translating a random subset of input sequences relative to the "
                         "labels' frame (simulated misregistration), in [0.0, 1.0); requires --use-augmentation")
    p.add_argument("--sequence-shift-max-voxels", type=float, default=1.2,
                    help="maximum translation per axis, in voxels of the grid being trained on (1.2 voxels is about "
                         "3 mm in-plane on the 96^3 grid, about 1.2 mm in --data-mode patch); must be > 0")
    return p


def parse_args(argv=None) -> argparse.Namespace:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if not (0.0 <= args.modality_dropout < 1.0):  # also rejects NaN
        parser.error(f"--modality-dropout must be in [0.0, 1.0), got {args.modality_dropout}")
    if not (0.0 <= args.sequence_shift < 1.0):  # also rejects NaN
        parser.error(f"--sequence-shift must be in [0.0, 1.0), got {args.sequence_shift}")
    if not (args.sequence_shift_max_voxels > 0.0 and math.isfinite(args.sequence_shift_max_voxels)):
        parser.error(f"--sequence-shift-max-voxels must be a positive finite number, got {args.sequence_shift_max_voxels}")
    if args.data_mode == "patch" and args.use_federation:
        parser.error("--data-mode patch is single-client only; it cannot be combined with --use-federation")
    if args.modality_dropout > 0.0 and not args.use_augmentation:
        parser.error("--modality-dropout requires --use-augmentation "
                     "(it is applied by the augmentation transform and would silently do nothing)")
    if args.sequence_shift > 0.0 and not args.use_augmentation:
        parser.error("--sequence-shift requires --use-augmentation "
                     "(it is applied by the augmentation transform and would silently do nothing)")
    return args


def main() -> None:
    args = parse_args()

    config = TrainConfig(
        use_augmentation=args.use_augmentation,
        use_federation=args.use_federation,
        use_domain_adaptation=args.use_domain_adaptation,
        data_mode=args.data_mode,
        cache_path=args.cache_path,
        patch_size=tuple(args.patch_size),
        patches_per_epoch=args.patches_per_epoch,
        patch_fractions=tuple(args.patch_fractions),
        lr=args.lr,
        coral_weight=args.coral_weight,
        coral_queue_size=args.coral_queue_size,
        coral_steps_per_round=args.coral_steps_per_round,
        run_id=args.run_id,
        checkpoint_dir=args.checkpoint_dir,
    )

    # --use-augmentation used to be a complete no-op end to end: the config
    # flag existed and train_single.py's loop already had the hook
    # (`augmentation_transform`), but nothing ever built a real transform to
    # pass through it (01_model_federated/BRIEF.md always said augmentation
    # was a separate section's job; that section never shipped a 3D version).
    augmentation_transform = (
        Augment3D(modality_dropout_prob=args.modality_dropout, sequence_shift_prob=args.sequence_shift,
                  sequence_shift_max_voxels=args.sequence_shift_max_voxels)
        if config.use_augmentation else None
    )

    if config.use_federation:
        _model, round_losses = train_federated(
            config=config,
            client_manifest_paths=args.client_manifests,
            num_rounds=args.rounds,
            local_epochs=args.epochs,
            augmentation_transform=augmentation_transform,
            resume=args.resume,
        )
        print(f"Federated training complete. Round losses: {round_losses}")
    else:
        _model, losses = train_single_client(
            config=config,
            manifest_path=args.manifest,
            num_epochs=args.epochs,
            augmentation_transform=augmentation_transform,
            resume=args.resume,
            deadline_unix=args.deadline_unix,
            lr_horizon=args.lr_horizon,
            loss_kind=args.loss,
        )
        stopped_early = args.deadline_unix is not None and time.time() >= args.deadline_unix
        print(f"Single-client training complete ({len(losses)} epoch(s) this call"
              f"{', stopped by deadline' if stopped_early else ''}). Losses: {losses}")


if __name__ == "__main__":
    main()
