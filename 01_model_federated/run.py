"""CLI entry point: single-client run, federated run, and resume of either."""
from __future__ import annotations

import argparse
import math
import random
import time

import numpy as np
import torch

from src.augment3d import Augment3D
from src.config import TrainConfig, load_config
from src.federated import train_federated
from src.train_single import train_single_client

DEFAULT_MANIFEST_DIR = "../00_shared/manifests"
OLD_DEFAULTS = {
    "lr": 1e-3,
    "loss": "dice_ce",
    "lr_horizon": None,
    "modality_dropout": 0.0,
    "sequence_shift": 0.0,
    "sequence_shift_max_voxels": 1.2,
}


class _StoreExplicit(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        explicit = set(getattr(namespace, "_explicit_cli", None) or ())
        explicit.add(self.dest)
        namespace._explicit_cli = explicit


def seed_everything(seed: int) -> None:
    """Seed every RNG a run draws from, so the same command twice gives the same model."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # harmless if no CUDA device


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Federated 3D U-Net training (BraTS-PEDs)")
    p.add_argument("--config", default=None, help="path to YAML configuration")
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
    p.add_argument("--lr", type=float, default=1e-3, action=_StoreExplicit)
    p.add_argument("--coral-weight", type=float, default=1.0)
    p.add_argument("--coral-queue-size", type=int, default=8)
    p.add_argument("--coral-steps-per-round", type=int, default=None)
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--deadline-unix", type=float, default=None,
                    help="unix timestamp; single-client loop stops between epochs once past it")
    p.add_argument("--lr-horizon", type=int, default=None, action=_StoreExplicit,
                    help="single-client only: cosine-anneal lr to 1e-5 over this many epochs "
                         "(fixed regardless of --epochs); default: no schedule")
    p.add_argument("--loss", choices=["dice_ce", "dice_focal", "region_dice_bce", "region_hybrid"],
                   default="dice_ce", action=_StoreExplicit,
                    help="single-client only: dice_focal down-weights easy/majority voxels, "
                         "a standard fix for the rare-class (ET) under-segmentation dice_ce shows; "
                         "region_dice_bce trains the three scored regions (WT/TC/ET) directly with "
                         "Dice + BCE on sums of the same 5-way softmax; region_hybrid adds dice_ce to "
                         "that with equal weight")
    p.add_argument("--region-terms", choices=["both", "dice", "bce"], default="both",
                    help="--loss region_dice_bce only (diagnostic): both = the full region loss, dice = only its "
                         "soft Dice half, bce = only its binary cross-entropy half")
    p.add_argument("--et-pos-weight", type=float, default=1.0,
                    help="--loss region_dice_bce only: weight on positive enhancing-tumour voxels in the BCE term "
                         "(counteracts ET being ~1 in 1,279 voxels); 1.0 = unweighted")
    p.add_argument("--small-et-weight", type=float, default=None,
                    help="draw patients with a small enhancing tumour (1..--small-et-max-voxels ET voxels) this many "
                         "times as often as patients with a larger one; tumour-free patients keep their share of "
                         "draws. Unset = today's loader; 1.0 = the matched random-draw control")
    p.add_argument("--small-et-max-voxels", type=int, default=None,
                    help="upper ET-voxel count for 'small' in --small-et-weight (default 600 on the 96^3 grid)")
    p.add_argument("--patch-size", type=int, nargs=3, default=[128, 128, 128], metavar=("D", "H", "W"),
                    help="--data-mode patch only: crop size in voxels, each a multiple of 16")
    p.add_argument("--patches-per-epoch", type=int, default=580,
                    help="--data-mode patch only: patches drawn per epoch")
    p.add_argument("--patch-fractions", type=float, nargs=3, default=[0.35, 0.45, 0.20],
                    metavar=("ET", "TUMOR", "RANDOM"),
                    help="--data-mode patch only: share of ET-centred / tumour-centred / random patches (sum 1)")
    p.add_argument("--modality-dropout", type=float, default=0.0, action=_StoreExplicit,
                    help="per-sequence probability of zeroing a whole input channel during augmentation, "
                         "in [0.0, 1.0); requires --use-augmentation")
    p.add_argument("--sequence-shift", type=float, default=0.0, action=_StoreExplicit,
                    help="probability per sample of translating a random subset of input sequences relative to the "
                         "labels' frame (simulated misregistration), in [0.0, 1.0); requires --use-augmentation")
    p.add_argument("--sequence-shift-max-voxels", type=float, default=1.2, action=_StoreExplicit,
                    help="maximum translation per axis, in voxels of the grid being trained on (1.2 voxels is about "
                         "3 mm in-plane on the 96^3 grid, about 1.2 mm in --data-mode patch); must be > 0")
    p.add_argument("--seed", type=int, default=None,
                    help="seed random / numpy / torch for a reproducible run; default: no seeding "
                         "(unchanged behaviour)")
    p.add_argument("--eval-every", type=int, default=0, metavar="N",
                    help="single-client only: every N epochs, score the checkpoint on --eval-manifest "
                         "(tools/eval_heldout_3d.py logic) and copy the one with the best worst-region "
                         "Dice to <checkpoint-dir>/<run-id>/best.pt; 0 = never (default, unchanged behaviour)")
    p.add_argument("--eval-manifest", default=f"{DEFAULT_MANIFEST_DIR}/heldout.json",
                    help="--eval-every only: held-out manifest to score; read from --cache-path in "
                         "--data-mode real, synthetic volumes in --data-mode dummy")
    return p


def resolve_settings(args: argparse.Namespace, cfg: dict | None) -> dict:
    """Resolve CLI values over config values over the legacy built-in defaults."""
    config = cfg or {}
    explicit = getattr(args, "_explicit_cli", None) or set()
    resolved = {}
    for name, default in OLD_DEFAULTS.items():
        config_value = None
        if name == "loss":
            config_value = (config.get("loss") or {}).get("kind")
        elif name in {"modality_dropout", "sequence_shift", "sequence_shift_max_voxels"}:
            augmentation_name = {
                "modality_dropout": "modality_dropout_prob",
                "sequence_shift": "sequence_shift_prob",
                "sequence_shift_max_voxels": "sequence_shift_max_voxels",
            }[name]
            config_value = (config.get("augmentation") or {}).get(augmentation_name)
        elif name == "lr":
            config_value = (config.get("schedule") or {}).get("lr")
        cli_value = getattr(args, name)
        resolved[name] = (
            cli_value if name in explicit else
            config_value if config_value is not None else default
        )

    resolved["augmentation"] = dict((config.get("augmentation") or {}))
    for key, arg_name in (
        ("modality_dropout_prob", "modality_dropout"),
        ("sequence_shift_prob", "sequence_shift"),
        ("sequence_shift_max_voxels", "sequence_shift_max_voxels"),
    ):
        resolved["augmentation"][key] = resolved[arg_name]

    resolved["model"] = {"width": 16, "depth": 5, **(config.get("model") or {})}
    resolved["loss_config"] = {
        "kind": "dice_ce", "class_weights": None, **(config.get("loss") or {})
    }
    resolved["schedule"] = {
        "kind": "none", "min_lr": 1.0e-5, **(config.get("schedule") or {})
    }
    if not cfg and args.lr_horizon is not None:
        resolved["schedule"]["kind"] = "cosine"
    return resolved


def parse_args(argv=None) -> argparse.Namespace:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    cfg = load_config(args.config) if args.config else None
    settings = resolve_settings(args, cfg)
    for name in OLD_DEFAULTS:
        setattr(args, name, settings[name])
    args.model = settings["model"]
    args.loss_config = settings["loss_config"]
    args.schedule = settings["schedule"]
    args.augmentation = settings["augmentation"]
    if not (0.0 <= args.modality_dropout < 1.0):  # also rejects NaN
        parser.error(f"--modality-dropout must be in [0.0, 1.0), got {args.modality_dropout}")
    if not (0.0 <= args.sequence_shift < 1.0):  # also rejects NaN
        parser.error(f"--sequence-shift must be in [0.0, 1.0), got {args.sequence_shift}")
    if not (args.sequence_shift_max_voxels > 0.0 and math.isfinite(args.sequence_shift_max_voxels)):
        parser.error(f"--sequence-shift-max-voxels must be a positive finite number, got {args.sequence_shift_max_voxels}")
    if args.loss in ("region_dice_bce", "region_hybrid") and args.use_federation:
        parser.error(f"--loss {args.loss} cannot be combined with --use-federation "
                     "(the federated loop never reads --loss and would silently ignore it)")
    if args.loss_config.get("class_weights") is not None and args.use_federation:
        parser.error("loss.class_weights cannot be combined with --use-federation")
    if args.loss in ("region_dice_bce", "region_hybrid") and args.loss_config.get("class_weights") is not None:
        parser.error(f"loss.class_weights cannot be combined with --loss {args.loss}")
    if not (args.et_pos_weight > 0.0 and math.isfinite(args.et_pos_weight)):  # also rejects NaN
        parser.error(f"--et-pos-weight must be a positive finite number, got {args.et_pos_weight}")
    if args.et_pos_weight != 1.0 and (args.loss != "region_dice_bce" or args.region_terms == "dice"):
        parser.error(f"--et-pos-weight {args.et_pos_weight} requires --loss region_dice_bce with a BCE term "
                     f"(got --loss {args.loss}, --region-terms {args.region_terms}); it would silently do nothing")
    if args.region_terms != "both" and args.loss != "region_dice_bce":
        parser.error(f"--region-terms {args.region_terms} requires --loss region_dice_bce (got --loss {args.loss}); "
                     "it would silently do nothing")
    if args.small_et_weight is not None and not (args.small_et_weight > 0.0 and math.isfinite(args.small_et_weight)):
        parser.error(f"--small-et-weight must be a positive finite number, got {args.small_et_weight}")
    if args.small_et_max_voxels is not None and args.small_et_weight is None:
        parser.error("--small-et-max-voxels requires --small-et-weight (it would silently do nothing)")
    if args.small_et_max_voxels is not None and args.small_et_max_voxels < 1:
        parser.error(f"--small-et-max-voxels must be >= 1, got {args.small_et_max_voxels}")
    if args.small_et_weight is not None and args.use_federation:
        parser.error("--small-et-weight cannot be combined with --use-federation (the federated loop has no sampler)")
    if args.small_et_weight is not None and args.data_mode == "patch":
        parser.error("--small-et-weight cannot be combined with --data-mode patch (patch mode has its own sampler)")
    if args.data_mode == "patch" and args.use_federation:
        parser.error("--data-mode patch is single-client only; it cannot be combined with --use-federation")
    if args.eval_every < 0:
        parser.error(f"--eval-every must be >= 0 (0 = never), got {args.eval_every}")
    if args.eval_every > 0 and args.use_federation:
        parser.error("--eval-every is single-client only; it cannot be combined with --use-federation "
                     "(the federated loop never reads it and would silently ignore it)")
    if args.eval_every > 0 and args.data_mode == "patch":
        parser.error("--eval-every does not support --data-mode patch (the held-out scorer reads the "
                     "96-cube cache; use tools/eval_patch3d.py for patch checkpoints)")
    if args.modality_dropout > 0.0 and not args.use_augmentation:
        parser.error("--modality-dropout requires --use-augmentation "
                     "(it is applied by the augmentation transform and would silently do nothing)")
    if args.sequence_shift > 0.0 and not args.use_augmentation:
        parser.error("--sequence-shift requires --use-augmentation "
                     "(it is applied by the augmentation transform and would silently do nothing)")
    return args


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        seed_everything(args.seed)

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
        model_width=args.model["width"],
        model_depth=args.model["depth"],
        class_weights=args.loss_config.get("class_weights"),
        schedule_kind=args.schedule["kind"],
        schedule_min_lr=args.schedule["min_lr"],
        lr_horizon=args.lr_horizon,
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
        Augment3D(**args.augmentation)
        if config.use_augmentation else None
    )
    if augmentation_transform is not None and args.seed is not None:
        augmentation_transform.set_random_state(args.seed)  # augmentation keeps its own RNG; seed it too

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
        heldout_scorer = None
        if args.eval_every > 0:
            from src.heldout_eval import build_heldout_scorer
            heldout_scorer = build_heldout_scorer(args.data_mode, args.cache_path, args.eval_manifest)
        _model, losses = train_single_client(
            config=config,
            manifest_path=args.manifest,
            num_epochs=args.epochs,
            augmentation_transform=augmentation_transform,
            resume=args.resume,
            deadline_unix=args.deadline_unix,
            lr_horizon=args.lr_horizon,
            loss_kind=args.loss,
            eval_every=args.eval_every,
            heldout_scorer=heldout_scorer,
            region_terms=args.region_terms,
            et_pos_weight=args.et_pos_weight,
            small_et_weight=args.small_et_weight,
            small_et_max_voxels=600 if args.small_et_max_voxels is None else args.small_et_max_voxels,
        )
        stopped_early = args.deadline_unix is not None and time.time() >= args.deadline_unix
        print(f"Single-client training complete ({len(losses)} epoch(s) this call"
              f"{', stopped by deadline' if stopped_early else ''}). Losses: {losses}")


if __name__ == "__main__":
    main()