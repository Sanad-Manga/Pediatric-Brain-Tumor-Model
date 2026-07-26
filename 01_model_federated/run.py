"""CLI entry point: single-client run, federated run, and resume of either."""
from __future__ import annotations

import argparse

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
    p.add_argument("--data-mode", choices=["dummy", "real"], default="dummy")
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
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--resume", action="store_true")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    config = TrainConfig(
        use_augmentation=args.use_augmentation,
        use_federation=args.use_federation,
        use_domain_adaptation=args.use_domain_adaptation,
        data_mode=args.data_mode,
        cache_path=args.cache_path,
        lr=args.lr,
        run_id=args.run_id,
        checkpoint_dir=args.checkpoint_dir,
    )

    if config.use_federation:
        _model, round_losses = train_federated(
            config=config,
            client_manifest_paths=args.client_manifests,
            num_rounds=args.rounds,
            local_epochs=args.epochs,
            resume=args.resume,
        )
        print(f"Federated training complete. Round losses: {round_losses}")
    else:
        _model, losses = train_single_client(
            config=config,
            manifest_path=args.manifest,
            num_epochs=args.epochs,
            resume=args.resume,
        )
        print(f"Single-client training complete. Losses: {losses}")


if __name__ == "__main__":
    main()
