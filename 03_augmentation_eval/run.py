#!/usr/bin/env python
"""Section 03 — 2D slice augmentation + ablation. The single entry point.

    python run.py test                  verify everything with no data, no model
    python run.py index                 scan the slice cache once (tumour/ET presence)
    python run.py plan --report         build the balanced per-hospital plan
    python run.py preview --n 8         sanity-check augmentation, save a PNG
    python run.py eval --experiment-name baseline --dummy-checkpoint --dummy-data

The 2D slices are produced from the original volumes by the team; this section
consumes them. Anything not reachable through this file does not count as
delivered. Every command states what it wrote and where.
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from src.config import load_config  # noqa: E402

COMMANDS = ("test", "index", "tumor-type", "plan", "pack", "train", "preview", "eval")


# ------------------------------------------------------------------ run.py test
def cmd_test(args) -> int:
    """Run the dummy test suite: CPU only, no real cache, no trained model."""
    import pytest

    tests_dir = HERE / "tests"
    argv = ["-q", str(tests_dir)]
    if args.verbose:
        argv = ["-v", str(tests_dir)]
    code = pytest.main(argv)
    print(f"\nwrote: nothing (test run only; suite at {tests_dir})")
    return int(code)


# ----------------------------------------------------------------- run.py index
def cmd_index(args) -> int:
    """Scan every mask once and record which slices contain tumour / ET.

    The slice cache stores image and mask only. `plan` needs tumour presence
    *before* it selects slices, and deriving that means opening every mask file
    (~90,000 for the full cohort), so it is done once here.
    """
    from src.slices import build_index, list_subjects, write_index

    cfg = load_config(args.config)
    cache_dir = Path(args.cache) if args.cache else cfg.resolve("cache_2d")
    planes = [p.strip() for p in args.planes.split(",")] if args.planes else cfg.planes

    subjects = list_subjects(cache_dir)
    if args.limit:
        subjects = subjects[: args.limit]

    index = build_index(cache_dir, planes=planes, subjects=subjects, update=args.update)
    path = write_index(index, cache_dir)

    print(f"planes indexed: {', '.join(planes)} (sagittal not used)")
    if args.update:
        print(f"carried over: {index['n_carried_over']}    newly scanned: {index['n_scanned_now']}")
    print(f"subjects: {index['n_subjects']}    slice files: {index['n_slice_files']}")
    if index.get("corrupt_files"):
        print(f"WARNING: {len(index['corrupt_files'])} unreadable slice file(s) skipped:")
        for p in index["corrupt_files"][:10]:
            print(f"  {p}")
        if len(index["corrupt_files"]) > 10:
            print(f"  ... and {len(index['corrupt_files']) - 10} more")
    tumor = sum(len(p["tumor"]) for s in index["subjects"].values() for p in s.values())
    et = sum(len(p["et"]) for s in index["subjects"].values() for p in s.values())
    if index["n_slice_files"]:
        print(f"slices with tumour: {tumor} ({tumor / index['n_slice_files']:.1%})    "
              f"with ET: {et} ({et / index['n_slice_files']:.1%})")
    print(f"wrote: {path.resolve()}")
    return 0


# ------------------------------------------------------------------ run.py plan
def cmd_plan(args) -> int:
    from src.plan import build_plans, format_report, write_plan

    cfg = load_config(args.config)
    cache_dir = Path(args.cache) if args.cache else None
    plans = build_plans(cfg, cache_dir)

    if args.report:
        print(format_report(plans))
        print()

    plans_dir = Path(args.out) if args.out else cfg.resolve("plans_dir")
    written = [write_plan(plan, plans_dir / f"{hospital}_plan.json")
               for hospital, plan in sorted(plans.items())]

    for hospital, plan in sorted(plans.items()):
        stats = plan["stats"]
        if stats["entries"] < stats["target"]:
            print(f"note: {hospital} emitted {stats['entries']} of {stats['target']} "
                  f"target entries (per-patient cap {stats['per_patient_cap']})")
        if stats["subjects_missing_from_cache"]:
            print(f"note: {hospital} has {len(stats['subjects_missing_from_cache'])} "
                  f"subjects missing from the cache")

    for path in written:
        print(f"wrote: {path.resolve()}")
    return 0


# ------------------------------------------------------------------ run.py pack
def cmd_pack(args) -> int:
    """Shrink the cache to what training and evaluation actually read."""
    from src.pack import pack

    cfg = load_config(args.config)
    manifest = pack(
        cfg,
        out_dir=Path(args.out),
        src_cache=Path(args.cache) if args.cache else None,
        half=not args.full_precision,
        include_heldout=not args.no_heldout,
    )
    print(f"slices copied: {manifest['slices_copied']} "
          f"({manifest['training_slices']} training + held-out)")
    print(f"held-out subjects: {manifest['heldout_subjects']}")
    print(f"precision: {'float16' if manifest['float16'] else 'float32'}")
    print(f"size: {manifest['gigabytes']} GB")
    print(f"wrote: {Path(args.out).resolve()}")
    print(f"wrote: {(Path(args.out) / 'plans').resolve()}  (ship these with the data)")
    return 0


# ------------------------------------------------------------ run.py tumor-type
def cmd_tumor_type(args) -> int:
    """Classify every subject with the geometric DMG-like/astrocytoma-like proxy.

    NOT histology -- see src/tumor_type.py. Used to train the auxiliary
    classification head in `run.py train`, and standalone here for anyone who
    wants the per-subject table without training.
    """
    from src.slices import load_manifest
    from src.tumor_type import build_type_index, write_type_index

    cfg = load_config(args.config)
    cache_dir = Path(args.cache) if args.cache else cfg.resolve("cache_2d")
    manifest_dir = cfg.resolve("manifests")

    subjects = []
    for name in ("hospitalA", "hospitalB", "heldout"):
        subjects.extend(load_manifest(manifest_dir, name))

    index = build_type_index(cache_dir, subjects, cfg, plane=args.plane)
    path = write_type_index(index, cache_dir)

    print("NOT histology -- geometric proxy from segmentation mask geometry only")
    print(f"subjects classified: {len(index['subjects'])} of {len(subjects)}")
    for stratum, n in sorted(index["counts"].items()):
        print(f"  {stratum:18s} {n}")
    print(f"wrote: {path.resolve()}")
    return 0


# ----------------------------------------------------------------- run.py train
def cmd_train(args) -> int:
    from src import train

    cfg = load_config(
        args.config,
        use_augmentation=args.use_augmentation,
        use_mixup=args.use_mixup,
    )
    summary = train.run(
        cfg,
        run_id=args.run_id,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_fraction=args.val_fraction,
        max_steps_per_epoch=args.max_steps,
        cache_dir=Path(args.cache) if args.cache else None,
        device=args.device,
        resume=not args.no_resume,
    )
    ckpt = Path(summary["checkpoint_dir"])
    print(f"\nbest mean_dice: {summary['best_mean_dice']}")
    print(f"validation patients ({len(summary['val_subjects'])}): "
          f"{', '.join(summary['val_subjects'])}")
    print(f"wrote: {(ckpt / 'best.pt').resolve()}")
    print(f"wrote: {(ckpt / 'last.pt').resolve()}")
    print(f"wrote: {(ckpt / 'history.json').resolve()}")
    return 0


# --------------------------------------------------------------- run.py preview
def cmd_preview(args) -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from src.dataset import SliceDataset

    cfg = load_config(args.config, use_augmentation=True)
    out_path = Path(args.out).resolve()

    with tempfile.TemporaryDirectory() as tmp:
        if args.dummy_data:
            from src.dummy import make_dummy_cache

            subjects = [f"DUMMY-{i:04d}" for i in range(max(2, args.n // 2))]
            cache_dir = Path(tmp)
            make_dummy_cache(cache_dir, subjects, cfg, volume_shape=(24, 24, 16), seed=0)
            entries = []
            for i in range(args.n):
                sid = subjects[i % len(subjects)]
                plane = cfg.planes[i % len(cfg.planes)]
                entries.append({
                    "hospital": "hospitalA", "source_subject_id": sid, "plane": plane,
                    "slice_index": 8 + (i % 4), "is_augmented": True, "aug_seed": 100 + i,
                })
        else:
            from src.plan import build_plans

            cache_dir = Path(args.cache) if args.cache else cfg.resolve("cache_2d")
            plans = build_plans(cfg, cache_dir)
            pool = plans["hospitalA"]["entries"]
            step = max(1, len(pool) // args.n)
            entries = [dict(pool[i * step], is_augmented=True, aug_seed=100 + i)
                       for i in range(args.n)]

        plain_cfg = load_config(args.config, use_augmentation=False)
        original = SliceDataset(entries, plain_cfg, cache_dir=cache_dir)
        augmented = SliceDataset(entries, cfg, cache_dir=cache_dir)

        n = len(entries)
        fig, axes = plt.subplots(2, n, figsize=(2.2 * n, 5.0), squeeze=False)
        for col in range(n):
            for row, ds, title in ((0, original, "original"), (1, augmented, "augmented")):
                sample = ds[col]
                image = sample["image"][0]
                label = sample["label"][0]
                ax = axes[row][col]
                ax.imshow(image, cmap="gray")
                ax.imshow(np.ma.masked_where(label == 0, label),
                          cmap="autumn", alpha=0.6, vmin=0, vmax=4)
                ax.set_axis_off()
                if col == 0:
                    ax.set_title(title, loc="left", fontsize=9)
            entry = entries[col]
            axes[0][col].set_title(f"{entry['plane']}[{entry['slice_index']}]", fontsize=8)

        fig.suptitle(f"augmentation preview — use_augmentation: false (top) vs true (bottom)",
                     fontsize=10)
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=110)
        plt.close(fig)

    print(f"previewed {len(entries)} slices (top row unaugmented, bottom row augmented)")
    print(f"wrote: {out_path}")
    return 0


# ------------------------------------------------------------------ run.py eval
def cmd_eval(args) -> int:
    from src import evaluate

    cfg = load_config(
        args.config,
        use_augmentation=args.use_augmentation,
        use_federation=args.use_federation,
        use_domain_adaptation=args.use_domain_adaptation,
    )
    if args.eval_plane:
        cfg.eval["plane"] = args.eval_plane

    with tempfile.TemporaryDirectory() as tmp:
        evaluate.run(
            cfg,
            experiment_name=args.experiment_name,
            checkpoint=args.checkpoint,
            dummy_checkpoint=args.dummy_checkpoint,
            dummy_data=args.dummy_data,
            dummy_n=args.dummy_n,
            cache_dir=Path(args.cache) if args.cache else None,
            csv_path=args.out_csv,
            device=args.device,
            tmp_dir=tmp,
        )
    return 0


# ------------------------------------------------------------------------ CLI
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run.py",
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="commands: " + " | ".join(COMMANDS),
    )
    parser.add_argument("--config", default=None, help="path to config.yaml")
    sub = parser.add_subparsers(dest="command", metavar="{" + ",".join(COMMANDS) + "}")

    p_test = sub.add_parser("test", help="run the dummy suite (no data, no model)")
    p_test.add_argument("-v", "--verbose", action="store_true")
    p_test.set_defaults(func=cmd_test)

    p_index = sub.add_parser("index", help="scan the slice cache, record tumour/ET presence")
    p_index.add_argument("--cache", default=None, help="override paths.cache_2d")
    p_index.add_argument("--limit", type=int, default=None, help="only the first N subjects")
    p_index.add_argument("--planes", default=None, help="comma-separated, default axial,coronal")
    p_index.add_argument("--update", action="store_true",
                         help="keep already-indexed subjects, scan only new ones "
                              "(use after each upload batch)")
    p_index.set_defaults(func=cmd_index)

    p_plan = sub.add_parser("plan", help="build the balanced per-hospital slice plan")
    p_plan.add_argument("--report", action="store_true", help="print the per-hospital table")
    p_plan.add_argument("--cache", default=None, help="override paths.cache_2d")
    p_plan.add_argument("--out", default=None, help="override paths.plans_dir")
    p_plan.set_defaults(func=cmd_plan)

    p_pack = sub.add_parser("pack", help="shrink the cache to what is actually used")
    p_pack.add_argument("--out", required=True, help="destination directory to upload")
    p_pack.add_argument("--cache", default=None, help="override paths.cache_2d (the source)")
    p_pack.add_argument("--full-precision", action="store_true",
                        help="keep float32 instead of halving to float16")
    p_pack.add_argument("--no-heldout", action="store_true",
                        help="training slices only; cannot run eval remotely")
    p_pack.set_defaults(func=cmd_pack)

    p_type = sub.add_parser("tumor-type", help="classify subjects: DMG-like vs astrocytoma-like proxy")
    p_type.add_argument("--cache", default=None, help="override paths.cache_2d")
    p_type.add_argument("--plane", default="axial", choices=["axial", "coronal"])
    p_type.set_defaults(func=cmd_tumor_type)

    p_train = sub.add_parser("train", help="train the segmentation model, save checkpoints")
    p_train.add_argument("--run-id", default="run1", help="checkpoints/<run_id>/")
    p_train.add_argument("--epochs", type=int, default=5)
    p_train.add_argument("--batch-size", type=int, default=8)
    p_train.add_argument("--lr", type=float, default=1e-3)
    p_train.add_argument("--val-fraction", type=float, default=0.25,
                         help="fraction of training PATIENTS held out for validation")
    p_train.add_argument("--max-steps", type=int, default=None,
                         help="cap steps per epoch (short runs on CPU)")
    p_train.add_argument("--cache", default=None, help="override paths.cache_2d")
    p_train.add_argument("--device", default="cpu")
    p_train.add_argument("--no-resume", action="store_true",
                         help="ignore an existing checkpoint and start fresh")
    for flag in ("use-augmentation", "use-mixup"):
        dest = flag.replace("-", "_")
        p_train.add_argument(f"--{flag}", dest=dest, action="store_true", default=None)
        p_train.add_argument(f"--no-{flag}", dest=dest, action="store_false", default=None)
    p_train.set_defaults(func=cmd_train)

    p_prev = sub.add_parser("preview", help="save a PNG grid of augmented slices")
    p_prev.add_argument("--n", type=int, default=8, help="number of slices")
    p_prev.add_argument("--out", default="preview.png")
    p_prev.add_argument("--cache", default=None, help="override paths.cache_2d")
    p_prev.add_argument("--dummy-data", action="store_true", help="use a synthetic cache")
    p_prev.set_defaults(func=cmd_preview)

    p_eval = sub.add_parser("eval", help="score a checkpoint, append one CSV row")
    p_eval.add_argument("--experiment-name", required=True, help="label for this CSV row")
    p_eval.add_argument("--checkpoint", default=None, help="trained checkpoint to evaluate")
    p_eval.add_argument("--dummy-checkpoint", action="store_true",
                        help="evaluate a randomly-initialised DummySegNet2D instead")
    p_eval.add_argument("--dummy-data", action="store_true",
                        help="use a synthetic cache instead of the real one")
    p_eval.add_argument("--dummy-n", type=int, default=4, help="number of dummy subjects")
    p_eval.add_argument("--cache", default=None, help="override paths.cache_2d")
    p_eval.add_argument("--out-csv", default=None, help="override paths.results_csv")
    p_eval.add_argument("--eval-plane", default=None, choices=["axial", "coronal", "both"],
                        help="override eval.plane")
    p_eval.add_argument("--device", default="cpu")
    for flag in ("use-augmentation", "use-federation", "use-domain-adaptation"):
        dest = flag.replace("-", "_")
        p_eval.add_argument(f"--{flag}", dest=dest, action="store_true", default=None)
        p_eval.add_argument(f"--no-{flag}", dest=dest, action="store_false", default=None)
    p_eval.set_defaults(func=cmd_eval)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if not getattr(args, "command", None):
        parser.print_help()
        return 0

    if args.command == "eval" and not args.checkpoint and not args.dummy_checkpoint:
        parser.error("eval requires --checkpoint <path> or --dummy-checkpoint")

    try:
        return args.func(args)
    except (FileNotFoundError, ValueError, AssertionError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
