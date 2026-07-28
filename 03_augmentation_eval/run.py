#!/usr/bin/env python
"""Section 03 — 2D slice augmentation + ablation. The single entry point.

    python run.py test                  verify everything with no data, no model
    python run.py slices --out cache_2d export axial + coronal slices
    python run.py plan --report         build the balanced per-hospital plan
    python run.py preview --n 8         sanity-check augmentation, save a PNG
    python run.py eval --experiment-name baseline --dummy-checkpoint --dummy-data

Anything not reachable through this file does not count as delivered. Every
command states what it wrote and where.
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

COMMANDS = ("test", "slices", "plan", "preview", "eval")


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


# ---------------------------------------------------------------- run.py slices
def cmd_slices(args) -> int:
    from src.slices import export_cache

    cfg = load_config(args.config)
    out_dir = Path(args.out) if args.out else cfg.resolve("cache_2d")
    planes = [p.strip() for p in args.planes.split(",")] if args.planes else None

    summary = export_cache(cfg, out_dir, manifest=args.manifest, limit=args.limit, planes=planes)

    print(f"planes exported: {', '.join(summary['planes'])} (sagittal dropped)")
    print(f"subjects written: {summary['subjects_written']} of {summary['subjects_requested']}")
    if summary["subjects_skipped"]:
        print(f"subjects skipped: {len(summary['subjects_skipped'])} "
              f"({', '.join(summary['subjects_skipped'][:5])}...)")
    print(f"files written: {len(summary['files_written'])}")
    print(f"wrote: {Path(summary['out_dir']).resolve()}")
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

    p_slices = sub.add_parser("slices", help="export axial + coronal slices from NIfTI")
    p_slices.add_argument("--out", default=None, help="output cache directory")
    p_slices.add_argument("--limit", type=int, default=None, help="only the first N subjects")
    p_slices.add_argument("--manifest", default="all",
                          help="hospitalA | hospitalB | heldout | all")
    p_slices.add_argument("--planes", default=None, help="comma-separated, default axial,coronal")
    p_slices.set_defaults(func=cmd_slices)

    p_plan = sub.add_parser("plan", help="build the balanced per-hospital slice plan")
    p_plan.add_argument("--report", action="store_true", help="print the per-hospital table")
    p_plan.add_argument("--cache", default=None, help="override paths.cache_2d")
    p_plan.add_argument("--out", default=None, help="override paths.plans_dir")
    p_plan.set_defaults(func=cmd_plan)

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
