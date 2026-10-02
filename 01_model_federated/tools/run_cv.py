#!/usr/bin/env python
"""Run five-fold training and validation using run.py and eval_heldout_3d.py."""
from __future__ import annotations

import argparse
import json
import os
import shlex
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np

SEC01 = Path(__file__).resolve().parent.parent
REPO_ROOT = SEC01.parent
DEFAULT_FOLDS_DIR = REPO_ROOT / "00_shared" / "manifests" / "cv5"
DEFAULT_OUT_DIR = SEC01 / "cv_runs"
REGIONS = ("ET", "NC", "WT")


def _extract_value(args: list[str], option: str, default: str | None = None) -> str | None:
    for index, token in enumerate(args):
        if token == option:
            return args[index + 1] if index + 1 < len(args) else default
        if token.startswith(option + "="):
            return token.split("=", 1)[1]
    return default


def _without_overridden_training_paths(args: list[str]) -> list[str]:
    removed = {"--manifest", "--run-id", "--checkpoint-dir"}
    result = []
    skip_value = False
    for token in args:
        if skip_value:
            skip_value = False
            continue
        if token in removed:
            skip_value = True
        elif not any(token.startswith(option + "=") for option in removed):
            result.append(token)
    return result


def _make_dummy_eval_cache(cache_dir: Path, patient_ids: list[str]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    shape = (96, 96, 96)
    for index, patient_id in enumerate(patient_ids):
        output = cache_dir / f"{patient_id}.npz"
        if output.exists():
            continue
        seg = np.zeros(shape, dtype=np.uint8)
        seg[24:72, 24:72, 24:72] = 4
        seg[36:60, 36:60, 36:60] = 2
        if index % 2 == 0:
            seg[44:52, 44:52, 44:52] = 1
        image = np.zeros(shape, dtype=np.float32)
        image[16:80, 16:80, 16:80] = 1.0
        np.savez_compressed(
            output,
            t1c=image,
            t1n=image,
            t2f=image,
            t2w=image,
            seg=seg,
        )


def _latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    checkpoints = []
    for path in checkpoint_dir.glob("epoch_*.pt"):
        try:
            epoch = int(path.stem.removeprefix("epoch_"))
        except ValueError:
            continue
        checkpoints.append((epoch, path))
    return max(checkpoints, default=(0, None), key=lambda item: item[0])[1]


def _command_text(command: list[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(command)
    return shlex.join(command)


def _load_training_args(argv: list[str] | None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run five-fold CV; remaining arguments are passed to run.py unchanged.",
        add_help=True,
    )
    parser.add_argument("--folds-dir", default=str(DEFAULT_FOLDS_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--dry-run", action="store_true")
    cv_args, training_args = parser.parse_known_args(argv)
    if any(arg in {"--folds-dir", "--out-dir", "--dry-run"} for arg in training_args):
        parser.error("CV options may only be specified once")
    return cv_args, _without_overridden_training_paths(training_args)


def _build_commands(
    fold: int,
    folds_dir: Path,
    out_dir: Path,
    cache_dir: Path,
    training_args: list[str],
    checkpoint: Path,
    score_path: Path,
) -> tuple[list[str], list[str]]:
    fold_dir = out_dir / f"fold{fold}"
    train_command = [
        sys.executable,
        str(SEC01 / "run.py"),
        "--manifest", str(folds_dir / f"fold{fold}_train.json"),
        "--run-id", f"cv_fold{fold}",
        "--checkpoint-dir", str(fold_dir / "checkpoints"),
        *training_args,
    ]
    score_command = [
        sys.executable,
        str(SEC01 / "tools" / "eval_heldout_3d.py"),
        "--checkpoint", str(checkpoint),
        "--cache-path", str(cache_dir),
        "--manifest", str(folds_dir / f"fold{fold}_val.json"),
        "--device", "cpu",
        "--out-json", str(score_path),
    ]
    return train_command, score_command


def _write_summary(fold_results: list[dict], out_dir: Path) -> Path:
    summary = {
        "folds": fold_results,
        "summary": {
            region: {
                "mean": statistics.fmean(item["dice"][region] for item in fold_results),
                "std": statistics.pstdev(item["dice"][region] for item in fold_results),
            }
            for region in REGIONS
        },
    }
    path = out_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path}")
    return path


def run_cv(argv: list[str] | None = None) -> int:
    cv_args, training_args = _load_training_args(argv)
    folds_dir = Path(cv_args.folds_dir).resolve()
    out_dir = Path(cv_args.out_dir).resolve()
    data_mode = _extract_value(training_args, "--data-mode", "dummy")
    cache_arg = _extract_value(training_args, "--cache-path")
    epochs = int(_extract_value(training_args, "--epochs", "1"))
    if epochs < 1:
        raise ValueError("--epochs must be at least 1")

    if data_mode == "dummy":
        cache_dir = out_dir / "dummy_eval_cache"
        patient_ids = []
        for fold in range(5):
            val_path = folds_dir / f"fold{fold}_val.json"
            if not val_path.is_file():
                raise FileNotFoundError(f"missing validation manifest: {val_path}")
            patient_ids.extend(json.loads(val_path.read_text(encoding="utf-8")))
        if not cv_args.dry_run:
            _make_dummy_eval_cache(cache_dir, patient_ids)
    else:
        if not cache_arg:
            raise ValueError("--cache-path is required for non-dummy cross-validation")
        cache_dir = Path(cache_arg).resolve()

    out_dir.mkdir(parents=True, exist_ok=True) if not cv_args.dry_run else None
    fold_results = []
    for fold in range(5):
        train_manifest = folds_dir / f"fold{fold}_train.json"
        val_manifest = folds_dir / f"fold{fold}_val.json"
        if not train_manifest.is_file() or not val_manifest.is_file():
            raise FileNotFoundError(f"fold {fold} needs {train_manifest} and {val_manifest}")
        fold_dir = out_dir / f"fold{fold}"
        score_path = fold_dir / "score.json"
        expected_checkpoint = fold_dir / "checkpoints" / f"cv_fold{fold}" / f"epoch_{epochs - 1}.pt"
        train_command, score_command = _build_commands(
            fold, folds_dir, out_dir, cache_dir, training_args, expected_checkpoint, score_path
        )
        if cv_args.dry_run:
            print(f"TRAIN fold {fold}: {_command_text(train_command)}")
            print(f"SCORE fold {fold}: {_command_text(score_command)}")
            continue

        checkpoint = _latest_checkpoint(fold_dir / "checkpoints" / f"cv_fold{fold}")
        if checkpoint and score_path.is_file():
            saved_score = json.loads(score_path.read_text(encoding="utf-8"))
            if Path(saved_score.get("checkpoint", "")) == checkpoint:
                fold_results.append({
                    "fold": fold,
                    "checkpoint": str(checkpoint),
                    "dice": {region: float(saved_score["dice"][f"dice_{region}"]) for region in REGIONS},
                    "n_subjects_scored": int(saved_score["n_subjects_scored"]),
                    "skipped": True,
                })
                print(f"fold {fold}: completed checkpoint and score found; skipping")
                continue

        fold_dir.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        if data_mode == "dummy":
            env["CUDA_VISIBLE_DEVICES"] = ""
            env["OMP_NUM_THREADS"] = "2"
            env["MKL_NUM_THREADS"] = "2"
            env["OPENBLAS_NUM_THREADS"] = "1"
        if checkpoint is None:
            subprocess.run(train_command, env=env, check=True)
            checkpoint = _latest_checkpoint(fold_dir / "checkpoints" / f"cv_fold{fold}")
        if checkpoint is None:
            raise RuntimeError(f"training fold {fold} finished without producing a checkpoint")
        score_command[score_command.index(str(expected_checkpoint))] = str(checkpoint)
        subprocess.run(score_command, env=env, check=True)
        score = json.loads(score_path.read_text(encoding="utf-8"))
        fold_results.append({
            "fold": fold,
            "checkpoint": str(checkpoint),
            "dice": {region: float(score["dice"][f"dice_{region}"]) for region in REGIONS},
            "n_subjects_scored": int(score["n_subjects_scored"]),
            "skipped": False,
        })

    if not cv_args.dry_run:
        if len(fold_results) != 5:
            raise RuntimeError("not all five folds produced a score")
        _write_summary(fold_results, out_dir)
    return 0


def main(argv: list[str] | None = None) -> int:
    return run_cv(argv)


if __name__ == "__main__":
    raise SystemExit(main())
