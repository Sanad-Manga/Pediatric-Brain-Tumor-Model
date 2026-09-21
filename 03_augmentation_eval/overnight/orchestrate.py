#!/usr/bin/env python
"""Unattended overnight training chain for NeuroPeds AI.

Runs a ladder of increasingly large models back to back. Each stage trains until
it either plateaus (no `min_region` gain for `patience` epochs) or runs out of
its slice of the clock; the next stage then starts automatically. Nothing here
waits for a human.

Two hard rules, because this runs while nobody is watching:

* **Never train on CPU.** A silent CPU fallback already cost this project a
  night: `run.py train` defaults to ``--device cpu``, so a missing CUDA wheel
  looks exactly like a slow GPU. Preflight refuses to start unless the
  interpreter, the CUDA build and the GPU all check out.
* **Never overrun the deadline.** The GPU has to be free when its owner sits
  down. Every stage carries a wall-clock deadline, and the finalisation step
  gets its time reserved up front rather than whatever happens to be left.

The shipped exhibition checkpoint is never overwritten in place. It is backed
up, and a new model replaces it only after beating it on the held-out split.
"""

from __future__ import annotations

import ctypes
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ------------------------------------------------------------- sleep guard
# 2026-08-09 incident: the machine suspended itself ~2 min into an unattended
# run (Sleep Reason: "Application API", nothing here requested it) and stayed
# asleep for 12h38m. Two things broke as a result, both explained by the same
# cause: training only *looked* stuck (the whole box, not the GPU, was
# suspended), and run_with_timeout's `proc.wait(timeout=...)` never fired
# because Windows' wait primitive is keyed to a tick count that does not
# advance while suspended -- from the timeout's point of view almost no time
# had passed. SetThreadExecutionState(ES_SYSTEM_REQUIRED) tells Windows this
# process needs the system to stay awake; it is scoped to the calling thread
# and automatically reverts when the process exits, so it never touches the
# user's actual power plan settings.
_ES_CONTINUOUS = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001


def prevent_sleep() -> None:
    ctypes.windll.kernel32.SetThreadExecutionState(_ES_CONTINUOUS | _ES_SYSTEM_REQUIRED)
    log("sleep guard ON: SetThreadExecutionState(ES_SYSTEM_REQUIRED) -- "
        "system will not suspend while this process is alive")


def allow_sleep() -> None:
    ctypes.windll.kernel32.SetThreadExecutionState(_ES_CONTINUOUS)
    log("sleep guard OFF: normal sleep behaviour restored")

# ----------------------------------------------------------------- environment
# Absolute path, never bare `python`: the machine has three interpreters and two
# of them carry a CPU-only torch.
PY = Path(r"C:\Users\ahmed\neuropeds_env\Scripts\python.exe")
# The training venv deliberately carries ONLY torch/monai/pyyaml -- Streamlit,
# pandas, plotly and scipy are not installed there and never should be, so a
# training-time `pip install` cannot silently change what torch build the venv
# resolves. The demo app needs its own interpreter with its own deps.
APP_PY = Path(r"C:\Users\ahmed\AppData\Local\Programs\Python\Python314\python.exe")
REPO = Path(r"C:\Users\ahmed\Pediatric-Brain-Tumor-Model")
SEC03 = REPO / "03_augmentation_eval"
SEC05 = REPO / "05_frontend_demo"
CACHE = Path(r"D:\NeuroPeds AI\pack_out_15k")
WORK = Path(r"C:\Users\ahmed\neuropeds_overnight")

#: GPU must be free by this time.
DEADLINE = datetime(2026, 8, 4, 14, 30, 0).timestamp()
#: Held out of the training window for eval + ROC rebuild + app restart.
FINALIZE_RESERVE = 3600
#: A fresh architecture below this many seconds cannot converge far enough to be
#: worth the GPU time, so the chain stops rather than starting one.
MIN_STAGE = 8000
#: Conservative seconds/epoch guess, used only to size the cosine LR horizon.
#: Calibrated from tonight's first attempt: s1@batch8 ran ~2600-2700s/epoch once
#: past the initial ramp. Halved batch size roughly doubles step count but each
#: step is cheaper and safer on VRAM, so wall time per epoch is assumed similar
#: rather than 2x -- biased slow on purpose either way, see plan_epochs().
EST_EPOCH = {"s1b_w64_d3": 2800, "s2b_w96_d3": 3200, "s3b_w64_d4": 3000}

NUM_WORKERS = 4

STAGES = [
    # New run_ids (not s1_w64_d3/s2_w96_d3/s3_w64_d4): tonight's first attempt
    # already wrote real results under those names -- s1_w64_d3 in particular
    # (epoch 6, min_region 0.5231, held-out mean 0.6687, held-out ET 0.5571,
    # beating the exhibition checkpoint's 0.4500 on ET) is a genuine candidate,
    # not scratch work, and --no-resume in run_stage() would overwrite it if
    # reused. main() loads it back in as a prior result before the stage loop
    # so it stays in the running against whatever these produce.
    #
    # batch_size dropped to 4 for the two stages that actually OOM'd tonight
    # (w96/d3 wedged in a fragmentation-driven allocator retry loop after ~50
    # min; w64/d4 OOM'd during a live smoke test earlier). w64/d3 stays at 8 --
    # it trained stably at that size for 6 epochs before the unrelated NaN.
    {"run_id": "s1b_w64_d3", "width": 64, "depth": 3, "patience": 10, "batch_size": 8},
    {"run_id": "s2b_w96_d3", "width": 96, "depth": 3, "patience": 8, "batch_size": 4},
    {"run_id": "s3b_w64_d4", "width": 64, "depth": 4, "patience": 8, "batch_size": 4},
]

LOG = WORK / "overnight.log"


def log(msg: str) -> None:
    line = f"[{datetime.now():%H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def remaining() -> float:
    return DEADLINE - time.time()


def hhmm(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"


def run_with_timeout(cmd: list[str], cwd: Path, timeout: float, log_path: Path,
                     env: dict | None = None) -> int:
    """`subprocess.run`, but a hang cannot hold the GPU past the deadline.

    `--deadline-unix` only stops train.py *between* epochs; a stuck DataLoader
    worker, a driver reset, or an I/O stall on the D: drive blocks inside one
    and nothing before this function noticed. A plain `.wait(timeout=...)`
    would raise but leave the process tree alive -- on Windows the parent
    `python.exe` is not the CUDA context, its child DataLoader workers are, so
    only `taskkill /T` (kill the whole tree) actually frees the GPU.
    """
    # The deadline is checked against the WALL clock (time.time()) in short
    # slices instead of one long `proc.wait(timeout=...)`. Windows' wait
    # primitive does not count time spent suspended, so a single long wait
    # silently stretches by however long the machine slept (12h38m on
    # 2026-08-09); the wall clock jumps forward on resume, so this fires the
    # moment the machine wakes past the deadline. prevent_sleep() is the
    # first line of defence; this is the backstop if it is ever bypassed.
    wall_deadline = time.time() + timeout
    with open(log_path, "a", encoding="utf-8") as fh:
        proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=fh,
                                stderr=subprocess.STDOUT, env=env)
        while True:
            left = wall_deadline - time.time()
            if left <= 0:
                break
            try:
                proc.wait(timeout=min(30.0, left))
                return proc.returncode
            except subprocess.TimeoutExpired:
                continue
        fh.write(f"\n[orchestrator] TIMEOUT after {timeout:.0f}s -- killing pid "
                 f"{proc.pid} and its process tree\n")
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                       capture_output=True)
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            fh.write("[orchestrator] process did not die within 60s of taskkill\n")
        return -9


# ------------------------------------------------------------------- preflight
def preflight() -> None:
    """Refuse to start on a broken environment. Every check is fatal.

    These are exactly the failures that produced a wasted night: the wrong
    interpreter, a CPU-only torch, a cache path that does not exist, and an
    unindexed cache that sends the loader scanning 55k files per epoch.
    """
    # Every round script calls preflight() first, so guarding here covers all
    # of them without each one having to remember. Held for the life of the
    # calling thread; Windows drops it automatically when the process exits.
    prevent_sleep()
    if not PY.exists():
        raise SystemExit(f"FATAL: training interpreter missing: {PY}")
    if not APP_PY.exists():
        raise SystemExit(f"FATAL: app interpreter missing: {APP_PY}")
    if shutil.which("taskkill") is None:
        raise SystemExit("FATAL: taskkill not on PATH -- the timeout/kill path "
                          "in run_with_timeout cannot free a hung GPU process")

    probe = subprocess.run(
        [str(PY), "-c",
         "import torch,json;"
         "print(json.dumps({'v':torch.__version__,"
         "'cuda':torch.cuda.is_available(),"
         "'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}))"],
        capture_output=True, text=True, timeout=120,
    )
    if probe.returncode != 0:
        raise SystemExit(f"FATAL: torch probe failed:\n{probe.stderr}")
    info = json.loads(probe.stdout.strip().splitlines()[-1])
    if not info["cuda"]:
        raise SystemExit(
            f"FATAL: torch {info['v']} reports CUDA unavailable. Refusing to "
            f"train on CPU. Reinstall with the cu126 index."
        )
    log(f"preflight: torch {info['v']} | {info['gpu']}")

    app_probe = subprocess.run(
        [str(APP_PY), "-c",
         "import streamlit, pandas, plotly, scipy, torch"],
        capture_output=True, text=True, timeout=60,
    )
    if app_probe.returncode != 0:
        raise SystemExit(
            f"FATAL: app interpreter {APP_PY} is missing a dependency "
            f"(streamlit needs streamlit+pandas+plotly, build_metrics_cache "
            f"needs torch+scipy):\n{app_probe.stderr}"
        )
    log(f"preflight: app interpreter has streamlit/pandas/plotly/scipy/torch")

    for path, what in ((CACHE, "slice cache"),
                       (CACHE / "index.json", "cache index"),
                       (SEC03 / "run.py", "trainer")):
        if not path.exists():
            raise SystemExit(f"FATAL: {what} missing: {path}")
    log(f"preflight: cache + index present at {CACHE}")

    free = shutil.disk_usage(SEC03.anchor).free / 2**30
    if free < 5:
        raise SystemExit(f"FATAL: only {free:.1f} GB free; checkpoints need room")
    log(f"preflight: {free:.0f} GB free on {SEC03.anchor}")


# ----------------------------------------------------------------- stage config
def write_stage_config(stage: dict) -> Path:
    """A per-stage copy of config.yaml with this stage's geometry patched in.

    `load_config`'s keyword overrides only reach top-level scalars, and run.py
    exposes no width/depth flag, so the geometry has to travel by file.

    The copy MUST live beside the original: `load_config` resolves every
    relative path in `paths:` against the directory holding the config file, so
    a copy anywhere else sends `manifests: "../00_shared/manifests"` somewhere
    that does not exist and the stage dies on startup.
    """
    text = (SEC03 / "config.yaml").read_text(encoding="utf-8")
    out, in_model = [], False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("model:"):
            in_model = True
            out.append(line)
            continue
        if in_model:
            if stripped.startswith("width:"):
                out.append(f"  width: {stage['width']}")
                continue
            if stripped.startswith("depth:"):
                out.append(f"  depth: {stage['depth']}")
                continue
            # any non-indented, non-comment line ends the model block
            if stripped and not line.startswith((" ", "\t")) and not stripped.startswith("#"):
                in_model = False
        out.append(line)

    path = SEC03 / f"config.overnight_{stage['run_id']}.yaml"
    path.write_text("\n".join(out) + "\n", encoding="utf-8")

    # Verify rather than trust the rewrite: a config that silently kept the old
    # width would make the whole stage a duplicate of the previous one.
    import yaml
    got = yaml.safe_load(path.read_text(encoding="utf-8"))["model"]
    if int(got["width"]) != stage["width"] or int(got["depth"]) != stage["depth"]:
        raise SystemExit(f"FATAL: stage config not patched: {got}")
    return path


def plan_epochs(stage: dict, budget: float) -> int:
    """Epoch count to hand the trainer, which is also the cosine LR horizon.

    `T_max` is set from this number, so it is not a cap that can be set high and
    forgotten: overshoot it and the run stops with the LR still near 1e-3, which
    is the oscillation the schedule exists to remove. The estimate is therefore
    biased slow -- finishing the budget early is harmless (the next stage simply
    starts sooner, with the LR fully annealed), while over-planning leaves the
    model unsettled.
    """
    return max(3, int(budget // EST_EPOCH[stage["run_id"]]))


def load_prior_result(run_id: str) -> dict | None:
    """Read back a stage's history.json from an earlier orchestrator run.

    No subprocess, no training -- just re-shapes an existing checkpoint's
    recorded history into the same dict run_stage() returns, so a genuinely
    good result from a prior attempt (e.g. tonight's s1_w64_d3, epoch 6,
    stopped by an unrelated NaN rather than a real plateau) stays in the
    running against fresh attempts instead of being silently discarded because
    this process happens to be starting over.
    """
    hist_path = SEC03 / "checkpoints" / run_id / "history.json"
    if not hist_path.exists():
        return None
    hist = json.loads(hist_path.read_text(encoding="utf-8"))
    done = hist.get("history", [])
    if not done:
        return None
    best = max(r.get("min_region_dice", 0.0) for r in done)
    best_mean = max(r.get("mean_dice", 0.0) for r in done)
    hist["_run_id"] = run_id
    hist["_best_min_region"] = best
    hist["_best_mean"] = best_mean
    log(f"    carrying forward prior result {run_id}: {len(done)} epochs | "
        f"best min_region {best:.4f} | best mean {best_mean:.4f}")
    return hist


# --------------------------------------------------------------------- running
def run_stage(stage: dict, budget: float) -> dict | None:
    """Train one stage to its own deadline. Returns its history.json, or None."""
    run_id = stage["run_id"]
    stage_deadline = time.time() + budget
    epochs = plan_epochs(stage, budget)
    cfg_path = write_stage_config(stage)
    stage_log = WORK / f"{run_id}.log"

    log(f"--- stage {run_id}: width={stage['width']} depth={stage['depth']} "
        f"| {epochs} epochs planned | budget {hhmm(budget)}")

    cmd = [
        str(PY), "run.py",
        "--config", str(cfg_path),          # global flag: must precede the subcommand
        "train",
        "--run-id", run_id,
        "--no-resume",                      # new geometry cannot load an old checkpoint
        "--epochs", str(epochs),
        "--batch-size", str(stage["batch_size"]),
        "--num-workers", str(NUM_WORKERS),
        "--device", "cuda",
        "--patience", str(stage["patience"]),
        "--deadline-unix", f"{stage_deadline:.0f}",
        "--cache", str(CACHE),
    ]

    import os
    full_env = {
        **os.environ,
        # Fragmentation, not true exhaustion, produced the 3.8 GB allocator OOM
        # during validation; expandable segments let those blocks be reused.
        # (No-op on Windows today, per the smoke test -- harmless to set.)
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }

    # `--deadline-unix` polices whole epochs; this polices the process itself,
    # in case one epoch never finishes. 20 min of slack covers one slow epoch
    # plus validation before the tree is killed outright.
    hard_timeout = budget + 1200
    stage_log.write_text("", encoding="utf-8")
    code = run_with_timeout(cmd, SEC03, hard_timeout, stage_log, env=full_env)

    hist_path = SEC03 / "checkpoints" / run_id / "history.json"
    if code == -9:
        log(f"    stage {run_id} HUNG and was killed after {hhmm(hard_timeout)} "
            f"(see {stage_log.name})")
    elif code != 0:
        log(f"    stage {run_id} exited {code} (see {stage_log.name})")
    if not hist_path.exists():
        log(f"    stage {run_id} produced no history; skipping")
        return None

    hist = json.loads(hist_path.read_text(encoding="utf-8"))
    done = hist.get("history", [])
    if not done:
        return None
    best = max(r.get("min_region_dice", 0.0) for r in done)
    best_mean = max(r.get("mean_dice", 0.0) for r in done)
    log(f"    stage {run_id}: {len(done)} epochs | best min_region {best:.4f} "
        f"| best mean {best_mean:.4f} | stop: {hist.get('stop_reason') or 'budget'}")
    hist["_run_id"] = run_id
    hist["_best_min_region"] = best
    hist["_best_mean"] = best_mean
    return hist


def held_out_eval(checkpoint: Path, name: str, eval_plane: str = "axial",
                  timeout: float = 1800) -> dict | None:
    """Score a checkpoint on the held-out manifest.

    ``eval_plane="both"`` averages axial+coronal softmax before argmax --
    ~1-3 Dice points more accurate per the project's own docs, at roughly 2x
    the cost. Caller must budget accordingly; the default timeout only fits
    axial (measured 5m45s/82 subjects), so pass a larger one for "both".
    """
    if not checkpoint.exists():
        log(f"    eval {name}: no checkpoint at {checkpoint}")
        return None
    csv_out = SEC03 / "results" / "overnight_eval.csv"
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(PY), "run.py", "eval",
        "--experiment-name", name,
        "--checkpoint", str(checkpoint),
        "--cache", str(CACHE),
        "--eval-plane", eval_plane,
        "--device", "cuda",
        "--out-csv", str(csv_out),
    ]
    log(f"    eval {name} (plane={eval_plane}) ...")
    eval_log = WORK / f"eval_{name}.log"
    eval_log.write_text("", encoding="utf-8")
    code = run_with_timeout(cmd, SEC03, timeout, eval_log)
    if code != 0:
        log(f"    eval {name} failed (exit {code}){' -- TIMED OUT' if code == -9 else ''}")
        return None
    try:
        import csv as _csv
        with open(csv_out, newline="", encoding="utf-8") as fh:
            rows = [r for r in _csv.DictReader(fh) if r.get("experiment_name") == name]
        if not rows:
            return None
        row = rows[-1]
        dice = {k: float(row[k]) for k in row if k.startswith("dice_") and row[k]}
        mean = sum(dice.values()) / len(dice) if dice else 0.0
        log(f"    eval {name}: " + "  ".join(f"{k}={v:.4f}" for k, v in dice.items())
            + f"  mean={mean:.4f}")
        return {"name": name, "dice": dice, "mean": mean, "checkpoint": str(checkpoint)}
    except Exception as exc:                      # noqa: BLE001 - report, never abort
        log(f"    eval {name}: could not parse CSV ({exc})")
        return None


def restart_streamlit() -> None:
    """Leave the demo serving, so the app is up when its owner returns.

    Verifies the port actually answers rather than assuming the launch
    succeeded -- APP_PY, not the training venv, is what has streamlit/pandas/
    plotly/scipy installed.
    """
    subprocess.run(["powershell", "-NoProfile", "-Command",
                    "Get-CimInstance Win32_Process | "
                    "Where-Object { $_.CommandLine -like '*streamlit*run*Home.py*' } | "
                    "ForEach-Object { Stop-Process -Id $_.ProcessId -Force }"],
                   capture_output=True)
    subprocess.Popen(
        ["powershell", "-NoProfile", "-Command",
         f"Start-Process -WindowStyle Minimized '{APP_PY}' "
         f"-ArgumentList '-m','streamlit','run','Home.py','--server.headless','true' "
         f"-WorkingDirectory '{SEC05}'"],
    )
    import urllib.request
    for _ in range(12):                       # ~60s for Streamlit to bind
        time.sleep(5)
        try:
            urllib.request.urlopen("http://localhost:8501", timeout=3)
            log("    streamlit confirmed serving on http://localhost:8501")
            return
        except Exception:                      # noqa: BLE001 - keep polling
            continue
    log("    WARNING: streamlit did not answer on :8501 after restart -- "
        "check manually; do not trust this line to mean the app is up")


# ---------------------------------------------------------------------- report
def write_report(stages: list[dict], evals: list[dict], promoted: str | None) -> None:
    lines = [
        "# Overnight run report",
        "",
        f"Finished {datetime.now():%Y-%m-%d %H:%M}. GPU is free.",
        "",
        "## Stages",
        "",
        "| stage | width x depth | epochs | best min-region | best mean (val) | stopped |",
        "|---|---|---|---|---|---|",
    ]
    for h in stages:
        lines.append(
            f"| {h['_run_id']} | {h.get('width')} x {h.get('depth')} | "
            f"{len(h.get('history', []))} | {h['_best_min_region']:.4f} | "
            f"{h['_best_mean']:.4f} | {h.get('stop_reason') or 'budget'} |"
        )
    lines += ["", "## Held-out evaluation", ""]
    if evals:
        regions = sorted({k for e in evals for k in e["dice"]})
        lines.append("| model | " + " | ".join(regions) + " | mean |")
        lines.append("|---" * (len(regions) + 2) + "|")
        for e in evals:
            lines.append(f"| {e['name']} | "
                         + " | ".join(f"{e['dice'].get(r, float('nan')):.4f}" for r in regions)
                         + f" | {e['mean']:.4f} |")
    else:
        lines.append("_No held-out evaluation completed._")
    lines += ["", "## Result", ""]
    lines.append(f"**Promoted to the app:** {promoted}" if promoted else
                 "**Nothing promoted** - the shipped exhibition checkpoint still "
                 "wins on held-out data and remains in place.")
    lines += ["", "Backup of the original: `checkpoints/overnight_run/best.pt.exhibition-backup`",
              "", "Per-stage logs are in this folder.", ""]
    (WORK / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    log(f"    wrote {WORK / 'REPORT.md'}")


# ------------------------------------------------------------------------ main
def main() -> int:
    WORK.mkdir(parents=True, exist_ok=True)
    log("=" * 68)
    log(f"overnight chain starting | deadline {datetime.fromtimestamp(DEADLINE):%H:%M} "
        f"({hhmm(remaining())} away)")
    preflight()

    completed: list[dict] = []
    prior = load_prior_result("s1_w64_d3")
    if prior:
        completed.append(prior)

    for stage in STAGES:
        budget = remaining() - FINALIZE_RESERVE
        if budget < MIN_STAGE:
            log(f"--- stopping the chain: {hhmm(budget)} left, below the "
                f"{hhmm(MIN_STAGE)} a fresh architecture needs")
            break
        hist = run_stage(stage, budget)
        if hist:
            completed.append(hist)
        if remaining() - FINALIZE_RESERVE < MIN_STAGE:
            log("--- clock spent; moving to finalisation")
            break

    # ---------------------------------------------------------- finalisation
    log(f"--- finalising ({hhmm(remaining())} to deadline)")
    evals: list[dict] = []
    promoted = None
    shipped = SEC03 / "checkpoints" / "overnight_run" / "best.pt"
    backup = shipped.with_suffix(".pt.exhibition-backup")

    if completed:
        winner = max(completed, key=lambda h: h["_best_min_region"])
        log(f"    best stage by validation min-region: {winner['_run_id']}")
        cand = SEC03 / "checkpoints" / winner["_run_id"] / "best.pt"

        new = held_out_eval(cand, winner["_run_id"])
        if new:
            evals.append(new)

        # `old is None` must mean ONLY "no shipped baseline exists" -- never
        # "the baseline eval failed". held_out_eval() also returns None on a
        # non-zero exit, a timeout, an empty CSV, or a parse error, none of
        # which say the baseline lost. Conflating them promotes an unproven
        # candidate over a real 0.682 model on the strength of eval.py having
        # a bad day. `baseline_ok` keeps the two cases distinguishable.
        old = None
        baseline_ok = not shipped.exists()      # no file -> nothing to beat
        if shipped.exists():
            old = held_out_eval(shipped, "exhibition_baseline")
            if old:
                evals.append(old)
                baseline_ok = True
            else:
                log("    baseline eval FAILED (not \"baseline lost\") -- "
                    "cannot compare, so nothing will be promoted this run")

        if new and baseline_ok and (old is None or new["mean"] > old["mean"]):
            if shipped.exists() and not backup.exists():
                shutil.copy2(shipped, backup)
                log(f"    backed up the exhibition checkpoint -> {backup.name}")
            shipped.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cand, shipped)
            promoted = f"{winner['_run_id']} (held-out mean {new['mean']:.4f})"
            log(f"    PROMOTED {winner['_run_id']} to the app checkpoint")

            rebuild_log = WORK / "roc_rebuild.log"
            rebuild_log.write_text("", encoding="utf-8")
            # APP_PY, not PY: build_metrics_cache imports inference.py, which
            # needs torch, and its Dice/HD95 path needs scipy -- neither is in
            # the training venv on purpose (see APP_PY's comment above).
            rebuild_code = run_with_timeout(
                [str(APP_PY), "-m", "utils.build_metrics_cache", "--cache", str(CACHE)],
                SEC05, 900, rebuild_log)
            if rebuild_code == 0:
                log("    roc_cache.json rebuilt")
            else:
                # roc_cache.json records which checkpoint produced it, and the
                # Dashboard refuses to show numbers belonging to a different
                # model. A promoted checkpoint with a stale cache is therefore a
                # visibly broken app -- roll the promotion back so the demo is
                # coherent, and say so in the report.
                log(f"    roc_cache rebuild FAILED (exit {rebuild_code}); rolling back")
                if backup.exists():
                    shutil.copy2(backup, shipped)
                    promoted = None
                    log("    restored the exhibition checkpoint; app left consistent")
        else:
            log("    new model did not beat the exhibition checkpoint; leaving it in place")
    else:
        log("    no stage completed; nothing to evaluate")

    write_report(completed, evals, promoted)
    restart_streamlit()
    log(f"DONE. GPU free at {datetime.now():%H:%M}.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception as exc:                      # noqa: BLE001
        log(f"FATAL: {type(exc).__name__}: {exc}")
        import traceback
        with open(LOG, "a", encoding="utf-8") as fh:
            traceback.print_exc(file=fh)
        raise
