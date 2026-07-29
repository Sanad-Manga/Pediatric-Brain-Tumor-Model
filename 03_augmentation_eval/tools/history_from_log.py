"""Rebuild `history.json` from a training log.

`train.py` used to write the per-epoch curve only after the whole epoch loop
finished, so a run stopped early (Colab timeout, manual interrupt, preemption)
left no history at all -- the numbers existed only in the console scrollback.
That is fixed going forward, but runs stopped before the fix still need their
curve recovered, and a log is sometimes the only artefact that survives.

Parses lines of the exact shape `train.py` prints::

    epoch 19: loss=0.2247  val ET=0.465  NC=0.753  WT=0.771  mean=0.663  type_acc=0.750  (293s)

and emits the same JSON structure `train.py` writes, so every downstream reader
(the demo's curve page, `loaders.load_training_history`) works unchanged.

Usage::

    python tools/history_from_log.py train.log -o checkpoints/<run_id>/history.json
    python tools/history_from_log.py train.log --run-id overnight_run --stdout
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

#: One printed epoch summary. Region names and the optional trailing fields are
#: matched loosely so a log from a run with the tumour-type head disabled, or
#: with a different region set, still parses.
EPOCH_RE = re.compile(
    r"epoch\s+(?P<epoch>\d+):\s*"
    r"loss=(?P<loss>[-+0-9.eE]+|nan)\s+"
    r"val\s+(?P<regions>.*?)"
    r"(?:\s+type_acc=(?P<type_acc>[0-9.]+))?"
    r"\s*\((?P<seconds>[0-9.]+)s\)"
)
REGION_RE = re.compile(r"(?P<name>[A-Za-z_]+)=(?P<value>[-+0-9.eE]+|nan)")


def parse_log(text: str) -> list[dict]:
    """Every epoch record found in ``text``, ordered by epoch.

    Later occurrences of the same epoch win: a resumed run re-prints epochs it
    redid, and the most recent value is the one that produced the checkpoint.
    """
    records: dict[int, dict] = {}
    for m in EPOCH_RE.finditer(text):
        record = {
            "epoch": int(m.group("epoch")),
            "loss": float(m.group("loss")),
            "seconds": float(m.group("seconds")),
        }
        for r in REGION_RE.finditer(m.group("regions")):
            name, value = r.group("name"), float(r.group("value"))
            key = "mean_dice" if name == "mean" else f"dice_{name}"
            record[key] = value
        if m.group("type_acc") is not None:
            record["type_accuracy"] = float(m.group("type_acc"))
        records[record["epoch"]] = record
    return [records[k] for k in sorted(records)]


def build_summary(history: list[dict], run_id: str, epochs: int | None = None) -> dict:
    """Wrap records in the structure `train.py` writes."""
    dices = [r["mean_dice"] for r in history if "mean_dice" in r]
    return {
        "run_id": run_id,
        "epochs": epochs if epochs is not None else (max(r["epoch"] for r in history) + 1 if history else 0),
        "epochs_completed": len(history),
        "completed": False,          # recovered from a log, so the run was cut short
        "best_mean_dice": round(max(dices), 4) if dices else None,
        "val_subjects": [],          # not recoverable from the log
        "train_entries": None,       # not recoverable from the log
        "history": history,
        "recovered_from_log": True,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", type=Path, help="file holding the copied training log")
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="where to write history.json")
    ap.add_argument("--run-id", default="overnight_run")
    ap.add_argument("--epochs", type=int, default=None,
                    help="the run's target epoch count, if known")
    ap.add_argument("--stdout", action="store_true", help="print instead of writing")
    args = ap.parse_args(argv)

    text = args.log.read_text(encoding="utf-8", errors="replace")
    history = parse_log(text)
    if not history:
        print("no epoch lines found -- expected lines like:\n"
              "  epoch 19: loss=0.2247  val ET=0.465  NC=0.753  WT=0.771  mean=0.663  (293s)",
              file=sys.stderr)
        return 1

    summary = build_summary(history, args.run_id, args.epochs)
    blob = json.dumps(summary, indent=2)

    if args.stdout or args.out is None:
        print(blob)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(blob + "\n", encoding="utf-8")
        print(f"wrote: {args.out}")

    gaps = [e for e in range(history[0]["epoch"], history[-1]["epoch"] + 1)
            if e not in {r["epoch"] for r in history}]
    print(f"epochs recovered: {len(history)} "
          f"({history[0]['epoch']}..{history[-1]['epoch']})", file=sys.stderr)
    if gaps:
        print(f"WARNING missing epochs in the log: {gaps}", file=sys.stderr)
    if summary["best_mean_dice"] is not None:
        print(f"best mean_dice: {summary['best_mean_dice']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
