#!/usr/bin/env python
"""Spread statistics from per_patient_scores.csv (CPU only, no torch)."""
from __future__ import annotations

import csv
import statistics as st
from pathlib import Path

CSV = Path(r"C:\Users\ahmed\neuropeds_overnight\per_patient_scores.csv")
REGIONS = ("ET", "NC", "WT")
MODELS = ("original", "epoch17", "ensemble")


def pct(xs, q):
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def main() -> None:
    rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
    print(f"{len(rows)} patients\n")
    for r in REGIONS:
        present = [x for x in rows if int(x[f"true_vox_{r}"]) > 0]
        print(f"== {r}: {len(present)} patients have this region in the ground truth "
              f"({len(rows) - len(present)} excluded: Dice is 1.0/0.0 by convention there)")
        print(f"{'model':10s} {'mean':>6s} {'median':>7s} {'std':>6s} {'p10':>6s} {'p25':>6s} "
              f"{'p75':>6s} {'min':>6s} {'<0.5':>5s} {'<0.3':>5s}")
        for m in MODELS:
            v = [float(x[f"{m}_{r}"]) for x in present]
            print(f"{m:10s} {st.mean(v):6.3f} {st.median(v):7.3f} {st.pstdev(v):6.3f} "
                  f"{pct(v, .10):6.3f} {pct(v, .25):6.3f} {pct(v, .75):6.3f} {min(v):6.3f} "
                  f"{sum(a < .5 for a in v):5d} {sum(a < .3 for a in v):5d}")
        d = [float(x[f"ensemble_{r}"]) - float(x[f"original_{r}"]) for x in present]
        e = [float(x[f"ensemble_{r}"]) - float(x[f"epoch17_{r}"]) for x in present]
        print(f"ensemble vs original: better on {sum(a > 0.005 for a in d)}/{len(d)}, "
              f"worse on {sum(a < -0.005 for a in d)}/{len(d)} (>0.005 margin)")
        print(f"ensemble vs epoch17 : better on {sum(a > 0.005 for a in e)}/{len(e)}, "
              f"worse on {sum(a < -0.005 for a in e)}/{len(e)}")
        worst = sorted(present, key=lambda x: float(x[f"ensemble_{r}"]))[:5]
        print("worst 5 (ensemble): " + ", ".join(
            f"{w['subject_id'][-9:-4]}={float(w[f'ensemble_{r}']):.2f}" for w in worst))
        print()
    mean_ens = [sum(float(x[f"ensemble_{r}"]) for r in REGIONS) / 3 for x in rows]
    print(f"per-patient mean-of-regions (ensemble): mean {st.mean(mean_ens):.4f} "
          f"median {st.median(mean_ens):.4f} p10 {pct(mean_ens, .10):.3f} "
          f"min {min(mean_ens):.3f}")


if __name__ == "__main__":
    main()
