"""Verification of the judge (SPEC Addendum 2, Req 24): does it get a known 1.5 mm answer right in both directions?

The three cases are built on a two-contrast phantom whose truth we set:

  A. the sequence is displaced by about 1.5 mm and the step recovers it   -> "aligned better"
  B. the sequence is already in place and the step is handed that same    -> "original better"
     1.5 mm transform anyway
  C. the sequence is already in place and the step is left to decide      -> "no disagreement"

Prints the markdown table in RESULTS_alignment.md, Addendum 2.

    python tools/check_judge.py
"""
import sys
import time
from pathlib import Path

import SimpleITK as sitk

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "tests"))
import phantoms  # noqa: E402
from prep import align, judge  # noqa: E402

ROTATION_DEG = (0.6, -0.4, 0.5)
SHIFT_MM = (1.1, -0.8, 0.4)


def cases():
    t1c, t2w = phantoms.brain_pair()
    moved, applied = phantoms.displaced(t2w, ROTATION_DEG, SHIFT_MM)
    pts = judge.mask_points(judge.brain_mask(t1c))
    truth = judge.transform_gap_mm(pts, applied, sitk.Euler3DTransform())
    _, recovered, _ = align.align_to_reference(moved, t1c)
    _, untouched, _ = align.align_to_reference(t2w, t1c)
    return truth, [("A: displaced %.2f mm, step recovers it" % truth, moved, t1c, recovered, "aligned better"),
                   ("B: in place, step proposes %.2f mm anyway" % truth, t2w, t1c, applied, "original better"),
                   ("C: in place, step decides for itself", t2w, t1c, untouched, "no disagreement")]


def main() -> int:
    truth, rows = cases()
    print(f"Phantom: rotation {ROTATION_DEG} deg, shift {SHIFT_MM} mm -> {truth:.3f} mm mean displacement\n")
    header = ("| Case | Expected | Verdict | Step moves it (mm) | " +
              " | ".join(k.replace("_", " ") for k in judge.CRITERIA) + " | re-registered |")
    print(header)
    print("|" + "---|" * (5 + len(judge.CRITERIA)))
    failures = 0
    for label, moving, t1c, ours, expected in rows:
        t0 = time.time()
        r = judge.judge(moving, t1c, ours)
        votes = " | ".join(r["criterion_votes"][k] for k in judge.CRITERIA)
        opinion = ", ".join(f"{m}: {o['vote']}" for m, o in r["second_opinions"].items())
        print(f"| {label} | {expected} | **{r['verdict']}** | {r['disagreement_mm']} | {votes} | {opinion} |")
        if r["verdict"] != expected:
            failures += 1
            print(f"  ^ MISMATCH (expected {expected}), {time.time() - t0:.0f}s", file=sys.stderr)
    print(f"\n{len(rows) - failures}/{len(rows)} cases as expected.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
