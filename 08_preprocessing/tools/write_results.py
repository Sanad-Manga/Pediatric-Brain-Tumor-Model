"""Write RESULTS_alignment.md from the logged verification output (check.out) and the C diagnosis (diag_c.txt).
The full check_alignment.py run stopped in part C when the rstar tool rejected an original BraTS scan (00115), so the
report is assembled from what was logged."""
import re
from pathlib import Path

import numpy as np

T = Path(__file__).resolve().parent
rows = []
for line in (T / "check.out").read_text(encoding="utf-8").splitlines():
    m = re.match(r"(\d+) (t2f|t2w): A mean ([\d.]+) max ([\d.]+) \| B mean ([\d.]+) max ([\d.]+) \| A-vs-B mean ([\d.]+) max ([\d.]+)", line)
    if m:
        rows.append(dict(sid=m[1], seq=m[2], A=float(m[3]), Amax=float(m[4]), B=float(m[5]), Bmax=float(m[6]), AB=float(m[7]), ABmax=float(m[8])))
c = [re.match(r"(\d+) R\* mean Dice: clean ([\d.]+) \| misaligned ([\d.]+) \| realigned ([\d.]+)", l)
     for l in (T / "check.out").read_text(encoding="utf-8").splitlines()]
c = [x for x in c if x]
okA = all(r["A"] < 0.5 and r["Amax"] < 1.0 for r in rows)
okB = all(r["B"] < 0.5 for r in rows)
nA = sum(r["A"] < 0.5 and r["Amax"] < 1.0 for r in rows)
nB = sum(r["B"] < 0.5 for r in rows)
nAB = sum(r["AB"] < 0.3 for r in rows)
L = ["# Alignment step: verification results (8 Oct 2026)", "",
     "BraTS-PEDs fresh-30 patients (first 10 by id). Residual error = distance between where a brain voxel should be and",
     "where the recovered transform puts it, over 20,000 brain points. Logged by `tools/check_alignment.py`",
     "(`tools/check.out`); C diagnosis by `tools/diag_c.py` (`tools/diag_c.txt`).", "",
     "## Summary", "",
     f"- **Req 9 (A, recovery) FAIL as written:** {nA}/{len(rows)} cases meet mean < 0.5 mm and max < 1.0 mm "
     f"(mean residual median {np.median([r['A'] for r in rows]):.2f} mm, worst {max(r['A'] for r in rows):.2f} mm).",
     f"- **Req 10 (B, do no harm) FAIL as written:** {nB}/{len(rows)} cases within 0.5 mm of identity "
     f"(median {np.median([r['B'] for r in rows]):.2f} mm, worst {max(r['B'] for r in rows):.2f} mm).",
     f"- **But the step is consistent:** in {nAB}/{len(rows)} cases the recovery in A lands within 0.3 mm of where B puts the",
     "  untouched scan (median " f"{np.median([r['AB'] for r in rows]):.2f} mm). So A's error is mostly the same offset B finds: the",
     "  registration disagrees with BraTS's own inter-sequence alignment by small rotations (mostly 0.1-1.2 degrees about one",
     "  axis). From these data alone it cannot be decided which of the two is closer to the truth.",
     "- **Req 11 (C, end to end) NOT MET / incomplete:** 2 of 5 patients scored before the run stopped. 00103: clean 0.780,",
     "  misaligned 0.782, realigned 0.779 (fine). 00077: clean 0.916, misaligned 0.903, realigned 0.575: an ET-free patient",
     "  where R* drew a false enhancing spot after realignment (ET Dice 1 -> 0; TC/WT 0.863/0.861 unaffected). The third",
     "  patient (00115) was rejected by the rstar input check before any alignment mattered: its original BraTS T1c has only",
     "  3.7% exactly-zero voxels (the check expects 30-60%).", "",
     "## A and B per case", "",
     "| Patient | Seq | A mean | A max | B mean | B max | A-vs-B mean | A-vs-B max |", "|---|---|---|---|---|---|---|---|"]
L += [f"| {r['sid']} | {r['seq']} | {r['A']:.3f} | {r['Amax']:.3f} | {r['B']:.3f} | {r['Bmax']:.3f} | {r['AB']:.3f} | {r['ABmax']:.3f} |" for r in rows]
L += ["", "## C so far", "", "| Patient | Clean | Misaligned (T2w + FLAIR moved 2 mm + 3 deg) | Realigned |", "|---|---|---|---|"]
L += [f"| {x[1]} | {x[2]} | {x[3]} | {x[4]} |" for x in c]
L += ["", "## What this means", "",
      "- The step reliably undoes misalignments of the size we stress-tested (2-3 mm, 2-5 degrees) to a consistent position.",
      "- It does not reproduce BraTS's alignment to better than about 0.5-1.5 mm, so on BraTS data it moves already-aligned",
      "  scans by that much. Before relying on it, decide which reference to trust, e.g. by checking a few cases visually or",
      "  with a second metric (correlation on T1n vs T1c, which share contrast).",
      "- End to end, the resampling it adds can flip R*'s fragile ET decision on a tumour-free patient (00077). On hospital",
      "  scans that genuinely need alignment this trade-off is clearly worth it (2 mm misalignment costs R* 0.078); on scans",
      "  that are already aligned the step should be skipped when the measured transform is below a threshold.",
      "- Separate finding: the rstar input check rejects at least one genuine BraTS scan (00115). Its 30-60% background rule",
      "  needs revisiting before hospital scans arrive.", ""]
(T.parent / "RESULTS_alignment.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L[:20]))
