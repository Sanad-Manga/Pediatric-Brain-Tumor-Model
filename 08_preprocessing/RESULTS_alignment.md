# Alignment step: verification results (8 Oct 2026)

BraTS-PEDs fresh-30 patients (first 10 by id). Residual error = distance between where a brain voxel should be and
where the recovered transform puts it, over 20,000 brain points. Logged by `tools/check_alignment.py`
(`tools/check.out`); C diagnosis by `tools/diag_c.py` (`tools/diag_c.txt`).

## Summary

- **Req 9 (A, recovery) FAIL as written:** 10/20 cases meet mean < 0.5 mm and max < 1.0 mm (mean residual median 0.48 mm, worst 1.34 mm).
- **Req 10 (B, do no harm) FAIL as written:** 11/20 cases within 0.5 mm of identity (median 0.50 mm, worst 1.62 mm).
- **But the step is consistent:** in 16/20 cases the recovery in A lands within 0.3 mm of where B puts the
  untouched scan (median 0.12 mm). So A's error is mostly the same offset B finds: the
  registration disagrees with BraTS's own inter-sequence alignment by small rotations (mostly 0.1-1.2 degrees about one
  axis). From these data alone it cannot be decided which of the two is closer to the truth.
- **Req 11 (C, end to end) NOT MET / incomplete:** 2 of 5 patients scored before the run stopped. 00103: clean 0.780,
  misaligned 0.782, realigned 0.779 (fine). 00077: clean 0.916, misaligned 0.903, realigned 0.575: an ET-free patient
  where R* drew a false enhancing spot after realignment (ET Dice 1 -> 0; TC/WT 0.863/0.861 unaffected). The third
  patient (00115) was rejected by the rstar input check before any alignment mattered: its original BraTS T1c has only
  3.7% exactly-zero voxels (the check expects 30-60%).

## A and B per case

| Patient | Seq | A mean | A max | B mean | B max | A-vs-B mean | A-vs-B max |
|---|---|---|---|---|---|---|---|
| 00077 | t2f | 0.567 | 1.325 | 0.523 | 1.311 | 0.109 | 0.264 |
| 00077 | t2w | 0.370 | 0.847 | 0.304 | 0.660 | 0.117 | 0.277 |
| 00103 | t2f | 0.363 | 0.826 | 0.341 | 0.742 | 0.110 | 0.239 |
| 00103 | t2w | 0.586 | 0.884 | 0.828 | 1.138 | 0.277 | 0.386 |
| 00115 | t2f | 0.554 | 0.900 | 0.607 | 0.808 | 0.152 | 0.412 |
| 00115 | t2w | 0.459 | 0.990 | 0.413 | 0.942 | 0.180 | 0.349 |
| 00127 | t2f | 0.494 | 1.162 | 0.498 | 1.099 | 0.121 | 0.222 |
| 00127 | t2w | 0.620 | 1.181 | 1.552 | 2.931 | 0.987 | 1.824 |
| 00128 | t2f | 0.105 | 0.114 | 0.071 | 0.135 | 0.079 | 0.123 |
| 00128 | t2w | 0.278 | 0.562 | 0.304 | 0.608 | 0.051 | 0.065 |
| 00129 | t2f | 0.376 | 0.605 | 0.323 | 0.513 | 0.062 | 0.103 |
| 00129 | t2w | 0.145 | 0.220 | 0.153 | 0.246 | 0.018 | 0.046 |
| 00131 | t2f | 0.393 | 0.949 | 0.357 | 0.834 | 0.062 | 0.141 |
| 00131 | t2w | 0.809 | 2.025 | 0.784 | 1.880 | 0.074 | 0.190 |
| 00132 | t2f | 0.230 | 0.468 | 0.288 | 0.683 | 0.179 | 0.410 |
| 00132 | t2w | 0.691 | 1.478 | 0.767 | 1.677 | 0.111 | 0.246 |
| 00133 | t2f | 1.344 | 3.064 | 1.617 | 3.531 | 0.314 | 0.679 |
| 00133 | t2w | 0.501 | 0.957 | 0.613 | 1.452 | 0.385 | 0.830 |
| 00136 | t2f | 0.975 | 2.274 | 1.276 | 2.684 | 0.365 | 0.774 |
| 00136 | t2w | 0.462 | 0.973 | 0.494 | 0.771 | 0.255 | 0.522 |

## C so far

| Patient | Clean | Misaligned (T2w + FLAIR moved 2 mm + 3 deg) | Realigned |
|---|---|---|---|
| 00077 | 0.9156 | 0.9027 | 0.5746 |
| 00103 | 0.7799 | 0.7815 | 0.7792 |

## What this means

- The step reliably undoes misalignments of the size we stress-tested (2-3 mm, 2-5 degrees) to a consistent position.
- It does not reproduce BraTS's alignment to better than about 0.5-1.5 mm, so on BraTS data it moves already-aligned
  scans by that much. Before relying on it, decide which reference to trust, e.g. by checking a few cases visually or
  with a second metric (correlation on T1n vs T1c, which share contrast).
- End to end, the resampling it adds can flip R*'s fragile ET decision on a tumour-free patient (00077). On hospital
  scans that genuinely need alignment this trade-off is clearly worth it (2 mm misalignment costs R* 0.078); on scans
  that are already aligned the step should be skipped when the measured transform is below a threshold.
- Separate finding: the rstar input check rejects at least one genuine BraTS scan (00115). Its 30-60% background rule
  needs revisiting before hospital scans arrive.


## Addendum 1: skip realignment when already aligned (Req 13-18)

13 unit tests pass (5 new for the skip rule). Mutation check: flipping the comparison or dropping the same-grid test is
caught; changing `> 0` to `>= 0` survives but is equivalent (a displacement is never below 0).

Real-scan check (`tools/check_skip.py`, untouched fresh-30 scans, default threshold 1.0 mm):

| Patient | Sequence | Displacement (mm) | Skipped | Output identical to input |
|---|---|---|---|---|
| 00077 | t1n | 0.204 | True | True |
| 00077 | t2f | 0.528 | True | True |
| 00077 | t2w | 0.306 | True | True |
| 00127 | t1n | 0.439 | True | True |
| 00127 | t2f | 0.544 | True | True |
| 00127 | t2w | 1.589 | False | False |
| 00128 | t1n | 0.216 | True | True |
| 00128 | t2f | 0.074 | True | True |
| 00128 | t2w | 0.313 | True | True |

**Req 18: PASS.** 8 of 9 sequences are left byte-identical; 00127 T2w (1.59 mm) is realigned. Note the rule protects
already-aligned scans only: a genuinely misaligned scan (C: 2 mm + 3 degrees) is still resampled, and 00077's flip in C
came from that realignment, so it is not prevented by this rule.
