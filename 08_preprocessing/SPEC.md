# Spec: Sequence alignment step (NeuroPeds preprocessing, part 1)

## 1. Goal
Rigidly align a patient's T1n, T2-FLAIR and T2w scans to the same patient's T1c, so the four sequences R* reads line up
voxel for voxel, and prove on BraTS data that the step recovers known misalignments and leaves aligned scans unchanged.

## 2. In Scope
- Package `prep` in `08_preprocessing/` with `align_to_reference(moving, fixed)`: rigid (6 parameters: 3 rotations,
  3 translations) registration by Mattes mutual information, multi-resolution, initialised by image geometry; the moving
  image resampled onto the reference grid with linear interpolation.
- `align_sequences(paths)`: align T1n, T2-FLAIR and T2w to T1c; return the aligned images, each transform's parameters,
  and the final metric value.
- Command line `python -m prep.align --t1c --t1n --t2f --t2w --out-dir`: writes the 4 aligned NIfTI files (T1c copied
  unchanged) and `alignment.json` (per sequence: rotation in degrees, translation in mm, metric).
- Verification script `tools/check_alignment.py` on BraTS-PEDs fresh-30 patients:
  - **A. Recovery:** move T2w and FLAIR by a known random rigid transform (translations 1-3 mm per axis, rotations
    2-5 degrees per axis, fixed seed), align, and measure the residual error over brain voxels between where each voxel
    should be and where the recovered transform puts it.
  - **B. Do no harm:** align scans that are already aligned; the recovered transform must be close to identity.
  - **C. End to end:** R* (deployed recipe, via the `rstar` command-line tool) on clean, misaligned and realigned scans of
    5 patients; report mean Dice for each.
- Unit tests in `08_preprocessing/tests/`.

## 3. Out of Scope
- Skull stripping, DICOM to NIfTI conversion, bias-field correction, resampling to an atlas or 1 mm grid, intensity
  normalisation (all wait for real hospital scans).
- Non-rigid (deformable) registration.
- Wiring the step into the R* package, the Streamlit app, or any training code.
- Any change to the deployed models.

## 4. Requirements
1. `align_to_reference` returns an image on the fixed image's grid (same size, spacing, origin, direction).
2. The transform is rigid: exactly 6 free parameters (Euler 3D), no scaling or shear.
3. Background outside the moving image is filled with 0.
4. `align_sequences` aligns T1n, T2-FLAIR and T2w to T1c and returns T1c unchanged (voxel-identical).
5. The CLI writes 4 NIfTI files named `<prefix>-{t1c,t1n,t2f,t2w}.nii.gz` and `alignment.json`; it exits non-zero with a
   message naming the file if an input is missing or unreadable.
6. Every run is deterministic for the same inputs (fixed sampling seed, single-threaded metric sampling).
7. Unit test: a synthetic volume shifted by a known (3, -2, 1) mm and rotated 4 degrees is recovered to within 0.5 mm
   mean residual error.
8. Unit test: aligning an image to itself gives a transform within 0.2 mm / 0.2 degrees of identity.
9. Verification A on 10 fresh-30 patients: mean residual error < 0.5 mm and maximum (over brain voxels, per patient and
   sequence) < 1.0 mm for every patient and both sequences.
10. Verification B on the same 10 patients: recovered transform within 0.5 mm mean residual of identity for every
    patient and sequence.
11. Verification C on 5 patients: realigned R* mean Dice within 0.01 of clean, and misaligned R* mean Dice reported.
12. Results are written to `08_preprocessing/RESULTS_alignment.md` with every number the requirements check.

**Assumption:** T1c is the reference, as in the BraTS preprocessing protocol (sequences co-registered to T1c before atlas
registration). **Assumption:** linear interpolation for images is acceptable; its blurring effect is what B and C measure.
**Assumption:** "brain voxels" = voxels where the reference T1n > 0 (BraTS images are skull-stripped with zero background).

## 5. Structure
```
08_preprocessing/
├── SPEC.md
├── README.md                 # what the step does, how to run it, what it does not do yet
├── RESULTS_alignment.md      # verification A/B/C numbers
├── prep/
│   ├── __init__.py
│   ├── align.py              # align_to_reference, align_sequences, CLI (python -m prep.align)
├── tests/
│   └── test_align.py         # synthetic recovery + identity tests (Req 7-8), grid/rigid/background checks (Req 1-4, 6)
└── tools/
    └── check_alignment.py    # verification A, B, C on BraTS-PEDs (Req 9-12)
```

## 6. Edge Cases
| Scenario | Expected Behaviour |
|---|---|
| An input file is missing | CLI exits 1: "missing input: <path>" |
| Moving and reference have different grids | Output is on the reference grid (Req 1) |
| Registration does not converge (metric NaN) | Raise RuntimeError naming the sequence |
| Moving image already aligned | Transform near identity (Req 8, 10) |

## 7. Done Checklist
- [ ] Req 1: `align_to_reference` output is on the fixed image's grid.
- [ ] Req 2: transform is rigid, 6 parameters.
- [ ] Req 3: background filled with 0.
- [ ] Req 4: `align_sequences` returns T1c unchanged and aligns the other three to it.
- [ ] Req 5: CLI writes 4 NIfTI files + alignment.json; non-zero exit with message on missing input.
- [ ] Req 6: deterministic for the same inputs.
- [ ] Req 7: synthetic shift + rotation recovered within 0.5 mm mean residual.
- [ ] Req 8: self-alignment within 0.2 mm / 0.2 degrees of identity.
- [ ] Req 9: verification A passes on 10 fresh-30 patients (mean < 0.5 mm, max < 1.0 mm).
- [ ] Req 10: verification B passes on the same 10 patients.
- [ ] Req 11: verification C: realigned within 0.01 of clean; misaligned reported.
- [ ] Req 12: RESULTS_alignment.md written with every checked number.
