# Plan: hospital DICOM → R*-ready input (preprocessing, parts 2–4)

Status: plan only, no code. The steps are built once the first Egyptian hospital scans arrive. Part 1 (alignment,
`prep/align.py`) is done.

## 1. Target: what R* will accept

R* checks its input strictly, so "1 mm NIfTI" is not enough:

- Each sequence is **exactly 240×240×155 voxels**, with **1 mm** voxels (0.9–1.1 mm tolerated)
  ([pipeline.py:207-211](../06_rstar_inference/rstar/pipeline.py#L207-L211), [config.py:28](../06_rstar_inference/rstar/config.py#L28)).
- The voxel array is read as-is, with no reorientation, and the 2D branch assumes the BraTS slice orientation
  ([preprocess.py:12-14](../06_rstar_inference/rstar/preprocess.py#L12-L14)). The scan must therefore be in **BraTS
  space (the SRI24 atlas grid)**, the space BraTS-PEDs was registered to [4]. Resampling to 1 mm alone does not
  achieve this.
- Skull-stripped, with the background **exactly 0**. R* rejects a sequence with fewer than 15 % zero voxels and warns
  below 25 %. It also rejects NaN values and more than 5 % negative voxels
  ([contract.py:48-63](../06_rstar_inference/rstar/contract.py#L48-L63)).
- No intensity normalisation or bias correction is needed: R* z-scores each sequence itself
  ([preprocess.py:20-33](../06_rstar_inference/rstar/preprocess.py#L20-L33)). A sequence the hospital did not acquire
  is simply left out (`rstar --t2w` omitted).

## 2. Recommended order

```
DICOM ─► 1 convert ─► 2 pick series ─► 3 align to T1c ─► 4 T1c → SRI24 ─► 5 resample once ─► 6 skull-strip ─► 7 R*
           dcm2niix     rules + human     prep.align       rigid, MI          onto 240×240×155   mask from T1c,
                        check                              (to build)         1 mm grid           applied to all 4
```

**Why this order:**
- **It is the BraTS-PEDs order.** Their scans were converted, co-registered, put on SRI24 at 1 mm, then skull-stripped
  [4]. Using the same order keeps hospital images as close as possible to what R* was trained on.
- **Align before stripping, using whole heads.** Registration uses mutual information, which matches better on whole
  heads than on masks that may differ slightly between sequences. Stripping once at the end gives one mask for all four
  sequences, so no sequence keeps a skull rim that another lacks.
- **Resample each sequence only once.** The alignment results showed that small resampling changes can flip R*'s
  enhancing-tumour decision ([RESULTS_alignment.md](RESULTS_alignment.md)). So step 4 should combine the alignment
  transform (already stored in `alignment.json`: rotation, translation, centre) with the atlas transform, and resample
  each original sequence once, with linear interpolation. Linear interpolation never creates negative values. Until
  step 4 exists, two resamplings (align, then atlas) are acceptable for testing.
- **Strip on the atlas grid.** Skull-stripping tools are trained on roughly atlas-oriented heads, and masking after
  resampling guarantees the background is exactly 0 on the final grid.

## 3. Tool per step

| Step | Recommended | Runs on | Licence | Evidence | Why |
|---|---|---|---|---|---|
| 1 Convert | **dcm2niix** (`pip install dcm2niix`, or the Windows zip) | CPU, Windows/Linux/macOS | BSD-2 [1] | De-facto standard; corrects gantry tilt and writes a BIDS `.json` sidecar per series [1] | Sidecars give series description, protocol, slice thickness, 2D/3D, TR/TE/TI for step 2. Alternatives (`dicom2nifti`, SimpleITK) handle fewer vendor quirks |
| 2 Pick series | Small rule script (to build) + a table a person confirms | CPU | — | — | Keyword rules on `SeriesDescription`/`ProtocolName` (FLAIR, T2, T1, post/+C/Gd), plus `InversionTime` (FLAIR) and contrast fields. Prefer 3D, else the thinnest slices; skip localisers and derived/reformatted series. Hospital naming varies, so a person signs off on the first cases |
| 3 Align | **`prep.align`** (done) | CPU, SimpleITK | in-repo | [RESULTS_alignment.md](RESULTS_alignment.md) | Already verified on BraTS-PEDs; T1c is the reference, as in BraTS |
| 4–5 Atlas + resample | **SimpleITK rigid MI**, same method as `prep.align`, onto the SRI24 T1 template (to build) | CPU | SRI24: free research licence (check) | Same method as step 3 | Rigid only (6 parameters), which suits children's heads that differ in size from the adult atlas. Reuses the tested code and lets the transforms be combined. Cross-check: BrainLes-preprocessing [6], the toolkit BraTS currently recommends, does steps 3–6 in one call and can serve as a reference on the first cases |
| 6 Skull-strip | **HD-BET** [2] as the default; **D3b paediatric model** [5] evaluated alongside | GPU < 10 s, CPU ≈ 2 min per scan; `pip`, Windows works (community-tested) | Code Apache-2.0, **weights CC BY-NC** (research use only) | Trained on multi-centre adult brain-tumour MRI; beat 6 other tools [2] | Tumour-trained, installs easily, integrated in BrainLes. **The D3b model is the one BraTS-PEDs itself used** [4,5] (nnU-Net, 527 paediatric tumour patients, Dice 0.98), so it may match R*'s training data best. But it needs nnU-Net v1 and Python 3.9, and its licence is unconfirmed. SynthStrip [3] (MIT/CC-BY weights, fast on CPU, works on any contrast, a paediatric variant exists) is the fallback; note that it keeps CSF in the mask by default |
| 7 Segment | `python -m rstar` | as now | in-repo | [06_rstar_inference](../06_rstar_inference/) | Its input checks (section 1) are the final gate |

## 4. Command chain (target; steps marked ★ are still to be built)

```
dcm2niix -z y -b y -f "%s_%p" -o case01/nifti case01/dicom
★ python -m prep.pick --nifti-dir case01/nifti --out case01/series.json      # human confirms series.json
python -m prep.align --t1c T1C --t1n T1N --t2f FLAIR --t2w T2 --out-dir case01/aligned
★ python -m prep.atlas --orig-dir case01/nifti --alignment case01/aligned/alignment.json \
      --atlas sri24_t1.nii.gz --out-dir case01/atlas                       # one resample, 240×240×155 @ 1 mm
★ python -m prep.strip --in-dir case01/atlas --tool hd-bet --out-dir case01/ready   # T1c mask, applied to all 4
python -m rstar --t1c case01/ready/t1c.nii.gz --t1n ... --t2f ... --t2w ... \
      --out case01/seg.nii.gz --json case01/report.json --review-mask case01/flags.nii.gz
```

## 5. How the built steps will be verified (later)

- **Grid check:** run a BraTS-PEDs case through steps 4–6 starting from its own files. The output must be on the
  BraTS-PEDs grid: same affine, same 240×240×155 size. Use a real BraTS-PEDs header as the reference, not a
  hand-written one.
- **Mask check:** the HD-BET and D3b masks must agree (Dice ≥ 0.95) with the zero background of BraTS-PEDs images.
- **First hospital cases:** a person reviews each one visually (series choice, alignment, mask, segmentation)
  before anything is reported. The R* `--json` warnings are recorded.

## 6. Risks

- **Thick-slice 2D scans** (for example 5 mm axial FLAIR) become blurry on a 1 mm grid. R* was trained on BraTS
  scans, which also came from mixed clinical protocols, but this is untested for us.
- **Very young children:** the rigid fit to the adult atlas and the tumour-trained HD-BET may both fail. This is a
  reason to evaluate the D3b model.
- **Post-operative cavities and shunts:** skull-strip masks can cut into them.
- **A missing sequence** lowers R*'s accuracy (R* runs with it absent and says so).

## 7. Questions for the hospital

1. Which sequences are routinely acquired for paediatric brain tumours? Are both T1 **pre-** and **post-contrast**
   acquired, plus FLAIR and T2? What are the protocol/series names as they appear on export?
2. Are the scans **2D or 3D**? What are the **slice thickness and spacing** for each sequence, and the in-plane resolution?
3. Scanner vendor(s), model(s) and field strength (1.5 T / 3 T)? Has the protocol changed over the years we would
   receive?
4. **Export format:** standard DICOM (classic or enhanced multi-frame)? Full series, or only selected images? Is
   there gantry tilt on any sequence? Are reformatted/derived series included?
5. **De-identification:** who anonymises the data, and is defacing needed? Is ethics approval or a data-sharing
   agreement in place? How will the data be transferred?
6. Patient age range, pre- versus post-operative scans, number of cases, and whether radiologist tumour outlines exist
   (needed to measure R* on their data).

## Sources

[1] dcm2niix — https://github.com/rordenlab/dcm2niix ; https://pypi.org/project/dcm2niix/
[2] HD-BET, Isensee et al., Hum Brain Mapp 2019, doi:10.1002/hbm.24750 — https://github.com/MIC-DKFZ/HD-BET
[3] SynthStrip, Hoopes et al., NeuroImage 2022 — https://surfer.nmr.mgh.harvard.edu/docs/synthstrip ; paediatric
    variant: Kelley et al. 2024, https://arxiv.org/abs/2402.16634
[4] BraTS-PEDs 2023 — https://arxiv.org/abs/2305.17033 ; https://arxiv.org/abs/2407.08855
[5] D3b paediatric skull-strip (nnU-Net) — medRxiv 10.1101/2024.07.29.24311006 ;
    https://github.com/d3b-center/peds-brain-auto-skull-strip
[6] BrainLes-preprocessing — https://pypi.org/project/brainles-preprocessing/ ;
    https://brats.readthedocs.io/en/latest/guides/preprocessing/
