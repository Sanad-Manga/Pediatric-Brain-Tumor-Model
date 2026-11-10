"""Decide whether our alignment step or the scan's own alignment puts a sequence in the better place (issue #65).

Verification B found that on some BraTS FLAIR/T2w scans the step moves an already-aligned sequence by 0.5-1.6 mm.
The numbers in RESULTS_alignment.md cannot say who is right, and neither can the eye: the two positions differ by
about one voxel, and the realigned scan is the blurrier of the two by construction (it has been resampled once more),
so in an ITK-SNAP overlay it looks worse whether or not it sits in the better place.

So the two candidates are scored here, on equal terms, by criteria the step never optimised.

**Equal terms.** Both candidates are resampled onto the same half-voxel-shifted copy of the T1c grid, as is T1c
itself. Every image compared has then been through exactly one linear interpolation at the same sub-voxel phase, so
sharpness cannot decide the outcome.

**Criteria the step never optimised.** The step maximises Mattes mutual information over 20% random samples of the
*whole* volume, background included, so scoring with that would be circular. Instead:

- `gradient_correlation` - do the two images' edges coincide (correlation of edge strength)?
- `brain_nmi` - normalised mutual information computed here, from a 64-bin joint histogram, brain only;
- `joint_mi` - ITK's joint-histogram mutual information, a different implementation, brain only, densely sampled;
- `ants_cc` - ANTS local neighbourhood correlation, which is not mutual information at all.

Each is evaluated at both candidate positions and votes for the better one, or ties when the difference is under
`REL_TOL` of its own magnitude. A fifth, independent opinion comes from registering the sequence again from scratch
with a different metric, sampling and optimiser, and asking which candidate that transform lands on; it is allowed to
abstain, because mutual information can find a spurious optimum several mm away and such a run is evidence about
nothing.

The brain mask is eroded by `ERODE_MM`: BraTS applies one skull-strip mask to all four sequences, so the brain outline
matches perfectly at identity no matter where the anatomy inside sits. Keeping it would be evidence for identity by
construction.

Verdicts: "no disagreement" when the candidates are less than `MIN_GAP_MM` apart (nothing to judge); "aligned better"
or "original better" when at least `MIN_VOTES` criteria vote that way and none votes the other way; "can't tell"
otherwise.
"""
from __future__ import annotations

import math

import numpy as np
import SimpleITK as sitk

from . import align

ERODE_MM = 3.0         # brain mask erosion; drops the shared skull-strip outline
MIN_GAP_MM = 0.3       # candidates closer than this are the same position for our purposes
REL_TOL = 0.001        # a criterion ties when the two candidates are within 0.1% of each other
MIN_VOTES = 2          # how many criteria must agree before a verdict is given
VOTE_RATIO = 2.0       # the re-registration must be this many times closer to one candidate than the other
MAX_OPINION_MM = 5.0   # a re-registration further than this from both candidates has failed, not voted
MAX_STEP_MM = 0.5      # bound on the re-registration's optimiser step (an unbounded line search diverges)
IDENTITY = "original"  # the scan's own (BraTS) alignment
ALIGNED = "aligned"    # our step's alignment
CRITERIA = ("gradient_correlation", "brain_nmi", "joint_mi", "ants_cc")


# --------------------------------------------------------------------------------------- grids, masks and points

def brain_mask(t1c: sitk.Image, erode_mm: float = ERODE_MM) -> sitk.Image:
    """Voxels inside the brain, eroded by `erode_mm`, as a uint8 image on t1c's grid (BraTS background is exactly 0)."""
    m = sitk.Cast(sitk.Cast(t1c, sitk.sitkFloat32) > 0, sitk.sitkUInt8)
    radius = [max(0, int(round(erode_mm / s))) for s in t1c.GetSpacing()]
    if any(radius):
        m = sitk.BinaryErode(m, radius, sitk.sitkBall)
    return m


def half_voxel_grid(ref: sitk.Image) -> sitk.Image:
    """`ref`'s grid shifted half a voxel along each axis, so resampling onto it always interpolates."""
    g = sitk.Image([int(s) for s in ref.GetSize()], sitk.sitkFloat32)
    g.SetSpacing(ref.GetSpacing())
    g.SetDirection(ref.GetDirection())
    g.SetOrigin(ref.TransformContinuousIndexToPhysicalPoint([0.5] * ref.GetDimension()))
    return g


def mask_points(mask: sitk.Image, n: int = 20000, seed: int = align.SEED) -> np.ndarray:
    """Physical coordinates of up to `n` voxels of `mask` (fixed-seed sample)."""
    idx = np.argwhere(sitk.GetArrayViewFromImage(mask) > 0)
    if len(idx) == 0:
        raise RuntimeError("brain mask is empty")
    pick = idx[np.random.default_rng(seed).choice(len(idx), min(n, len(idx)), replace=False)]
    return np.array([mask.TransformIndexToPhysicalPoint([int(i) for i in p[::-1]]) for p in pick])


def transform_gap_mm(points: np.ndarray, a: sitk.Transform, b: sitk.Transform) -> float:
    """Mean distance (mm) between where `a` and `b` send each point: how far apart the two alignments are."""
    pa = np.array([a.TransformPoint(tuple(float(v) for v in p)) for p in points])
    pb = np.array([b.TransformPoint(tuple(float(v) for v in p)) for p in points])
    return float(np.linalg.norm(pa - pb, axis=1).mean())


# --------------------------------------------------------------------------------------------------- criteria

def gradient_magnitude(img: sitk.Image, sigma: float = 1.0) -> np.ndarray:
    return sitk.GetArrayFromImage(sitk.GradientMagnitudeRecursiveGaussian(sitk.Cast(img, sitk.sitkFloat32), sigma))


def gradient_correlation(a: sitk.Image, b: sitk.Image, mask: np.ndarray, sigma: float = 1.0) -> float:
    """Pearson correlation of the two images' edge strength inside `mask`: 1.0 = edges coincide everywhere."""
    ga, gb = gradient_magnitude(a, sigma)[mask], gradient_magnitude(b, sigma)[mask]
    if ga.std() == 0 or gb.std() == 0:
        return 0.0
    return float(np.corrcoef(ga, gb)[0, 1])


def normalised_mutual_information(a: sitk.Image, b: sitk.Image, mask: np.ndarray, bins: int = 64) -> float:
    """(H(a) + H(b)) / H(a, b) inside `mask`; 1.0 = independent, higher = more shared structure."""
    x = sitk.GetArrayFromImage(sitk.Cast(a, sitk.sitkFloat32))[mask]
    y = sitk.GetArrayFromImage(sitk.Cast(b, sitk.sitkFloat32))[mask]
    joint, _, _ = np.histogram2d(x, y, bins=bins)
    p = joint / joint.sum()

    def entropy(q):
        q = q[q > 0]
        return -float(np.sum(q * np.log(q)))

    hxy = entropy(p.ravel())
    return (entropy(p.sum(1)) + entropy(p.sum(0))) / hxy if hxy > 0 else 0.0


def itk_metric(fixed: sitk.Image, moving: sitk.Image, mask: sitk.Image, metric: str) -> float:
    """ITK's `metric` ("joint" or "ants") for `moving` against `fixed` inside `mask`, with no registration run:
    just the value at the position `moving` is already in. Returned so that higher is better."""
    reg = sitk.ImageRegistrationMethod()
    if metric == "joint":
        reg.SetMetricAsJointHistogramMutualInformation(numberOfHistogramBins=32)
    elif metric == "ants":
        reg.SetMetricAsANTSNeighborhoodCorrelation(4)
    else:
        raise ValueError(f"unknown metric {metric!r}")
    reg.SetMetricFixedMask(mask)
    reg.SetMetricSamplingStrategy(reg.NONE)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetInitialTransform(sitk.Euler3DTransform())
    reg.SetNumberOfThreads(1)
    value = float(reg.MetricEvaluate(sitk.Cast(fixed, sitk.sitkFloat32), sitk.Cast(moving, sitk.sitkFloat32)))
    return -value  # ITK metrics are minimised


def candidate_scores(moving: sitk.Image, t1c: sitk.Image, ours: sitk.Transform,
                     mask_img: sitk.Image) -> tuple[dict, int]:
    """All four criteria for both candidates, each image resampled exactly once onto the same shifted grid."""
    identity = sitk.Euler3DTransform()
    grid = half_voxel_grid(t1c)
    fair_t1c = sitk.Resample(t1c, grid, identity, sitk.sitkLinear, 0.0, sitk.sitkFloat32)
    fair_mask_img = sitk.Resample(mask_img, grid, identity, sitk.sitkNearestNeighbor, 0)
    fair_mask = sitk.GetArrayFromImage(fair_mask_img) > 0
    scores = {}
    for name, t in ((IDENTITY, identity), (ALIGNED, ours)):
        cand = sitk.Resample(moving, grid, t, sitk.sitkLinear, 0.0, sitk.sitkFloat32)
        scores[name] = {
            "gradient_correlation": round(gradient_correlation(fair_t1c, cand, fair_mask), 6),
            "brain_nmi": round(normalised_mutual_information(fair_t1c, cand, fair_mask), 6),
            "joint_mi": round(itk_metric(fair_t1c, cand, fair_mask_img, "joint"), 6),
            "ants_cc": round(itk_metric(fair_t1c, cand, fair_mask_img, "ants"), 6),
        }
    return scores, int(fair_mask.sum())


def criterion_votes(scores: dict) -> dict[str, str]:
    """Per criterion: which candidate scores higher, or "tie" within `REL_TOL` of the criterion's own magnitude."""
    votes = {}
    for key in CRITERIA:
        a, b = scores[ALIGNED][key], scores[IDENTITY][key]
        scale = max(abs(a), abs(b), 1e-12)
        delta = (a - b) / scale
        votes[key] = ALIGNED if delta > REL_TOL else IDENTITY if delta < -REL_TOL else "tie"
    return votes


# ------------------------------------------------------------------------------------------ independent opinion

def second_opinion(moving: sitk.Image, fixed: sitk.Image, mask: sitk.Image, metric: str = "joint",
                   seed: int = align.SEED) -> sitk.Euler3DTransform:
    """Register `moving` to `fixed` again, independently of `prep.align`: a different metric, regular sampling inside
    `mask` only, and a step-bounded gradient descent. Same rigid model and same reference, different evidence."""
    fixed_f, moving_f = sitk.Cast(fixed, sitk.sitkFloat32), sitk.Cast(moving, sitk.sitkFloat32)
    transform = sitk.Euler3DTransform(sitk.CenteredTransformInitializer(
        fixed_f, moving_f, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY))

    reg = sitk.ImageRegistrationMethod()
    if metric == "joint":
        reg.SetMetricAsJointHistogramMutualInformation(numberOfHistogramBins=32)
        reg.SetMetricSamplingPercentage(0.5, seed)
    elif metric == "ants":
        reg.SetMetricAsANTSNeighborhoodCorrelation(4)
        reg.SetMetricSamplingPercentage(1.0, seed)
    else:
        raise ValueError(f"unknown metric {metric!r}")
    reg.SetMetricSamplingStrategy(reg.REGULAR)
    reg.SetMetricFixedMask(mask)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=300, convergenceMinimumValue=1e-7,
                                      convergenceWindowSize=10, estimateLearningRate=reg.EachIteration,
                                      maximumStepSizeInPhysicalUnits=MAX_STEP_MM)
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetShrinkFactorsPerLevel([4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([2.0, 1.0, 0.0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    reg.SetInitialTransform(transform, inPlace=True)
    reg.SetNumberOfThreads(1)

    threads = sitk.ProcessObject.GetGlobalDefaultNumberOfThreads()
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    try:
        reg.Execute(fixed_f, moving_f)
    finally:
        sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(threads)
    if not math.isfinite(float(reg.GetMetricValue())):
        raise RuntimeError(f"second opinion ({metric}) did not converge (metric is not finite)")
    return transform


def opinion_vote(gap_to_identity: float, gap_to_ours: float) -> str:
    """Which candidate a re-registration sits on; "failed" when it landed near neither, "" when it separates nothing."""
    if min(gap_to_identity, gap_to_ours) > MAX_OPINION_MM:
        return "failed"
    if gap_to_ours > 0 and gap_to_identity >= VOTE_RATIO * gap_to_ours:
        return ALIGNED
    if gap_to_identity > 0 and gap_to_ours >= VOTE_RATIO * gap_to_identity:
        return IDENTITY
    return ""


# ----------------------------------------------------------------------------------------------------- verdict

def combine(votes: list[str]) -> str:
    """A verdict only when at least `MIN_VOTES` criteria agree and nothing points the other way."""
    cast = [v for v in votes if v in (ALIGNED, IDENTITY)]
    if len(set(cast)) != 1 or len(cast) < MIN_VOTES:
        return "can't tell"
    return f"{cast[0]} better"


def judge(moving: sitk.Image, t1c: sitk.Image, ours: sitk.Transform, reregister: tuple[str, ...] = ("joint",),
          seed: int = align.SEED, erode_mm: float = ERODE_MM) -> dict:
    """Judge our step's transform `ours` for `moving` against leaving `moving` where the scan already has it.

    `moving` must be on t1c's grid (BraTS scans are). Returns the verdict, every criterion's score for both
    candidates, and what each piece of evidence voted."""
    if not align.same_grid(moving, t1c):
        raise ValueError("moving is not on the T1c grid; this comparison only applies to already-aligned inputs")
    mask_img = brain_mask(t1c, erode_mm)
    identity = sitk.Euler3DTransform()
    pts = mask_points(mask_img, seed=seed)
    disagreement = transform_gap_mm(pts, ours, identity)

    scores, mask_voxels = candidate_scores(moving, t1c, ours, mask_img)
    votes = criterion_votes(scores)

    opinions = {}
    for metric in reregister:
        other = second_opinion(moving, t1c, mask_img, metric, seed)
        gap_identity = transform_gap_mm(pts, other, identity)
        gap_ours = transform_gap_mm(pts, other, ours)
        vote = opinion_vote(gap_identity, gap_ours)
        opinions[metric] = {"gap_to_original_mm": round(gap_identity, 3), "gap_to_aligned_mm": round(gap_ours, 3),
                            "vote": vote or "undecided", **align.transform_summary(other)}

    all_votes = list(votes.values()) + [o["vote"] for o in opinions.values()]
    verdict = "no disagreement" if disagreement < MIN_GAP_MM else combine(all_votes)
    return {"verdict": verdict,
            "disagreement_mm": round(disagreement, 3),
            "scores": scores,
            "criterion_votes": votes,
            "second_opinions": opinions,
            "mask_voxels": mask_voxels,
            "settings": {"erode_mm": erode_mm, "min_gap_mm": MIN_GAP_MM, "rel_tol": REL_TOL,
                         "min_votes": MIN_VOTES, "vote_ratio": VOTE_RATIO, "max_opinion_mm": MAX_OPINION_MM,
                         "max_step_mm": MAX_STEP_MM, "seed": seed}}
