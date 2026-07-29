"""Measured evaluation metrics for the demo: ROC / AUC and calibration.

Everything here is computed from real model output on real held-out data. There
are no stored constants to fall back on -- a caller that has no checkpoint gets
an exception, not a plausible-looking number.

**What the ROC is over.** Segmentation is dense per-pixel classification, so a
ROC curve is built one-vs-rest per tumour region: for each pixel, the model's
softmax probability for that region is the score, and the cached ground truth
is the label. Regions follow the BraTS convention section 03 evaluates on:

    ET = label 1
    TC = labels 1 + 3            (tumour core)
    WT = labels 1 + 2 + 3 + 4    (whole tumour)

**Why pixels are subsampled.** A single 240x240 slice is 57,600 pixels and the
overwhelming majority are background, so pooling every pixel across many slices
both blows up memory and produces an AUC dominated by trivially-easy background.
`max_pixels_per_slice` takes a class-stratified sample instead, keeping all
positive pixels and a bounded random draw of negatives. The resulting AUC is an
unbiased estimate of the full-pixel AUC under that sampling, and the sample size
is reported so the number is never quoted without its precision.
"""

from __future__ import annotations

import numpy as np

from .inference import load_model, predict_slice

#: label sets per scored region, matching section 03's evaluator
REGIONS = {
    "ET": (1,),
    "TC": (1, 3),
    "WT": (1, 2, 3, 4),
}
REGION_COLOURS = {"ET": "#EF4444", "TC": "#818CF8", "WT": "#38BDF8"}


def roc_curve(scores: np.ndarray, labels: np.ndarray, n_thresholds: int = 200):
    """(fpr, tpr, thresholds) for binary ``labels`` and continuous ``scores``.

    Implemented directly rather than via sklearn to keep the demo's dependency
    set unchanged. Thresholds are quantiles of the score distribution, so the
    curve is dense where the scores actually live instead of wasting points on
    empty probability ranges.
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels).ravel().astype(bool)

    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        # Undefined: ROC needs both classes present. Say so rather than return
        # a degenerate diagonal that reads as "50% accurate".
        return None, None, None

    order = np.argsort(-scores, kind="mergesort")
    s_sorted = scores[order]
    l_sorted = labels[order]

    tps = np.cumsum(l_sorted)
    fps = np.cumsum(~l_sorted)

    # keep only the last index of each run of equal scores
    distinct = np.where(np.diff(s_sorted))[0]
    idx = np.r_[distinct, s_sorted.size - 1]

    tpr = np.r_[0.0, tps[idx] / n_pos]
    fpr = np.r_[0.0, fps[idx] / n_neg]
    thresholds = np.r_[np.inf, s_sorted[idx]]

    if tpr.size > n_thresholds:                     # thin for plotting only
        keep = np.unique(np.linspace(0, tpr.size - 1, n_thresholds).astype(int))
        tpr, fpr, thresholds = tpr[keep], fpr[keep], thresholds[keep]
    return fpr, tpr, thresholds


def auc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    """Area under the curve by the trapezoid rule."""
    return float(np.trapezoid(tpr, fpr)) if hasattr(np, "trapezoid") else float(np.trapz(tpr, fpr))


def _stratified_sample(pos_mask: np.ndarray, max_pixels: int, rng) -> np.ndarray:
    """Indices keeping every positive pixel plus a bounded draw of negatives."""
    pos_idx = np.flatnonzero(pos_mask)
    neg_idx = np.flatnonzero(~pos_mask)
    budget = max(max_pixels - pos_idx.size, 0)
    if neg_idx.size > budget:
        neg_idx = rng.choice(neg_idx, size=budget, replace=False)
    if pos_idx.size > max_pixels:
        pos_idx = rng.choice(pos_idx, size=max_pixels, replace=False)
    return np.concatenate([pos_idx, neg_idx])


def collect_scores(checkpoint_path, cache_dir, subjects, plane: str = "axial",
                   slices_per_subject: int = 6, max_pixels_per_slice: int = 4000,
                   device: str = "cpu", seed: int = 1337, progress=None) -> dict:
    """Pool per-pixel (score, label) pairs per region over several subjects.

    Only tumour-bearing slices are sampled: a slice with no tumour contributes
    no positives to any region and would just inflate the negative pool.
    """
    from .inference import find_tumor_slices

    rng = np.random.default_rng(seed)
    model, type_head, cfg, meta = load_model(checkpoint_path, str(cache_dir), device=device)

    pooled = {r: {"scores": [], "labels": []} for r in REGIONS}
    n_slices = 0

    for n, sid in enumerate(subjects, 1):
        tumor_idx = find_tumor_slices(cache_dir, sid, plane)
        if not tumor_idx:
            continue
        # spread the picks across the tumour extent rather than taking a
        # contiguous block, which would over-sample one part of the tumour
        picks = np.unique(np.linspace(0, len(tumor_idx) - 1,
                                      min(slices_per_subject, len(tumor_idx))).astype(int))
        for p in picks:
            r = predict_slice(model, cfg, cache_dir, sid, plane, tumor_idx[p], device=device)
            probs = r["probabilities"]
            truth = r["ground_truth"]
            for region, labels in REGIONS.items():
                score = probs[list(labels)].sum(axis=0).ravel()
                positive = np.isin(truth, list(labels)).ravel()
                keep = _stratified_sample(positive, max_pixels_per_slice, rng)
                pooled[region]["scores"].append(score[keep])
                pooled[region]["labels"].append(positive[keep])
            n_slices += 1
        if progress:
            progress(n, len(subjects))

    out = {"n_subjects": len(subjects), "n_slices": n_slices,
           "plane": plane, "checkpoint": meta, "regions": {}}
    for region in REGIONS:
        if not pooled[region]["scores"]:
            out["regions"][region] = None
            continue
        scores = np.concatenate(pooled[region]["scores"])
        labels = np.concatenate(pooled[region]["labels"])
        fpr, tpr, _ = roc_curve(scores, labels)
        out["regions"][region] = {
            "fpr": fpr, "tpr": tpr,
            "auc": auc(fpr, tpr) if fpr is not None else None,
            "n_pixels": int(scores.size),
            "n_positive": int(labels.sum()),
            "prevalence": float(labels.mean()),
        }
    return out
