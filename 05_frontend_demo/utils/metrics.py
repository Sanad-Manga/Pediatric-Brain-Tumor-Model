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


def roc_curve(scores: np.ndarray, labels: np.ndarray, n_thresholds: int | None = None):
    """(fpr, tpr, thresholds) for binary ``labels`` and continuous ``scores``.

    Implemented directly rather than via sklearn to keep the demo's dependency
    set unchanged.

    ``n_thresholds`` down-samples the returned curve **for plotting only**.
    Leave it None — the default — whenever the result feeds :func:`auc`:
    thinning picks evenly spaced indices, which under-samples the elbow where
    the curve turns, and integrating that shifts the AUC in the third or fourth
    decimal. Compute the area on the full curve, thin afterwards with
    :func:`thin_curve`.
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

    if n_thresholds is not None and tpr.size > n_thresholds:
        fpr, tpr, thresholds = thin_curve(fpr, tpr, thresholds, n_thresholds)
    return fpr, tpr, thresholds


def thin_curve(fpr, tpr, thresholds=None, n: int = 300):
    """Reduce a ROC curve to ~``n`` points **for plotting**, keeping its shape.

    Sampling evenly by index flattens the elbow, which is the only visually
    interesting part of a curve with AUC ~0.99. Instead, walk the curve by arc
    length so points are spent where it actually bends, and always keep the
    first and last vertex so the axes still terminate at (0,0) and (1,1).
    """
    fpr = np.asarray(fpr, dtype=float)
    tpr = np.asarray(tpr, dtype=float)
    if fpr.size <= n:
        return fpr, tpr, thresholds

    step = np.hypot(np.diff(fpr), np.diff(tpr))
    arc = np.concatenate([[0.0], np.cumsum(step)])
    if arc[-1] <= 0:
        keep = np.unique(np.linspace(0, fpr.size - 1, n).astype(int))
    else:
        targets = np.linspace(0.0, arc[-1], n)
        keep = np.unique(np.searchsorted(arc, targets).clip(0, fpr.size - 1))
    keep = np.unique(np.concatenate([[0], keep, [fpr.size - 1]]))
    return fpr[keep], tpr[keep], (None if thresholds is None else np.asarray(thresholds)[keep])


def auc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    """Area under the curve by the trapezoid rule."""
    return float(np.trapezoid(tpr, fpr)) if hasattr(np, "trapezoid") else float(np.trapz(tpr, fpr))


def confusion_rates(pred_mask: np.ndarray, true_mask: np.ndarray) -> dict:
    """TP/FP/FN/TN counts for one binary region on one slice.

    Returned as raw counts, not rates: rates have to be pooled over the whole
    evaluation set, and averaging per-slice rates would weight a slice with
    three tumour pixels the same as one with three thousand.
    """
    p = np.asarray(pred_mask, dtype=bool)
    t = np.asarray(true_mask, dtype=bool)
    return {
        "tp": int((p & t).sum()),
        "fp": int((p & ~t).sum()),
        "fn": int((~p & t).sum()),
        "tn": int((~p & ~t).sum()),
    }


def rates_from_counts(c: dict) -> dict:
    """Sensitivity / specificity / precision / Dice from pooled counts."""
    tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
    def div(a, b):
        return (a / b) if b else None
    return {
        "sensitivity": div(tp, tp + fn),
        "specificity": div(tn, tn + fp),
        "precision": div(tp, tp + fp),
        "dice": div(2 * tp, 2 * tp + fp + fn),
        **c,
    }


def hd95(pred_mask: np.ndarray, true_mask: np.ndarray, spacing_mm: float = 1.0):
    """95th-percentile Hausdorff distance in mm between two 2D masks.

    Returns None when either mask is empty -- the distance is undefined, and
    substituting 0 would read as a perfect boundary match.
    """
    from scipy.ndimage import distance_transform_edt

    p = np.asarray(pred_mask, dtype=bool)
    t = np.asarray(true_mask, dtype=bool)
    if not p.any() or not t.any():
        return None

    # distance from every pixel to the nearest surface pixel of the other mask
    dist_to_t = distance_transform_edt(~t) * spacing_mm
    dist_to_p = distance_transform_edt(~p) * spacing_mm

    p_border = p & ~_erode(p)
    t_border = t & ~_erode(t)
    if not p_border.any() or not t_border.any():
        return None

    d = np.concatenate([dist_to_t[p_border], dist_to_p[t_border]])
    return float(np.percentile(d, 95))


def _erode(mask: np.ndarray) -> np.ndarray:
    from scipy.ndimage import binary_erosion

    return binary_erosion(mask, border_value=0)


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
    counts = {r: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for r in REGIONS}
    hd_samples = {r: [] for r in REGIONS}
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
            pred = r["prediction"]
            for region, labels in REGIONS.items():
                score = probs[list(labels)].sum(axis=0)
                positive = np.isin(truth, list(labels))
                predicted = np.isin(pred, list(labels))

                # counts over EVERY pixel -- these are exact, not sampled
                c = confusion_rates(predicted, positive)
                for k in counts[region]:
                    counts[region][k] += c[k]
                d = hd95(predicted, positive)
                if d is not None:
                    hd_samples[region].append(d)

                # ROC pools a stratified sample; see the module docstring
                keep = _stratified_sample(positive.ravel(), max_pixels_per_slice, rng)
                pooled[region]["scores"].append(score.ravel()[keep])
                pooled[region]["labels"].append(positive.ravel()[keep])
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
        hd = hd_samples[region]
        out["regions"][region] = {
            "fpr": fpr, "tpr": tpr,
            "auc": auc(fpr, tpr) if fpr is not None else None,
            "n_pixels": int(scores.size),
            "n_positive": int(labels.sum()),
            "prevalence": float(labels.mean()),
            # exact, over all pixels of every scored slice
            **rates_from_counts(counts[region]),
            "hd95_median_mm": float(np.median(hd)) if hd else None,
            "hd95_n_slices": len(hd),
        }
    return out
