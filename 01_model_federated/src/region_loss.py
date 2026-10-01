"""Region-based loss on the existing 5-class head (SPEC.md Addendum 5, option A).

The model keeps its 5-channel softmax output (background, ET, NET, CC, ED). The three regions that
are actually scored -- WT (labels 1-4), TC (labels 1-3, "NC" in this project's metrics) and
ET (label 1) -- are read off that output by summing probabilities, and each is trained directly
with soft Dice + binary cross-entropy (nnU-Net's BraTS recipe, minus the separate sigmoid head).

Known limitation, accepted for this version: the loss only sees the sum P1+P2+P3, so nothing here
separates label 2 (non-enhancing) from label 3 (cystic). Their split is arbitrary; the three scored
regions are unaffected.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from monai.losses import DiceCELoss

NUM_CLASSES = 5
REGION_NAMES = ("WT", "TC", "ET")  # channel order of every (B, 3, ...) tensor in this module


def labels_to_region_targets(y: torch.Tensor) -> torch.Tensor:
    """Label volume (B,D,H,W) or (B,1,D,H,W), values in {0..4} -> float32 (B,3,D,H,W) of (WT, TC, ET)."""
    if y.ndim == 5 and y.shape[1] == 1:
        y = y[:, 0]
    if y.ndim != 4:
        raise ValueError(f"labels must have shape (B,D,H,W) or (B,1,D,H,W), got {tuple(y.shape)}")
    bad = (y < 0) | (y > NUM_CLASSES - 1)
    if bool(bad.any()):
        raise ValueError(f"label values must be in 0..{NUM_CLASSES - 1}, found {y[bad][0].item()}")
    return torch.stack([y > 0, (y >= 1) & (y <= 3), y == 1], dim=1).to(torch.float32)


def region_probs_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """5-channel logits (B,5,...) -> float32 region probabilities (B,3,...) = (WT, TC, ET).

    Built from sums of the softmax (WT = P1+P2+P3+P4, TC = P1+P2+P3, ET = P1), which is 1-P0 up to
    rounding, so ET <= TC <= WT holds exactly in floating point, not just in exact arithmetic.
    """
    if logits.ndim < 3 or logits.shape[1] != NUM_CLASSES:
        raise ValueError(f"logits must have {NUM_CLASSES} channels in dim 1, got shape {tuple(logits.shape)}")
    p = torch.softmax(logits.float(), dim=1)
    et = p[:, 1]
    tc = et + p[:, 2] + p[:, 3]
    wt = tc + p[:, 4]
    return torch.stack([wt, tc, et], dim=1).clamp(0.0, 1.0)


class RegionDiceBCELoss(nn.Module):
    """Mean over (WT, TC, ET) of soft Dice loss + binary cross-entropy, from 5-class logits.

    Dice is computed over the whole volume (batch size is fixed at 1 by 00_shared/CONTRACTS.md).
    ``smooth`` is 1.0, not nnU-Net's 1e-5, on purpose: with 1e-5 an empty-target region (no ET, on
    ~40% of patients) still costs ~1 even when almost nothing is predicted, while with 1.0 a correct
    "nothing" costs ~0 and a 500-voxel false blob costs ~1. The BCE is evaluated in log space from
    the logits, and everything runs in float32 whatever the logit dtype, so it is safe under AMP.
    """

    def __init__(self, smooth: float = 1.0, terms: str = "both") -> None:
        super().__init__()
        if isinstance(smooth, bool) or not isinstance(smooth, (int, float)) or not math.isfinite(smooth) or smooth <= 0:
            raise ValueError(f"smooth must be a positive finite number, got {smooth!r}")
        if terms not in ("both", "dice", "bce"):  # Addendum 7: train one half alone, as a diagnostic
            raise ValueError(f"terms must be 'both', 'dice' or 'bce', got {terms!r}")
        self.smooth = float(smooth)
        self.terms = terms

    def forward(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        lg = logits.float()
        target = labels_to_region_targets(y).to(lg.device)
        prob = region_probs_from_logits(lg)
        if prob.shape != target.shape:
            raise ValueError(f"logits {tuple(logits.shape)} and labels {tuple(y.shape)} do not describe the same volume")

        # log p and log(1-p) for each region, straight from the logits (no log of a probability sum)
        lse = torch.logsumexp(lg, dim=1)
        l0, l1, l2, l3, l4 = lg.unbind(dim=1)
        log_p = torch.stack([torch.logsumexp(lg[:, 1:], dim=1), torch.logsumexp(lg[:, 1:4], dim=1), l1], dim=1)
        log_q = torch.stack([
            l0,
            torch.logsumexp(torch.stack([l0, l4], dim=1), dim=1),
            torch.logsumexp(torch.stack([l0, l2, l3, l4], dim=1), dim=1),
        ], dim=1)
        log_p = log_p - lse.unsqueeze(1)
        log_q = log_q - lse.unsqueeze(1)

        dims = (0,) + tuple(range(2, lg.ndim))  # everything except the region channel
        dice = 1.0 - (2.0 * (prob * target).sum(dims) + self.smooth) / (prob.sum(dims) + target.sum(dims) + self.smooth)
        bce = -(target * log_p + (1.0 - target) * log_q).mean(dims)
        if self.terms == "dice":
            return dice.mean()
        if self.terms == "bce":
            return bce.mean()
        return (dice + bce).mean()


class RegionHybridLoss(nn.Module):
    """Region loss + the existing per-label Dice-CE, equal weights (SPEC.md Addendum 6).

    The first real run of RegionDiceBCELoss alone (2026-10-01) cut false ET but under-segmented real
    enhancing tumour (about two-thirds of the true volume). The per-label term keeps direct per-class
    pressure on label 1 and supervises the label 2 vs label 3 split the region term cannot see.
    """

    def __init__(self) -> None:
        super().__init__()
        self.region = RegionDiceBCELoss()
        self.per_label = DiceCELoss(to_onehot_y=True, softmax=True, include_background=True)  # = _build_loss("dice_ce")

    def forward(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        region_term = self.region(logits, y)                    # validates the label values first
        y5 = y if (y.ndim == 5 and y.shape[1] == 1) else y.unsqueeze(1)
        return region_term + self.per_label(logits.float(), y5).float()
