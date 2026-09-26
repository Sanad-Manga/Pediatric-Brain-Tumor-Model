"""Pre-processing exactly as it was measured on 2026-09-26 (see SPEC.md, In Scope).

3D branch: trilinear resample of the raw volume to 96^3, float16 round trip, per-channel z-score over voxels > 0.
2D branch: per-channel brain-only z-score at full resolution, slices in the team's slice-cache orientation, padded to the common size.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

#: The 2D pipeline's slices are stored flipped, so restacked 2D probabilities sit in a frame that differs from the NIfTI
#: volume by a flip of spatial axes 0 and 1. Verified on 111 patients (2026-09-26). Read at call time, never copied by value.
FRAME_FLIPS = (0, 1)
CUBE = 96
FLIPS_3D = ((), (0,), (1,), (2,))


# ------------------------------------------------------------------------------------------------- 3D branch
def zscore_3d(volume: np.ndarray) -> np.ndarray:
    """Zero-mean unit-variance over brain voxels (> 0); background stays exactly 0. Same rule as section 01's _zscore_normalize."""
    brain = volume > 0
    if not brain.any():
        return volume
    values = volume[brain]
    mean = values.mean()
    std = values.std()
    out = np.zeros_like(volume)
    if std < 1e-8:
        out[brain] = values - mean
    else:
        out[brain] = (values - mean) / std
    return out


def to_96cube(volume: np.ndarray, present) -> np.ndarray:
    """(4, X, Y, Z) raw -> (4, 96, 96, 96) float32, float16-representable. Absent channels are zeros."""
    out = np.zeros((volume.shape[0], CUBE, CUBE, CUBE), dtype=np.float32)
    for c in range(volume.shape[0]):
        if not present[c]:
            continue
        t = torch.from_numpy(np.ascontiguousarray(volume[c], dtype=np.float32))[None, None]
        r = F.interpolate(t, size=(CUBE, CUBE, CUBE), mode="trilinear", align_corners=False)[0, 0].numpy()
        out[c] = r.astype(np.float16).astype(np.float32)
    return out


def prepare_3d_input(volume: np.ndarray, present) -> np.ndarray:
    """(1, 4, 96, 96, 96) float32 model input: z-scored present channels, absent channels exactly zero AFTER the z-score."""
    cube = to_96cube(volume, present)
    out = np.zeros_like(cube)
    for c in range(cube.shape[0]):
        if present[c]:
            out[c] = zscore_3d(cube[c])
    return out[None]


# ------------------------------------------------------------------------------------------------- 2D branch
def normalize_2d(volume: np.ndarray):
    """Brain-only z-score at full resolution with float64 statistics. Returns (norm float32, brain mask (X, Y, Z))."""
    norm = np.zeros(volume.shape, dtype=np.float32)
    for c in range(volume.shape[0]):
        b = volume[c] > 0
        if not b.any():
            continue
        vb = volume[c][b].astype(np.float64)
        std = vb.std()
        if std < 1e-8:
            norm[c][b] = (volume[c][b] - vb.mean()).astype(np.float32)
        else:
            norm[c][b] = ((volume[c][b] - vb.mean()) / std).astype(np.float32)
    brain = (volume > 0).any(axis=0)
    return norm, brain


def extract_slices(norm: np.ndarray, brain: np.ndarray, plane: str):
    """Slices in the slice-cache orientation. Returns (indices, images (N, 4, H, W) float32, float16-representable)."""
    indices, images = [], []
    if plane == "axial":
        for k in range(norm.shape[3]):
            if brain[:, :, k].any():
                indices.append(k)
                images.append(norm[:, :, :, k][:, ::-1, ::-1])
    elif plane == "coronal":
        for k in range(norm.shape[2]):
            j = norm.shape[2] - 1 - k
            if brain[:, j, :].any():
                indices.append(k)
                images.append(norm[:, :, j, :][:, ::-1, :])
    else:
        raise ValueError(f"plane must be 'axial' or 'coronal', got {plane!r}")
    if not images:
        raise ValueError("no slice contains brain voxels")
    stack = np.stack([np.ascontiguousarray(im).astype(np.float16).astype(np.float32) for im in images])
    return indices, stack


def restack(slices: np.ndarray, plane: str, indices, spatial_shape) -> np.ndarray:
    """(C, N, H, W) per-slice arrays -> (C, X, Y, Z), each slice at its true index; slices without brain stay 0.
    The result is in the 2D frame (see FRAME_FLIPS), exactly like section 03's restack_volume."""
    out = np.zeros((slices.shape[0], *spatial_shape), dtype=slices.dtype)
    if plane == "axial":
        out[:, :, :, list(indices)] = np.moveaxis(slices, 1, 3)
    elif plane == "coronal":
        out[:, :, list(indices), :] = np.moveaxis(slices, 1, 2)
    else:
        raise ValueError(f"plane must be 'axial' or 'coronal', got {plane!r}")
    return out


def to_nifti_frame(probs: np.ndarray) -> np.ndarray:
    """(C, X, Y, Z) in the 2D frame -> the NIfTI frame."""
    return np.ascontiguousarray(np.flip(probs, axis=tuple(a + 1 for a in FRAME_FLIPS)))


def compute_pad(shape, target):
    pads = []
    for size, tgt in zip(shape, target):
        if size > tgt:
            raise ValueError(f"slice shape {tuple(shape)} exceeds common_size {tuple(target)}; padding cannot shrink a slice")
        total = tgt - size
        before = total // 2
        pads.append((before, total - before))
    return tuple(pads)


def pad_to(arr: np.ndarray, target):
    """Zero-pad the trailing two axes to `target` (centred; an odd remainder goes at the end). Same as section 03."""
    pad = compute_pad(arr.shape[-2:], target)
    lead = ((0, 0),) * (arr.ndim - 2)
    return np.pad(arr, lead + pad, mode="constant", constant_values=0), pad


def unpad(arr: np.ndarray, pad) -> np.ndarray:
    (top, bottom), (left, right) = pad
    return arr[..., top:arr.shape[-2] - bottom, left:arr.shape[-1] - right]
