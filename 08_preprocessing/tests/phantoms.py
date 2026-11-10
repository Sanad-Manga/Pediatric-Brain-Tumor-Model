"""A two-contrast phantom head: a 'T1c' and a 'T2w' of the same anatomy with opposite contrast.

Different contrast is the point - it is what makes mutual information necessary and what the judge has to cope with,
and the single-contrast blob phantom in test_align.py does not exercise it.
"""
from __future__ import annotations

import math

import numpy as np
import SimpleITK as sitk


def brain_pair(size: int = 88, seed: int = 3, spacing: float = 1.0) -> tuple[sitk.Image, sitk.Image]:
    """('T1c', 'T2w') of one phantom head: brain, cortical ribbon, ventricles, a rim-enhancing tumour, noise."""
    rng = np.random.default_rng(seed)
    z, y, x = np.mgrid[:size, :size, :size].astype(np.float32)
    c = (size - 1) / 2
    r = np.sqrt(((z - c) / (size * 0.40)) ** 2 + ((y - c) / (size * 0.44)) ** 2 + ((x - c) / (size * 0.36)) ** 2)
    brain = r <= 1.0
    cortex = brain & (r > 0.82)
    vent = np.zeros_like(brain)
    for dx in (-size / 16, size / 16):
        vent |= (((z - c) / (size * 0.15)) ** 2 + ((y - c - 2) / (size * 0.055)) ** 2
                 + ((x - c - dx) / (size * 0.033)) ** 2) <= 1.0
    tr = np.sqrt((z - c - size * 0.10) ** 2 + (y - c + size * 0.12) ** 2 + (x - c - size * 0.09) ** 2)
    core, rim = tr <= size * 0.073, (tr > size * 0.073) & (tr <= size * 0.099)
    blobs = np.zeros_like(z)
    for _ in range(14):
        bz, by, bx = rng.uniform(size * 0.3, size * 0.7, 3)
        s = rng.uniform(3, 7)
        blobs += rng.uniform(0.4, 1.0) * np.exp(-((z - bz) ** 2 + (y - by) ** 2 + (x - bx) ** 2) / (2 * s * s))

    def build(wm, gm, csf, tumour_core, tumour_rim, blob_gain, noise_seed):
        v = np.zeros((size,) * 3, np.float32)
        v[brain], v[cortex], v[vent] = wm, gm, csf
        v[core], v[rim] = tumour_core, tumour_rim
        v += blob_gain * blobs
        v[~brain] = 0
        v = sitk.GetArrayFromImage(sitk.SmoothingRecursiveGaussian(sitk.GetImageFromArray(v), 0.8))
        v += np.random.default_rng(noise_seed).normal(0, 0.02, v.shape).astype(np.float32)
        v[~brain] = 0
        img = sitk.GetImageFromArray(np.clip(v, 0, None))
        img.SetSpacing((spacing,) * 3)
        return img

    t1c = build(0.70, 0.45, 0.10, 0.25, 1.00, 0.06, 11)   # WM bright, CSF dark, enhancing rim
    t2w = build(0.30, 0.55, 0.95, 0.85, 0.40, 0.05, 12)   # CSF bright, bright tumour core
    return t1c, t2w


def euler(img: sitk.Image, rot_deg=(0, 0, 0), shift=(0, 0, 0)) -> sitk.Euler3DTransform:
    t = sitk.Euler3DTransform()
    t.SetCenter(img.TransformContinuousIndexToPhysicalPoint([(s - 1) / 2 for s in img.GetSize()]))
    t.SetRotation(*[math.radians(a) for a in rot_deg])
    t.SetTranslation([float(v) for v in shift])
    return t


def displaced(img: sitk.Image, rot_deg=(0.6, -0.4, 0.5), shift=(1.1, -0.8, 0.4)) -> tuple[sitk.Image, sitk.Euler3DTransform]:
    """`img` moved by a small rigid transform of about the size the BraTS disagreement is (~1.5 mm)."""
    t = euler(img, rot_deg, shift)
    return sitk.Resample(img, img, t, sitk.sitkLinear, 0.0, img.GetPixelID()), t
