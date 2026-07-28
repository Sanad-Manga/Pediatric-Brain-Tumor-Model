"""Dummy 2D network and synthetic cache, so everything runs with no real data.

This is the only module that fabricates tensors. The previous section was built
and fully tested before any real data or model existed and that worked well --
this keeps that property for the 2D rewrite.

``DummySegNet2D`` honours the shared model interface from CONTRACTS.md::

    model(x) -> (seg_logits, features)

``features`` is the bottleneck embedding that section 02's CORAL and the PCA/LDA
plots consume. It is returned here purely so the interface is exercised.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .slices import write_subject_plane


class DummySegNet2D(nn.Module):
    """A tiny 2D encoder/decoder with the shared ``(seg_logits, features)`` interface.

    Deliberately not a real architecture -- section 01 owns that. This exists so
    the evaluation path, the CSV writer and the tests can run end to end without
    a trained model.
    """

    def __init__(self, in_channels: int = 4, num_classes: int = 5, width: int = 8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=3, padding=1),
            nn.InstanceNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width * 2, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(width * 2),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(width * 2, width, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, num_classes, kernel_size=1),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        bottleneck = self.encoder(x)
        seg_logits = self.decoder(bottleneck)
        features = self.pool(bottleneck).flatten(1)
        return seg_logits, features


def save_dummy_checkpoint(path: Path, model: nn.Module | None = None, **meta) -> Path:
    """Write a checkpoint a later ``eval --checkpoint`` run can load."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model = model if model is not None else DummySegNet2D()
    torch.save({"model_state_dict": model.state_dict(), **meta}, path)
    return path


def load_checkpoint(path: Path, model: nn.Module) -> nn.Module:
    """Load a checkpoint saved either as a bare state dict or wrapped in a dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict", payload) if isinstance(payload, dict) else payload
    model.load_state_dict(state)
    return model


# ------------------------------------------------------------- synthetic cache
def make_dummy_cache(
    out_dir: Path,
    subjects,
    cfg,
    volume_shape=(24, 24, 16),
    hospital_of=None,
    seed: int = 0,
    et_fraction: float = 0.6,
) -> Path:
    """Write a tiny cache in the real format, sliced from random volumes.

    Small volumes keep the CPU test suite fast while exercising the genuine
    axial/coronal shape difference: with ``(24, 24, 16)`` axial slices are 24x24
    and coronal are 24x16, the same mismatch as the real 240x240 vs 240x155.

    Every subject gets a tumour -- this is a tumour segmentation dataset, so a
    subject with no tumour at all is not a case the pipeline has to handle. What
    varies is tumour *extent*, which is what drives the per-patient slice counts
    the cap and the up-sampling exist to balance. ``et_fraction`` controls how
    many subjects have an enhancing core, mirroring the held-out set where 34 of
    82 subjects have no ET.
    """
    from .slices import extract_plane_slices

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    modalities = cfg.modalities

    for s, subject_id in enumerate(subjects):
        volumes = {}
        for m, mod in enumerate(modalities):
            # Raw, un-normalised, per-modality scale -- like the real cache.
            vol = rng.random(volume_shape).astype(np.float32) * (200.0 * (m + 1))
            vol[vol < 20.0] = 0.0          # a background the brain mask can find
            volumes[mod] = vol

        seg = np.zeros(volume_shape, dtype=np.uint8)
        sub_rng = np.random.default_rng(seed + s + 1)
        x, y, z = volume_shape

        # Tumour extent varies per subject, so patients contribute different
        # numbers of tumour-bearing slices.
        radius = int(sub_rng.integers(2, max(3, min(x, y) // 4)))
        depth = int(sub_rng.integers(2, max(3, z // 2)))
        cx = int(sub_rng.integers(radius + 1, x - radius - 1))
        cy = int(sub_rng.integers(radius + 1, y - radius - 1))
        lo = int(sub_rng.integers(0, max(1, z - depth)))
        hi = min(z, lo + depth)

        # Edema (4) outermost, non-enhancing (2) inside it, enhancing (1) core.
        seg[cx - radius:cx + radius, cy - radius:cy + radius, lo:hi] = 4
        if radius > 1:
            seg[cx - radius + 1:cx + radius - 1, cy - radius + 1:cy + radius - 1, lo:hi] = 2
        if sub_rng.random() < et_fraction:
            seg[cx - 1:cx + 1, cy - 1:cy + 1, lo:hi] = 1

        means, stds = [], []
        for mod in modalities:
            brain = volumes[mod][volumes[mod] != 0]
            means.append(float(brain.mean()) if brain.size else 0.0)
            stds.append(float(max(brain.std(), cfg.min_std)) if brain.size else cfg.min_std)

        hospital = hospital_of(subject_id) if hospital_of else "hospitalA"
        for plane in cfg.planes:
            axis = cfg.plane_axis(plane)
            write_subject_plane(
                out_dir, subject_id, plane, hospital,
                {m: extract_plane_slices(v, axis) for m, v in volumes.items()},
                extract_plane_slices(seg, axis),
                np.asarray(means, dtype=np.float32),
                np.asarray(stds, dtype=np.float32),
            )
    return out_dir


def make_dummy_manifests(manifest_dir: Path, counts=(3, 3, 3), prefix: str = "DUMMY") -> dict:
    """Write disjoint hospitalA / hospitalB / heldout manifests."""
    import json

    manifest_dir = Path(manifest_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    names = ("hospitalA", "hospitalB", "heldout")

    out, start = {}, 0
    for name, n in zip(names, counts):
        ids = [f"{prefix}-{i:04d}" for i in range(start, start + n)]
        start += n
        out[name] = ids
        with open(manifest_dir / f"{name}.json", "w", encoding="utf-8") as fh:
            json.dump(ids, fh, indent=2)
    return out
