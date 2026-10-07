"""RStarSegmenter: one co-registered BraTS-space scan in, one label volume (and honest diagnostics) out."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import fusion, guards, models as model_loading, preprocess, review_flags
from .config import SEQUENCES, RStarConfig
from .contract import ContractError, validate_input

INFER_CHUNK = 16                     # slices per 2D forward pass (same as section 03)
_SELF_CHECK_DONE = {"done": False}      # one automatic self-check per process


@dataclass
class RStarResult:
    labels: np.ndarray | None
    mode: str                                   # 'R*' or '3D-only'
    status: str                                 # 'ok' or 'review'
    warnings: list = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)
    review_mask: np.ndarray | None = None       # uint8, 0 = not flagged, k = flagged spot k (only when review_flags is on)
    review_spots: list | None = None            # issue #44 list (only when review_flags is on)


def _softmax(logits: np.ndarray, axis: int = 1) -> np.ndarray:
    shifted = logits - logits.max(axis=axis, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=axis, keepdims=True)


class RStarSegmenter:
    def __init__(self, config: RStarConfig | None = None, models_2d=None, models_3d=None, run_self_check=None):
        self.cfg = config or RStarConfig()
        injected = models_2d is not None and models_3d is not None
        self._checkpoints: list = []
        if models_2d is None:
            models_2d, info_2d = model_loading.load_2d_ensemble(self.cfg)
            self._checkpoints += info_2d
        if models_3d is None:
            models_3d, info_3d = model_loading.load_3d_family(self.cfg)
            self._checkpoints += info_3d
        self.models_2d = list(models_2d)
        self.models_3d = list(models_3d)
        self.device = self.cfg.resolved_device()
        if run_self_check is True:
            self.self_check()
        elif run_self_check is None and not injected and not _SELF_CHECK_DONE["done"]:
            self.self_check()
            _SELF_CHECK_DONE["done"] = True

    # ------------------------------------------------------------------------------------------ the two branches
    def _probs_2d(self, volume: np.ndarray, models=None) -> np.ndarray:
        """Shipped 2D ensemble on axial + coronal slices, averaged over members and planes, in the NIfTI frame. (5, X, Y, Z)."""
        models = self.models_2d if models is None else models
        norm, brain = preprocess.normalize_2d(volume)
        common = (256, 256)
        planes = []
        for plane in ("axial", "coronal"):
            indices, images = preprocess.extract_slices(norm, brain, plane)
            padded, pads = zip(*[preprocess.pad_to(im, common) for im in images])
            x = np.stack(padded)
            acc = 0
            for model in models:
                logits = []
                for start in range(0, x.shape[0], INFER_CHUNK):
                    chunk = torch.as_tensor(x[start:start + INFER_CHUNK], dtype=torch.float32).to(self.device)
                    with torch.no_grad(), torch.amp.autocast("cuda", enabled=str(self.device).startswith("cuda")):
                        out = model(chunk)[0]
                    logits.append(out.detach().float().cpu().numpy())
                probs = preprocess.unpad(_softmax(np.concatenate(logits, axis=0)), pads[0])
                acc = acc + preprocess.restack(probs.transpose(1, 0, 2, 3), plane, indices, volume.shape[1:])
            planes.append(acc / len(models))
        return preprocess.to_nifti_frame(sum(planes) / len(planes)).astype(np.float32)

    def _probs_3d(self, volume: np.ndarray, present, models=None) -> np.ndarray:
        """3D family (flip-TTA-4 per member), upsampled to the input grid. (5, X, Y, Z)."""
        models = self.models_3d if models is None else models
        x = torch.from_numpy(preprocess.prepare_3d_input(volume, present)).to(self.device)
        total = 0
        with torch.no_grad():
            for model in models:
                for flips in preprocess.FLIPS_3D:
                    xi = torch.flip(x, dims=[d + 2 for d in flips]) if flips else x
                    pr = torch.softmax(model(xi)[0], dim=1)
                    total = total + (torch.flip(pr, dims=[d + 2 for d in flips]) if flips else pr)
        mean = total / (len(models) * len(preprocess.FLIPS_3D))
        full = F.interpolate(mean.float(), size=tuple(volume.shape[1:]), mode="trilinear", align_corners=False)[0]
        return full.cpu().numpy().astype(np.float32)

    # ------------------------------------------------------------------------------------------ self check
    def self_check(self) -> bool:
        """Identity stub models through the module's own 2D and 3D paths; raises SelfCheckError on a geometry bug."""
        stub = [guards.IdentityStub()]
        return guards.run_self_check(
            lambda vol: self._probs_2d(vol, models=[m.to(self.device) for m in stub]),
            lambda vol, present: self._probs_3d(vol, present, models=[m.to(self.device) for m in stub]),
        )

    # ------------------------------------------------------------------------------------------ public API
    def segment(self, volume, present=None, voxel_mm3: float | None = None) -> RStarResult:
        t0 = time.time()
        present, warnings = validate_input(volume, present, self.cfg)
        vol = np.asarray(volume, dtype=np.float32)
        missing = [SEQUENCES[c] for c in range(len(SEQUENCES)) if not present[c]]
        if missing:
            mode = "3D-only"
            warnings.append(
                f"sequence(s) {', '.join(missing)} absent: 3D-only mode (the 2D models collapse without every sequence)"
                + ("; accuracy with two or more sequences missing is not validated" if len(missing) >= 2 else ""))
        else:
            mode = "R*"

        p3 = self._probs_3d(vol, present)
        agree = None
        status = "review" if len(missing) >= 2 else "ok"
        if mode == "R*":
            p2 = self._probs_2d(vol)
            agree = fusion.agreement(p2, p3)
            labels = fusion.fuse(p2, p3, self.cfg.w3d, self.cfg.background_scale)
            agree_status, agree_warnings = guards.agreement_status(agree, self.cfg)
            warnings += agree_warnings
            if agree_status == "review":
                status = "review"
        else:
            labels = fusion.argmax_labels(p3)               # measured 3D-only path: plain argmax, no background scaling

        review_mask = review_spots = None
        if self.cfg.review_flags:                           # from the labels BEFORE the 500 mm^3 rule (it may erase them)
            et_prob = p3[1] if mode == "3D-only" else self.cfg.w3d * p3[1] + (1.0 - self.cfg.w3d) * p2[1]
            review_mask, review_spots = self._review_flags(labels, np.clip(et_prob, 0.0, 1.0))

        cleanup = None
        if self.cfg.fragment_cleanup:                       # after the flags (they see everything), before the 500 mm^3 rule
            labels, wt_removed, et_cleaned = fusion.remove_fragments(labels, self.cfg.cleanup_min_voxels)
            cleanup = {"fragment_voxels_removed": wt_removed, "et_fragment_voxels_relabelled": et_cleaned}

        labels, et_before, relabelled = fusion.apply_small_et_rule(
            labels, self.cfg.et_min_mm3, self.cfg.voxel_mm3 if voxel_mm3 is None else voxel_mm3)
        diagnostics = {
            "mode": mode,
            "agreement": agree,
            "et_voxels_before_rule": et_before,
            "et_relabelled": relabelled,
            "present": present,
            "checkpoints": [{"name": c["name"], "sha256": c["sha256"]} for c in self._checkpoints],
            "elapsed_s": round(time.time() - t0, 2),
        }
        if review_spots is not None:
            diagnostics["review_spot_count"] = len(review_spots)
        if cleanup is not None:
            diagnostics.update(cleanup)
        return RStarResult(labels=labels.astype(np.uint8), mode=mode, status=status, warnings=warnings, diagnostics=diagnostics,
                           review_mask=review_mask, review_spots=review_spots)

    def _review_flags(self, pre_rule_labels: np.ndarray, et_prob: np.ndarray):
        """Spots of pre-rule ET with >= review_min_voxels voxels and mean ET probability < review_prob_cut (SPEC Addendum 2)."""
        spot_ids, spots = review_flags.find_et_spots(pre_rule_labels, et_prob)
        big = [s for s in spots if s.voxels >= self.cfg.review_min_voxels]
        decisions = review_flags.decide(big, size_cut=0, prob_cut=self.cfg.review_prob_cut, agree_cut=0.0, small_cut=0)
        return review_flags.review_outputs(spot_ids, big, decisions)

    def segment_paths(self, paths: dict):
        """paths maps 't1c','t1n','t2f','t2w' to NIfTI files (a missing or None entry = absent sequence).
        Returns (result, affine)."""
        import nibabel as nib

        volume = np.zeros((len(SEQUENCES), *self.cfg.expected_shape), dtype=np.float32)
        present, affine, zooms = [], None, None
        for c, name in enumerate(SEQUENCES):
            p = paths.get(name)
            present.append(p is not None)
            if p is None:
                continue
            img = nib.load(str(p))
            data = np.asarray(img.dataobj, dtype=np.float32)
            if tuple(data.shape) != tuple(self.cfg.expected_shape):
                raise ContractError(f"{name} has shape {tuple(data.shape)}, expected {tuple(self.cfg.expected_shape)}")
            z = tuple(float(v) for v in img.header.get_zooms()[:3])
            if any(not 0.9 - 1e-3 <= v <= 1.1 + 1e-3 for v in z):          # tolerance: NIfTI stores zooms as float32
                raise ContractError(f"{name} has voxel size {z} mm; the models were trained at 1 mm isotropic (0.9-1.1 mm); resample first")
            if affine is None:
                affine, zooms = img.affine, z
            volume[c] = data
        if affine is None:
            raise ContractError("no sequence file was given: provide at least one of t1c, t1n, t2f, t2w")
        return self.segment(volume, tuple(present), voxel_mm3=float(np.prod(zooms))), affine
