r"""Add a display image (T1c, contrast-enhanced) to each patient of the comparison package, so the Model Comparison
page can draw the real scan under the outlines on every slice.

  comparison_cache/<patient>/t1c.npz      {"t1c": uint8 (240, 240, 155)}
  comparison_cache/<patient>/expert.npz   {"labels": uint8 (240, 240, 155)}  the expert segmentation (BraTS -seg), 0-4

Same array frame as labels_*.npz (both read straight from the BraTS NIfTI files), so slice z of the image matches
slice z of the labels with no rotation or flip. Intensities: brain voxels windowed to their 0.5-99.5 percentile range
and scaled to 0-255; background stays 0. About 4 MB per patient. Display only: never used by any model.
Needs the raw NIfTI (not in git).
"""
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from precompute_comparison import DATA_ROOT, OUT_DIR  # noqa: E402


def window_uint8(volume: np.ndarray) -> np.ndarray:
    brain = volume > 0
    if not brain.any():
        return np.zeros(volume.shape, np.uint8)
    lo, hi = np.percentile(volume[brain], [0.5, 99.5])
    scaled = np.clip((volume - lo) / max(hi - lo, 1e-6), 0, 1) * 255
    return np.where(brain, scaled, 0).round().astype(np.uint8)


def main():
    for out in sorted(p for p in OUT_DIR.iterdir() if p.is_dir()):
        pid = out.name
        vol = np.asarray(nib.load(str(DATA_ROOT / pid / f"{pid}-t1c.nii.gz")).dataobj, dtype=np.float32)
        labels = np.load(out / "labels_rstar.npz")["labels"]
        if vol.shape != labels.shape:
            raise SystemExit(f"{pid}: image shape {vol.shape} does not match the labels {labels.shape}")
        np.savez_compressed(out / "t1c.npz", t1c=window_uint8(vol))
        expert = np.rint(np.asarray(nib.load(str(DATA_ROOT / pid / f"{pid}-seg.nii.gz")).dataobj)).astype(np.uint8)
        np.savez_compressed(out / "expert.npz", labels=expert)
        print(pid, f"{(out / 't1c.npz').stat().st_size / 1e6:.1f} MB", flush=True)


if __name__ == "__main__":
    main()
