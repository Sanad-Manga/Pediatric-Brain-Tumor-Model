"""Unit tests for prep.align (SPEC Req 1-8) on small synthetic volumes."""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prep import align  # noqa: E402


def phantom(size=64, seed=0, spacing=1.0):
    """Smooth, asymmetric blobs inside a sphere, zero outside (like a skull-stripped brain)."""
    rng = np.random.default_rng(seed)
    z, y, x = np.mgrid[:size, :size, :size].astype(np.float32)
    c = (size - 1) / 2
    vol = np.zeros((size,) * 3, np.float32)
    for _ in range(12):
        cz, cy, cx = rng.uniform(size * 0.25, size * 0.75, 3)
        s = rng.uniform(3, 8)
        vol += rng.uniform(0.5, 2.0) * np.exp(-((z - cz) ** 2 + (y - cy) ** 2 + (x - cx) ** 2) / (2 * s * s))
    vol[(z - c) ** 2 + (y - c) ** 2 + (x - c) ** 2 > (size * 0.42) ** 2] = 0
    img = sitk.GetImageFromArray(vol)
    img.SetSpacing((spacing,) * 3)
    return img


def euler(img, rot_deg=(0, 0, 0), shift=(0, 0, 0)):
    t = sitk.Euler3DTransform()
    t.SetCenter(img.TransformContinuousIndexToPhysicalPoint([(s - 1) / 2 for s in img.GetSize()]))
    t.SetRotation(*[math.radians(a) for a in rot_deg])
    t.SetTranslation(shift)
    return t


def brain_points(img, n=3000, seed=0):
    arr = sitk.GetArrayFromImage(img)
    idx = np.argwhere(arr > 0)
    pick = idx[np.random.default_rng(seed).choice(len(idx), min(n, len(idx)), replace=False)]
    return np.array([img.TransformIndexToPhysicalPoint([int(i) for i in p[::-1]]) for p in pick])


def test_output_on_fixed_grid_and_background_zero():  # Req 1, 3
    fixed = phantom(64)
    moving = sitk.RegionOfInterest(phantom(64), [48, 48, 48], [8, 8, 8])  # smaller grid, offset origin
    out, t, _ = align.align_to_reference(moving, fixed)
    assert out.GetSize() == fixed.GetSize()
    assert out.GetSpacing() == fixed.GetSpacing()
    assert out.GetOrigin() == fixed.GetOrigin()
    assert out.GetDirection() == fixed.GetDirection()
    assert sitk.GetArrayFromImage(out)[0, 0, 0] == 0  # outside the moving image's extent


def test_transform_is_rigid_six_parameters():  # Req 2
    fixed = phantom(48)
    _, t, _ = align.align_to_reference(sitk.Resample(fixed, fixed, euler(fixed, (2, 0, 0), (1, 0, 0))), fixed)
    assert isinstance(t, sitk.Euler3DTransform)
    assert len(t.GetParameters()) == 6


def test_recovers_known_shift_and_rotation():  # Req 7
    fixed = phantom(64)
    p = euler(fixed, rot_deg=(0, 0, 4), shift=(3, -2, 1))
    moving = sitk.Resample(fixed, fixed, p, sitk.sitkLinear, 0.0)  # moving(x) = fixed(P(x))
    _, t, _ = align.align_to_reference(moving, fixed)
    err = align.residual_error_mm(brain_points(fixed), t, p)  # perfect recovery: P(T(x)) = x
    assert err.mean() < 0.5, err.mean()


def test_self_alignment_is_identity():  # Req 8
    fixed = phantom(64, seed=3)
    _, t, _ = align.align_to_reference(fixed, fixed)
    s = align.transform_summary(t)
    assert max(abs(a) for a in s["rotation_deg"]) < 0.2
    assert max(abs(v) for v in s["translation_mm"]) < 0.2
    assert align.residual_error_mm(brain_points(fixed), t).mean() < 0.2


def test_deterministic():  # Req 6
    fixed = phantom(48, seed=5)
    moving = sitk.Resample(fixed, fixed, euler(fixed, (1, 2, 0), (2, 0, -1)), sitk.sitkLinear, 0.0)
    a = align.align_to_reference(moving, fixed)[1].GetParameters()
    b = align.align_to_reference(moving, fixed)[1].GetParameters()
    assert a == b


def _write_case(tmp, misalign=True):
    ref = phantom(48, seed=7)
    paths = {}
    for i, name in enumerate(align.SEQUENCES):
        img = ref if name == "t1c" or not misalign else sitk.Resample(ref, ref, euler(ref, (0, 1, 2), (1, -1, i)), sitk.sitkLinear, 0.0)
        paths[name] = tmp / f"case-{name}.nii.gz"
        sitk.WriteImage(img, str(paths[name]))
    return ref, paths


def test_align_sequences_keeps_t1c_and_aligns_rest(tmp_path):  # Req 4
    ref, paths = _write_case(tmp_path)
    out, report = align.align_sequences(paths)
    assert set(out) == set(align.SEQUENCES)
    np.testing.assert_array_equal(sitk.GetArrayFromImage(out["t1c"]), sitk.GetArrayFromImage(sitk.ReadImage(str(paths["t1c"]))))
    for name in ("t1n", "t2f", "t2w"):
        assert out[name].GetSize() == ref.GetSize() and "rotation_deg" in report[name]


def test_cli_writes_outputs(tmp_path):  # Req 5
    _, paths = _write_case(tmp_path)
    args = sum(([f"--{n}", str(p)] for n, p in paths.items()), []) + ["--out-dir", str(tmp_path / "out"), "--prefix", "p1"]
    assert align.main(args) == 0
    for n in align.SEQUENCES:
        assert (tmp_path / "out" / f"p1-{n}.nii.gz").is_file()
    rep = json.loads((tmp_path / "out" / "alignment.json").read_text())
    assert rep["t1c"] == {"reference": True} and set(rep) == set(align.SEQUENCES) | {"skip_below_mm"}


def test_cli_missing_input(tmp_path, capsys):  # Req 5, edge case
    _, paths = _write_case(tmp_path)
    paths["t2w"] = tmp_path / "nope.nii.gz"
    args = sum(([f"--{n}", str(p)] for n, p in paths.items()), []) + ["--out-dir", str(tmp_path / "out")]
    assert align.main(args) == 1
    err = capsys.readouterr().err
    assert "missing input" in err and "nope.nii.gz" in err


# ---- Addendum 1: skip realignment when already aligned (Req 13-17) ----

def _case(tmp, shifts):
    """t1c = phantom; each other sequence moved by shifts[name] = (rot_deg, shift_mm) or None (left exactly in place)."""
    ref = phantom(48, seed=11)
    paths = {}
    for name in align.SEQUENCES:
        s = shifts.get(name)
        img = ref if s is None else sitk.Resample(ref, ref, euler(ref, *s), sitk.sitkLinear, 0.0)
        paths[name] = tmp / f"s-{name}.nii.gz"
        sitk.WriteImage(img, str(paths[name]))
    return paths


def test_report_has_displacement_and_skipped(tmp_path):  # Req 13
    _, rep = align.align_sequences(_case(tmp_path, {}))
    for name in ("t1n", "t2f", "t2w"):
        assert isinstance(rep[name]["displacement_mm"], float) and isinstance(rep[name]["skipped"], bool)


def test_aligned_sequence_returned_identical(tmp_path):  # Req 14
    paths = _case(tmp_path, {})
    out, rep = align.align_sequences(paths)
    for name in ("t1n", "t2f", "t2w"):
        assert rep[name]["skipped"], rep[name]
        np.testing.assert_array_equal(sitk.GetArrayFromImage(out[name]),
                                      sitk.GetArrayFromImage(sitk.ReadImage(str(paths[name]))))


def test_misaligned_sequence_is_realigned(tmp_path):  # Req 15
    out, rep = align.align_sequences(_case(tmp_path, {"t2w": ((0, 0, 4), (3, -2, 1))}))
    assert not rep["t2w"]["skipped"] and rep["t2w"]["displacement_mm"] > 1.0
    assert rep["t1n"]["skipped"]


def test_threshold_zero_never_skips_and_other_grid_resampled(tmp_path):  # Req 16
    paths = _case(tmp_path, {})
    _, rep = align.align_sequences(paths, skip_below_mm=0)
    assert not any(rep[n]["skipped"] for n in ("t1n", "t2f", "t2w"))
    ref = sitk.ReadImage(str(paths["t1c"]))
    shifted = sitk.Image(ref)
    shifted.SetOrigin(tuple(o + 0.5 for o in ref.GetOrigin()))  # same voxels, different grid
    sitk.WriteImage(shifted, str(paths["t2f"]))
    out, rep = align.align_sequences(paths, skip_below_mm=100)
    assert not rep["t2f"]["skipped"]
    assert align.same_grid(out["t2f"], ref)


def test_cli_threshold_recorded(tmp_path):  # Req 17
    paths = _case(tmp_path, {})
    args = sum(([f"--{n}", str(p)] for n, p in paths.items()), []) + ["--out-dir", str(tmp_path / "o"), "--skip-below-mm", "0"]
    assert align.main(args) == 0
    rep = json.loads((tmp_path / "o" / "alignment.json").read_text())
    assert rep["skip_below_mm"] == 0 and not rep["t2w"]["skipped"]
    assert align.main(args[:-1] + ["-1"]) == 1
