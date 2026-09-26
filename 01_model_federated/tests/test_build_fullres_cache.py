import json
from pathlib import Path

import nibabel as nib
import numpy as np

from tools.build_fullres_cache import build, process_one

SHAPE = (12, 10, 8)
# Written out here on purpose (not imported from the tool): the spec fixes the channel order, and a
# test that reads the order from the code under test can never notice the code changing it.
ORDER = ("t1c", "t1n", "t2f", "t2w")


def _save_nii(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.asarray(array, dtype=np.float32), np.eye(4)), str(path))


def _write_patient(raw_dir, sid, seed=0, seg_values=(0, 1, 2, 3, 4), zero_modality=None, skip=None,
                   odd_shape_for=None, odd_seg_shape=False):
    rng = np.random.default_rng(seed)
    raws = {}
    for m in ORDER:
        shape = (SHAPE[0] + 1, *SHAPE[1:]) if odd_shape_for == m else SHAPE
        arr = rng.random(shape).astype(np.float32) * 200.0 + 1.0
        arr[::4] = 0.0
        if m == zero_modality:
            arr[:] = 0.0
        raws[m] = arr
        if skip != m:
            _save_nii(raw_dir / sid / f"{sid}-{m}.nii.gz", arr)
    seg_shape = (SHAPE[0] + 1, *SHAPE[1:]) if odd_seg_shape else SHAPE
    seg = rng.choice(np.array(seg_values), size=seg_shape).astype(np.float32)
    _save_nii(raw_dir / sid / f"{sid}-seg.nii.gz", seg)
    return raws, seg


def test_writes_expected_files_dtypes_shapes_and_stats(tmp_path):  # Req 35
    raw, out = tmp_path / "raw", tmp_path / "out"
    raws, seg = _write_patient(raw, "S1")
    assert process_one(str(raw), str(out), "S1")[1] == "ok"

    img = np.load(out / "S1.img.npy")
    assert img.dtype == np.float16 and img.shape == (4, *SHAPE)
    for c, m in enumerate(ORDER):  # channel order t1c, t1n, t2f, t2w
        assert np.allclose(img[c].astype(np.float32), raws[m], rtol=1e-3, atol=0.05)
    saved_seg = np.load(out / "S1.seg.npy")
    assert saved_seg.dtype == np.uint8 and saved_seg.shape == SHAPE
    assert np.array_equal(saved_seg, seg.astype(np.uint8))

    stats = np.load(out / "S1.stats.npy")
    assert stats.dtype == np.float32 and stats.shape == (4, 2)
    for c, m in enumerate(ORDER):
        brain = raws[m][raws[m] > 0].astype(np.float64)
        assert np.allclose(stats[c], [brain.mean(), brain.std()], rtol=1e-4)


def test_modality_without_any_positive_voxel_gets_default_stats(tmp_path):  # Req 35
    raw, out = tmp_path / "raw", tmp_path / "out"
    _write_patient(raw, "S1", zero_modality="t2w")
    assert process_one(str(raw), str(out), "S1")[1] == "ok"
    assert np.load(out / "S1.stats.npy")[3].tolist() == [0.0, 1.0]


def test_atomic_writes_and_a_second_run_skips_existing_patients(tmp_path):  # Req 36
    raw, out = tmp_path / "raw", tmp_path / "out"
    for i in range(3):
        _write_patient(raw, f"S{i}", seed=i)
    first = build(raw, out, workers=1)
    assert first["ok"] == 3 and first["skipped"] == 0 and not first["errors"]
    assert not list(out.glob("*.tmp")), "a temp file was left behind"

    before = {p.name: p.stat().st_mtime_ns for p in out.glob("*.npy")}
    assert len(before) == 9
    second = build(raw, out, workers=1)
    assert second["ok"] == 0 and second["skipped"] == 3
    assert {p.name: p.stat().st_mtime_ns for p in out.glob("*.npy")} == before, "existing files were rewritten"


def test_bad_patients_are_isolated_leave_no_files_and_do_not_stop_the_batch(tmp_path):  # Req 36
    raw, out = tmp_path / "raw", tmp_path / "out"
    _write_patient(raw, "GOOD", seed=1)
    _write_patient(raw, "MISSING", seed=2, skip="t1n")
    _write_patient(raw, "SHAPE", seed=3, odd_shape_for="t2f")
    _write_patient(raw, "LABEL", seed=4, seg_values=(0, 1, 7))
    _write_patient(raw, "SEGSHAPE", seed=5, odd_seg_shape=True)  # the modalities agree; only the seg differs

    summary = build(raw, out, workers=1)
    assert summary["ok"] == 1 and summary["skipped"] == 0
    assert {e["subject_id"] for e in summary["errors"]} == {"MISSING", "SHAPE", "LABEL", "SEGSHAPE"}
    for bad in ("MISSING", "SHAPE", "LABEL", "SEGSHAPE"):
        assert list(out.glob(f"{bad}.*")) == [], f"{bad} left output files behind"
    assert len(list(out.glob("GOOD.*"))) == 3
    on_disk = json.loads((out / "_build_summary.json").read_text())
    assert on_disk["ok"] == 1 and len(on_disk["errors"]) == 4


def test_arrays_are_written_to_temp_files_and_only_then_renamed(tmp_path, monkeypatch):  # Req 36
    raw, out = tmp_path / "raw", tmp_path / "out"
    _write_patient(raw, "S1")
    names, real_save = [], np.save

    def spy(fh, arr, *args, **kwargs):
        names.append(Path(fh.name).name)
        return real_save(fh, arr, *args, **kwargs)

    monkeypatch.setattr(np, "save", spy)
    assert process_one(str(raw), str(out), "S1")[1] == "ok"
    assert len(names) == 3 and all(n.endswith(".tmp") for n in names), names
    assert not list(out.glob("*.tmp")) and len(list(out.glob("S1.*"))) == 3


def test_a_failure_part_way_through_writing_leaves_nothing_behind(tmp_path, monkeypatch):  # Req 36
    raw, out = tmp_path / "raw", tmp_path / "out"
    _write_patient(raw, "S1")
    calls, real_save = {"n": 0}, np.save

    def flaky(fh, arr, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("disk full")
        return real_save(fh, arr, *args, **kwargs)

    monkeypatch.setattr(np, "save", flaky)
    assert process_one(str(raw), str(out), "S1")[1] == "error"
    assert list(out.iterdir()) == [], "a half-written patient left files behind"
