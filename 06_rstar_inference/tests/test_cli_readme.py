"""Req 18 and 20: the path API, the command line, and the README."""
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from _helpers import TINY, make_volume, tiny_config

from rstar import ContractError, RStarSegmenter
from rstar.cli import main
from rstar.guards import IdentityStub

README = Path(__file__).resolve().parents[1] / "README.md"
AFFINE = np.array([[-1.0, 0, 0, 120.0], [0, -1.0, 0, 130.0], [0, 0, 1.0, -70.0], [0, 0, 0, 1.0]])


def _write(tmp_path, name, data, zooms=(1.0, 1.0, 1.0), affine=AFFINE):
    img = nib.Nifti1Image(np.asarray(data, dtype=np.float32), affine)
    img.header.set_zooms(zooms)
    path = tmp_path / f"{name}.nii.gz"
    nib.save(img, path)
    return str(path)


def _seg():
    return RStarSegmenter(tiny_config(), models_2d=[IdentityStub()], models_3d=[IdentityStub()])


def _paths(tmp_path, vol, names=("t1c", "t1n", "t2f", "t2w"), **kw):
    order = ("t1c", "t1n", "t2f", "t2w")
    return {n: _write(tmp_path, n, vol[order.index(n)], **kw) for n in names}


# ------------------------------------------------------------------------------------------------ Req 18
def test_segment_paths_reads_nifti_and_treats_an_omitted_sequence_as_absent(tmp_path):
    vol, _ = make_volume()
    paths = _paths(tmp_path, vol, names=("t1c", "t1n", "t2w"))
    result, affine = _seg().segment_paths({**paths, "t2f": None})
    assert result.mode == "3D-only" and result.diagnostics["present"] == (True, True, False, True)
    assert np.allclose(affine, AFFINE) and result.labels.shape == TINY
    assert any("t2f" in w for w in result.warnings)


@pytest.mark.parametrize("zooms, ok", [((1.0, 1.0, 1.0), True), ((0.95, 1.05, 1.0), True), ((2.0, 2.0, 2.0), False),
                                       ((1.0, 1.0, 3.0), False), ((1.2, 1.0, 1.0), False), ((0.8, 1.0, 1.0), False)])
def test_the_voxel_size_must_be_one_millimetre_isotropic_within_ten_percent(tmp_path, zooms, ok):
    vol, _ = make_volume()
    paths = _paths(tmp_path, vol, zooms=zooms)
    if ok:
        result, _ = _seg().segment_paths(paths)
        assert result.mode == "R*"
    else:
        with pytest.raises(ContractError, match="voxel size"):
            _seg().segment_paths(paths)


def test_the_et_rule_uses_the_voxel_volume_from_the_header(tmp_path):
    vol, _ = make_volume()
    plain = _seg().segment_paths(_paths(tmp_path, vol))[0]
    n = plain.diagnostics["et_voxels_before_rule"]
    assert n > 100
    threshold = n * 1.15                                     # between n * 1.0 mm3 and n * 1.331 mm3
    seg = RStarSegmenter(tiny_config(et_min_mm3=threshold), models_2d=[IdentityStub()], models_3d=[IdentityStub()])
    assert seg.segment_paths(_paths(tmp_path, vol))[0].diagnostics["et_relabelled"] is True          # 1 mm voxels: n mm3 < threshold
    (tmp_path / "grown").mkdir()
    grown = _paths(tmp_path / "grown", vol, zooms=(1.1, 1.1, 1.1))
    assert seg.segment_paths(grown)[0].diagnostics["et_relabelled"] is False                         # 1.1 mm voxels: 1.331 n >= threshold


def test_a_wrong_shape_or_no_files_is_a_contract_error(tmp_path):
    vol, _ = make_volume()
    bad = _write(tmp_path, "bad", vol[0][:, :, :30])
    with pytest.raises(ContractError, match="shape"):
        _seg().segment_paths({"t1c": bad})
    with pytest.raises(ContractError, match="no sequence file"):
        _seg().segment_paths({})


def test_the_cli_writes_a_uint8_nifti_with_the_input_affine_and_a_json_report(tmp_path, capsys):
    vol, _ = make_volume()
    paths = _paths(tmp_path, vol, names=("t1c", "t1n", "t2w"))
    out, report = tmp_path / "seg.nii.gz", tmp_path / "report.json"
    code = main(["--t1c", paths["t1c"], "--t1n", paths["t1n"], "--t2w", paths["t2w"], "--out", str(out), "--json", str(report)], segmenter=_seg())
    assert code == 0
    img = nib.load(str(out))
    assert img.shape == TINY and img.get_data_dtype() == np.uint8 and np.allclose(img.affine, AFFINE)
    data = json.loads(report.read_text(encoding="utf-8"))
    assert set(data) == {"status", "mode", "warnings", "diagnostics"} and data["mode"] == "3D-only"
    assert data["diagnostics"]["present"] == [True, True, False, True]
    assert "t2f" in " ".join(data["warnings"]) and "3D-only" in capsys.readouterr().out


def test_the_cli_exits_2_with_the_message_on_stderr_for_a_contract_error(tmp_path, capsys):
    vol, _ = make_volume()
    bad = _write(tmp_path, "bad", vol[0][:, :, :30])
    out = tmp_path / "seg.nii.gz"
    code = main(["--t1c", bad, "--out", str(out)], segmenter=_seg())
    assert code == 2 and not out.exists()
    assert "input rejected" in capsys.readouterr().err


def test_the_cli_requires_an_output_path(tmp_path):
    with pytest.raises(SystemExit) as exc:
        main(["--t1c", "x.nii.gz"], segmenter=_seg())
    assert exc.value.code == 2


# ------------------------------------------------------------------------------------------------ Req 20
def test_the_readme_carries_the_validity_disclaimer_and_documents_the_contract_and_modes():
    text = README.read_text(encoding="utf-8")
    for phrase in ("US BraTS-PEDs only", "radiologist review", "not a medical device", "Input contract", "R*", "3D-only"):
        assert phrase in text, phrase
