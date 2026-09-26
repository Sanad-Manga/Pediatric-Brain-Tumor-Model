"""Req 11 and 12: section alias imports, hash verification, the pinned manifest, and the loaders' wiring."""
import hashlib
import json
import sys

import pytest
import torch

from rstar import ModelIntegrityError, RStarConfig
from rstar import models as model_loading
from rstar.config import DEFAULT_MANIFEST, load_manifest, validate_manifest
from rstar.sections import alias_name, import_submodule, load_section_package


# ------------------------------------------------------------------------------------------------ Req 11
def test_both_sections_load_under_aliases_without_touching_sys_path_or_src():
    path_before = list(sys.path)
    src_before = sys.modules.get("src")
    for section in ("01", "03"):
        pkg = load_section_package(section)
        assert pkg.__name__ == alias_name(section)
    assert hasattr(import_submodule("03", "model"), "build_model")
    assert hasattr(import_submodule("01", "model"), "FederatedUNet3D")
    assert sys.path == path_before
    assert sys.modules.get("src") is src_before


def test_the_two_aliases_are_different_packages_and_are_cached():
    a, b = load_section_package("01"), load_section_package("03")
    assert a is not b and a.__file__ != b.__file__
    assert load_section_package("01") is a and load_section_package("03") is b


def test_a_missing_section_directory_raises_file_not_found_naming_it(tmp_path, monkeypatch):
    monkeypatch.delitem(sys.modules, alias_name("01"), raising=False)
    with pytest.raises(FileNotFoundError, match="01_model_federated"):
        load_section_package("01", repo_root=tmp_path)


def test_an_unknown_section_is_rejected():
    with pytest.raises(ValueError, match="section"):
        load_section_package("02")


# ------------------------------------------------------------------------------------------------ Req 12
def _file(tmp_path, name="ckpt.bin", payload=b"weights"):
    p = tmp_path / name
    p.write_bytes(payload)
    return p, hashlib.sha256(payload).hexdigest()


def test_a_matching_hash_passes_and_returns_it(tmp_path):
    p, h = _file(tmp_path)
    assert model_loading.verify_sha256(p, h) == h


def test_a_mismatch_names_the_file_and_both_hashes(tmp_path):
    p, h = _file(tmp_path)
    wrong = "0" * 64
    with pytest.raises(ModelIntegrityError) as exc:
        model_loading.verify_sha256(p, wrong)
    message = str(exc.value)
    assert str(p) in message and wrong[:12] in message and h[:12] in message


def test_a_missing_file_raises_file_not_found_naming_the_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="nope.bin"):
        model_loading.verify_sha256(tmp_path / "nope.bin", "0" * 64)


def _manifest_for(tmp_path, entry_file, family_files):
    def entry(name, path):
        return {"name": name, "path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    manifest = {"ensemble_2d": entry("e2d", entry_file), "family_3d": [entry(f"m{i}", p) for i, p in enumerate(family_files)]}
    mpath = tmp_path / "manifest.json"
    mpath.write_text(json.dumps(manifest), encoding="utf-8")
    return mpath


def test_verify_hashes_can_be_switched_off_but_a_missing_file_still_fails(tmp_path):
    good, _ = _file(tmp_path, "a.bin")
    mpath = _manifest_for(tmp_path, good, [good] * 4)
    (tmp_path / "a.bin").write_bytes(b"changed")                  # the hash no longer matches
    strict = RStarConfig(models_root=tmp_path, manifest_path=mpath, device="cpu")
    with pytest.raises(ModelIntegrityError):
        model_loading.load_3d_family(strict)
    relaxed = RStarConfig(models_root=tmp_path, manifest_path=mpath, device="cpu", verify_hashes=False)
    with pytest.raises(Exception) as exc:                         # past the hash check the bytes are not a checkpoint
        model_loading.load_3d_family(relaxed)
    assert not isinstance(exc.value, ModelIntegrityError)
    (tmp_path / "a.bin").unlink()
    with pytest.raises(FileNotFoundError):
        model_loading.load_3d_family(relaxed)


def test_the_default_manifest_lists_one_2d_file_and_four_3d_files_with_relative_paths_and_hashes():
    manifest = load_manifest()
    assert DEFAULT_MANIFEST.exists()
    entries = [manifest["ensemble_2d"], *manifest["family_3d"]]
    assert len(manifest["family_3d"]) == 4 and len(entries) == 5
    for e in entries:
        assert len(e["sha256"]) == 64 and set(e["sha256"]) <= set("0123456789abcdef")
        assert not e["path"].startswith(("/", "\\")) and ":" not in e["path"]
    assert len({e["sha256"] for e in entries}) == 5


@pytest.mark.parametrize("mutate", [
    lambda m: m["family_3d"].pop(),
    lambda m: m["ensemble_2d"].update(path="C:/models/best.pt"),
    lambda m: m["ensemble_2d"].update(path="/models/best.pt"),
    lambda m: m["family_3d"][0].update(sha256="abc"),
    lambda m: m["family_3d"][1].update(sha256="G" * 64),
    lambda m: m.pop("ensemble_2d"),
])
def test_a_malformed_manifest_is_rejected(mutate):
    manifest = json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))
    mutate(manifest)
    with pytest.raises(ValueError):
        validate_manifest(manifest)


def test_the_3d_loader_builds_and_loads_a_checkpoint_after_verifying_it(tmp_path):
    model = import_submodule("01", "model").FederatedUNet3D()
    path = tmp_path / "m3d.pt"
    torch.save({"model_state": model.state_dict(), "epoch": 1}, path)
    mpath = _manifest_for(tmp_path, path, [path] * 4)
    cfg = RStarConfig(models_root=tmp_path, manifest_path=mpath, device="cpu")
    models, info = model_loading.load_3d_family(cfg)
    assert len(models) == 4 and len(info) == 4 and not models[0].training
    for a, b in zip(models[0].state_dict().values(), model.state_dict().values()):
        assert torch.equal(a, b)


def test_the_2d_loader_reads_an_ensemble_payload_and_rejects_anything_else(tmp_path):
    sec03 = import_submodule("03", "model")
    config_mod = import_submodule("03", "config")
    from rstar.config import REPO_ROOT

    cfg2d = config_mod.load_config(REPO_ROOT / "03_augmentation_eval" / "config.yaml")
    members = []
    for width in (4, 8):
        net = sec03.build_model(cfg2d, spatial_dims=2, width=width, depth=2)
        members.append({"model_state_dict": net.state_dict(), "width": width, "depth": 2})
    good = tmp_path / "ens.pt"
    torch.save({"ensemble": True, "members": members, "spatial_dims": 2}, good)
    other = tmp_path / "other.bin"
    other.write_bytes(b"x")
    mpath = _manifest_for(tmp_path, good, [other] * 4)
    cfg = RStarConfig(models_root=tmp_path, manifest_path=mpath, device="cpu")
    models, info = model_loading.load_2d_ensemble(cfg)
    assert len(models) == 2 and len(info) == 1
    bad = tmp_path / "plain.pt"
    torch.save({"model_state_dict": {}}, bad)
    with pytest.raises(ModelIntegrityError, match="not an ensemble"):
        model_loading.load_2d_ensemble(RStarConfig(models_root=tmp_path, manifest_path=_manifest_for(tmp_path, bad, [other] * 4), device="cpu"))
