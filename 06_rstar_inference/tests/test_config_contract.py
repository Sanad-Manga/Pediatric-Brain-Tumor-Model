"""Req 1, 2 and 19: configuration defaults/validation, the input contract, import hygiene."""
import math
import os
import subprocess
import sys

import numpy as np
import pytest
from _helpers import TINY, make_volume, tiny_config

from rstar import ContractError, RStarConfig
from rstar.contract import validate_input


# ------------------------------------------------------------------------------------------------ Req 1
def test_defaults_are_the_measured_recipe():
    c = RStarConfig()
    assert (c.w3d, c.background_scale, c.et_min_mm3, c.voxel_mm3) == (0.5, 0.5, 500.0, 1.0)
    assert (c.agreement_review, c.agreement_strong) == (0.70, 0.10)
    assert c.expected_shape == (240, 240, 155)
    assert c.verify_hashes is True and c.device == "auto"


@pytest.mark.parametrize("kwargs, field", [
    ({"w3d": -0.1}, "w3d"), ({"w3d": 1.1}, "w3d"), ({"w3d": math.nan}, "w3d"),
    ({"background_scale": 0.0}, "background_scale"), ({"background_scale": -1.0}, "background_scale"),
    ({"background_scale": math.inf}, "background_scale"),
    ({"et_min_mm3": -1.0}, "et_min_mm3"),
    ({"voxel_mm3": 0.0}, "voxel_mm3"), ({"voxel_mm3": -2.0}, "voxel_mm3"),
    ({"agreement_review": 1.5}, "agreement_review"), ({"agreement_strong": -0.1}, "agreement_strong"),
    ({"agreement_strong": 0.9, "agreement_review": 0.5}, "agreement_strong"),
    ({"expected_shape": (240, 240)}, "expected_shape"),
])
def test_invalid_config_raises_naming_the_field(kwargs, field):
    with pytest.raises(ValueError, match=field):
        RStarConfig(**kwargs)


def test_boundary_config_values_are_accepted():
    RStarConfig(w3d=0.0)
    RStarConfig(w3d=1.0)
    RStarConfig(et_min_mm3=0.0)
    RStarConfig(agreement_strong=0.5, agreement_review=0.5)


# ------------------------------------------------------------------------------------------------ Req 2
def _zero_fraction_channel(fraction):
    ch = np.full(TINY, 50.0, dtype=np.float32)
    ch.flat[: int(round(fraction * ch.size))] = 0.0
    return ch


def test_a_valid_volume_passes_without_warnings():
    vol, _ = make_volume()
    present, warnings = validate_input(vol, None, tiny_config())
    assert present == (True, True, True, True) and warnings == []


@pytest.mark.parametrize("bad, must_say", [
    (lambda v: v[0], "shape"),                                   # 3 dims
    (lambda v: v[:3], "shape"),                                  # 3 channels
    (lambda v: np.zeros((4, 48, 56, 39), np.float32) + 1.0, "shape"),
])
def test_wrong_shape_raises_contract_error(bad, must_say):
    vol, _ = make_volume()
    with pytest.raises(ContractError, match=must_say):
        validate_input(bad(vol), None, tiny_config())


@pytest.mark.parametrize("poison", [np.nan, np.inf, -np.inf])
def test_non_finite_values_in_a_present_sequence_raise(poison):
    vol, _ = make_volume()
    vol[1, 5, 5, 5] = poison
    with pytest.raises(ContractError, match="NaN or infinite"):
        validate_input(vol, None, tiny_config())


def _with_negative_fraction(fraction):
    vol, _ = make_volume()
    brain = np.flatnonzero(vol[0] > 0)
    vol[0].reshape(-1)[brain[: int(round(fraction * vol[0].size))]] = -50.0
    return vol


@pytest.mark.parametrize("fraction, outcome", [(0.0, "silent"), (1.0 / 107520, "silent"), (0.005, "silent"), (0.02, "warning"), (0.04, "warning"), (0.06, "error")])
def test_negative_voxels_are_background_up_to_a_limit(fraction, outcome):
    vol = _with_negative_fraction(fraction)
    if outcome == "error":
        with pytest.raises(ContractError, match="negative"):
            validate_input(vol, None, tiny_config())
        return
    present, warnings = validate_input(vol, None, tiny_config())
    assert present[0]
    assert (warnings == []) if outcome == "silent" else (len(warnings) == 1 and "negative" in warnings[0] and "t1c" in warnings[0])


def test_a_present_sequence_that_is_all_zero_raises_and_names_it():
    vol, _ = make_volume()
    vol[2] = 0.0
    with pytest.raises(ContractError, match="t2f.*entirely zero"):
        validate_input(vol, None, tiny_config())


def test_a_sequence_with_a_skull_background_raises_with_the_fix():
    vol, _ = make_volume()
    vol[3] = _zero_fraction_channel(0.10)
    with pytest.raises(ContractError, match="t2w.*skull-strip"):
        validate_input(vol, None, tiny_config())


def test_zero_fraction_just_under_the_error_limit_raises_and_just_over_only_warns():
    vol, _ = make_volume()
    vol[0] = _zero_fraction_channel(0.14)
    with pytest.raises(ContractError):
        validate_input(vol, None, tiny_config())
    vol[0] = _zero_fraction_channel(0.16)
    present, warnings = validate_input(vol, None, tiny_config())
    assert present[0] and len(warnings) == 1 and "t1c" in warnings[0]
    vol[0] = _zero_fraction_channel(0.24)
    assert len(validate_input(vol, None, tiny_config())[1]) == 1
    vol[0] = _zero_fraction_channel(0.26)
    assert validate_input(vol, None, tiny_config())[1] == []


@pytest.mark.parametrize("present", [(True, True, True), (1, 1, 1, 1), "tttt", 5])
def test_malformed_present_raises(present):
    vol, _ = make_volume()
    with pytest.raises(ContractError, match="present"):
        validate_input(vol, present, tiny_config())


def test_no_sequence_present_raises():
    vol, _ = make_volume()
    with pytest.raises(ContractError, match="no sequence"):
        validate_input(vol, (False, False, False, False), tiny_config())


@pytest.mark.parametrize("garbage", [np.nan, -5.0, 0.0])
def test_the_content_of_an_absent_sequence_is_ignored(garbage):
    vol, _ = make_volume()
    vol[2] = garbage
    present, warnings = validate_input(vol, (True, True, False, True), tiny_config())
    assert present == (True, True, False, True) and warnings == []


def test_validation_never_modifies_the_input():
    vol, _ = make_volume()
    vol[2, 0, 0, 0] = np.nan                                     # absent channel with a NaN: compare bytes, not values
    before = vol.copy()
    validate_input(vol, (True, True, False, True), tiny_config())
    assert vol.tobytes() == before.tobytes()
    vol.setflags(write=False)                                    # a read-only view is accepted
    validate_input(vol, (True, True, False, True), tiny_config())


# ------------------------------------------------------------------------------------------------ Req 19
def test_importing_rstar_leaves_sys_path_alone_and_does_not_load_the_sections():
    code = (
        "import sys\n"
        "before = list(sys.path)\n"
        "import rstar\n"
        "assert sys.path == before, 'sys.path changed'\n"
        "assert not [m for m in sys.modules if m.startswith('rstar_sec')], 'a section package was imported'\n"
        "assert 'src' not in sys.modules, 'a top-level src package was imported'\n"
    )
    package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ, PYTHONPATH=package_dir)
    proc = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, cwd=package_dir)
    assert proc.returncode == 0, proc.stderr
