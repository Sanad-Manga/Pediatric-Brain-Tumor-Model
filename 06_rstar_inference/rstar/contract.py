"""Input contract: what a scan must look like before R* will touch it."""
from __future__ import annotations

import numpy as np

from .config import SEQUENCES, RStarConfig

ZERO_FRACTION_ERROR = 0.15      # below this, a sequence looks like it still has a skull/noise background
ZERO_FRACTION_WARN = 0.25       # between the two limits: run, but say so
NEGATIVE_FRACTION_WARN = 0.01   # negative voxels are background to the models; real BraTS scans hold up to ~0.5%
NEGATIVE_FRACTION_ERROR = 0.05  # beyond this the scan is not in raw non-negative intensities


class ContractError(ValueError):
    """The input does not meet the contract. The message names the problem and the fix."""


def normalise_present(present) -> tuple[bool, ...]:
    if present is None:
        return (True,) * len(SEQUENCES)
    try:
        values = tuple(present)
    except TypeError:
        raise ContractError(f"`present` must be a sequence of {len(SEQUENCES)} booleans (t1c, t1n, t2f, t2w), got {present!r}") from None
    if len(values) != len(SEQUENCES) or not all(isinstance(v, (bool, np.bool_)) for v in values):
        raise ContractError(f"`present` must be a sequence of {len(SEQUENCES)} booleans (t1c, t1n, t2f, t2w), got {present!r}")
    return tuple(bool(v) for v in values)


def validate_input(volume, present, cfg: RStarConfig):
    """Check `volume` against the contract. Returns (present tuple, warnings). Never modifies `volume`."""
    present = normalise_present(present)
    if not any(present):
        raise ContractError("no sequence is present: declare at least one of t1c, t1n, t2f, t2w as present")
    arr = np.asarray(volume)
    expected = (len(SEQUENCES), *cfg.expected_shape)
    if arr.ndim != 4 or arr.shape != expected:
        raise ContractError(
            f"volume has shape {tuple(arr.shape)}, expected {expected} (channels t1c, t1n, t2f, t2w in BraTS space); "
            "register and resample the scan to the BraTS grid first")
    warnings: list[str] = []
    for c, name in enumerate(SEQUENCES):
        if not present[c]:
            continue                                    # the content of an absent sequence is ignored
        channel = arr[c]
        if not np.isfinite(channel).all():
            raise ContractError(f"sequence {name} contains NaN or infinite values; fix the export or declare it absent")
        negative_fraction = float((channel < 0).mean())
        if negative_fraction > NEGATIVE_FRACTION_ERROR:
            raise ContractError(
                f"sequence {name} has negative intensities in {negative_fraction:.1%} of its voxels; the models expect raw "
                "non-negative intensities (a few negative voxels are tolerated); check the export")
        if negative_fraction > NEGATIVE_FRACTION_WARN:
            warnings.append(f"sequence {name} has negative intensities in {negative_fraction:.1%} of its voxels; they are treated as background")
        if float(channel.max()) <= 0.0:
            raise ContractError(f"sequence {name} is declared present but is entirely zero; declare it absent instead")
        zero_fraction = float((channel == 0).mean())
        if zero_fraction < ZERO_FRACTION_ERROR:
            raise ContractError(
                f"sequence {name} has only {zero_fraction:.0%} exactly-zero voxels, so it looks like it is not skull-stripped "
                "(BraTS volumes are 30-60% background); skull-strip the scan first")
        if zero_fraction < ZERO_FRACTION_WARN:
            warnings.append(f"sequence {name} has only {zero_fraction:.0%} exactly-zero voxels; check that the scan is skull-stripped")
    return present, warnings
