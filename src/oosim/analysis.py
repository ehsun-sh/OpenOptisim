"""Measurements computed from signals.

These are plain functions rather than components on purpose: the same reductions
are needed by the test suite, by scripts, and later by measurement blocks and the
GUI. Keeping them here means one implementation and one set of tests.

They are also where result data gets *reduced*. A browser must never receive a
million raw samples; it receives what these functions return.
"""

from __future__ import annotations

import numpy as np

from .signals import Band


def instantaneous_power(band: Band) -> np.ndarray:
    """Instantaneous power [W] per sample, summed over both polarizations."""
    return (np.abs(band.Ex.astype(np.complex128)) ** 2) + (
        np.abs(band.Ey.astype(np.complex128)) ** 2
    )


def rms_time_width(band: Band) -> float:
    """RMS width of the intensity envelope in time [s].

    The second central moment of ``|A(t)|**2``. For a Gaussian intensity profile
    ``exp(-(T/T0)**2)`` this equals ``T0 / sqrt(2)``, so *ratios* of this quantity
    track ``T1/T0`` exactly — which is what the dispersion validation compares
    against the analytical broadening factor.

    The time window is periodic, so this is only meaningful while the pulse stays
    well inside it. A pulse that has spread far enough to wrap around will report
    a width that is wrong rather than merely imprecise.
    """
    power = instantaneous_power(band)
    total = float(power.sum())
    if total <= 0.0:
        raise ValueError("cannot measure the width of a signal carrying no power")

    t = np.arange(band.num_samples, dtype=np.float64) / band.fs
    mean = float((power * t).sum() / total)
    variance = float((power * (t - mean) ** 2).sum() / total)
    return float(np.sqrt(variance))


def peak_power(band: Band) -> float:
    """Highest instantaneous power in the window [W]."""
    return float(instantaneous_power(band).max())
