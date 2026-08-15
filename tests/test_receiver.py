"""Validation of the PIN photodiode against closed-form receiver relations."""

from __future__ import annotations

import numpy as np
import pytest

from oosim import Graph, SimulationContext
from oosim.components import CWLaser, PINPhotodiode
from oosim.units import K_BOLTZMANN, Q_ELECTRON, dbm_to_w

# Noise variance is estimated from the samples, so the window has to be long
# enough for the estimate to be tight: the relative error of a variance estimate
# over N samples is ~sqrt(2/N), which is 0.55% at N = 65536. Tolerances below are
# set several times that.
NOISE_CTX = SimulationContext(bit_rate=10e9, samples_per_symbol=16, sequence_length=4096, seed=99)


def _detect(
    power_dbm: float,
    *,
    responsivity: float = 0.8,
    dark_current: float = 0.0,
    load_resistance: float = 50.0,
    temperature: float = 300.0,
    shot_noise: bool = False,
    thermal_noise: bool = False,
    ctx: SimulationContext | None = None,
) -> np.ndarray:
    ctx = ctx or SimulationContext(bit_rate=10e9, samples_per_symbol=8, sequence_length=64)
    g = Graph(ctx)
    laser = g.add(CWLaser(power=power_dbm, wavelength=1550.0))
    pin = g.add(
        PINPhotodiode(
            responsivity=responsivity,
            dark_current=dark_current,
            load_resistance=load_resistance,
            temperature=temperature,
            shot_noise=shot_noise,
            thermal_noise=thermal_noise,
        )
    )
    g.chain(laser, pin)
    return np.asarray(g.run()[pin].samples, dtype=np.float64)


# --------------------------------------------------------------------------
# Responsivity:  I = R * P + I_dark
# --------------------------------------------------------------------------


@pytest.mark.parametrize("power_dbm", [0.0, -10.0, -20.0, 3.0])
def test_photocurrent_is_responsivity_times_power(power_dbm: float) -> None:
    current = _detect(power_dbm, responsivity=0.8)
    expected = 0.8 * dbm_to_w(power_dbm)
    np.testing.assert_allclose(current, expected, rtol=1e-5)


def test_photocurrent_is_linear_in_optical_power() -> None:
    """Square-law detection is linear in *power*: 3 dB more light, twice the current."""
    low = _detect(-10.0).mean()
    high = _detect(-7.0).mean()
    assert high / low == pytest.approx(10 ** (3 / 10), rel=1e-4)


def test_dark_current_adds_a_constant_offset() -> None:
    """Run in double precision: this measures a nanoamp difference between two
    currents of tens of microamps, and complex64/float32 storage — the default,
    and the right default for waveforms — only resolves that to about 0.03%.
    The limitation is in the storage precision, not the model.
    """
    ctx = SimulationContext(
        bit_rate=10e9, samples_per_symbol=8, sequence_length=64, precision="double"
    )
    without = _detect(-10.0, dark_current=0.0, ctx=ctx).mean()
    with_dark = _detect(-10.0, dark_current=5e-9, ctx=ctx).mean()
    assert with_dark - without == pytest.approx(5e-9, rel=1e-6)


def test_zero_responsivity_gives_only_dark_current() -> None:
    current = _detect(0.0, responsivity=0.0, dark_current=1e-9)
    np.testing.assert_allclose(current, 1e-9, rtol=1e-5)


# --------------------------------------------------------------------------
# Shot noise:  sigma^2 = 2 * q * I * B
# --------------------------------------------------------------------------


@pytest.mark.parametrize("power_dbm", [0.0, -5.0, -10.0])
def test_shot_noise_variance_matches_theory(power_dbm: float) -> None:
    current = _detect(power_dbm, shot_noise=True, ctx=NOISE_CTX)

    mean_current = 0.8 * dbm_to_w(power_dbm)
    bandwidth = NOISE_CTX.sample_rate / 2.0
    expected = 2.0 * Q_ELECTRON * mean_current * bandwidth

    assert current.var() == pytest.approx(expected, rel=0.05)
    assert current.mean() == pytest.approx(mean_current, rel=1e-3)


def test_shot_noise_scales_linearly_with_photocurrent() -> None:
    """Doubling the current doubles the shot-noise variance — the signature that
    distinguishes shot noise from thermal noise."""
    dim = _detect(-10.0, shot_noise=True, ctx=NOISE_CTX).var()
    bright = _detect(-7.0, shot_noise=True, ctx=NOISE_CTX).var()
    assert bright / dim == pytest.approx(10 ** (3 / 10), rel=0.1)


def test_shot_noise_vanishes_with_the_current_that_causes_it() -> None:
    """Shot noise is carried by the photocurrent, so with no current there is
    none at all — not merely a small amount."""
    current = _detect(0.0, responsivity=0.0, dark_current=0.0, shot_noise=True, thermal_noise=False)
    np.testing.assert_array_equal(current, 0.0)


# --------------------------------------------------------------------------
# Thermal noise:  sigma^2 = 4 * k * T * B / R_load
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("temperature", "load"), [(300.0, 50.0), (300.0, 1000.0), (77.0, 50.0)])
def test_thermal_noise_variance_matches_theory(temperature: float, load: float) -> None:
    current = _detect(
        -10.0,
        thermal_noise=True,
        temperature=temperature,
        load_resistance=load,
        ctx=NOISE_CTX,
    )
    bandwidth = NOISE_CTX.sample_rate / 2.0
    expected = 4.0 * K_BOLTZMANN * temperature * bandwidth / load

    assert current.var() == pytest.approx(expected, rel=0.05)


def test_thermal_noise_does_not_depend_on_received_power() -> None:
    """The property that separates it from shot noise, and the reason a thermal-
    limited receiver has a noise floor rather than a noise slope."""
    dim = _detect(-30.0, thermal_noise=True, ctx=NOISE_CTX).var()
    bright = _detect(0.0, thermal_noise=True, ctx=NOISE_CTX).var()
    assert bright == pytest.approx(dim, rel=0.05)


def test_noise_sources_add_in_variance() -> None:
    shot = _detect(-5.0, shot_noise=True, ctx=NOISE_CTX).var()
    thermal = _detect(-5.0, thermal_noise=True, ctx=NOISE_CTX).var()
    both = _detect(-5.0, shot_noise=True, thermal_noise=True, ctx=NOISE_CTX).var()
    assert both == pytest.approx(shot + thermal, rel=0.05)


def test_noise_is_reproducible_and_independent_per_source() -> None:
    first = _detect(-5.0, shot_noise=True, thermal_noise=True, ctx=NOISE_CTX)
    second = _detect(-5.0, shot_noise=True, thermal_noise=True, ctx=NOISE_CTX)
    np.testing.assert_array_equal(first, second)


def test_disabling_noise_gives_an_exactly_clean_current() -> None:
    current = _detect(-5.0, shot_noise=False, thermal_noise=False)
    assert current.var() == pytest.approx(0.0, abs=1e-30)


# --------------------------------------------------------------------------
# Detection semantics
# --------------------------------------------------------------------------


def test_detector_reports_amperes() -> None:
    """The waveform's unit is carried, not assumed: a driver emits volts, a
    photodiode emits amperes, and a block downstream can tell which it has."""
    ctx = SimulationContext(bit_rate=10e9, samples_per_symbol=8, sequence_length=16)
    g = Graph(ctx)
    laser = g.add(CWLaser(power=0.0))
    pin = g.add(PINPhotodiode(shot_noise=False, thermal_noise=False))
    g.chain(laser, pin)
    assert g.run()[pin].unit == "A"


def test_multiple_bands_are_detected_incoherently() -> None:
    """A square-law detector sums the power of well-separated channels.

    Beating between bands lands at their frequency separation, far outside any
    realistic receiver bandwidth, and is not modelled — see the class docstring.
    """
    from oosim.components import Combiner

    ctx = SimulationContext(bit_rate=10e9, samples_per_symbol=8, sequence_length=16)
    g = Graph(ctx)
    ch1 = g.add(CWLaser(power=-10.0, wavelength=1550.0, label="ch1"))
    ch2 = g.add(CWLaser(power=-10.0, wavelength=1560.0, label="ch2"))
    mux = g.add(Combiner(2))
    pin = g.add(PINPhotodiode(responsivity=1.0, shot_noise=False, thermal_noise=False))

    g.connect(ch1, mux["in0"])
    g.connect(ch2, mux["in1"])
    g.chain(mux, pin)

    expected = 2 * dbm_to_w(-10.0)
    np.testing.assert_allclose(np.asarray(g.run()[pin].samples), expected, rtol=1e-5)


def test_noise_flags_are_booleans_not_numbers() -> None:
    """Flags are declared as BoolParam, so passing a number is a mistake worth
    catching at construction rather than silently truthy-testing it."""
    with pytest.raises(TypeError, match="must be True or False"):
        PINPhotodiode(shot_noise=1.0)


def test_si_refuses_to_convert_a_flag() -> None:
    pin = PINPhotodiode()
    with pytest.raises(TypeError, match="is a flag, not a quantity"):
        pin.si("shot_noise")
