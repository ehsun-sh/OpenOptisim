"""Unit tests for the core data model: units, context, and signals."""

from __future__ import annotations

import math

import numpy as np
import pytest

from oosim import Band, NoiseBin, OpticalSignal, SimulationContext
from oosim.units import (
    dbm_to_w,
    frequency_to_wavelength,
    from_si,
    known_units,
    to_si,
    w_to_dbm,
    wavelength_to_frequency,
)

# --------------------------------------------------------------------------
# Units
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("dbm", "watts"), [(0.0, 1e-3), (10.0, 1e-2), (-30.0, 1e-6), (30.0, 1.0)])
def test_dbm_watt_conversion(dbm: float, watts: float) -> None:
    assert dbm_to_w(dbm) == pytest.approx(watts, rel=1e-12)
    assert w_to_dbm(watts) == pytest.approx(dbm, abs=1e-12)


def test_zero_power_is_minus_infinity_not_an_error() -> None:
    """A dark port is a normal state, not a domain error."""
    assert w_to_dbm(0.0) == -math.inf


@pytest.mark.parametrize("unit", sorted(known_units()))
def test_unit_conversions_round_trip(unit: str) -> None:
    value = 12.5
    assert from_si(to_si(value, unit), unit) == pytest.approx(value, rel=1e-12)


def test_unknown_unit_is_rejected_with_a_useful_message() -> None:
    with pytest.raises(ValueError, match="unknown unit 'furlong'"):
        to_si(1.0, "furlong")


def test_wavelength_frequency_round_trip() -> None:
    assert frequency_to_wavelength(wavelength_to_frequency(1550e-9)) == pytest.approx(
        1550e-9, rel=1e-15
    )


# --------------------------------------------------------------------------
# SimulationContext
# --------------------------------------------------------------------------


def test_context_derives_consistent_time_and_frequency_parameters() -> None:
    ctx = SimulationContext(bit_rate=10e9, samples_per_symbol=16, sequence_length=64)
    assert ctx.sample_rate == 160e9
    assert ctx.num_samples == 1024
    assert ctx.time_window == pytest.approx(1024 / 160e9)
    assert ctx.time_step == pytest.approx(1 / 160e9)
    assert ctx.time_axis().shape == (1024,)


def test_context_rejects_sub_nyquist_sampling() -> None:
    with pytest.raises(ValueError, match="Nyquist"):
        SimulationContext(bit_rate=10e9, samples_per_symbol=1, sequence_length=8)


def test_precision_selects_the_field_dtype() -> None:
    single = SimulationContext(bit_rate=1e9, samples_per_symbol=2, sequence_length=4)
    double = SimulationContext(
        bit_rate=1e9, samples_per_symbol=2, sequence_length=4, precision="double"
    )
    assert single.complex_dtype == np.complex64
    assert double.complex_dtype == np.complex128


def test_rng_streams_are_reproducible_and_independent() -> None:
    """Each block's noise must depend on its own identity and nothing else.

    If streams were drawn from a shared generator, adding a block anywhere would
    change the noise realisation of every block downstream of it, and no result
    involving noise could ever be regression-tested.
    """
    ctx = SimulationContext(bit_rate=1e9, samples_per_symbol=2, sequence_length=4, seed=42)

    assert ctx.rng("laser1").normal(size=5).tolist() == ctx.rng("laser1").normal(size=5).tolist()
    assert ctx.rng("laser1").normal(size=5).tolist() != ctx.rng("laser2").normal(size=5).tolist()

    other_seed = SimulationContext(bit_rate=1e9, samples_per_symbol=2, sequence_length=4, seed=43)
    assert (
        ctx.rng("laser1").normal(size=5).tolist()
        != other_seed.rng("laser1").normal(size=5).tolist()
    )


# --------------------------------------------------------------------------
# Signals
# --------------------------------------------------------------------------


def _band(n: int = 8, amplitude: complex = 1.0 + 0j, f0: float = 193.4e12) -> Band:
    return Band(
        Ex=np.full(n, amplitude, dtype=np.complex64),
        Ey=np.zeros(n, dtype=np.complex64),
        f0=f0,
        fs=160e9,
    )


def test_band_fields_are_read_only() -> None:
    """Blocks receive signals they must not modify; the arrays enforce it.

    Immutability is what lets metadata-only blocks share buffers instead of
    copying hundreds of megabytes per span.
    """
    band = _band()
    with pytest.raises(ValueError, match="read-only"):
        band.Ex[0] = 5.0


def test_band_average_power_sums_both_polarizations() -> None:
    n = 16
    band = Band(
        Ex=np.full(n, 0.1, dtype=np.complex64),
        Ey=np.full(n, 0.2, dtype=np.complex64),
        f0=193.4e12,
        fs=160e9,
    )
    assert band.average_power() == pytest.approx(0.01 + 0.04, rel=1e-6)


def test_band_rejects_mismatched_polarizations() -> None:
    with pytest.raises(ValueError, match="same length"):
        Band(
            Ex=np.zeros(8, dtype=np.complex64),
            Ey=np.zeros(4, dtype=np.complex64),
            f0=193.4e12,
            fs=160e9,
        )


def test_band_rejects_real_fields() -> None:
    """A real array here means someone forgot the envelope is complex."""
    with pytest.raises(TypeError, match="must be complex"):
        Band(Ex=np.zeros(8), Ey=np.zeros(8, dtype=np.complex64), f0=193.4e12, fs=160e9)


def test_scaling_amplitude_scales_power_quadratically() -> None:
    band = _band(amplitude=1.0 + 0j)
    assert band.scale_amplitude(0.5).average_power() == pytest.approx(0.25, rel=1e-6)


def test_optical_signal_rejects_duplicate_centre_frequencies() -> None:
    with pytest.raises(ValueError, match="coherent addition"):
        OpticalSignal(bands=(_band(f0=193.4e12), _band(f0=193.4e12)))


def test_signal_and_noise_power_are_reported_separately() -> None:
    """Keeping ASE out of the sampled bands is what allows a realistic sample rate."""
    signal = OpticalSignal(
        bands=(_band(amplitude=0.1 + 0j),),
        noise=(NoiseBin(f_start=193.0e12, f_end=193.1e12, psd_x=1e-18, psd_y=1e-18),),
    )
    assert signal.signal_power() == pytest.approx(0.01, rel=1e-6)
    assert signal.noise_power() == pytest.approx(2e-18 * 0.1e12, rel=1e-9)
    assert signal.total_power() == pytest.approx(
        signal.signal_power() + signal.noise_power(), rel=1e-9
    )


def test_band_lookup_by_centre_frequency() -> None:
    signal = OpticalSignal(bands=(_band(f0=193.4e12), _band(f0=193.5e12)))
    assert signal.band_at(193.5e12).f0 == 193.5e12
    with pytest.raises(KeyError):
        signal.band_at(190.0e12)
