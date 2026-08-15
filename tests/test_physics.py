"""Validation against closed-form results.

Every physics block is checked against an analytical expression, not against
another simulator. Comparing with a commercial tool needs a licence and cannot
run in CI, so it can confirm a model once but cannot keep it correct.

Each test names the relation it verifies.
"""

from __future__ import annotations

import numpy as np
import pytest

from oosim import Graph, SimulationContext
from oosim.components import Attenuator, Combiner, CWLaser, Fiber, PowerMeter
from oosim.units import dbm_to_w, w_to_dbm

# dBm tolerance. Fields are complex64 by default (~1e-7 relative), so a power
# error of 1e-6 relative is ~4e-6 dB; 1e-4 dB is comfortably above the noise
# floor of the representation and well below anything physically meaningful.
DB_TOL = 1e-4


@pytest.fixture
def ctx() -> SimulationContext:
    return SimulationContext(bit_rate=10e9, samples_per_symbol=16, sequence_length=64, seed=1234)


# --------------------------------------------------------------------------
# Fiber attenuation:  P_out = P_in * 10 ** (-alpha_dB_per_km * L_km / 10)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("p_in_dbm", "length_km", "alpha_db_per_km"),
    [
        (0.0, 80.0, 0.2),  # 16 dB — the canonical C-band span
        (10.0, 100.0, 0.2),  # 20 dB
        (-3.0, 40.0, 0.25),  # 10 dB
        (0.0, 0.0, 0.2),  # zero length: lossless
        (5.0, 25.0, 0.35),  # O-band-ish loss coefficient
    ],
)
def test_fiber_attenuation_matches_beer_lambert(
    ctx: SimulationContext, p_in_dbm: float, length_km: float, alpha_db_per_km: float
) -> None:
    g = Graph(ctx)
    laser = g.add(CWLaser(power=p_in_dbm, wavelength=1550.0))
    fiber = g.add(Fiber(length=length_km, attenuation=alpha_db_per_km))
    meter = g.add(PowerMeter())
    g.chain(laser, fiber, meter)

    reading = g.run()[meter]

    expected_dbm = p_in_dbm - alpha_db_per_km * length_km
    assert reading.power_dbm == pytest.approx(expected_dbm, abs=DB_TOL)

    # And in linear units, against the exponential form directly.
    expected_w = dbm_to_w(p_in_dbm) * 10 ** (-alpha_db_per_km * length_km / 10)
    assert reading.power_w == pytest.approx(expected_w, rel=1e-5)


def test_attenuation_is_multiplicative_across_cascaded_spans(ctx: SimulationContext) -> None:
    """Two 40 km spans must equal one 80 km span. Catches per-span offset errors."""
    g = Graph(ctx)
    laser = g.add(CWLaser(power=0.0))
    span_a = g.add(Fiber(length=40.0, attenuation=0.2, label="span_a"))
    span_b = g.add(Fiber(length=40.0, attenuation=0.2, label="span_b"))
    meter = g.add(PowerMeter())
    g.chain(laser, span_a, span_b, meter)

    assert g.run()[meter].power_dbm == pytest.approx(-16.0, abs=DB_TOL)


def test_attenuator_and_fiber_agree_for_equal_loss(ctx: SimulationContext) -> None:
    """A 16 dB attenuator and an 80 km / 0.2 dB/km span must give the same power."""
    readings = []
    for element in (Attenuator(attenuation=16.0), Fiber(length=80.0, attenuation=0.2)):
        g = Graph(ctx)
        laser = g.add(CWLaser(power=0.0))
        loss = g.add(element)
        meter = g.add(PowerMeter())
        g.chain(laser, loss, meter)
        readings.append(g.run()[meter].power_w)

    assert readings[0] == pytest.approx(readings[1], rel=1e-6)


# --------------------------------------------------------------------------
# Source
# --------------------------------------------------------------------------


def test_laser_power_is_independent_of_time_window() -> None:
    """Average power must not depend on how many samples we happen to simulate.

    A normalisation that divides by the wrong length passes with one window and
    fails with another, so this is checked explicitly rather than assumed.
    """
    powers = []
    for sequence_length in (16, 64, 256):
        ctx = SimulationContext(
            bit_rate=10e9, samples_per_symbol=8, sequence_length=sequence_length
        )
        g = Graph(ctx)
        laser = g.add(CWLaser(power=3.0))
        meter = g.add(PowerMeter())
        g.chain(laser, meter)
        powers.append(g.run()[meter].power_dbm)

    assert powers == pytest.approx([3.0] * 3, abs=DB_TOL)


def test_phase_noise_changes_phase_but_not_average_power() -> None:
    """Linewidth broadens the line; it must not add or remove power."""
    ctx = SimulationContext(bit_rate=10e9, samples_per_symbol=16, sequence_length=256, seed=7)

    g = Graph(ctx)
    laser = g.add(CWLaser(power=0.0, linewidth=1000.0))  # 1 MHz
    meter = g.add(PowerMeter())
    g.chain(laser, meter)
    results = g.run(keep=[laser])

    assert results[meter].power_dbm == pytest.approx(0.0, abs=DB_TOL)

    band = results.port(laser, "out").bands[0]
    phase = np.angle(band.Ex)
    assert np.std(phase) > 0.0, "linewidth was declared but the phase is constant"


def test_laser_wavelength_maps_to_the_expected_optical_frequency(ctx: SimulationContext) -> None:
    g = Graph(ctx)
    laser = g.add(CWLaser(wavelength=1550.0))
    meter = g.add(PowerMeter())
    g.chain(laser, meter)

    (band,) = g.run()[meter].bands
    # c / 1550 nm = 193.4145 THz, the ITU-T G.694.1 grid anchor region.
    assert band.f0 == pytest.approx(193.414489e12, rel=1e-6)
    assert band.wavelength_nm == pytest.approx(1550.0, rel=1e-9)


# --------------------------------------------------------------------------
# Multi-band (WDM) — proves the signal model is real, not decorative
# --------------------------------------------------------------------------


def test_two_carriers_propagate_as_separate_bands(ctx: SimulationContext) -> None:
    """Two wavelengths through one fiber must stay two independently sampled bands.

    This is the test that keeps `OpticalSignal.bands` honest. A single-carrier
    model passes every attenuation test above while being fundamentally unable
    to represent a WDM system; only exercising a second band shows the
    difference, which is why this is here in the first week and not in Phase 3.
    """
    g = Graph(ctx)
    ch1 = g.add(CWLaser(power=0.0, wavelength=1550.0, label="ch1"))
    ch2 = g.add(CWLaser(power=-3.0, wavelength=1551.0, label="ch2"))
    mux = g.add(Combiner(2))
    fiber = g.add(Fiber(length=80.0, attenuation=0.2))
    meter = g.add(PowerMeter())

    g.connect(ch1, mux["in0"])
    g.connect(ch2, mux["in1"])
    g.chain(mux, fiber, meter)

    reading = g.run()[meter]

    assert len(reading.bands) == 2, "the two carriers did not survive as separate bands"

    lower, upper = reading.bands  # sorted by frequency: 1551 nm is the lower one
    assert upper.wavelength_nm == pytest.approx(1550.0, rel=1e-6)
    assert lower.wavelength_nm == pytest.approx(1551.0, rel=1e-6)

    # Each channel is attenuated by its own 16 dB, independently.
    assert upper.power_dbm == pytest.approx(0.0 - 16.0, abs=DB_TOL)
    assert lower.power_dbm == pytest.approx(-3.0 - 16.0, abs=DB_TOL)

    # And the total is their incoherent sum.
    expected_total = w_to_dbm(dbm_to_w(-16.0) + dbm_to_w(-19.0))
    assert reading.power_dbm == pytest.approx(expected_total, abs=DB_TOL)


def test_bands_stay_narrow_regardless_of_channel_spacing(ctx: SimulationContext) -> None:
    """Channel spacing must not drive the sample rate.

    1550 nm and 1500 nm are ~6.4 THz apart. If bands were forced onto one grid,
    representing both would need a sample rate in the terahertz. Each band keeps
    the context sample rate instead — which is the entire reason for the design.
    """
    g = Graph(ctx)
    ch1 = g.add(CWLaser(wavelength=1550.0, label="ch1"))
    ch2 = g.add(CWLaser(wavelength=1500.0, label="ch2"))
    mux = g.add(Combiner(2))
    meter = g.add(PowerMeter())

    g.connect(ch1, mux["in0"])
    g.connect(ch2, mux["in1"])
    g.chain(mux, meter)
    results = g.run(keep=[mux])

    signal = results.port(mux, "out")
    spacing = abs(signal.bands[0].f0 - signal.bands[1].f0)
    assert spacing > 6e12, "expected multi-THz spacing for this test to mean anything"
    for band in signal.bands:
        assert band.fs == ctx.sample_rate
        assert band.num_samples == ctx.num_samples


def test_combining_co_located_carriers_is_rejected(ctx: SimulationContext) -> None:
    """Identical centre frequencies interfere; multiplexing them is not the same
    operation as adding them, so the engine refuses rather than guessing."""
    g = Graph(ctx)
    ch1 = g.add(CWLaser(wavelength=1550.0, label="ch1"))
    ch2 = g.add(CWLaser(wavelength=1550.0, label="ch2"))
    mux = g.add(Combiner(2))
    meter = g.add(PowerMeter())

    g.connect(ch1, mux["in0"])
    g.connect(ch2, mux["in1"])
    g.chain(mux, meter)

    with pytest.raises(ValueError, match="coherently"):
        g.run()


def test_combiner_insertion_loss_applies_to_every_channel(ctx: SimulationContext) -> None:
    g = Graph(ctx)
    ch1 = g.add(CWLaser(power=0.0, wavelength=1550.0, label="ch1"))
    ch2 = g.add(CWLaser(power=0.0, wavelength=1551.0, label="ch2"))
    mux = g.add(Combiner(2, insertion_loss=3.0))
    meter = g.add(PowerMeter())

    g.connect(ch1, mux["in0"])
    g.connect(ch2, mux["in1"])
    g.chain(mux, meter)

    for band in g.run()[meter].bands:
        assert band.power_dbm == pytest.approx(-3.0, abs=DB_TOL)
