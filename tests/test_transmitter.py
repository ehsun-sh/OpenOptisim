"""Validation of the transmitter chain: PRBS, NRZ driver, and the MZM."""

from __future__ import annotations

import numpy as np
import pytest

from oosim import Graph, SimulationContext
from oosim.components import (
    CWLaser,
    DCVoltage,
    MachZehnderModulator,
    NRZDriver,
    PowerMeter,
    PRBSGenerator,
)
from oosim.components.electrical import PRBS_TAPS
from oosim.units import db_to_linear

DB_TOL = 1e-4


def _prbs_bits(order: int, length: int, seed: int = 5) -> np.ndarray:
    ctx = SimulationContext(bit_rate=10e9, samples_per_symbol=2, sequence_length=length, seed=seed)
    g = Graph(ctx)
    source = g.add(PRBSGenerator(order=float(order)))
    return np.asarray(g.run()[source].bits)


# --------------------------------------------------------------------------
# PRBS — a test pattern is defined by these properties, not by "looks random"
# --------------------------------------------------------------------------


@pytest.mark.parametrize("order", [7, 9, 11])
def test_prbs_period_is_two_to_the_n_minus_one(order: int) -> None:
    period = 2**order - 1
    bits = _prbs_bits(order, 2 * period)
    np.testing.assert_array_equal(bits[:period], bits[period : 2 * period])


@pytest.mark.parametrize("order", [7, 9, 11])
def test_prbs_is_balanced_over_one_period(order: int) -> None:
    """A maximal-length sequence has exactly 2**(n-1) ones per period."""
    period = 2**order - 1
    bits = _prbs_bits(order, period)
    assert int(bits.sum()) == 2 ** (order - 1)


@pytest.mark.parametrize("order", [7, 9])
def test_every_nonzero_window_appears_exactly_once(order: int) -> None:
    """The defining property of a maximal-length LFSR: over one period the
    register visits every state except all-zeros, once each."""
    period = 2**order - 1
    bits = _prbs_bits(order, period)
    wrapped = np.concatenate([bits, bits[: order - 1]])
    windows = {tuple(wrapped[i : i + order]) for i in range(period)}
    assert len(windows) == period
    assert tuple([0] * order) not in windows


def test_prbs_never_gets_stuck_in_the_all_zero_state() -> None:
    """All-zeros is a fixed point of the LFSR, so seeding into it would produce
    a dead output. Checked across many seeds rather than assumed."""
    for seed in range(30):
        bits = _prbs_bits(7, 127, seed=seed)
        assert int(bits.sum()) == 64


def test_prbs_is_reproducible_for_a_given_seed() -> None:
    np.testing.assert_array_equal(_prbs_bits(7, 127, seed=3), _prbs_bits(7, 127, seed=3))
    assert not np.array_equal(_prbs_bits(7, 127, seed=3), _prbs_bits(7, 127, seed=4))


def test_unsupported_prbs_order_is_rejected_by_name() -> None:
    ctx = SimulationContext(bit_rate=10e9, samples_per_symbol=2, sequence_length=16)
    g = Graph(ctx)
    g.add(PRBSGenerator(order=8.0))
    with pytest.raises(ValueError, match="not a standard maximal-length"):
        g.run()


def test_supported_orders_are_the_standard_telecom_polynomials() -> None:
    assert sorted(PRBS_TAPS) == [7, 9, 11, 15, 23, 31]


# --------------------------------------------------------------------------
# NRZ driver
# --------------------------------------------------------------------------


def test_nrz_holds_each_bit_for_a_full_symbol() -> None:
    ctx = SimulationContext(bit_rate=10e9, samples_per_symbol=8, sequence_length=32, seed=2)
    g = Graph(ctx)
    source = g.add(PRBSGenerator(order=7.0))
    driver = g.add(NRZDriver(v_low=-1.0, v_high=2.0))
    g.chain(source, driver)

    results = g.run(keep=[source])
    bits = np.asarray(results.port(source, "out").bits)
    waveform = np.asarray(results[driver].samples)

    assert waveform.shape == (ctx.num_samples,)
    expected = np.repeat(np.where(bits.astype(bool), 2.0, -1.0), ctx.samples_per_symbol)
    np.testing.assert_allclose(waveform, expected, rtol=1e-6)

    # Within a symbol the level is flat — that is what NRZ means.
    per_symbol = waveform.reshape(ctx.sequence_length, ctx.samples_per_symbol)
    np.testing.assert_allclose(per_symbol.std(axis=1), 0.0, atol=1e-7)


# --------------------------------------------------------------------------
# Mach-Zehnder modulator:  P_out/P_in = cos^2(pi*V / (2*V_pi))
# --------------------------------------------------------------------------


def _modulated_power_dbm(
    voltage: float,
    *,
    v_pi: float = 4.0,
    v_bias: float = 0.0,
    extinction_ratio: float = 100.0,
    insertion_loss: float = 0.0,
) -> float:
    ctx = SimulationContext(bit_rate=10e9, samples_per_symbol=4, sequence_length=8)
    g = Graph(ctx)
    laser = g.add(CWLaser(power=0.0, wavelength=1550.0))
    drive = g.add(DCVoltage(voltage=voltage))
    mzm = g.add(
        MachZehnderModulator(
            v_pi=v_pi,
            v_bias=v_bias,
            extinction_ratio=extinction_ratio,
            insertion_loss=insertion_loss,
        )
    )
    meter = g.add(PowerMeter())
    g.connect(laser, mzm["optical_in"])
    g.connect(drive, mzm["electrical_in"])
    g.chain(mzm, meter)
    return float(g.run()[meter].power_dbm)


@pytest.mark.parametrize("v_over_vpi", [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0])
def test_mzm_follows_the_cosine_squared_transfer_curve(v_over_vpi: float) -> None:
    """Extinction ratio is set very high so the ideal curve is what is measured."""
    v_pi = 4.0
    measured = _modulated_power_dbm(v_over_vpi * v_pi, v_pi=v_pi, extinction_ratio=100.0)
    ideal = np.cos(np.pi * v_over_vpi / 2.0) ** 2

    if ideal < 1e-9:  # the null: dominated by the extinction-ratio floor
        assert measured < -90.0
    else:
        assert measured == pytest.approx(10 * np.log10(ideal), abs=1e-3)


def test_mzm_peaks_at_zero_volts_and_nulls_at_vpi() -> None:
    assert _modulated_power_dbm(0.0) == pytest.approx(0.0, abs=DB_TOL)
    assert _modulated_power_dbm(4.0) < -90.0


def test_quadrature_bias_gives_half_power() -> None:
    """V_pi/2 is the standard bias point for intensity modulation."""
    assert _modulated_power_dbm(2.0, v_pi=4.0) == pytest.approx(-3.0103, abs=1e-3)


def test_extinction_ratio_sets_the_null_depth_exactly() -> None:
    """The on/off ratio measured across the transfer curve must equal the spec.

    This is the property the ER parameter names, so it is checked directly rather
    than inferred from the shape of the curve.
    """
    for er_db in (10.0, 20.0, 30.0):
        on = _modulated_power_dbm(0.0, extinction_ratio=er_db)
        off = _modulated_power_dbm(4.0, extinction_ratio=er_db)
        assert on - off == pytest.approx(er_db, abs=1e-3)


def test_insertion_loss_offsets_the_whole_curve() -> None:
    for v in (0.0, 1.0, 2.0):
        lossless = _modulated_power_dbm(v, insertion_loss=0.0)
        lossy = _modulated_power_dbm(v, insertion_loss=3.0)
        assert lossless - lossy == pytest.approx(3.0, abs=1e-3)


def test_bias_shifts_the_curve_along_the_voltage_axis() -> None:
    """Biasing at V_pi and driving at -V_pi must return to peak transmission."""
    assert _modulated_power_dbm(-4.0, v_bias=4.0) == pytest.approx(0.0, abs=DB_TOL)
    assert _modulated_power_dbm(0.0, v_bias=2.0) == pytest.approx(-3.0103, abs=1e-3)


def test_transfer_curve_is_periodic_in_two_vpi() -> None:
    assert _modulated_power_dbm(8.0) == pytest.approx(_modulated_power_dbm(0.0), abs=DB_TOL)


def test_mzm_rejects_a_zero_vpi_rather_than_dividing_by_it() -> None:
    with pytest.raises(ValueError, match="v_pi must be positive"):
        _modulated_power_dbm(1.0, v_pi=0.0)


# --------------------------------------------------------------------------
# The full transmitter
# --------------------------------------------------------------------------


def test_modulated_average_power_matches_the_mark_density() -> None:
    """A PRBS-driven MZM biased for OOK emits close to half the CW power.

    With ~50% marks and a high extinction ratio the average is 3 dB below the
    peak, offset by the sequence's actual mark density — which is not exactly
    1/2 for a finite PRBS, so the expected value is computed from the bits.
    """
    ctx = SimulationContext(bit_rate=10e9, samples_per_symbol=8, sequence_length=127, seed=11)
    g = Graph(ctx)
    laser = g.add(CWLaser(power=0.0, wavelength=1550.0))
    prbs = g.add(PRBSGenerator(order=7.0))
    driver = g.add(NRZDriver(v_low=0.0, v_high=4.0))
    mzm = g.add(MachZehnderModulator(v_pi=4.0, extinction_ratio=40.0))
    meter = g.add(PowerMeter())

    g.chain(prbs, driver)
    g.connect(laser, mzm["optical_in"])
    g.connect(driver, mzm["electrical_in"])
    g.chain(mzm, meter)

    results = g.run(keep=[prbs])
    mark_density = results.port(prbs, "out").ones_fraction()

    floor = 1.0 / db_to_linear(40.0)
    # Zeros sit at peak transmission here (0 V), ones are driven to the null.
    expected = (1.0 - mark_density) * 1.0 + mark_density * floor
    assert results[meter].power_w == pytest.approx(expected * 1e-3, rel=1e-4)
