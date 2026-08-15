"""Validation of the receiver filter, Q-factor, and BER.

The central test here compares two independent routes to the same number: the
BER predicted from Q under the Gaussian approximation, and the BER obtained by
counting how many bits the decision circuit actually got wrong. Nothing in the
code makes those agree — if the noise model, the filter's noise bandwidth, the
threshold placement, or the Q definition is wrong, they diverge.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from oosim import Graph, SimulationContext
from oosim.analysis import ber_from_q, eye_histogram, measure_eye, optimal_threshold, q_factor
from oosim.components import (
    BERAnalyzer,
    CWLaser,
    ElectricalFilter,
    EyeDiagram,
    Fiber,
    MachZehnderModulator,
    NRZDriver,
    PINPhotodiode,
    PRBSGenerator,
)
from oosim.kernels import gaussian_lowpass_response, gaussian_noise_bandwidth, lowpass_filter
from oosim.units import K_BOLTZMANN

V_PI = 4.0
FILTER_GHZ = 7.0


# --------------------------------------------------------------------------
# BER from Q — pure arithmetic, no simulation involved
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("q", "expected"),
    [
        (0.0, 0.5),
        (1.0, 1.5866e-1),
        (2.0, 2.2750e-2),
        (3.0, 1.3499e-3),
        (4.0, 3.1671e-5),
        (6.0, 9.8659e-10),
        (7.0, 1.2799e-12),
    ],
)
def test_ber_from_q_matches_the_gaussian_tail(q: float, expected: float) -> None:
    """Q = 6 giving ~1e-9 is the origin of the industry's "Q of 6" shorthand."""
    assert ber_from_q(q) == pytest.approx(expected, rel=1e-3)


def test_q_factor_and_threshold_definitions() -> None:
    assert q_factor(1.0, 0.1, 0.0, 0.1) == pytest.approx(5.0)
    # With equal rail noise the optimal threshold is the midpoint.
    assert optimal_threshold(1.0, 0.1, 0.0, 0.1) == pytest.approx(0.5)
    # With a noisier mark rail it moves towards the quieter space rail.
    assert optimal_threshold(1.0, 0.3, 0.0, 0.1) == pytest.approx(0.25)


# --------------------------------------------------------------------------
# Receiver filter
# --------------------------------------------------------------------------


def test_filter_is_three_db_down_at_its_stated_bandwidth() -> None:
    """The definition of the 3 dB point, checked rather than assumed."""
    bandwidth = 7e9
    response = gaussian_lowpass_response(np.array([0.0, bandwidth]), bandwidth)
    assert response[0] == pytest.approx(1.0)
    assert response[1] ** 2 == pytest.approx(0.5, rel=1e-12)


def test_gaussian_noise_bandwidth_closed_form() -> None:
    """B_n = B * sqrt(pi / 4 ln2), verified against numerical integration."""
    bandwidth = 7e9
    f = np.linspace(0.0, 100e9, 2_000_001)
    integrated = float(np.trapezoid(gaussian_lowpass_response(f, bandwidth) ** 2, f))
    assert gaussian_noise_bandwidth(bandwidth) == pytest.approx(integrated, rel=1e-6)
    assert gaussian_noise_bandwidth(bandwidth) / bandwidth == pytest.approx(1.0645, rel=1e-3)


def test_filtering_white_noise_reduces_variance_by_the_bandwidth_ratio() -> None:
    """White noise spread over fs/2, filtered to B_n, keeps the fraction B_n/(fs/2).

    This is the relation that makes a simulated BER correspond to a real one: the
    detector's noise spans the simulated bandwidth, which is an artefact of the
    oversampling factor, and the filter is what converts it to a physical number.
    """
    rng = np.random.default_rng(1)
    fs = 160e9
    bandwidth = 7e9
    noise = rng.normal(0.0, 1.0, size=1 << 20)

    filtered = lowpass_filter(noise, fs, bandwidth)
    expected = gaussian_noise_bandwidth(bandwidth) / (fs / 2.0)

    assert filtered.var() == pytest.approx(expected, rel=0.02)


def test_filter_passes_dc_unchanged() -> None:
    constant = np.full(1024, 3.0)
    np.testing.assert_allclose(lowpass_filter(constant, 160e9, 7e9), 3.0, rtol=1e-9)


def test_filter_has_no_group_delay() -> None:
    """Zero-phase filtering keeps the output aligned with the input, so nothing
    downstream has to compensate a delay.

    The tone is placed on an exact FFT bin so the window holds a whole number of
    cycles; otherwise the circular wrap rings and would hide the phase result
    being tested. The output must equal the input scaled by |H(f)| — same phase,
    same sample positions, only the amplitude changed.
    """
    n, fs, bandwidth = 4096, 160e9, 7e9
    frequency = 25 * fs / n  # exactly 25 cycles in the window
    tone = np.sin(2 * np.pi * frequency * np.arange(n) / fs)

    filtered = lowpass_filter(tone, fs, bandwidth)
    gain = float(gaussian_lowpass_response(np.array([frequency]), bandwidth)[0])

    np.testing.assert_allclose(filtered, gain * tone, atol=1e-9)


# --------------------------------------------------------------------------
# End-to-end: does the counted BER match the one predicted from Q?
# --------------------------------------------------------------------------


def _link(
    launch_dbm: float,
    *,
    sequence_length: int = 4096,
    samples_per_symbol: int = 8,
    span_km: float = 0.0,
    dispersion: float = 0.0,
    shot_noise: bool = False,
    thermal_noise: bool = True,
    seed: int = 4242,
) -> tuple[Graph, BERAnalyzer]:
    """An OOK link: PRBS -> NRZ -> laser -> MZM -> fiber -> PIN -> filter -> BER."""
    ctx = SimulationContext(
        bit_rate=10e9,
        samples_per_symbol=samples_per_symbol,
        sequence_length=sequence_length,
        seed=seed,
    )
    g = Graph(ctx)

    prbs = g.add(PRBSGenerator(order=15.0))
    # A 1 drives the modulator to peak transmission, a 0 to the null.
    driver = g.add(NRZDriver(v_low=V_PI, v_high=0.0))
    laser = g.add(CWLaser(power=launch_dbm, wavelength=1550.0))
    mzm = g.add(MachZehnderModulator(v_pi=V_PI, extinction_ratio=30.0))
    fiber = g.add(Fiber(length=span_km, attenuation=0.2, dispersion=dispersion))
    pin = g.add(PINPhotodiode(responsivity=0.8, shot_noise=shot_noise, thermal_noise=thermal_noise))
    lpf = g.add(ElectricalFilter(bandwidth=FILTER_GHZ))
    ber = g.add(BERAnalyzer())

    g.chain(prbs, driver)
    g.connect(laser, mzm["optical_in"])
    g.connect(driver, mzm["electrical_in"])
    g.chain(mzm, fiber, pin, lpf)
    g.connect(lpf, ber["in"])
    g.connect(prbs["out"], ber["reference"])
    return g, ber


@pytest.mark.parametrize("launch_dbm", [-23.0, -22.0, -21.0, -20.0, -19.0])
def test_counted_ber_agrees_with_the_gaussian_prediction(launch_dbm: float) -> None:
    """The central validation of the receiver chain.

    Q is measured from the rail statistics; the BER it predicts is compared with
    the errors the decision circuit actually made. The tolerance is derived from
    counting statistics rather than picked: with N_err expected errors the count
    has standard deviation sqrt(N_err), so five of those is a wide margin that
    still fails decisively if the physics is wrong.

    Thermal noise only, so both rails carry identical Gaussian noise and the
    Gaussian formula is exact rather than approximate.

    The launch powers are low on purpose. A working link runs at a BER no
    simulation can count — 1e-9 needs 1e10 bits — so agreement is established
    where counting is possible, between roughly 1e-4 and 1e-1, and the formula
    is then trusted to extrapolate. That is the same bargain a lab makes with a
    BER-vs-power curve.
    """
    g, ber = _link(launch_dbm, sequence_length=16384)
    measurement = g.run()[ber]

    predicted = measurement.ber_gaussian
    expected_errors = predicted * measurement.bits_evaluated
    assert expected_errors > 10, "launch power chosen too high to count errors reliably"

    margin = 5.0 * math.sqrt(expected_errors)
    assert abs(measurement.errors - expected_errors) < margin, (
        f"counted {measurement.errors} errors, Gaussian prediction {expected_errors:.1f} "
        f"(Q = {measurement.q_factor:.3f})"
    )


def test_q_scales_with_received_power_in_a_thermal_limited_receiver() -> None:
    """Thermal noise does not depend on the signal, so Q is proportional to power:
    3 dB more light must double Q. A receiver where that is not true is not
    thermal-limited, and this is how you tell."""
    low, ber_low = _link(-18.0)
    high, ber_high = _link(-15.0)

    q_low = low.run()[ber_low].q_factor
    q_high = high.run()[ber_high].q_factor

    assert q_high / q_low == pytest.approx(10 ** (3 / 10), rel=0.05)


def test_rail_noise_matches_the_filtered_thermal_variance() -> None:
    """Ties the measured eye back to the closed-form receiver noise.

    sigma = sqrt(4kTB_n/R_L) with B_n from the filter — no fitted constants.
    """
    ctx_bandwidth = gaussian_noise_bandwidth(FILTER_GHZ * 1e9)
    expected_sigma = math.sqrt(4.0 * K_BOLTZMANN * 300.0 * ctx_bandwidth / 50.0)

    g, ber = _link(-16.0, sequence_length=16384)
    measurement = g.run()[ber]

    assert measurement.std_zero == pytest.approx(expected_sigma, rel=0.05)
    assert measurement.std_one == pytest.approx(expected_sigma, rel=0.05)


def test_shot_noise_makes_the_mark_rail_noisier_than_the_space_rail() -> None:
    """The asymmetry the Q definition exists to handle: sigma1 != sigma0, so the
    optimal threshold sits below the midpoint, nearer the quieter space rail.

    The ratio is far smaller than the ~30x the rails' powers alone would suggest,
    and that is correct rather than a defect: the receiver filter's impulse
    response spans about a symbol, so each space sample is a weighted average
    that includes noise from its neighbours — and half of those are marks. The
    mark rail's shot noise smears into the spaces beside it.
    """
    g, ber = _link(-10.0, shot_noise=True, thermal_noise=False, sequence_length=8192)
    m = g.run()[ber]

    assert m.std_one > 1.3 * m.std_zero
    midpoint = 0.5 * (m.mean_one + m.mean_zero)
    assert m.threshold < midpoint


def test_more_launch_power_lowers_the_bit_error_rate() -> None:
    rates = []
    for launch in (-19.0, -18.0, -17.0, -16.0):
        g, ber = _link(launch, sequence_length=8192)
        rates.append(g.run()[ber].ber_gaussian)
    assert rates == sorted(rates, reverse=True)


def test_error_free_link_reports_no_errors() -> None:
    g, ber = _link(0.0, sequence_length=4096)
    measurement = g.run()[ber]
    assert measurement.errors == 0
    assert measurement.q_factor > 7.0


def test_span_loss_is_equivalent_to_launching_that_much_lower() -> None:
    """An end-to-end consistency check across the whole chain.

    120 km of 0.2 dB/km fiber is 24 dB. Launching 0 dBm through it must give the
    same Q as launching -24 dBm back to back — the loss has to arrive at the
    decision circuit as exactly that and nothing else. Modulator, fiber,
    detector, filter and analyzer all have to agree for this to hold, and any
    stray gain or normalisation error in any of them breaks it.
    """
    through_span, ber_span = _link(0.0, span_km=120.0, dispersion=0.0)
    back_to_back, ber_b2b = _link(-24.0, span_km=0.0, dispersion=0.0)

    q_span = through_span.run()[ber_span].q_factor
    q_b2b = back_to_back.run()[ber_b2b].q_factor

    assert q_span == pytest.approx(q_b2b, rel=0.02)


def test_dispersion_closes_the_eye() -> None:
    """The Phase 1 headline result: a link that works at 0 km fails at 200 km on
    dispersion alone, with the launch power held constant."""
    back_to_back, ber_b2b = _link(-10.0, span_km=0.0, dispersion=0.0)
    dispersed, ber_disp = _link(-10.0, span_km=200.0, dispersion=17.0)

    q_b2b = back_to_back.run()[ber_b2b].q_factor
    q_dispersed = dispersed.run()[ber_disp].q_factor

    assert q_b2b > 7.0
    assert q_dispersed < q_b2b / 2.0


# --------------------------------------------------------------------------
# Decision circuit behaviour
# --------------------------------------------------------------------------


def test_adaptive_timing_finds_the_centre_of_the_symbol() -> None:
    """With no ISI the best sampling instant is mid-symbol, so this is a check
    that the search works rather than a physical claim."""
    g, ber = _link(-12.0, samples_per_symbol=16)
    offset = g.run()[ber].sample_offset
    assert 4 <= offset <= 12


def test_sampling_at_the_symbol_edge_degrades_q() -> None:
    ctx_kwargs = {"samples_per_symbol": 16, "sequence_length": 4096}
    best, ber_best = _link(-14.0, **ctx_kwargs)  # type: ignore[arg-type]
    q_best = best.run()[ber_best].q_factor

    edge, ber_edge = _link(-14.0, **ctx_kwargs)  # type: ignore[arg-type]
    analyzer = next(c for c in edge.components if isinstance(c, BERAnalyzer))
    analyzer._values["adaptive_timing"] = False
    analyzer._values["sample_offset"] = 0.0
    q_edge = edge.run()[ber_edge].q_factor

    assert q_edge < q_best


def test_measure_eye_rejects_a_length_mismatch() -> None:
    with pytest.raises(ValueError, match="needs 80"):
        measure_eye(np.zeros(64), np.zeros(10, dtype=np.uint8), 8)


def test_measure_eye_needs_both_rails() -> None:
    with pytest.raises(ValueError, match="only one symbol value"):
        measure_eye(np.zeros(80), np.ones(10, dtype=np.uint8), 8)


# --------------------------------------------------------------------------
# Eye diagram is a data reduction
# --------------------------------------------------------------------------


def test_eye_histogram_size_is_independent_of_the_simulation_length() -> None:
    """The architectural claim, asserted: a longer run must not produce a larger
    result. This is what keeps raw sample buffers out of the UI.
    """
    shapes = []
    for sequence_length in (512, 4096):
        g = Graph(
            SimulationContext(
                bit_rate=10e9, samples_per_symbol=8, sequence_length=sequence_length, seed=1
            )
        )
        prbs = g.add(PRBSGenerator(order=15.0))
        driver = g.add(NRZDriver(v_low=0.0, v_high=1.0))
        eye = g.add(EyeDiagram(time_bins=64.0, amplitude_bins=32.0))
        g.chain(prbs, driver, eye)
        shapes.append(g.run()[eye].shape)

    assert shapes[0] == shapes[1] == (32, 64)


def test_eye_histogram_counts_every_sample_it_is_given() -> None:
    samples = np.random.default_rng(0).normal(size=8 * 100)
    histogram = eye_histogram(samples, 8, 10e9, span_symbols=2, time_bins=16, amplitude_bins=16)
    assert histogram.counts.sum() == samples.size


def test_eye_histogram_needs_at_least_one_trace() -> None:
    with pytest.raises(ValueError, match="shorter than one"):
        eye_histogram(np.zeros(4), 8, 10e9, span_symbols=2)
