"""The coherent transceiver, end to end.

The chain under test is PRBS -> QAM mapper -> IQ driver -> IQ modulator ->
coherent receiver -> sampler -> analyser. What makes these tests worth having is
that most of them compare against something derived outside the code: the
shot-noise-limited SNR against a closed form, the counted symbol error rate
against :func:`oosim.modulation.ser_qam`, the modulator's intrinsic loss against
the 3 dB a dual-parallel structure costs by construction.
"""

from __future__ import annotations

import math
from typing import Any, cast

import numpy as np
import pytest

from oosim import Graph, SimulationContext
from oosim.components import (
    CoherentReceiver,
    ConstellationAnalyzer,
    ConstellationDiagram,
    CWLaser,
    IQDriver,
    IQModulator,
    IQSampler,
    PowerMeter,
    PRBSGenerator,
    QAMMapper,
)
from oosim.modulation import qam_constellation, ser_qam
from oosim.signals import Band, ConstellationMeasurement, OpticalSignal, SymbolSignal
from oosim.units import C_LIGHT, Q_ELECTRON

SQUARE_ORDERS = [2, 4, 6]
ALL_ORDERS = [1, *SQUARE_ORDERS]

#: Frequency shift produced by one nanometre of wavelength at 1550 nm [Hz/nm].
GHZ_PER_NM = C_LIGHT / (1550e-9**2) * 1e-9


def build(
    *,
    bits_per_symbol: int = 2,
    tx_dbm: float = 0.0,
    lo_dbm: float = 10.0,
    lo_nm: float = 1550.0,
    shot_noise: bool = False,
    thermal_noise: bool = False,
    predistort: bool = True,
    sequence_length: int = 2048,
    samples_per_symbol: int = 4,
    seed: int = 7,
    remove_frequency_offset: bool = True,
    **modulator: float,
) -> tuple[Graph, ConstellationAnalyzer, PowerMeter]:
    """A single-polarization coherent link, back to back."""
    ctx = SimulationContext(
        bit_rate=32e9,
        samples_per_symbol=samples_per_symbol,
        sequence_length=sequence_length,
        seed=seed,
        precision="double",
    )
    graph = Graph(ctx)
    prbs = graph.add(
        PRBSGenerator(order=23.0, bits_per_symbol=float(bits_per_symbol), label="prbs")
    )
    mapper = graph.add(QAMMapper(bits_per_symbol=float(bits_per_symbol), label="map"))
    driver = graph.add(IQDriver(v_pi=4.0, predistort=predistort, label="drv"))
    laser = graph.add(CWLaser(power=tx_dbm, label="tx"))
    modulator_block = graph.add(IQModulator(v_pi=4.0, label="mod", **modulator))
    meter = graph.add(PowerMeter(label="pm"))
    lo = graph.add(CWLaser(power=lo_dbm, wavelength=lo_nm, label="lo"))
    receiver = graph.add(
        CoherentReceiver(shot_noise=shot_noise, thermal_noise=thermal_noise, label="rx")
    )
    sampler = graph.add(IQSampler(label="smp"))
    analyzer = graph.add(
        ConstellationAnalyzer(remove_frequency_offset=remove_frequency_offset, label="vsa")
    )

    graph.chain(prbs, mapper, driver)
    graph.connect(laser, modulator_block["optical_in"])
    graph.connect(driver["i"], modulator_block["i"])
    graph.connect(driver["q"], modulator_block["q"])
    graph.connect(modulator_block, meter["in"])
    graph.connect(modulator_block, receiver["in"])
    graph.connect(lo, receiver["lo"])
    graph.connect(receiver["i"], sampler["i"])
    graph.connect(receiver["q"], sampler["q"])
    graph.connect(mapper["out"], sampler["reference"])
    graph.connect(sampler["out"], analyzer["in"])
    graph.connect(mapper["out"], analyzer["reference"])
    return graph, analyzer, meter


# --------------------------------------------------------------------------
# Back to back
# --------------------------------------------------------------------------


@pytest.mark.parametrize("bits_per_symbol", ALL_ORDERS)
def test_a_noiseless_link_recovers_every_symbol_exactly(bits_per_symbol: int) -> None:
    """With nothing added and the modulator linearised, the link is transparent.

    Exact rather than approximate: any residual EVM here would be a bug in the
    chain rather than an impairment, and would put a floor under every later
    measurement.
    """
    graph, analyzer, _ = build(bits_per_symbol=bits_per_symbol)
    result = graph.run(keep=[])[analyzer]

    assert result.evm == pytest.approx(0.0, abs=1e-9)
    assert result.bit_errors == 0
    assert result.symbol_errors == 0
    assert result.bits_evaluated == 2048 * bits_per_symbol


@pytest.mark.parametrize("bits_per_symbol", ALL_ORDERS)
def test_the_information_rate_is_the_symbol_rate_times_the_bits(bits_per_symbol: int) -> None:
    """32 GBd of 16-QAM is 128 Gb/s, and the symbol signal knows it."""
    graph, _, _ = build(bits_per_symbol=bits_per_symbol)
    mapper = next(c for c in graph.components if isinstance(c, QAMMapper))
    symbols = graph.run(keep=[mapper]).port(mapper, "out")

    assert symbols.bits_per_symbol == bits_per_symbol
    assert symbols.symbol_rate == pytest.approx(32e9)
    assert symbols.bit_rate == pytest.approx(32e9 * bits_per_symbol)


# --------------------------------------------------------------------------
# The modulator
# --------------------------------------------------------------------------


def test_the_iq_modulator_costs_three_decibels() -> None:
    """A dual-parallel structure halves the power at the constellation corner.

    Not a modelling loss — it is what splitting into two arms and recombining in
    quadrature costs, and it is why a coherent transmitter's power budget starts
    3 dB behind an intensity-modulated one.
    """
    graph, _, meter = build(bits_per_symbol=2, tx_dbm=0.0)
    launched = graph.run(keep=[])[meter].power_dbm
    assert launched == pytest.approx(-10.0 * math.log10(2.0), abs=1e-6)


def test_qpsk_is_immune_to_the_modulator_sine_but_16qam_is_not() -> None:
    """QPSK only ever visits the extremes, so the sine merely scales it.

    This is the sharpest available check that the arm transfer really is
    ``sin(pi*V/(2*V_pi))`` and not a linearised stand-in: a linear model would
    leave *both* formats undistorted, and this test would stop separating them.
    """
    _, qpsk_analyzer, _ = _run(bits_per_symbol=2, predistort=False)
    _, qam16_analyzer, _ = _run(bits_per_symbol=4, predistort=False)

    assert qpsk_analyzer.evm == pytest.approx(0.0, abs=1e-9)
    assert qam16_analyzer.evm > 0.10


def test_the_compressed_inner_level_is_exactly_the_sine_ratio() -> None:
    """16-QAM's inner quadrature should land at sin(pi/6)/sin(pi/2) = 1/2.

    Driven at full swing the outer level maps to V_pi and the inner to V_pi/3, so
    the field levels become 1 and 0.5 instead of 1 and 1/3. Checking the number
    rather than merely "some distortion" is what makes this a test of the
    transfer characteristic.
    """
    graph, _, _ = build(bits_per_symbol=4, predistort=False)
    sampler = next(c for c in graph.components if isinstance(c, IQSampler))
    symbols = np.asarray(graph.run(keep=[sampler]).port(sampler, "out").symbols)

    levels = np.unique(np.round(np.abs(symbols.real), 6))
    assert levels.shape == (2,)
    assert levels[0] / levels[1] == pytest.approx(0.5, rel=1e-6)


@pytest.mark.parametrize("bits_per_symbol", SQUARE_ORDERS)
def test_predistortion_removes_the_modulator_nonlinearity(bits_per_symbol: int) -> None:
    """Pre-applying the inverse sine is what a transmitter DSP does."""
    _, distorted, _ = _run(bits_per_symbol=bits_per_symbol, predistort=False)
    _, corrected, _ = _run(bits_per_symbol=bits_per_symbol, predistort=True)
    assert corrected.evm <= distorted.evm
    assert corrected.evm == pytest.approx(0.0, abs=1e-9)


def test_backing_the_drive_off_also_linearises_it() -> None:
    """The other way out of the sine: use only the part of it that is straight."""
    graph, analyzer, _ = build(bits_per_symbol=4, predistort=False)
    driver = next(c for c in graph.components if isinstance(c, IQDriver))

    full = graph.run(keep=[], overrides={(driver, "drive_ratio"): 1.0})[analyzer].evm
    backed_off = graph.run(keep=[], overrides={(driver, "drive_ratio"): 0.2})[analyzer].evm
    assert backed_off < full / 5.0


def test_a_bias_error_leaks_carrier_when_the_drive_is_backed_off() -> None:
    """An arm off its null passes an unmodulated component: the residual carrier.

    Tested on the transfer characteristic directly rather than through the link,
    because a finite PRBS does not have an exactly balanced symbol count and its
    own imbalance would swamp the effect being measured.
    """
    modulator = IQModulator(v_pi=4.0, bias_i=0.6, label="mod")
    drive = np.array([4.0 / 3.0, -4.0 / 3.0] * 64)
    transmission = modulator.field_transmission(drive, np.zeros_like(drive))

    assert abs(float(np.mean(transmission.real))) > 0.05


def test_at_full_swing_a_symmetric_bias_leaks_no_carrier() -> None:
    """A property that is not obvious and would be easy to model away.

    For a two-level drive ``+-V``, ``t(+V) + t(-V) = 2*sin(b)*cos(a)`` with
    ``a = pi*V/(2*V_pi)`` and ``b`` the bias. At full swing ``a = pi/2``, the
    cosine vanishes, and the leak disappears however large the bias is — the two
    levels are pushed off the null by equal and opposite amounts.

    This is why the preceding test has to back the drive off to see anything, and
    why QPSK at full swing is unusually forgiving of a bias error.
    """
    modulator = IQModulator(v_pi=4.0, bias_i=1.2, label="mod")
    drive = np.array([4.0, -4.0] * 64)
    transmission = modulator.field_transmission(drive, np.zeros_like(drive))

    assert float(np.mean(transmission.real)) == pytest.approx(0.0, abs=1e-12)


def test_a_bias_error_costs_evm_on_a_multilevel_format() -> None:
    """16-QAM has inner levels that do not sit at full swing, so it does feel it."""
    _, clean, _ = _run(bits_per_symbol=4)
    _, biased, _ = _run(bits_per_symbol=4, bias_i=0.6)
    assert clean.evm == pytest.approx(0.0, abs=1e-9)
    assert biased.evm > 0.02


def test_quadrature_error_skews_the_constellation() -> None:
    """The two arms no longer 90 degrees apart is a real transmitter impairment."""
    _, clean, _ = _run(bits_per_symbol=4)
    _, skewed, _ = _run(bits_per_symbol=4, quadrature_error=8.0)
    assert clean.evm == pytest.approx(0.0, abs=1e-9)
    assert skewed.evm > 0.05


# --------------------------------------------------------------------------
# The receiver
# --------------------------------------------------------------------------


@pytest.mark.parametrize("tx_dbm", [-40.0, -36.0, -32.0])
def test_shot_noise_limited_snr_matches_the_closed_form(tx_dbm: float) -> None:
    """``SNR = R * P_s / (2 * q * B)`` for QPSK, from the receiver's noise convention.

    The LO dominates the shot noise, so each quadrature carries variance
    ``q*R*P_lo*B`` while the beat carries ``R**2 * P_s * P_lo`` of signal power
    across both quadratures. The LO power cancels — which is the whole reason a
    coherent receiver can be made shot-noise limited by turning the LO up.
    """
    graph, analyzer, meter = build(
        bits_per_symbol=2, tx_dbm=tx_dbm, shot_noise=True, sequence_length=8192, seed=11
    )
    results = graph.run(keep=[])
    measured = results[analyzer].snr
    signal_power = results[meter].power_w

    receiver = next(c for c in graph.components if isinstance(c, CoherentReceiver))
    bandwidth = receiver.noise_bandwidth(graph.ctx)
    predicted = receiver.si("responsivity") * signal_power / (2.0 * Q_ELECTRON * bandwidth)

    assert measured == pytest.approx(predicted, rel=0.05)


def test_turning_the_local_oscillator_up_does_not_change_the_snr() -> None:
    """The LO amplifies signal and its own shot noise together.

    That cancellation is the defining property of coherent detection, and it is
    why an LO is specified by how far it lifts the receiver above its thermal
    floor rather than by an SNR it buys.
    """
    graph, analyzer, _ = build(
        bits_per_symbol=2, tx_dbm=-36.0, shot_noise=True, sequence_length=8192, seed=5
    )
    lo = next(c for c in graph.components if c.label == "lo")

    quiet = graph.run(keep=[], overrides={(lo, "power"): 6.0})[analyzer].snr
    loud = graph.run(keep=[], overrides={(lo, "power"): 16.0})[analyzer].snr
    assert loud == pytest.approx(quiet, rel=0.08)


@pytest.mark.parametrize("tx_dbm", [-40.0, -37.0, -34.0])
def test_counted_symbol_errors_match_the_analytical_rate(tx_dbm: float) -> None:
    """The decision circuit checked against theory rather than against itself.

    Launch powers are chosen to land in the countable range: high enough to see
    thousands of errors in the window, low enough that the closed form has not
    yet run past what the sequence can resolve.
    """
    graph, analyzer, _ = build(
        bits_per_symbol=2, tx_dbm=tx_dbm, shot_noise=True, sequence_length=16384, seed=23
    )
    result = graph.run(keep=[])[analyzer]

    assert result.symbol_errors > 30, "chosen operating point produced too few errors to count"
    assert result.ser_counted == pytest.approx(ser_qam(result.snr, 2), rel=0.2)


def _beat(signal_state: tuple[complex, complex], lo_state: tuple[complex, complex]) -> float:
    """Peak I photocurrent for a CW signal and LO in the given Jones states.

    Double precision because the diagonal case is asserted against ``1/sqrt(2)``
    to nine digits, and the limit at single precision is how the photocurrent is
    *stored*, not how it is computed.
    """
    ctx = SimulationContext(
        bit_rate=10e9, samples_per_symbol=4, sequence_length=64, seed=1, precision="double"
    )
    receiver = CoherentReceiver(responsivity=1.0, shot_noise=False, thermal_noise=False, label="rx")
    n = ctx.num_samples
    f0 = C_LIGHT / 1550e-9

    def field(state: tuple[complex, complex]) -> OpticalSignal:
        return OpticalSignal(
            bands=(
                Band(
                    Ex=np.full(n, state[0], dtype=np.complex128),
                    Ey=np.full(n, state[1], dtype=np.complex128),
                    f0=f0,
                    fs=ctx.sample_rate,
                ),
            )
        )

    out = receiver.run(ctx, {"in": field(signal_state), "lo": field(lo_state)})
    return float(np.abs(np.asarray(out["i"].samples)).max())


def test_the_beat_follows_the_jones_inner_product_in_every_polarization() -> None:
    """Polarization fading falls out of the model rather than being added to it.

    A single-polarization coherent receiver goes deaf when the fibre rotates the
    signal onto the state orthogonal to the LO — and, just as importantly, hears
    it perfectly when both sit on the *other* axis. Testing only the orthogonal
    case would pass for a receiver that had quietly dropped the Y term
    altogether, because that receiver also reports zero, for the wrong reason.
    A deliberate sabotage of exactly that kind is what this test was written to
    catch.

    At 45 degrees the projection is ``cos(45) = 1/sqrt(2)``, so the beat is not
    merely "reduced" but reduced by a number.
    """
    aligned_x = _beat((1.0, 0.0), (1.0, 0.0))
    aligned_y = _beat((0.0, 1.0), (0.0, 1.0))
    crossed = _beat((0.0, 1.0), (1.0, 0.0))
    crossed_other_way = _beat((1.0, 0.0), (0.0, 1.0))
    diagonal = _beat((2.0**-0.5, 2.0**-0.5), (1.0, 0.0))

    assert aligned_x == pytest.approx(1.0)
    assert aligned_y == pytest.approx(1.0), "the Y polarization is not being detected at all"
    assert crossed == 0.0
    assert crossed_other_way == 0.0
    assert diagonal == pytest.approx(2.0**-0.5, rel=1e-9)


def test_the_local_oscillator_must_be_a_single_tone() -> None:
    ctx = SimulationContext(bit_rate=10e9, samples_per_symbol=4, sequence_length=16, seed=1)
    receiver = CoherentReceiver(label="rx")
    n = ctx.num_samples
    ones = np.ones(n, dtype=np.complex128)
    zeros = np.zeros(n, dtype=np.complex128)
    two_tone = OpticalSignal(
        bands=(
            Band(Ex=ones, Ey=zeros, f0=193.1e12, fs=ctx.sample_rate),
            Band(Ex=ones, Ey=zeros, f0=193.2e12, fs=ctx.sample_rate),
        )
    )
    with pytest.raises(ValueError, match="single band"):
        receiver.run(ctx, {"in": two_tone, "lo": two_tone})


# --------------------------------------------------------------------------
# Carrier frequency offset
# --------------------------------------------------------------------------


@pytest.mark.parametrize("delta_nm", [0.004, 0.008, 0.04])
def test_a_frequency_offset_is_estimated_with_the_right_sign_and_size(delta_nm: float) -> None:
    """Intradyne operation: the constellation rotates at the carrier difference.

    The offset is never configured anywhere — it falls out of the two lasers'
    own centre frequencies, which is exactly why the bands carry ``f0`` instead
    of the run assuming one carrier.
    """
    lo_nm = 1550.0 + delta_nm
    expected = C_LIGHT / 1550e-9 - C_LIGHT / (lo_nm * 1e-9)

    graph, analyzer, _ = build(bits_per_symbol=2, lo_nm=lo_nm, sequence_length=4096)
    result = graph.run(keep=[])[analyzer]

    assert result.frequency_offset == pytest.approx(expected, rel=1e-3)
    assert result.evm == pytest.approx(0.0, abs=1e-6)


def test_an_uncorrected_offset_destroys_the_measurement() -> None:
    """What carrier recovery is for, stated as a test."""
    graph, analyzer, _ = build(
        bits_per_symbol=2, lo_nm=1550.008, sequence_length=4096, remove_frequency_offset=False
    )
    assert graph.run(keep=[])[analyzer].evm > 1.0


def test_offset_removal_never_makes_the_measurement_worse() -> None:
    """The invariant that fixes a real bug found while building this.

    The estimator averages the error phasor's turn per symbol, which assumes what
    is left after dividing out the reference is noise. Deterministic distortion
    breaks that assumption: it is correlated with the data, and biases the
    average by a few parts in ten thousand of the symbol rate. That is
    physically nothing, but a frequency error accumulates, and over thousands of
    symbols it becomes radians — turning a 14% EVM into a meaningless one. The
    fix is to keep the correction only when it helps, and this asserts it.
    """
    _, uncorrected, _ = _run(bits_per_symbol=4, predistort=False, remove_frequency_offset=False)
    _, corrected, _ = _run(bits_per_symbol=4, predistort=False, remove_frequency_offset=True)

    assert corrected.evm <= uncorrected.evm + 1e-12
    assert corrected.evm == pytest.approx(0.1427, rel=0.05)


# --------------------------------------------------------------------------
# Wiring and reduction
# --------------------------------------------------------------------------


def test_the_two_ends_disagreeing_on_the_format_is_refused() -> None:
    """A receiver slicing for the wrong alphabet still produces a BER.

    Which is exactly why it has to raise instead.
    """
    ctx = SimulationContext(bit_rate=32e9, samples_per_symbol=4, sequence_length=64, seed=1)
    analyzer = ConstellationAnalyzer(label="vsa")
    qpsk = qam_constellation(2)
    qam16 = qam_constellation(4)
    flat = np.zeros(64, dtype=int)
    received = SymbolSignal(symbols=qpsk[flat], symbol_rate=32e9, constellation=qpsk)
    reference = SymbolSignal(symbols=qam16[flat], symbol_rate=32e9, constellation=qam16)
    with pytest.raises(ValueError, match="disagree on the format"):
        analyzer.run(ctx, {"in": received, "reference": reference})


def test_a_mapper_fed_the_wrong_number_of_bits_refuses() -> None:
    ctx = SimulationContext(bit_rate=32e9, samples_per_symbol=4, sequence_length=100, seed=1)
    graph = Graph(ctx)
    prbs = graph.add(PRBSGenerator(order=15.0, bits_per_symbol=1.0, label="prbs"))
    mapper = graph.add(QAMMapper(bits_per_symbol=4.0, label="map"))
    graph.chain(prbs, mapper)
    with pytest.raises(ValueError, match="needs 400"):
        graph.run(keep=[])


@pytest.mark.parametrize("sequence_length", [512, 4096])
def test_the_constellation_diagram_is_the_same_size_whatever_the_run_length(
    sequence_length: int,
) -> None:
    """The second reduction type, and it reduces for the same reason the first does."""
    ctx = SimulationContext(
        bit_rate=32e9, samples_per_symbol=4, sequence_length=sequence_length, seed=2
    )
    graph = Graph(ctx)
    prbs = graph.add(PRBSGenerator(order=15.0, bits_per_symbol=4.0, label="prbs"))
    mapper = graph.add(QAMMapper(bits_per_symbol=4.0, label="map"))
    diagram = graph.add(ConstellationDiagram(bins=64.0, label="cd"))
    graph.chain(prbs, mapper, diagram)

    histogram = graph.run(keep=[])[diagram]
    assert histogram.shape == (64, 64)
    assert int(histogram.counts.sum()) == sequence_length
    assert histogram.reference.shape == (16,)


def test_the_diagram_frames_the_alphabet_not_the_outliers() -> None:
    """A window set by the noise would squeeze the constellation into the centre."""
    ctx = SimulationContext(bit_rate=32e9, samples_per_symbol=4, sequence_length=256, seed=2)
    graph = Graph(ctx)
    prbs = graph.add(PRBSGenerator(order=15.0, bits_per_symbol=2.0, label="prbs"))
    mapper = graph.add(QAMMapper(bits_per_symbol=2.0, label="map"))
    diagram = graph.add(ConstellationDiagram(bins=32.0, extent=1.6, label="cd"))
    graph.chain(prbs, mapper, diagram)

    histogram = graph.run(keep=[])[diagram]
    outermost = float(np.abs(qam_constellation(2)).max())
    assert float(histogram.inphase_edges[-1]) == pytest.approx(1.6 * outermost)


def test_agc_makes_the_measurement_independent_of_the_receiver_gain() -> None:
    """An EVM that moved when the LO was turned up would be measuring the receiver."""
    _, weak, _ = _run(bits_per_symbol=4, lo_dbm=0.0)
    _, strong, _ = _run(bits_per_symbol=4, lo_dbm=20.0)
    assert weak.evm == pytest.approx(strong.evm, abs=1e-9)


def _run(**kwargs: Any) -> tuple[Graph, ConstellationMeasurement, PowerMeter]:
    """Build a link, run it, and return the analyser's measurement."""
    graph, analyzer, meter = build(**kwargs)
    measurement = cast(ConstellationMeasurement, graph.run(keep=[])[analyzer])
    return graph, measurement, meter


def test_shot_noise_scales_the_way_the_formula_says() -> None:
    """Halving the noise bandwidth should double the SNR.

    Independent of the absolute convention, which is what makes it worth having
    alongside the closed-form check: a constant factor error would pass that one
    only if it also passed this, and the two fail differently.
    """
    _, narrow, _ = _run(
        bits_per_symbol=2, tx_dbm=-36.0, shot_noise=True, samples_per_symbol=4, sequence_length=8192
    )
    _, wide, _ = _run(
        bits_per_symbol=2, tx_dbm=-36.0, shot_noise=True, samples_per_symbol=8, sequence_length=8192
    )
    assert narrow.snr / wide.snr == pytest.approx(2.0, rel=0.1)


def test_snr_is_linear_in_received_power() -> None:
    _, low, _ = _run(bits_per_symbol=2, tx_dbm=-40.0, shot_noise=True, sequence_length=8192)
    _, high, _ = _run(bits_per_symbol=2, tx_dbm=-37.0, shot_noise=True, sequence_length=8192)
    assert high.snr / low.snr == pytest.approx(10.0**0.3, rel=0.08)


def test_mer_is_the_evm_stated_in_decibels() -> None:
    _, result, _ = _run(bits_per_symbol=4, tx_dbm=-36.0, shot_noise=True)
    assert result.mer_db == pytest.approx(-20.0 * math.log10(result.evm), rel=1e-9)
