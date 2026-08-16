"""Pulse shaping and differential encoding.

Two transmitter-side refinements with sharply different jobs. Root-raised-cosine
shaping bounds the spectrum, which is what lets channels be packed onto a grid at
all. Differential quadrant encoding closes the quarter-turn ambiguity that every
blind receiver stage upstream leaves behind.

The second is tested *without* the constellation analyser, deliberately. That
block removes a common rotation data-aided, using the transmitted sequence — so
it is immune to the ambiguity by construction and would report success either
way. A real receiver has no reference, so the comparison here is made where it
actually bites: on the decoded symbols themselves.
"""

from __future__ import annotations

import numpy as np
import pytest

from oosim import Graph, SimulationContext
from oosim.components import (
    CoherentReceiver,
    ConstellationAnalyzer,
    CWLaser,
    DifferentialDecoder,
    IQDriver,
    IQModulator,
    IQSampler,
    PRBSGenerator,
    QAMMapper,
)
from oosim.dsp import circular_filter, root_raised_cosine, shape_symbols
from oosim.modulation import (
    differential_decode,
    differential_encode,
    nearest_indices,
    qam_constellation,
    quadrant_constellation,
)

QUARTER = 1j

# --------------------------------------------------------------------------
# Root-raised cosine
# --------------------------------------------------------------------------


@pytest.mark.parametrize("roll_off", [0.0, 0.1, 0.2, 0.5, 1.0])
def test_the_filter_carries_unit_energy(roll_off: float) -> None:
    taps = root_raised_cosine(roll_off, 16, 8)
    assert float(np.sum(taps**2)) == pytest.approx(1.0)


def test_the_filter_length_is_odd_so_the_peak_lands_on_a_sample() -> None:
    """An even-length filter has its peak between samples and a half-sample delay."""
    for span, sps in ((16, 8), (15, 8), (10, 4)):
        assert root_raised_cosine(0.2, span, sps).shape[0] % 2 == 1


@pytest.mark.parametrize("roll_off", [0.2, 0.5, 1.0])
def test_two_roots_make_a_nyquist_pulse(roll_off: float) -> None:
    """The property the whole scheme rests on: zero at every other symbol instant.

    Cascading the transmitter's root with the receiver's matched root gives a full
    raised cosine, which is exactly zero at every symbol instant but its own — so
    neighbouring symbols contribute nothing to a decision. One root alone has no
    such property, which is why the shaping is split rather than done at one end.
    """
    sps = 8
    taps = root_raised_cosine(roll_off, 24, sps)
    cascade = np.convolve(taps, taps)
    at_instants = cascade[(cascade.shape[0] - 1) // 2 :: sps]

    assert at_instants[0] == pytest.approx(1.0, rel=1e-6)
    assert float(np.max(np.abs(at_instants[1:]))) < 0.02


def test_a_sharper_roll_off_needs_a_longer_filter_to_stay_nyquist() -> None:
    """The trade the span parameter exists for: a narrow spectrum decays slowly.

    Truncating it is what leaves residual inter-symbol interference, and at
    roll-off zero the tails are so long that a modest span leaves plainly more of
    it than a generous roll-off does.
    """
    sps = 8

    def residual(roll_off: float) -> float:
        taps = root_raised_cosine(roll_off, 10, sps)
        cascade = np.convolve(taps, taps)
        return float(np.max(np.abs(cascade[(cascade.shape[0] - 1) // 2 :: sps][1:])))

    assert residual(0.0) > residual(0.3) > residual(1.0)


@pytest.mark.parametrize(("roll_off", "span"), [(-0.1, 16), (1.5, 16), (0.2, 0)])
def test_the_filter_rejects_impossible_settings(roll_off: float, span: int) -> None:
    with pytest.raises(ValueError):
        root_raised_cosine(roll_off, span, 8)


def test_shaping_preserves_the_constellations_scale() -> None:
    """A unit-energy filter would otherwise rescale the constellation silently.

    The claim is about the *scale*, not about individual symbols: a single root is
    not ISI-free — only the cascade with the receiver's root is — so the shaped
    waveform sampled at symbol instants carries interference from its neighbours
    by design. What must not change is the level the drive voltages are derived
    from.
    """
    sps = 8
    points = qam_constellation(4)
    rng = np.random.default_rng(11)
    symbols = points[rng.integers(0, 16, 4096)]
    taps = root_raised_cosine(0.2, 24, sps)

    at_instants = shape_symbols(symbols, sps, taps)[::sps]
    scale = float(np.sqrt(np.mean(np.abs(at_instants) ** 2) / np.mean(np.abs(symbols) ** 2)))
    assert scale == pytest.approx(1.0, rel=0.05)


def test_a_shaped_pulse_survives_its_own_matched_filter() -> None:
    """Shape, match, sample: the symbols come back."""
    sps = 8
    points = qam_constellation(4)
    rng = np.random.default_rng(0)
    indices = rng.integers(0, 16, 512)
    taps = root_raised_cosine(0.2, 24, sps)

    received = circular_filter(shape_symbols(points[indices], sps, taps), taps)[::sps]
    gain = complex(np.mean(received * np.conj(points[indices])) / np.mean(np.abs(points) ** 2))

    assert np.array_equal(nearest_indices(received / gain, points), indices)


def spectral_fraction(
    samples: np.ndarray, sample_rate: float, symbol_rate: float, edge: float
) -> float:
    """Fraction of the power inside ``+-edge`` symbol rates."""
    spectrum = np.abs(np.fft.fftshift(np.fft.fft(samples))) ** 2
    frequency = np.fft.fftshift(np.fft.fftfreq(samples.shape[0], 1.0 / sample_rate)) / symbol_rate
    return float(spectrum[np.abs(frequency) <= edge].sum() / spectrum.sum())


def build(
    *,
    pulse_shaping: bool = False,
    roll_off: float = 0.2,
    drive_ratio: float = 0.4,
    bits_per_symbol: int = 4,
    differential: bool = False,
    sequence_length: int = 2048,
) -> tuple[Graph, ConstellationAnalyzer, IQModulator]:
    ctx = SimulationContext(
        bit_rate=32e9,
        samples_per_symbol=8,
        sequence_length=sequence_length,
        seed=5,
        precision="double",
    )
    graph = Graph(ctx)
    prbs = graph.add(
        PRBSGenerator(order=23.0, bits_per_symbol=float(bits_per_symbol), label="prbs")
    )
    mapper = graph.add(
        QAMMapper(
            bits_per_symbol=float(bits_per_symbol), differential=differential, label="map"
        )
    )
    reference = graph.add(
        QAMMapper(bits_per_symbol=float(bits_per_symbol), differential=False, label="ref")
    )
    driver = graph.add(
        IQDriver(
            pulse_shaping=pulse_shaping,
            roll_off=roll_off,
            drive_ratio=drive_ratio,
            label="drv",
        )
    )
    laser = graph.add(CWLaser(power=-6.0, linewidth=0.0, label="tx"))
    modulator = graph.add(IQModulator(label="mod"))
    lo = graph.add(CWLaser(power=13.0, linewidth=0.0, label="lo"))
    receiver = graph.add(CoherentReceiver(shot_noise=False, thermal_noise=False, label="rx"))
    sampler = graph.add(
        IQSampler(matched_filter=pulse_shaping, roll_off=roll_off, label="smp")
    )

    graph.connect(prbs["out"], mapper["in"])
    graph.connect(prbs["out"], reference["in"])
    graph.connect(mapper["out"], driver["in"])
    graph.connect(laser, modulator["optical_in"])
    graph.connect(driver["i"], modulator["i"])
    graph.connect(driver["q"], modulator["q"])
    graph.connect(modulator, receiver["in"])
    graph.connect(lo, receiver["lo"])
    graph.connect(receiver["i"], sampler["i"])
    graph.connect(receiver["q"], sampler["q"])
    graph.connect(mapper["out"], sampler["reference"])

    source = sampler["out"]
    if differential:
        decoder = graph.add(DifferentialDecoder(label="dec"))
        graph.connect(source, decoder["in"])
        source = decoder["out"]

    analyzer = graph.add(ConstellationAnalyzer(ignore_edges=64.0, label="vsa"))
    graph.connect(source, analyzer["in"])
    graph.connect(reference["out"], analyzer["reference"])
    return graph, analyzer, modulator


def test_shaping_bounds_the_spectrum_and_a_held_symbol_does_not() -> None:
    """The reason pulse shaping exists at all.

    A raised-cosine spectrum stops dead at ``(1 + roll_off) / 2`` symbol rates
    from the carrier. A held symbol has a sinc spectrum, which never stops — fine
    for one channel alone, useless the moment neighbours are packed beside it on
    a grid.
    """
    edge = (1.0 + 0.2) / 2.0

    graph, _, modulator = build(pulse_shaping=True)
    shaped = np.asarray(graph.run(keep=[modulator]).port(modulator, "out").bands[0].Ex)

    graph, _, modulator = build(pulse_shaping=False)
    held = np.asarray(graph.run(keep=[modulator]).port(modulator, "out").bands[0].Ex)

    contained = spectral_fraction(shaped, 32e9 * 8, 32e9, edge)
    leaking = spectral_fraction(held, 32e9 * 8, 32e9, edge)

    assert contained > 0.99, f"only {contained:.1%} of the shaped power is in band"
    assert leaking < 0.9, f"a held symbol should leak; {leaking:.1%} was contained"


def test_a_shaped_link_still_recovers_every_symbol() -> None:
    graph, analyzer, _ = build(pulse_shaping=True)
    result = graph.run(keep=[])[analyzer]
    assert result.symbol_errors == 0
    assert result.evm < 0.03


def test_shaping_raises_the_peak_so_a_full_swing_drive_clips() -> None:
    """A real effect with a real remedy, and both are asserted.

    A held symbol never leaves the constellation's levels; a shaped one overshoots
    between them. Driving at full swing runs the pre-distortion past ``arcsin(1)``
    and clips. Backing off is what a transmitter does about it.
    """
    graph, analyzer, _ = build(pulse_shaping=True, drive_ratio=1.0)
    clipped = graph.run(keep=[])[analyzer].evm

    graph, analyzer, _ = build(pulse_shaping=True, drive_ratio=0.4)
    backed_off = graph.run(keep=[])[analyzer].evm

    assert clipped > 0.04
    assert backed_off < clipped / 3.0


def test_a_held_symbol_does_not_clip_at_full_swing() -> None:
    """Which is what identifies the clipping above as the shaping's doing."""
    graph, analyzer, _ = build(pulse_shaping=False, drive_ratio=1.0)
    assert graph.run(keep=[])[analyzer].evm == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------
# Differential quadrant encoding
# --------------------------------------------------------------------------


@pytest.mark.parametrize("bits_per_symbol", [2, 4, 6])
def test_the_quadrant_alphabet_is_the_same_points_relabelled(bits_per_symbol: int) -> None:
    relabelled = quadrant_constellation(bits_per_symbol)
    plain = qam_constellation(bits_per_symbol)

    assert relabelled.shape == plain.shape
    assert float(np.mean(np.abs(relabelled) ** 2)) == pytest.approx(1.0)
    assert np.allclose(
        np.sort_complex(np.round(relabelled, 9)), np.sort_complex(np.round(plain, 9))
    )


@pytest.mark.parametrize("bits_per_symbol", [2, 4, 6])
def test_a_quarter_turn_is_a_constant_shift_in_the_relabelled_index(
    bits_per_symbol: int,
) -> None:
    """The property that makes differential encoding possible.

    Under the plain Gray labelling a quarter turn permutes the bits differently
    for different points, because rotating swaps the roles of the I and Q
    magnitudes. Under this labelling it does exactly one thing, to every point.
    """
    points = quadrant_constellation(bits_per_symbol)
    quarter = points.shape[0] // 4

    turned = nearest_indices(points * QUARTER, points)
    expected = (np.arange(points.shape[0]) + quarter) % points.shape[0]
    assert np.array_equal(turned, expected)


def test_the_plain_labelling_does_not_have_that_property() -> None:
    """Stated as a test so the relabelling cannot be mistaken for redundant."""
    points = qam_constellation(4)
    turned = nearest_indices(points * QUARTER, points)
    shifts = (turned - np.arange(points.shape[0])) % points.shape[0]
    assert len(set(shifts.tolist())) > 1, "a plain Gray quarter turn should not be a constant shift"


def test_bpsk_has_no_quadrant_structure_to_encode() -> None:
    with pytest.raises(ValueError, match="at least 2 bits"):
        quadrant_constellation(1)


@pytest.mark.parametrize("bits_per_symbol", [2, 4, 6])
def test_differential_coding_round_trips(bits_per_symbol: int) -> None:
    points = quadrant_constellation(bits_per_symbol)
    quarter = points.shape[0] // 4
    rng = np.random.default_rng(1)
    indices = rng.integers(0, points.shape[0], size=2000)

    recovered = differential_decode(differential_encode(indices, quarter), quarter)
    assert np.array_equal(recovered[1:], indices[1:])


@pytest.mark.parametrize("bits_per_symbol", [2, 4, 6])
@pytest.mark.parametrize("turns", [0, 1, 2, 3])
def test_the_data_survives_any_quarter_turn(bits_per_symbol: int, turns: int) -> None:
    """What the whole scheme is for: the ambiguity becomes a constant that cancels."""
    points = quadrant_constellation(bits_per_symbol)
    quarter = points.shape[0] // 4
    rng = np.random.default_rng(2)
    indices = rng.integers(0, points.shape[0], size=2000)

    transmitted = points[differential_encode(indices, quarter)]
    received = transmitted * (QUARTER**turns)
    decoded = differential_decode(nearest_indices(received, points), quarter)

    assert np.array_equal(decoded[1:], indices[1:])


@pytest.mark.parametrize("turns", [1, 2, 3])
def test_absolute_labelling_loses_everything_to_the_same_turn(turns: int) -> None:
    """The comparison that gives the previous test its meaning."""
    points = qam_constellation(4)
    rng = np.random.default_rng(2)
    indices = rng.integers(0, points.shape[0], size=2000)

    received = points[indices] * (QUARTER**turns)
    decoded = nearest_indices(received, points)

    wrong = int(np.count_nonzero(decoded != indices))
    assert wrong > 1500, f"only {wrong} of 2000 symbols were corrupted"


@pytest.mark.parametrize("turns", [0, 1, 2, 3])
def test_the_decoder_block_undoes_a_turn_applied_to_real_symbols(turns: int) -> None:
    """The component, on symbols a mapper actually produced.

    The decoder re-emits under the ordinary Gray labelling, so its output can be
    compared directly against a plain mapper fed the same bits — which is what the
    link graph does, and what makes the analyser need no special handling.
    """
    ctx = SimulationContext(
        bit_rate=32e9, samples_per_symbol=4, sequence_length=512, seed=3, precision="double"
    )
    graph = Graph(ctx)
    prbs = graph.add(PRBSGenerator(order=15.0, bits_per_symbol=4.0, label="prbs"))
    encoded = graph.add(QAMMapper(bits_per_symbol=4.0, differential=True, label="map"))
    plain = graph.add(QAMMapper(bits_per_symbol=4.0, differential=False, label="ref"))
    graph.connect(prbs["out"], encoded["in"])
    graph.connect(prbs["out"], plain["in"])
    results = graph.run(keep=[encoded, plain])

    from oosim.signals import SymbolSignal

    sent = results.port(encoded, "out")
    turned = SymbolSignal(
        symbols=np.asarray(sent.symbols) * (QUARTER**turns),
        symbol_rate=sent.symbol_rate,
        constellation=np.asarray(sent.constellation),
    )
    decoded = DifferentialDecoder(label="dec").run(ctx, {"in": turned})["out"]
    expected = np.asarray(results.port(plain, "out").symbols)

    assert np.allclose(np.asarray(decoded.symbols)[1:], expected[1:])


def test_a_differentially_encoded_link_runs_end_to_end() -> None:
    """And costs nothing when there is no ambiguity to absorb."""
    graph, analyzer, _ = build(differential=True)
    result = graph.run(keep=[])[analyzer]
    assert result.symbol_errors == 0
    assert result.evm < 0.01


def test_shaping_and_differential_coding_compose() -> None:
    graph, analyzer, _ = build(pulse_shaping=True, differential=True)
    result = graph.run(keep=[])[analyzer]
    assert result.symbol_errors == 0
