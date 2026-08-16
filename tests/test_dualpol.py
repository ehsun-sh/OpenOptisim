"""Polarization multiplexing: two tributaries at one wavelength.

The claim under test is stated most sharply by its failure case. A fibre rotates
the launched polarization arbitrarily, and after any real rotation a
dual-polarization receiver's two branches are *mixtures* rather than channels —
not degraded data, no data. The butterfly equaliser is what turns them back into
channels, and both halves of that are asserted here: unrecoverable without it,
error-free with it, at rotations up to and including the 45-degree worst case.

This is also the increment that finally exercises ``Ey``, which has been in
:class:`~oosim.signals.Band` since the first commit for exactly this reason.
"""

from __future__ import annotations

import math
from typing import Any, cast

import numpy as np
import pytest

from oosim import Graph, SimulationContext
from oosim.components import (
    ButterflyEqualizer,
    CarrierRecovery,
    ConstellationAnalyzer,
    CWLaser,
    DualPolarizationReceiver,
    IQDriver,
    IQModulator,
    IQSampler,
    PolarizationCombiner,
    PolarizationRotator,
    PRBSGenerator,
    QAMMapper,
    Splitter,
)
from oosim.dsp import butterfly_equalize, constellation_radii, godard_radius
from oosim.modulation import qam_constellation
from oosim.signals import Band, ConstellationMeasurement, OpticalSignal
from oosim.units import C_LIGHT

F0 = C_LIGHT / 1550e-9


def cw(ex: complex, ey: complex, ctx: SimulationContext) -> OpticalSignal:
    n = ctx.num_samples
    return OpticalSignal(
        bands=(
            Band(
                Ex=np.full(n, ex, dtype=np.complex128),
                Ey=np.full(n, ey, dtype=np.complex128),
                f0=F0,
                fs=ctx.sample_rate,
            ),
        )
    )


def context(**kwargs: Any) -> SimulationContext:
    defaults: dict[str, Any] = {
        "bit_rate": 32e9,
        "samples_per_symbol": 4,
        "sequence_length": 64,
        "seed": 1,
        "precision": "double",
    }
    defaults.update(kwargs)
    return SimulationContext(**defaults)


# --------------------------------------------------------------------------
# The passives
# --------------------------------------------------------------------------


def test_the_combiner_puts_the_two_arms_on_orthogonal_axes() -> None:
    """And is lossless, unlike a power combiner — orthogonal states do not interfere."""
    ctx = context()
    combiner = PolarizationCombiner(label="pbc")
    out = combiner.run(ctx, {"x": cw(1.0, 0.0, ctx), "y": cw(2.0, 0.0, ctx)})["out"]

    band = out.bands[0]
    assert np.allclose(band.Ex, 1.0)
    assert np.allclose(band.Ey, 2.0)
    assert band.average_power() == pytest.approx(5.0)


def test_the_combiner_is_the_opposite_rule_to_the_wavelength_combiner() -> None:
    """``Combiner`` refuses two inputs at one frequency; this one requires it.

    The distinction is the whole reason both exist: co-located carriers interfere
    unless they are orthogonally polarized, and then they do not.
    """
    ctx = context()
    out = PolarizationCombiner(label="pbc").run(
        ctx, {"x": cw(1.0, 0.0, ctx), "y": cw(1.0, 0.0, ctx)}
    )["out"]
    assert out.num_bands == 1


@pytest.mark.parametrize("angle", [0.0, 17.0, 45.0, 90.0, 137.0])
@pytest.mark.parametrize("phase", [0.0, 33.0])
def test_the_rotator_conserves_power_exactly(angle: float, phase: float) -> None:
    """A rotation that changed the power would not be a rotation."""
    ctx = context()
    rotator = PolarizationRotator(angle=angle, phase=phase, label="rot")
    source = cw(0.6, 0.8, ctx)
    out = rotator.run(ctx, {"in": source})["out"]

    assert out.bands[0].average_power() == pytest.approx(source.bands[0].average_power(), rel=1e-12)


def test_the_rotator_is_unitary() -> None:
    jones = PolarizationRotator(angle=37.0, phase=61.0, label="rot").jones_matrix()
    assert np.allclose(jones @ jones.conj().T, np.eye(2), atol=1e-12)


def test_a_quarter_turn_swaps_the_two_axes() -> None:
    ctx = context()
    out = PolarizationRotator(angle=90.0, label="rot").run(ctx, {"in": cw(1.0, 0.0, ctx)})["out"]
    band = out.bands[0]
    assert np.allclose(np.abs(band.Ex), 0.0, atol=1e-12)
    assert np.allclose(np.abs(band.Ey), 1.0)


@pytest.mark.parametrize("ways", [2, 4])
def test_the_splitter_divides_the_power(ways: int) -> None:
    """A graph edge fans out for free; an optical field does not."""
    ctx = context()
    splitter = Splitter(ways, label="sp")
    out = splitter.run(ctx, {"in": cw(1.0, 0.0, ctx)})

    assert len(out) == ways
    for signal in out.values():
        assert signal.bands[0].average_power() == pytest.approx(1.0 / ways)


# --------------------------------------------------------------------------
# The receiver
# --------------------------------------------------------------------------


def test_the_dual_pol_receiver_hears_both_axes() -> None:
    """The property a single-polarization receiver does not have.

    A signal on Y alone produces no photocurrent at all in a single-pol receiver
    whose LO sits on X. This one must see it, or polarization multiplexing cannot
    work.
    """
    ctx = context()
    receiver = DualPolarizationReceiver(
        responsivity=1.0, shot_noise=False, thermal_noise=False, label="rx"
    )
    lo = cw(1.0, 0.0, ctx)

    on_x = receiver.run(ctx, {"in": cw(1.0, 0.0, ctx), "lo": lo})
    on_y = receiver.run(ctx, {"in": cw(0.0, 1.0, ctx), "lo": lo})

    peak = lambda s: float(np.abs(np.asarray(s.samples)).max())  # noqa: E731
    assert peak(on_x["xi"]) > 0.1
    assert peak(on_x["yi"]) == pytest.approx(0.0, abs=1e-12)
    assert peak(on_y["yi"]) > 0.1
    assert peak(on_y["xi"]) == pytest.approx(0.0, abs=1e-12)


def test_the_dual_pol_receiver_emits_four_photocurrents() -> None:
    assert set(DualPolarizationReceiver.outputs) == {"xi", "xq", "yi", "yq"}


# --------------------------------------------------------------------------
# The equaliser, on its own
# --------------------------------------------------------------------------


def test_qpsk_has_one_radius_and_16qam_has_three() -> None:
    """Which is exactly why one algorithm does not serve both."""
    assert constellation_radii(qam_constellation(2)).shape == (1,)
    assert constellation_radii(qam_constellation(4)).shape == (3,)


def test_the_godard_radius_is_the_modulus_for_a_constant_modulus_format() -> None:
    points = qam_constellation(2)
    assert godard_radius(points) == pytest.approx(float(np.abs(points[0]) ** 2))


def test_the_godard_radius_sits_on_no_16qam_point() -> None:
    """Which is why CMA opens the eye but cannot close it, and needs a second stage."""
    radii = constellation_radii(qam_constellation(4)) ** 2
    target = godard_radius(qam_constellation(4))
    assert all(abs(target - r) > 0.05 for r in radii)


@pytest.mark.parametrize("taps", [0, 2, 8])
def test_the_equaliser_needs_an_odd_positive_tap_count(taps: int) -> None:
    with pytest.raises(ValueError, match="odd"):
        butterfly_equalize(
            np.zeros(64, dtype=np.complex128),
            np.zeros(64, dtype=np.complex128),
            qam_constellation(2),
            taps=taps,
        )


def test_the_equaliser_rejects_tributaries_of_different_lengths() -> None:
    with pytest.raises(ValueError, match="differ in length"):
        butterfly_equalize(
            np.zeros(64, dtype=np.complex128),
            np.zeros(32, dtype=np.complex128),
            qam_constellation(2),
        )


# --------------------------------------------------------------------------
# The whole link
# --------------------------------------------------------------------------


def build(
    *,
    rotation: float = 30.0,
    retardation: float = 25.0,
    equalize: bool = True,
    bits_per_symbol: int = 4,
    launch: float = 0.0,
    linewidth: float = 100.0,
    sequence_length: int = 4096,
    seed: int = 17,
) -> tuple[Graph, dict[str, ConstellationAnalyzer], dict[str, QAMMapper]]:
    """A dual-polarization coherent link through a rotated channel."""
    ctx = context(sequence_length=sequence_length, seed=seed)
    graph = Graph(ctx)

    laser = graph.add(CWLaser(power=launch, linewidth=linewidth, label="tx"))
    splitter = graph.add(Splitter(2, label="sp"))
    graph.connect(laser, splitter["in"])

    mappers: dict[str, QAMMapper] = {}
    modulators: dict[str, IQModulator] = {}
    for index, axis in enumerate(("x", "y")):
        # Different PRBS orders so the two tributaries carry different data —
        # identical tributaries would hide a failure to separate them.
        prbs = graph.add(
            PRBSGenerator(
                order=23.0 if axis == "x" else 15.0,
                bits_per_symbol=float(bits_per_symbol),
                label=f"prbs_{axis}",
            )
        )
        mapper = graph.add(QAMMapper(bits_per_symbol=float(bits_per_symbol), label=f"map_{axis}"))
        driver = graph.add(IQDriver(label=f"drv_{axis}"))
        modulator = graph.add(IQModulator(label=f"mod_{axis}"))
        graph.chain(prbs, mapper, driver)
        graph.connect(splitter[f"out{index}"], modulator["optical_in"])
        graph.connect(driver["i"], modulator["i"])
        graph.connect(driver["q"], modulator["q"])
        mappers[axis] = mapper
        modulators[axis] = modulator

    combiner = graph.add(PolarizationCombiner(label="pbc"))
    graph.connect(modulators["x"], combiner["x"])
    graph.connect(modulators["y"], combiner["y"])

    rotator = graph.add(PolarizationRotator(angle=rotation, phase=retardation, label="rot"))
    graph.connect(combiner, rotator["in"])

    lo = graph.add(CWLaser(power=13.0, linewidth=linewidth, label="lo"))
    receiver = graph.add(DualPolarizationReceiver(label="rx"))
    graph.connect(rotator, receiver["in"])
    graph.connect(lo, receiver["lo"])

    samplers: dict[str, IQSampler] = {}
    for axis in ("x", "y"):
        sampler = graph.add(IQSampler(label=f"smp_{axis}"))
        graph.connect(receiver[f"{axis}i"], sampler["i"])
        graph.connect(receiver[f"{axis}q"], sampler["q"])
        graph.connect(mappers[axis]["out"], sampler["reference"])
        samplers[axis] = sampler

    if equalize:
        equalizer = graph.add(ButterflyEqualizer(label="eq"))
        graph.connect(samplers["x"]["out"], equalizer["x"])
        graph.connect(samplers["y"]["out"], equalizer["y"])
        sources = {"x": equalizer["x_out"], "y": equalizer["y_out"]}
    else:
        sources = {axis: samplers[axis]["out"] for axis in ("x", "y")}

    # Each output is measured against *both* references. Nothing in a blind cost
    # function labels the tributaries, so a channel that swaps them — anything
    # past 45 degrees — is separated perfectly and delivered the other way round.
    # A deployed link resolves the pairing by framing; two analysers per output
    # resolve it here, in one run.
    analyzers: dict[str, ConstellationAnalyzer] = {}
    for axis in ("x", "y"):
        recovery = graph.add(CarrierRecovery(label=f"cr_{axis}"))
        graph.connect(sources[axis], recovery["in"])
        for reference in ("x", "y"):
            analyzer = graph.add(
                ConstellationAnalyzer(ignore_edges=128.0, label=f"vsa_{axis}{reference}")
            )
            graph.connect(recovery["out"], analyzer["in"])
            graph.connect(mappers[reference]["out"], analyzer["reference"])
            analyzers[axis + reference] = analyzer
    return graph, analyzers, mappers


def measure(**kwargs: Any) -> dict[str, ConstellationMeasurement]:
    """Both tributaries, with the output-to-reference pairing resolved."""
    graph, analyzers, _ = build(**kwargs)
    results = graph.run(keep=[])
    taken = {
        key: cast(ConstellationMeasurement, results[analyzer])
        for key, analyzer in analyzers.items()
    }

    direct = taken["xx"].symbol_errors + taken["yy"].symbol_errors
    swapped = taken["xy"].symbol_errors + taken["yx"].symbol_errors
    if direct <= swapped:
        return {"x": taken["xx"], "y": taken["yy"]}
    return {"x": taken["xy"], "y": taken["yx"]}


@pytest.mark.parametrize("rotation", [30.0, 45.0])
def test_a_rotated_channel_is_unrecoverable_without_the_equaliser(rotation: float) -> None:
    """Not degraded — destroyed. Both tributaries, at every rotation that mixes them.

    This is the test that gives the next one its meaning. 45 degrees is included
    because it is the worst case *and* the one a symmetric filter initialisation
    cannot converge on at all.
    """
    result = measure(rotation=rotation, equalize=False)
    for axis, measurement in result.items():
        assert measurement.evm > 0.5, f"{axis} should be destroyed, EVM was {measurement.evm}"
        assert measurement.symbol_errors > 1000, f"{axis} recovered symbols it should not have"


@pytest.mark.parametrize("rotation", [0.0, 30.0, 45.0, 72.0])
def test_the_equaliser_recovers_both_tributaries(rotation: float) -> None:
    """256 Gb/s through an arbitrary polarization rotation, error-free."""
    result = measure(rotation=rotation, equalize=True)
    for axis, measurement in result.items():
        assert measurement.symbol_errors == 0, f"{axis}: {measurement.symbol_errors} errors"
        assert measurement.evm < 0.06


def test_the_equaliser_costs_nothing_when_there_is_nothing_to_undo() -> None:
    """An unrotated channel needs no separation, and the filter must not damage it."""
    without = measure(rotation=0.0, equalize=False)
    with_it = measure(rotation=0.0, equalize=True)
    for axis in ("x", "y"):
        assert with_it[axis].evm < without[axis].evm + 0.005


def test_polarization_multiplexing_doubles_the_line_rate() -> None:
    """Two independent tributaries at one wavelength — the whole point of the exercise."""
    graph, _, mappers = build()
    results = graph.run(keep=list(mappers.values()))

    rates = [results.port(mapper, "out").bit_rate for mapper in mappers.values()]
    assert rates == [pytest.approx(128e9), pytest.approx(128e9)]
    assert sum(rates) == pytest.approx(256e9)


def test_the_two_tributaries_really_do_carry_different_data() -> None:
    """Otherwise a filter that failed to separate them would still look successful."""
    graph, _, mappers = build()
    results = graph.run(keep=list(mappers.values()))
    x = np.asarray(results.port(mappers["x"], "out").symbols)
    y = np.asarray(results.port(mappers["y"], "out").symbols)

    overlap = abs(complex(np.mean(x * np.conj(y))))
    assert overlap < 0.1, "the two tributaries are correlated; the test would prove nothing"


@pytest.mark.parametrize("bits_per_symbol", [2, 4])
def test_dual_polarization_works_for_every_format(bits_per_symbol: int) -> None:
    result = measure(rotation=40.0, bits_per_symbol=bits_per_symbol)
    for measurement in result.values():
        assert measurement.symbol_errors == 0


def test_a_retardation_alone_still_needs_the_equaliser() -> None:
    """Rotation is not the only way a fibre mixes states; a phase between the axes
    does it too, and the filter has to handle a complex channel and not just a
    real one."""
    without = measure(rotation=20.0, retardation=80.0, equalize=False)
    with_it = measure(rotation=20.0, retardation=80.0, equalize=True)
    assert max(m.evm for m in without.values()) > 0.5
    assert all(m.symbol_errors == 0 for m in with_it.values())


def test_the_equaliser_refuses_mismatched_formats() -> None:
    ctx = context()
    equalizer = ButterflyEqualizer(label="eq")
    from oosim.signals import SymbolSignal

    qpsk, qam16 = qam_constellation(2), qam_constellation(4)
    flat = np.zeros(64, dtype=int)
    with pytest.raises(ValueError, match="different constellations"):
        equalizer.run(
            ctx,
            {
                "x": SymbolSignal(symbols=qpsk[flat], symbol_rate=32e9, constellation=qpsk),
                "y": SymbolSignal(symbols=qam16[flat], symbol_rate=32e9, constellation=qam16),
            },
        )


def test_the_recovered_symbols_keep_their_shape_and_alphabet() -> None:
    graph, _, _ = build()
    equalizer = next(c for c in graph.components if isinstance(c, ButterflyEqualizer))
    sampler = next(c for c in graph.components if c.label == "smp_x")
    results = graph.run(keep=[equalizer, sampler])

    before = results.port(sampler, "out")
    after = results.port(equalizer, "x_out")
    assert after.num_symbols == before.num_symbols
    assert after.order == before.order
    assert after.symbol_rate == before.symbol_rate


def test_a_quarter_turn_delivers_the_tributaries_swapped() -> None:
    """The output labels are not meaningful, and this is where that becomes visible.

    At 90 degrees the channel maps x onto y outright. A filter that genuinely
    inverts the channel therefore hands back the tributaries the other way round
    — and it is *right* to, since nothing blind can know which was which. The
    pairing is recovered here by measuring against both references; a real link
    recovers it from framing.
    """
    graph, analyzers, _ = build(rotation=90.0, retardation=0.0, equalize=True)
    results = graph.run(keep=[])
    errors = {key: results[analyzer].symbol_errors for key, analyzer in analyzers.items()}

    assert errors["xy"] == 0 and errors["yx"] == 0, "the swapped pairing should be clean"
    assert errors["xx"] > 0 and errors["yy"] > 0, "the direct pairing should not be"


def test_the_45_degree_case_converges_at_all() -> None:
    """Guards the symmetry tilt in the filter's initialisation.

    A symmetric start is equidistant from both valid solutions when the channel
    mixes half and half, which is a saddle rather than a minimum: the adaptation
    stalls, and running it longer makes it worse. Without the tilt this fails.
    """
    result = measure(rotation=45.0, bits_per_symbol=4, equalize=True)
    assert all(m.symbol_errors == 0 for m in result.values())
    assert all(m.evm < 0.06 for m in result.values())


def test_the_link_still_needs_carrier_recovery_as_well() -> None:
    """The equaliser separates; it does not track the laser. Both stages earn their place."""
    narrow = measure(rotation=30.0, linewidth=1.0)
    wide = measure(rotation=30.0, linewidth=100.0)
    # Carrier recovery is in the chain for both, so a wide linewidth should cost
    # little. Without it the wide case would collapse; that is tested in test_dsp.
    assert all(m.symbol_errors == 0 for m in narrow.values())
    assert all(m.symbol_errors == 0 for m in wide.values())


def test_power_is_conserved_from_the_combiner_to_the_rotator() -> None:
    """End-to-end on the optical side: nothing invents or loses power in between."""
    ctx = context()
    combined = PolarizationCombiner(label="pbc").run(
        ctx, {"x": cw(0.7, 0.0, ctx), "y": cw(0.4, 0.0, ctx)}
    )["out"]
    rotated = PolarizationRotator(angle=33.0, phase=48.0, label="rot").run(
        ctx, {"in": combined}
    )["out"]

    assert rotated.signal_power() == pytest.approx(0.7**2 + 0.4**2, rel=1e-12)
    assert math.isclose(combined.signal_power(), rotated.signal_power(), rel_tol=1e-12)
