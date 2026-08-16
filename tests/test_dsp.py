"""Receiver DSP in the link: what carrier recovery is actually for.

The claim these tests exist to check is not "the block runs" but a physical one:
**with an uncompensated carrier, launching more power stops helping.** Laser
phase noise is a random walk, so it is not removable by subtracting a constant or
a line, and it puts a ceiling on the measured SNR that no power budget lifts.
Carrier recovery removes the ceiling. Both halves are asserted.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from oosim import Graph, SimulationContext
from oosim.components import (
    CarrierRecovery,
    CoherentReceiver,
    ConstellationAnalyzer,
    CWLaser,
    IQDriver,
    IQModulator,
    IQSampler,
    PRBSGenerator,
    QAMMapper,
)
from oosim.signals import ConstellationMeasurement

#: Enough symbols for the walk to actually wander, and a window edge to discard.
SEQUENCE = 4096
EDGES = 64.0


def build(
    *,
    bits_per_symbol: int = 4,
    linewidth: float = 100.0,
    launch: float = -10.0,
    recovery: bool = True,
    seed: int = 31,
    **recovery_params: float,
) -> tuple[Graph, ConstellationAnalyzer]:
    ctx = SimulationContext(
        bit_rate=32e9,
        samples_per_symbol=8,
        sequence_length=SEQUENCE,
        seed=seed,
        precision="double",
    )
    graph = Graph(ctx)
    prbs = graph.add(
        PRBSGenerator(order=23.0, bits_per_symbol=float(bits_per_symbol), label="prbs")
    )
    mapper = graph.add(QAMMapper(bits_per_symbol=float(bits_per_symbol), label="map"))
    driver = graph.add(IQDriver(label="drv"))
    tx = graph.add(CWLaser(power=launch, linewidth=linewidth, label="tx"))
    modulator = graph.add(IQModulator(label="mod"))
    lo = graph.add(CWLaser(power=10.0, linewidth=linewidth, label="lo"))
    receiver = graph.add(CoherentReceiver(label="rx"))
    sampler = graph.add(IQSampler(label="smp"))
    analyzer = graph.add(ConstellationAnalyzer(ignore_edges=EDGES, label="vsa"))

    graph.chain(prbs, mapper, driver)
    graph.connect(tx, modulator["optical_in"])
    graph.connect(driver["i"], modulator["i"])
    graph.connect(driver["q"], modulator["q"])
    graph.connect(modulator, receiver["in"])
    graph.connect(lo, receiver["lo"])
    graph.connect(receiver["i"], sampler["i"])
    graph.connect(receiver["q"], sampler["q"])
    graph.connect(mapper["out"], sampler["reference"])

    if recovery:
        stage = graph.add(CarrierRecovery(label="cr", **recovery_params))
        graph.connect(sampler["out"], stage["in"])
        graph.connect(stage["out"], analyzer["in"])
    else:
        graph.connect(sampler["out"], analyzer["in"])
    graph.connect(mapper["out"], analyzer["reference"])
    return graph, analyzer


def measure(**kwargs: Any) -> ConstellationMeasurement:
    graph, analyzer = build(**kwargs)
    return cast(ConstellationMeasurement, graph.run(keep=[])[analyzer])


# --------------------------------------------------------------------------
# The floor, and its removal
# --------------------------------------------------------------------------


def test_without_recovery_more_power_stops_buying_snr() -> None:
    """The ceiling. 12 dB of extra launch power buys barely one.

    This is the test that makes the next one mean something: if power still
    worked, carrier recovery would be solving a problem the model does not have.
    """
    quiet = measure(recovery=False, launch=-14.0)
    loud = measure(recovery=False, launch=-2.0)

    assert loud.snr_db - quiet.snr_db < 2.0
    assert loud.ber_estimated > 1e-4, "the link should still be failing at any power"


def test_with_recovery_power_works_again() -> None:
    """The same 12 dB now buys most of 12 dB, and the link closes."""
    quiet = measure(recovery=True, launch=-14.0)
    loud = measure(recovery=True, launch=-2.0)

    assert loud.snr_db - quiet.snr_db > 7.0
    assert loud.ber_estimated < 1e-12


@pytest.mark.parametrize("launch", [-14.0, -10.0, -6.0])
def test_recovery_is_worth_several_decibels_at_every_power(launch: float) -> None:
    without = measure(recovery=False, launch=launch)
    with_it = measure(recovery=True, launch=launch)
    assert with_it.snr_db > without.snr_db + 5.0


def test_a_wider_linewidth_costs_more_without_recovery_than_with() -> None:
    """The impairment being removed is specifically the laser's, not the shot noise."""
    narrow_off = measure(recovery=False, linewidth=1.0)
    wide_off = measure(recovery=False, linewidth=100.0)
    narrow_on = measure(recovery=True, linewidth=1.0)
    wide_on = measure(recovery=True, linewidth=100.0)

    assert narrow_off.snr_db - wide_off.snr_db > 5.0
    assert narrow_on.snr_db - wide_on.snr_db < 3.0


def test_recovery_does_not_damage_an_already_clean_signal() -> None:
    """A correction stage that hurt when there was nothing to correct would be worse
    than no stage at all, because it would be paid for on every link."""
    without = measure(recovery=False, linewidth=0.0, launch=-6.0)
    with_it = measure(recovery=True, linewidth=0.0, launch=-6.0)
    assert with_it.snr_db > without.snr_db - 0.5


# --------------------------------------------------------------------------
# Shape and wiring
# --------------------------------------------------------------------------


def test_recovery_preserves_the_sequence_and_the_alphabet() -> None:
    graph, _ = build()
    sampler = next(c for c in graph.components if isinstance(c, IQSampler))
    stage = next(c for c in graph.components if isinstance(c, CarrierRecovery))
    results = graph.run(keep=[sampler, stage])

    before = results.port(sampler, "out")
    after = results.port(stage, "out")

    assert after.num_symbols == before.num_symbols
    assert after.symbol_rate == before.symbol_rate
    assert np.array_equal(np.asarray(after.constellation), np.asarray(before.constellation))


def test_recovery_only_rotates_and_never_rescales() -> None:
    """Gain belongs to the AGC. A phase stage that also touched amplitude would be
    quietly doing two jobs, and only one of them would be tested."""
    graph, _ = build()
    sampler = next(c for c in graph.components if isinstance(c, IQSampler))
    stage = next(c for c in graph.components if isinstance(c, CarrierRecovery))
    results = graph.run(keep=[sampler, stage])

    before = np.abs(np.asarray(results.port(sampler, "out").symbols))
    after = np.abs(np.asarray(results.port(stage, "out").symbols))
    assert np.allclose(before, after, rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("bits_per_symbol", [2, 4, 6])
def test_recovery_works_for_every_square_format(bits_per_symbol: int) -> None:
    without = measure(bits_per_symbol=bits_per_symbol, recovery=False, launch=-6.0)
    with_it = measure(bits_per_symbol=bits_per_symbol, recovery=True, launch=-6.0)
    assert with_it.snr_db > without.snr_db + 5.0
