"""A 256 Gb/s dual-polarization coherent link through a rotated channel.

Two independent 16-QAM tributaries share one wavelength on orthogonal
polarizations. A fibre rotates the launched state arbitrarily, so what the
receiver's two branches carry is a *mixture* of both tributaries rather than one
each — and past a small angle neither is recoverable at all. The butterfly
equaliser is what separates them again.

Run: ``python examples/dualpol_link.py``
"""

from __future__ import annotations

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

SYMBOL_RATE = 32e9
BITS_PER_SYMBOL = 4


def build(rotation: float, *, equalize: bool, sequence_length: int = 4096) -> tuple:
    ctx = SimulationContext(
        bit_rate=SYMBOL_RATE,
        samples_per_symbol=4,
        sequence_length=sequence_length,
        seed=2026,
        precision="double",
    )
    graph = Graph(ctx)
    laser = graph.add(CWLaser(power=0.0, linewidth=100.0, label="tx"))
    splitter = graph.add(Splitter(2, label="sp"))
    graph.connect(laser, splitter["in"])

    mappers, modulators = {}, {}
    for index, axis in enumerate(("x", "y")):
        prbs = graph.add(
            PRBSGenerator(
                order=23.0 if axis == "x" else 15.0,
                bits_per_symbol=float(BITS_PER_SYMBOL),
                label=f"prbs_{axis}",
            )
        )
        mapper = graph.add(QAMMapper(bits_per_symbol=float(BITS_PER_SYMBOL), label=f"map_{axis}"))
        driver = graph.add(IQDriver(label=f"drv_{axis}"))
        modulator = graph.add(IQModulator(label=f"mod_{axis}"))
        graph.chain(prbs, mapper, driver)
        graph.connect(splitter[f"out{index}"], modulator["optical_in"])
        graph.connect(driver["i"], modulator["i"])
        graph.connect(driver["q"], modulator["q"])
        mappers[axis], modulators[axis] = mapper, modulator

    combiner = graph.add(PolarizationCombiner(label="pbc"))
    graph.connect(modulators["x"], combiner["x"])
    graph.connect(modulators["y"], combiner["y"])
    rotator = graph.add(PolarizationRotator(angle=rotation, phase=25.0, label="rot"))
    graph.connect(combiner, rotator["in"])

    lo = graph.add(CWLaser(power=13.0, linewidth=100.0, label="lo"))
    receiver = graph.add(DualPolarizationReceiver(label="rx"))
    graph.connect(rotator, receiver["in"])
    graph.connect(lo, receiver["lo"])

    samplers = {}
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

    analyzers = {}
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
    return graph, analyzers


def measure(rotation: float, *, equalize: bool) -> tuple:
    graph, analyzers = build(rotation, equalize=equalize)
    results = graph.run(keep=[])
    taken = {key: results[a] for key, a in analyzers.items()}
    # Nothing blind labels the tributaries, so a channel that swaps them is
    # separated correctly and delivered the other way round. Framing resolves
    # this in a real link; both references resolve it here.
    direct = taken["xx"].symbol_errors + taken["yy"].symbol_errors
    swapped = taken["xy"].symbol_errors + taken["yx"].symbol_errors
    if direct <= swapped:
        return taken["xx"], taken["yy"], False
    return taken["xy"], taken["yx"], True


def main() -> None:
    rate = SYMBOL_RATE * BITS_PER_SYMBOL * 2 / 1e9
    print(
        f"Dual-polarization 16-QAM, {SYMBOL_RATE / 1e9:.0f} GBd x 2 pol = {rate:.0f} Gb/s\n"
        f"Two independent tributaries on one wavelength, through a rotated channel.\n"
    )
    print("rotation    without equaliser              with equaliser")
    print("-" * 70)
    for rotation in (0.0, 15.0, 30.0, 45.0, 72.0, 90.0):
        off_x, off_y, _ = measure(rotation, equalize=False)
        on_x, on_y, swapped = measure(rotation, equalize=True)
        note = "  (tributaries swapped)" if swapped else ""
        print(
            f"{rotation:5.0f} deg  EVM {off_x.evm * 100:6.1f} / {off_y.evm * 100:6.1f} %"
            f"  {off_x.symbol_errors + off_y.symbol_errors:5} err"
            f"      EVM {on_x.evm * 100:4.2f} / {on_y.evm * 100:4.2f} %"
            f"  {on_x.symbol_errors + on_y.symbol_errors:3} err{note}"
        )
    print(
        "\nPast a few degrees the unequalised branches are not degraded — they carry\n"
        "no recoverable data at all. Separating them is linear algebra the receiver\n"
        "has to learn blind, with no training sequence anywhere in the link."
    )


if __name__ == "__main__":
    main()
