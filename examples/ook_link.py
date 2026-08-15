"""A 10 Gb/s on-off-keyed link, characterised two ways.

Run it with::

    python examples/ook_link.py

It builds the link once, sweeps it, and writes the schematic to
``examples/ook_link.oosim`` — the same file the GUI will eventually open.
No plotting dependency: results print as tables.
"""

from __future__ import annotations

from pathlib import Path

from oosim import Graph, SimulationContext, save, sweep
from oosim.components import (
    BERAnalyzer,
    CWLaser,
    ElectricalFilter,
    Fiber,
    MachZehnderModulator,
    NRZDriver,
    PINPhotodiode,
    PRBSGenerator,
)

BIT_RATE = 10e9
V_PI = 4.0

# Where the GUI would put the blocks. Kept out of the physics, in its own
# section of the project file, so a diff shows model changes and not moved boxes.
LAYOUT = {
    "prbs": {"x": 40.0, "y": 40.0},
    "driver": {"x": 200.0, "y": 40.0},
    "laser": {"x": 40.0, "y": 160.0},
    "mzm": {"x": 360.0, "y": 100.0},
    "fiber": {"x": 520.0, "y": 100.0},
    "pin": {"x": 680.0, "y": 100.0},
    "lpf": {"x": 840.0, "y": 100.0},
    "ber": {"x": 1000.0, "y": 100.0},
}


def build(sequence_length: int = 8192) -> Graph:
    """PRBS -> NRZ -> laser -> MZM -> fiber -> PIN -> filter -> BER analyzer."""
    ctx = SimulationContext(
        bit_rate=BIT_RATE, samples_per_symbol=8, sequence_length=sequence_length, seed=2026
    )
    g = Graph(ctx)

    prbs = g.add(PRBSGenerator(order=15.0, label="prbs"))
    driver = g.add(NRZDriver(v_low=V_PI, v_high=0.0, label="driver"))  # a 1 opens the modulator
    laser = g.add(CWLaser(power=0.0, wavelength=1550.0, label="laser"))
    mzm = g.add(MachZehnderModulator(v_pi=V_PI, extinction_ratio=30.0, label="mzm"))
    fiber = g.add(Fiber(length=0.0, attenuation=0.2, dispersion=17.0, label="fiber"))
    pin = g.add(PINPhotodiode(responsivity=0.8, shot_noise=True, thermal_noise=True, label="pin"))
    lpf = g.add(ElectricalFilter(bandwidth=0.7 * BIT_RATE / 1e9, label="lpf"))
    analyzer = g.add(BERAnalyzer(label="ber"))

    g.chain(prbs, driver)
    g.connect(laser, mzm["optical_in"])
    g.connect(driver, mzm["electrical_in"])
    g.chain(mzm, fiber, pin, lpf)
    g.connect(lpf, analyzer["in"])
    g.connect(prbs["out"], analyzer["reference"])
    return g


def sensitivity_curve(graph: Graph) -> None:
    """BER against launch power, back to back.

    Below about -19 dBm the link makes enough errors to count, so the Gaussian
    estimate can be checked against reality. Above it, only the estimate is
    available — which is exactly the situation in a lab.
    """
    analyzer = graph.components[-1]
    result = sweep(
        graph,
        {("laser", "power"): [float(p) for p in range(-24, -13)]},
        fixed={("fiber", "length"): 0.0, ("fiber", "dispersion"): 0.0},
    )

    print("\nReceiver sensitivity (back to back, no dispersion)")
    print("  launch      Q      BER (from Q)    errors / bits")
    print("  " + "-" * 52)
    for point in result:
        m = point.runs[0][analyzer]
        counted = f"{m.errors} / {m.bits_evaluated}" if m.errors else "none counted"
        launch = point.values["laser.power"]
        print(f"  {launch:>4.0f} dBm  {m.q_factor:6.2f}   {m.ber_gaussian:12.3e}    {counted}")


def dispersion_limited_reach(graph: Graph) -> None:
    """Q against distance at a launch power high enough that loss is not the limit.

    The eye closes from pulse overlap, not from lack of light: the amplifier-free
    reach of a 10 Gb/s NRZ link on standard fiber is set by dispersion long
    before it is set by power.
    """
    analyzer = graph.components[-1]
    spans = [0.0, 20.0, 40.0, 60.0, 80.0, 100.0, 120.0]

    for label, dispersion in (("D = 17 ps/nm/km", 17.0), ("dispersion off", 0.0)):
        result = sweep(
            graph,
            {("fiber", "length"): spans},
            fixed={("laser", "power"): 0.0, ("fiber", "dispersion"): dispersion},
        )
        print(f"\nReach at 0 dBm launch, {label}")
        print("  distance     Q      BER (from Q)")
        print("  " + "-" * 40)
        for point in result:
            m = point.runs[0][analyzer]
            print(
                f"  {point.values['fiber.length']:>4.0f} km   "
                f"{m.q_factor:6.2f}   {m.ber_gaussian:12.3e}"
            )


def monte_carlo(graph: Graph) -> None:
    """Repeat one marginal operating point to see the spread on the estimate.

    A single BER at a marginal point is one sample. The spread is the reason
    repeats exist, and the reason a single number should not be quoted as if it
    were the answer.
    """
    analyzer = graph.components[-1]
    result = sweep(
        graph,
        {("laser", "power"): [-20.0]},
        fixed={("fiber", "length"): 0.0, ("fiber", "dispersion"): 0.0},
        runs=8,
    )
    errors = result.metric(analyzer, lambda m: float(m.errors))[0]
    q = result.metric(analyzer, lambda m: m.q_factor)[0]

    print("\nEight repeats at -20 dBm (independent noise, same link)")
    print(f"  errors: {[int(e) for e in errors]}")
    print(f"  Q:      {q.mean():.3f} +/- {q.std():.3f}")


if __name__ == "__main__":
    graph = build()
    sensitivity_curve(graph)
    dispersion_limited_reach(graph)
    monte_carlo(graph)

    destination = save(graph, Path(__file__).with_suffix(".oosim"), ui=LAYOUT)
    print(f"\nSchematic written to {destination.name}")
