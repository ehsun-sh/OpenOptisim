"""A 10 Gb/s on-off-keyed link, characterised two ways.

Run it with::

    python examples/ook_link.py

No plotting dependency: the results print as tables. Everything here goes
through the public API, so anything this script can do the GUI will be able to
do too.
"""

from __future__ import annotations

from oosim import Graph, SimulationContext
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


def build(
    launch_dbm: float, span_km: float, *, dispersion: float = 17.0, sequence_length: int = 8192
) -> tuple[Graph, BERAnalyzer]:
    """PRBS -> NRZ -> laser -> MZM -> fiber -> PIN -> filter -> BER analyzer."""
    ctx = SimulationContext(
        bit_rate=BIT_RATE, samples_per_symbol=8, sequence_length=sequence_length, seed=2026
    )
    g = Graph(ctx)

    prbs = g.add(PRBSGenerator(order=15.0))
    driver = g.add(NRZDriver(v_low=V_PI, v_high=0.0))  # a 1 opens the modulator
    laser = g.add(CWLaser(power=launch_dbm, wavelength=1550.0))
    mzm = g.add(MachZehnderModulator(v_pi=V_PI, extinction_ratio=30.0))
    fiber = g.add(Fiber(length=span_km, attenuation=0.2, dispersion=dispersion))
    pin = g.add(PINPhotodiode(responsivity=0.8, shot_noise=True, thermal_noise=True))
    lpf = g.add(ElectricalFilter(bandwidth=0.7 * BIT_RATE / 1e9))
    analyzer = g.add(BERAnalyzer())

    g.chain(prbs, driver)
    g.connect(laser, mzm["optical_in"])
    g.connect(driver, mzm["electrical_in"])
    g.chain(mzm, fiber, pin, lpf)
    g.connect(lpf, analyzer["in"])
    g.connect(prbs["out"], analyzer["reference"])
    return g, analyzer


def sensitivity_curve() -> None:
    """BER against launch power, back to back.

    Below about -19 dBm the link makes enough errors to count, so the Gaussian
    estimate can be checked against reality. Above it, only the estimate is
    available — which is exactly the situation in a lab.
    """
    print("\nReceiver sensitivity (back to back, no dispersion)")
    print("  launch      Q      BER (from Q)    errors / bits")
    print("  " + "-" * 52)
    for launch in range(-24, -13):
        g, analyzer = build(float(launch), span_km=0.0, dispersion=0.0)
        m = g.run()[analyzer]
        counted = f"{m.errors} / {m.bits_evaluated}" if m.errors else "none counted"
        print(f"  {launch:>4} dBm  {m.q_factor:6.2f}   {m.ber_gaussian:12.3e}    {counted}")


def dispersion_limited_reach() -> None:
    """Q against distance at a launch power high enough that loss is not the limit.

    The eye closes from pulse overlap, not from lack of light: the amplifier-free
    reach of a 10 Gb/s NRZ link on standard fiber is set by dispersion long
    before it is set by power.
    """
    print("\nDispersion-limited reach (0 dBm launch, D = 17 ps/nm/km)")
    print("  distance     Q      BER (from Q)")
    print("  " + "-" * 40)
    for span in (0, 20, 40, 60, 80, 100, 120):
        g, analyzer = build(0.0, span_km=float(span), dispersion=17.0)
        m = g.run()[analyzer]
        print(f"  {span:>4} km   {m.q_factor:6.2f}   {m.ber_gaussian:12.3e}")

    print("\n  Compare with loss alone, dispersion switched off:")
    for span in (0, 60, 120):
        g, analyzer = build(0.0, span_km=float(span), dispersion=0.0)
        m = g.run()[analyzer]
        print(f"  {span:>4} km   {m.q_factor:6.2f}   {m.ber_gaussian:12.3e}")


if __name__ == "__main__":
    sensitivity_curve()
    dispersion_limited_reach()
