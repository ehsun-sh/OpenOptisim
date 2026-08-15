"""An amplified, nonlinear link — where optical engineering gets interesting.

Run it with::

    python examples/amplified_link.py

Two effects fight each other here. Launch too little power and amplifier noise
dominates; launch too much and the Kerr effect distorts the signal. The optimum
between them is the whole reason long-haul links are designed rather than merely
assembled.
"""

from __future__ import annotations

import math

from oosim import Component, Graph, SimulationContext, sweep
from oosim.analysis import peak_power, rms_time_width
from oosim.components import EDFA, CWLaser, Fiber, OSNRMeter, SechPulse
from oosim.kernels import (
    dispersion_to_beta2,
    soliton_peak_power,
    soliton_period,
)
from oosim.units import w_to_dbm

SPAN_KM = 80.0
SPAN_LOSS_DB = 0.2 * SPAN_KM


def amplified_chain(spans: int) -> tuple[Graph, OSNRMeter]:
    """A chain of identical span-plus-amplifier sections, each exactly transparent."""
    ctx = SimulationContext(bit_rate=10e9, samples_per_symbol=8, sequence_length=64, seed=7)
    g = Graph(ctx)

    previous: Component = g.add(CWLaser(power=0.0, wavelength=1550.0, label="laser"))
    for i in range(spans):
        fiber = g.add(Fiber(length=SPAN_KM, attenuation=0.2, label=f"span{i}"))
        amp = g.add(EDFA(gain=SPAN_LOSS_DB, noise_figure=5.0, label=f"edfa{i}"))
        g.connect(previous, fiber)
        g.connect(fiber, amp)
        previous = amp

    meter = g.add(OSNRMeter(label="osnr"))
    g.connect(previous, meter)
    return g, meter


def osnr_versus_span_count() -> None:
    """Each identical span costs 10*log10(N) of OSNR. Doubling the count costs 3 dB.

    Nothing in the model is written in those terms — the noise bins simply
    accumulate — so the rule falling out is the check.
    """
    print("\nOSNR down a chain of 80 km spans, each amplified back to 0 dBm")
    print("  spans   reach     OSNR      vs. one span")
    print("  " + "-" * 46)
    reference = None
    for spans in (1, 2, 4, 8, 16):
        g, meter = amplified_chain(spans)
        measured = g.run()[meter]
        reference = reference if reference is not None else measured
        print(
            f"  {spans:>5}  {spans * SPAN_KM:>5.0f} km  {measured:6.2f} dB   "
            f"{measured - reference:+6.2f} dB  (theory {-10 * math.log10(spans):+6.2f})"
        )


def launch_power_optimum() -> None:
    """OSNR always improves with launch power; the eye does not.

    This sweep shows only the noise side of the trade-off, because OSNR does not
    know about distortion. The nonlinear cost is what the soliton section below
    makes visible.
    """
    g, meter = amplified_chain(8)
    result = sweep(g, {("laser", "power"): [float(p) for p in (-6, -3, 0, 3, 6)]})

    print("\nOSNR after 8 spans, against launch power")
    print("  launch     OSNR")
    print("  " + "-" * 22)
    for point in result:
        print(f"  {point.values['laser.power']:>4.0f} dBm  {point.runs[0][meter]:6.2f} dB")


def soliton_demonstration() -> None:
    """A pulse that dispersion and the Kerr effect leave alone between them.

    At the soliton power the chirp each effect imposes cancels the other exactly.
    Change the power and the balance breaks; switch off either effect and the
    pulse spreads or distorts.
    """
    ctx = SimulationContext(bit_rate=10e9, samples_per_symbol=256, sequence_length=4)
    t0_ps, dispersion, gamma = 10.0, 17.0, 1.3
    beta2 = dispersion_to_beta2(dispersion * 1e-6, 1550e-9)
    p0 = soliton_peak_power(beta2, gamma * 1e-3, t0_ps * 1e-12)
    distance_km = 4 * soliton_period(beta2, t0_ps * 1e-12) / 1e3

    print(f"\nSech pulse, T0 = {t0_ps:.0f} ps, over {distance_km:.1f} km (4 soliton periods)")
    print(f"  soliton power P0 = {p0 * 1e3:.1f} mW ({w_to_dbm(p0):.1f} dBm)")
    print("  configuration                width      peak power    steps")
    print("  " + "-" * 60)

    cases = (
        ("soliton (N = 1)", p0, dispersion, gamma),
        ("half the power", p0 / 2, dispersion, gamma),
        ("no nonlinearity", p0, dispersion, 0.0),
        ("no dispersion", p0, 0.0, gamma),
    )
    for name, power, d, g_nl in cases:
        g = Graph(ctx)
        source = g.add(SechPulse(peak_power=w_to_dbm(power), width=t0_ps, label="src"))
        fiber = g.add(
            Fiber(
                length=distance_km,
                attenuation=0.0,
                dispersion=d,
                nonlinearity=g_nl,
                label="fiber",
            )
        )
        meter = g.add(OSNRMeter(label="osnr"))
        g.chain(source, fiber, meter)

        results = g.run(keep=[source, fiber])
        launched = results.port(source, "out").bands[0]
        received = results.port(fiber, "out").bands[0]
        diagnostics = results.port(fiber, "diagnostics")

        width_ratio = rms_time_width(received) / rms_time_width(launched)
        peak_ratio = peak_power(received) / peak_power(launched)
        print(
            f"  {name:<26} x{width_ratio:5.2f}      x{peak_ratio:5.2f}     {diagnostics.steps:>6}"
        )


if __name__ == "__main__":
    osnr_versus_span_count()
    launch_power_optimum()
    soliton_demonstration()
