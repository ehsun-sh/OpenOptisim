"""A single-polarization coherent link, characterised across modulation formats.

Builds PRBS -> Gray-coded M-QAM -> IQ driver -> IQ modulator -> 90-degree hybrid
with balanced detection, and sweeps the launch power for each format. The result
is the trade the tool exists to let someone explore: every extra bit per symbol
buys spectral efficiency and costs sensitivity, by an amount you can read off.

Run: ``python examples/coherent_link.py``
"""

from __future__ import annotations

from oosim import Graph, SimulationContext, sweep
from oosim.components import (
    CoherentReceiver,
    ConstellationAnalyzer,
    CWLaser,
    IQDriver,
    IQModulator,
    IQSampler,
    PowerMeter,
    PRBSGenerator,
    QAMMapper,
)

V_PI = 4.0
SYMBOL_RATE = 32e9
FORMATS = {1: "BPSK", 2: "QPSK", 4: "16-QAM", 6: "64-QAM", 8: "256-QAM"}


def build(bits_per_symbol: int, sequence_length: int = 8192) -> Graph:
    ctx = SimulationContext(
        bit_rate=SYMBOL_RATE,
        samples_per_symbol=4,
        sequence_length=sequence_length,
        seed=2026,
        precision="double",
    )
    graph = Graph(ctx)
    prbs = graph.add(
        PRBSGenerator(order=23.0, bits_per_symbol=float(bits_per_symbol), label="prbs")
    )
    mapper = graph.add(QAMMapper(bits_per_symbol=float(bits_per_symbol), label="map"))
    driver = graph.add(IQDriver(v_pi=V_PI, predistort=True, label="drv"))
    laser = graph.add(CWLaser(power=-30.0, wavelength=1550.0, label="tx"))
    modulator = graph.add(IQModulator(v_pi=V_PI, label="mod"))
    meter = graph.add(PowerMeter(label="pm"))
    lo = graph.add(CWLaser(power=10.0, wavelength=1550.0, label="lo"))
    receiver = graph.add(CoherentReceiver(responsivity=0.8, label="rx"))
    sampler = graph.add(IQSampler(label="smp"))
    analyzer = graph.add(ConstellationAnalyzer(label="vsa"))

    graph.chain(prbs, mapper, driver)
    graph.connect(laser, modulator["optical_in"])
    graph.connect(driver["i"], modulator["i"])
    graph.connect(driver["q"], modulator["q"])
    graph.connect(modulator, meter["in"])
    graph.connect(modulator, receiver["in"])
    graph.connect(lo, receiver["lo"])
    graph.connect(receiver["i"], sampler["i"])
    graph.connect(receiver["q"], sampler["q"])
    graph.connect(mapper["out"], sampler["reference"])
    graph.connect(sampler["out"], analyzer["in"])
    graph.connect(mapper["out"], analyzer["reference"])
    return graph


def main() -> None:
    print(f"Single-polarization coherent link, {SYMBOL_RATE / 1e9:.0f} GBd, back to back\n")
    print("format     rate      launch for BER 1e-3     SNR there   EVM there")
    print("-" * 68)

    for bits_per_symbol, name in FORMATS.items():
        graph = build(bits_per_symbol)
        laser = next(c for c in graph.components if c.label == "tx")
        analyzer = next(c for c in graph.components if isinstance(c, ConstellationAnalyzer))
        meter = next(c for c in graph.components if isinstance(c, PowerMeter))

        powers = [float(p) for p in range(-46, -8)]
        curve = sweep(graph, {(laser, "power"): powers})

        threshold = None
        for point in curve:
            result = point.runs[0][analyzer]
            if result.ber_estimated <= 1e-3:
                threshold = (point.runs[0][meter].power_dbm, result)
                break

        rate = SYMBOL_RATE * bits_per_symbol / 1e9
        if threshold is None:
            print(f"{name:<10} {rate:5.0f} Gb/s   not reached in the swept range")
            continue
        received_dbm, result = threshold
        print(
            f"{name:<10} {rate:5.0f} Gb/s   {received_dbm:8.1f} dBm received"
            f"      {result.snr_db:5.1f} dB    {result.evm * 100:5.1f}%"
        )

    print(
        "\nLaunch is quoted at the modulator output, which already includes the 3 dB\n"
        "a dual-parallel IQ structure costs by construction."
    )


if __name__ == "__main__":
    main()
