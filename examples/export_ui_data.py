"""Export real engine output for the GUI mockup.

Everything the interface shows should come from a real run. This writes the
component manifests, a real constellation histogram, a real eye, and a real
per-format sweep to JSON so the mockup can be built against true data — which
also proves the data the session server will eventually serve is the data the
engine already produces.

The link is the coherent one, because it is the one that exercises the whole
port-type system: binary into symbols, symbols into two electrical drives, an
optical field, two photocurrents back, and symbols out again.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from oosim import Graph, SimulationContext, manifests, sweep
from oosim.analysis import eye_histogram
from oosim.components import (
    CarrierRecovery,
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

V_PI = 4.0
SYMBOL_RATE = 32e9
BITS_PER_SYMBOL = 4
FORMATS = {1: "BPSK", 2: "QPSK", 4: "16-QAM", 6: "64-QAM", 8: "256-QAM"}


def build(sequence_length: int = 4096) -> Graph:
    ctx = SimulationContext(
        bit_rate=SYMBOL_RATE,
        samples_per_symbol=16,
        sequence_length=sequence_length,
        seed=2026,
        precision="double",
    )
    graph = Graph(ctx)
    prbs = graph.add(
        PRBSGenerator(order=23.0, bits_per_symbol=float(BITS_PER_SYMBOL), label="prbs")
    )
    mapper = graph.add(QAMMapper(bits_per_symbol=float(BITS_PER_SYMBOL), label="map"))
    driver = graph.add(IQDriver(v_pi=V_PI, predistort=True, label="drv"))
    # 100 kHz, which is an ordinary coherent-transmitter laser rather than a
    # specially quiet one. An earlier version of this export had to run at 10 kHz
    # because there was no carrier recovery in the chain and the accumulated
    # phase walk put a hard floor near 18 dB SNR however much power was
    # launched. The CarrierRecovery stage below is what makes the realistic
    # number usable again.
    laser = graph.add(CWLaser(power=-8.0, wavelength=1550.0, linewidth=100.0, label="tx"))
    modulator = graph.add(IQModulator(v_pi=V_PI, label="mod"))
    meter = graph.add(PowerMeter(label="pm"))
    lo = graph.add(CWLaser(power=10.0, wavelength=1550.0, linewidth=100.0, label="lo"))
    receiver = graph.add(CoherentReceiver(responsivity=0.8, label="rx"))
    sampler = graph.add(IQSampler(label="smp"))
    recovery = graph.add(CarrierRecovery(window=64.0, test_phases=32.0, label="cr"))
    analyzer = graph.add(ConstellationAnalyzer(ignore_edges=64.0, label="vsa"))
    diagram = graph.add(ConstellationDiagram(bins=96.0, extent=1.5, label="cd"))

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
    graph.connect(sampler["out"], recovery["in"])
    graph.connect(recovery["out"], analyzer["in"])
    graph.connect(mapper["out"], analyzer["reference"])
    graph.connect(recovery["out"], diagram["in"])
    return graph


def of_type(graph: Graph, kind: type) -> Any:
    return next(c for c in graph.components if isinstance(c, kind))


def main() -> None:
    graph = build()
    analyzer = of_type(graph, ConstellationAnalyzer)
    diagram = of_type(graph, ConstellationDiagram)
    meter = of_type(graph, PowerMeter)
    receiver = of_type(graph, CoherentReceiver)
    laser = next(c for c in graph.components if c.label == "tx")

    results = graph.run(keep=[receiver])
    measurement = results[analyzer]
    histogram = results[diagram]

    # The I-quadrature eye of a coherent receiver: a real thing to look at, and
    # for 16-QAM it shows the four levels the format actually carries.
    current = results.port(receiver, "i")
    eye = eye_histogram(
        np.asarray(current.samples),
        graph.ctx.samples_per_symbol,
        graph.ctx.bit_rate,
        span_symbols=2,
        time_bins=64,
        amplitude_bins=72,
        unit=current.unit,
    )

    # Required received power per format, from the same graph re-run.
    sensitivity: list[dict[str, Any]] = []
    for bits_per_symbol, name in FORMATS.items():
        prbs = of_type(graph, PRBSGenerator)
        mapper = of_type(graph, QAMMapper)
        points = [float(p) for p in range(-44, -6, 3)]
        curve = sweep(
            graph,
            {
                (laser, "power"): points,
                (prbs, "bits_per_symbol"): [float(bits_per_symbol)],
                (mapper, "bits_per_symbol"): [float(bits_per_symbol)],
            },
        )
        sensitivity.append(
            {
                "name": name,
                "bits_per_symbol": bits_per_symbol,
                "gbps": SYMBOL_RATE * bits_per_symbol / 1e9,
                "points": [
                    {
                        "received_dbm": point.runs[0][meter].power_dbm,
                        "snr_db": point.runs[0][analyzer].snr_db,
                        "ber": point.runs[0][analyzer].ber_estimated,
                    }
                    for point in curve
                ],
            }
        )

    payload: dict[str, Any] = {
        "manifests": manifests(),
        "measurement": {
            "evm": measurement.evm,
            "snr_db": measurement.snr_db,
            "mer_db": measurement.mer_db,
            "ber": measurement.ber_estimated,
            "symbol_errors": measurement.symbol_errors,
            "symbols": measurement.symbols_evaluated,
            "bit_errors": measurement.bit_errors,
            "bits": measurement.bits_evaluated,
            "frequency_offset_mhz": measurement.frequency_offset / 1e6,
            "bits_per_symbol": measurement.bits_per_symbol,
            "received_dbm": results[meter].power_dbm,
        },
        "constellation": {
            "counts": np.asarray(histogram.counts).astype(int).tolist(),
            "inphase_edges": np.asarray(histogram.inphase_edges).round(5).tolist(),
            "quadrature_edges": np.asarray(histogram.quadrature_edges).round(5).tolist(),
            "reference": [[float(p.real), float(p.imag)] for p in np.asarray(histogram.reference)],
        },
        "eye": {
            "counts": np.asarray(eye.counts).astype(int).tolist(),
            "time_ps": (np.asarray(eye.time_edges) * 1e12).round(3).tolist(),
            "amplitude_ua": (np.asarray(eye.amplitude_edges) * 1e6).round(4).tolist(),
            "unit": eye.unit,
        },
        "sensitivity": sensitivity,
        "context": {
            "symbol_rate": graph.ctx.bit_rate,
            "samples_per_symbol": graph.ctx.samples_per_symbol,
            "sequence_length": graph.ctx.sequence_length,
            "num_samples": graph.ctx.num_samples,
            "seed": graph.ctx.seed,
            "format": FORMATS[BITS_PER_SYMBOL],
            "gbps": SYMBOL_RATE * BITS_PER_SYMBOL / 1e9,
        },
    }

    destination = Path(__file__).parent / "ui_data.json"
    destination.write_text(json.dumps(payload, indent=1), encoding="utf-8")

    print(
        f"{FORMATS[BITS_PER_SYMBOL]} at {SYMBOL_RATE * BITS_PER_SYMBOL / 1e9:.0f} Gb/s: "
        f"EVM = {measurement.evm * 100:.2f}%, SNR = {measurement.snr_db:.2f} dB, "
        f"BER = {measurement.ber_estimated:.3e}, "
        f"{measurement.symbol_errors} symbol errors in {measurement.symbols_evaluated}"
    )
    print(f"{len(payload['manifests'])} component manifests")
    counts = payload["constellation"]["counts"]
    print(f"constellation histogram {len(counts)}x{len(counts[0])}")
    print(f"wrote {destination.name} ({destination.stat().st_size / 1024:.0f} kB)")


if __name__ == "__main__":
    main()
