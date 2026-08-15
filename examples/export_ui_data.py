"""Export real engine output for the GUI mockup.

Everything the interface shows should come from a real run. This writes the
component manifests, a real eye histogram, and a real sensitivity sweep to JSON
so the mockup can be built against true data — which also proves the data the
session server will eventually serve is the data the engine already produces.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from oosim import Graph, SimulationContext, manifests, sweep
from oosim.analysis import eye_histogram
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

V_PI = 4.0


def build(sequence_length: int = 2048) -> Graph:
    ctx = SimulationContext(
        bit_rate=10e9, samples_per_symbol=64, sequence_length=sequence_length, seed=2026
    )
    g = Graph(ctx)
    prbs = g.add(PRBSGenerator(order=15.0, label="prbs"))
    driver = g.add(NRZDriver(v_low=V_PI, v_high=0.0, label="driver"))
    laser = g.add(CWLaser(power=-8.0, wavelength=1550.0, linewidth=100.0, label="laser"))
    mzm = g.add(MachZehnderModulator(v_pi=V_PI, extinction_ratio=30.0, label="mzm"))
    fiber = g.add(Fiber(length=40.0, attenuation=0.2, dispersion=17.0, label="fiber"))
    pin = g.add(PINPhotodiode(responsivity=0.8, label="pin"))
    lpf = g.add(ElectricalFilter(bandwidth=7.0, label="lpf"))
    analyzer = g.add(BERAnalyzer(label="ber"))

    g.chain(prbs, driver)
    g.connect(laser, mzm["optical_in"])
    g.connect(driver, mzm["electrical_in"])
    g.chain(mzm, fiber, pin, lpf)
    g.connect(lpf, analyzer["in"])
    g.connect(prbs["out"], analyzer["reference"])
    return g


def main() -> None:
    graph = build()
    analyzer = next(c for c in graph.components if isinstance(c, BERAnalyzer))
    lpf = next(c for c in graph.components if isinstance(c, ElectricalFilter))

    results = graph.run(keep=[lpf])
    measurement = results[analyzer]
    waveform = results.port(lpf, "out")

    histogram = eye_histogram(
        np.asarray(waveform.samples),
        graph.ctx.samples_per_symbol,
        graph.ctx.bit_rate,
        span_symbols=2,
        time_bins=128,
        amplitude_bins=72,
        unit=waveform.unit,
    )

    laser = next(c for c in graph.components if isinstance(c, CWLaser))
    curve = sweep(graph, {(laser, "power"): [float(p) for p in range(-24, -11)]})

    payload: dict[str, Any] = {
        "manifests": manifests(),
        "measurement": {
            "q_factor": measurement.q_factor,
            "q_db": measurement.q_db,
            "ber": measurement.ber_gaussian,
            "errors": measurement.errors,
            "bits": measurement.bits_evaluated,
            "mean_one": measurement.mean_one,
            "mean_zero": measurement.mean_zero,
            "std_one": measurement.std_one,
            "std_zero": measurement.std_zero,
            "threshold": measurement.threshold,
            "sample_offset": measurement.sample_offset,
        },
        "eye": {
            "counts": np.asarray(histogram.counts).astype(int).tolist(),
            "time_ps": (np.asarray(histogram.time_edges) * 1e12).round(3).tolist(),
            "amplitude_ua": (np.asarray(histogram.amplitude_edges) * 1e6).round(4).tolist(),
            "unit": histogram.unit,
        },
        "sensitivity": [
            {
                "launch_dbm": point.values["laser.power"],
                "q": point.runs[0][analyzer].q_factor,
                "ber": point.runs[0][analyzer].ber_gaussian,
            }
            for point in curve
        ],
        "context": {
            "bit_rate": graph.ctx.bit_rate,
            "samples_per_symbol": graph.ctx.samples_per_symbol,
            "sequence_length": graph.ctx.sequence_length,
            "num_samples": graph.ctx.num_samples,
            "seed": graph.ctx.seed,
        },
    }

    destination = Path(__file__).parent / "ui_data.json"
    destination.write_text(json.dumps(payload, indent=1), encoding="utf-8")

    print(f"Q = {measurement.q_factor:.3f}, BER = {measurement.ber_gaussian:.3e}")
    print(f"{len(payload['manifests'])} component manifests")
    print(f"eye histogram {len(payload['eye']['counts'])}x{len(payload['eye']['counts'][0])}")
    print(f"wrote {destination.name} ({destination.stat().st_size / 1024:.0f} kB)")


if __name__ == "__main__":
    main()
