"""Measurement blocks that reduce a waveform to numbers a person can act on."""

from __future__ import annotations

import numpy as np

from ..analysis import eye_histogram, measure_eye
from ..component import BoolParam, Component, Param, PortType
from ..context import SimulationContext
from ..signals import BinarySignal, ElectricalSignal, Signal


class BERAnalyzer(Component):
    """Decision circuit: samples the eye, decides bits, and counts the errors.

    Takes the transmitted sequence on a second input so errors can be *counted*
    rather than only inferred from Q. Having both is the point — the Gaussian
    estimate is what gets quoted at realistic error rates, and the count is what
    proves the estimate is trustworthy at rates high enough to measure.

    The decision threshold is placed for equal error probability from the
    measured rail statistics, and the sampling instant is chosen to maximise Q,
    which is what a receiver's clock recovery converges to. A fixed sampling
    instant can be forced to study mis-timed sampling.
    """

    display_name = "BER Analyzer"
    category = "Measurements"

    ignore_edges = Param(
        4.0,
        unit="",
        min=0.0,
        doc="Symbols to discard at each end of the window (circular-filter wrap)",
    )
    adaptive_timing = BoolParam(True, doc="Choose the sampling instant that maximises Q")
    sample_offset = Param(
        0.0, unit="", min=0.0, doc="Sampling instant within the symbol when timing is fixed"
    )

    inputs = {"in": PortType.ELECTRICAL, "reference": PortType.BINARY}
    outputs = {"out": PortType.METRIC}

    def run(self, ctx: SimulationContext, inputs: dict[str, Signal]) -> dict[str, Signal]:
        waveform: ElectricalSignal = inputs["in"]
        reference: BinarySignal = inputs["reference"]

        offset = None if self.adaptive_timing else int(self.sample_offset)
        measurement = measure_eye(
            np.asarray(waveform.samples),
            np.asarray(reference.bits),
            ctx.samples_per_symbol,
            sample_offset=offset,
            ignore_edges=int(self.ignore_edges),
        )
        return {"out": measurement}


class EyeDiagram(Component):
    """Bins a received waveform into an eye diagram.

    Emits a fixed-size histogram rather than the waveform: the size depends on
    the requested resolution and not on how long the simulation ran. That is what
    keeps a multi-million-sample buffer from ever reaching a browser.
    """

    display_name = "Eye Diagram"
    category = "Measurements"

    span_symbols = Param(2.0, unit="", min=1.0, doc="Symbols across the horizontal axis")
    time_bins = Param(128.0, unit="", min=8.0, doc="Horizontal resolution")
    amplitude_bins = Param(128.0, unit="", min=8.0, doc="Vertical resolution")

    inputs = {"in": PortType.ELECTRICAL}
    outputs = {"out": PortType.METRIC}

    def run(self, ctx: SimulationContext, inputs: dict[str, Signal]) -> dict[str, Signal]:
        waveform: ElectricalSignal = inputs["in"]
        histogram = eye_histogram(
            np.asarray(waveform.samples),
            ctx.samples_per_symbol,
            ctx.bit_rate,
            span_symbols=int(self.span_symbols),
            time_bins=int(self.time_bins),
            amplitude_bins=int(self.amplitude_bins),
            unit=waveform.unit,
        )
        return {"out": histogram}
