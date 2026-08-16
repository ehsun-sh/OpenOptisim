"""Receiver DSP blocks that operate on symbols."""

from __future__ import annotations

import numpy as np

from ..component import Component, Param, PortType
from ..context import SimulationContext
from ..dsp import butterfly_equalize
from ..signals import Signal, SymbolSignal


class ButterflyEqualizer(Component):
    """Separates two polarization tributaries that the channel has mixed.

    A dual-polarization receiver measures the field on its own axes, and a fibre
    rotates the launched state arbitrarily before it gets there. What arrives is
    two *mixtures*, not two channels. This 2x2 adaptive filter is what turns them
    back into channels, and without it a dual-polarization link recovers nothing
    at all past a small rotation — not a degraded version of the data, nothing.

    It is blind: no training sequence, no reference. The filters are adapted to
    drive each output onto a modulus the constellation actually uses, which a
    clean tributary satisfies and a mixture of two independent ones does not.
    See :func:`oosim.dsp.butterfly_equalize` for the two-stage scheme and for the
    45-degree saddle that the initialisation is tilted to avoid.

    **Which output is which is not determined.** Nothing in a blind cost function
    labels the tributaries, so the filter may deliver them swapped, and each
    carries its own arbitrary phase from the same quadrant ambiguity that
    :class:`~oosim.components.coherent.CarrierRecovery` has. A deployed link
    resolves both by framing and differential encoding; here the measurement
    block resolves them because it holds the reference.
    """

    display_name = "Butterfly Equalizer"
    category = "DSP"

    taps = Param(7.0, unit="", min=1.0, max=65.0, doc="Filter length; must be odd")
    step = Param(3e-3, unit="", min=1e-6, doc="Adaptation step size")
    passes = Param(2.0, unit="", min=1.0, max=8.0, doc="Times the sequence is run through")

    inputs = {"x": PortType.SYMBOL, "y": PortType.SYMBOL}
    outputs = {"x_out": PortType.SYMBOL, "y_out": PortType.SYMBOL}

    def run(self, ctx: SimulationContext, inputs: dict[str, Signal]) -> dict[str, Signal]:
        tributary_x: SymbolSignal = inputs["x"]
        tributary_y: SymbolSignal = inputs["y"]

        if tributary_x.num_symbols != tributary_y.num_symbols:
            raise ValueError(
                f"{self.label}: tributaries differ in length, "
                f"{tributary_x.num_symbols} and {tributary_y.num_symbols}"
            )
        if tributary_x.order != tributary_y.order:
            raise ValueError(
                f"{self.label}: tributaries carry different constellations, "
                f"{tributary_x.order} and {tributary_y.order} points"
            )

        taps = int(self.taps)
        if taps % 2 == 0:
            raise ValueError(f"{self.label}: taps must be odd, got {taps}")

        constellation = np.asarray(tributary_x.constellation)
        out_x, out_y, _ = butterfly_equalize(
            np.asarray(tributary_x.symbols),
            np.asarray(tributary_y.symbols),
            constellation,
            taps=taps,
            step=self.step,
            passes=int(self.passes),
        )
        return {
            "x_out": SymbolSignal(
                symbols=out_x, symbol_rate=tributary_x.symbol_rate, constellation=constellation
            ),
            "y_out": SymbolSignal(
                symbols=out_y, symbol_rate=tributary_y.symbol_rate, constellation=constellation
            ),
        }
