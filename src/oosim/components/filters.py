"""Electrical filtering."""

from __future__ import annotations

from ..component import Component, Param, PortType
from ..context import SimulationContext
from ..kernels import gaussian_noise_bandwidth, lowpass_filter
from ..signals import ElectricalSignal, Signal


class ElectricalFilter(Component):
    """Gaussian low-pass filter for a received waveform.

    Every real receiver band-limits before deciding, and without it a simulation
    is not merely optimistic but meaningless: the noise a detector adds spans the
    whole simulated bandwidth ``fs/2``, which is set by the oversampling factor
    rather than by anything physical. Filtering to a receiver bandwidth is what
    makes a Q-factor or a BER correspond to a real link.

    Standard practice is a fourth-order Bessel at roughly 0.7 times the symbol
    rate. The Gaussian shape used here is close in effect and has a closed-form
    noise bandwidth, ``B_n = B * sqrt(pi / 4 ln2) ~ 1.0645 * B``, which makes the
    effect on noise checkable exactly rather than approximately.

    The filter is zero-phase, so there is no group delay to compensate; see
    :func:`oosim.kernels.lowpass_filter`.
    """

    display_name = "Electrical Filter"
    category = "Electrical"

    bandwidth = Param(7.0, unit="GHz", min=0.0, doc="3 dB power bandwidth")

    inputs = {"in": PortType.ELECTRICAL}
    outputs = {"out": PortType.ELECTRICAL}

    def noise_bandwidth(self) -> float:
        """Equivalent noise bandwidth [Hz]."""
        return gaussian_noise_bandwidth(self.si("bandwidth"))

    def run(self, ctx: SimulationContext, inputs: dict[str, Signal]) -> dict[str, Signal]:
        waveform: ElectricalSignal = inputs["in"]
        filtered = lowpass_filter(waveform.samples, waveform.fs, self.si("bandwidth"))
        return {
            "out": ElectricalSignal(
                samples=filtered.astype(ctx.real_dtype), fs=waveform.fs, unit=waveform.unit
            )
        }
