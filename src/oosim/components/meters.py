"""Measurement components."""

from __future__ import annotations

from ..component import Component, PortType
from ..context import SimulationContext
from ..signals import BandPower, OpticalSignal, PowerReading, Signal
from ..units import frequency_to_wavelength


class PowerMeter(Component):
    """Ideal optical power meter.

    Reports total power and a per-band breakdown. The breakdown matters as soon
    as more than one carrier is present: a single total is exactly what makes a
    WDM result impossible to interpret.
    """

    display_name = "Optical Power Meter"
    category = "Measurements"

    inputs = {"in": PortType.OPTICAL}
    outputs = {"out": PortType.METRIC}

    def run(self, ctx: SimulationContext, inputs: dict[str, Signal]) -> dict[str, Signal]:
        signal: OpticalSignal = inputs["in"]
        bands = tuple(
            BandPower(
                f0=band.f0,
                wavelength_nm=frequency_to_wavelength(band.f0) * 1e9,
                power_w=band.average_power(),
            )
            for band in sorted(signal.bands, key=lambda b: b.f0)
        )
        return {
            "out": PowerReading(
                signal_power_w=signal.signal_power(),
                noise_power_w=signal.noise_power(),
                bands=bands,
            )
        }
