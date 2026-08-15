"""Measurement components."""

from __future__ import annotations

from ..analysis import OSNR_REFERENCE_BANDWIDTH, osnr
from ..component import Component, Param, PortType
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


class OSNRMeter(Component):
    """Measures optical signal-to-noise ratio in a reference bandwidth.

    OSNR is what actually predicts whether an amplified link will work, and it
    is only meaningful because noise is carried as a spectral density rather
    than mixed into the samples: the ratio depends on the noise in a 0.1 nm
    slice, not on however much of it the simulation happened to sample.
    """

    display_name = "OSNR Meter"
    category = "Measurements"

    reference_bandwidth = Param(
        OSNR_REFERENCE_BANDWIDTH / 1e9,
        unit="GHz",
        min=0.0,
        doc="Reference bandwidth; 12.5 GHz is 0.1 nm at 1550 nm",
    )

    inputs = {"in": PortType.OPTICAL}
    outputs = {"out": PortType.METRIC}

    def run(self, ctx: SimulationContext, inputs: dict[str, Signal]) -> dict[str, Signal]:
        signal: OpticalSignal = inputs["in"]
        return {"out": osnr(signal, reference_bandwidth=self.si("reference_bandwidth"))}
