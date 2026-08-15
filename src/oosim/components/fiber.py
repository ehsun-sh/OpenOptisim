"""Optical fiber.

Currently the linear, lossy model only. Chromatic dispersion is the next slice;
Kerr nonlinearity and PMD follow in Phase 1.5 with an adaptive-step SSFM.
"""

from __future__ import annotations

from ..component import Component, Param, PortType
from ..context import SimulationContext
from ..signals import OpticalSignal, Signal
from ..units import db_to_linear


class Fiber(Component):
    """Single-mode fiber, attenuation only.

    Power obeys ``P_out = P_in * 10**(-alpha_dB_per_km * L_km / 10)``, so the
    field amplitude is scaled by the square root of that factor. Attenuation is
    wavelength-independent in this model: every band and every noise bin is
    scaled identically. Making loss wavelength-dependent is a later refinement
    and does not change the interface.
    """

    display_name = "Optical Fiber"
    category = "Fiber"

    length = Param(80.0, unit="km", min=0.0, doc="Fiber span length")
    attenuation = Param(0.2, unit="dB/km", min=0.0, doc="Attenuation coefficient")

    inputs = {"in": PortType.OPTICAL}
    outputs = {"out": PortType.OPTICAL}

    def loss_db(self) -> float:
        """Total span loss [dB]."""
        # si() gives dB/m and metres, so their product is dB.
        return self.si("attenuation") * self.si("length")

    def run(self, ctx: SimulationContext, inputs: dict[str, Signal]) -> dict[str, Signal]:
        signal: OpticalSignal = inputs["in"]
        power_factor = db_to_linear(-self.loss_db())
        amplitude_factor = power_factor**0.5

        return {
            "out": OpticalSignal(
                bands=tuple(b.scale_amplitude(amplitude_factor) for b in signal.bands),
                noise=tuple(n.scale_power(power_factor) for n in signal.noise),
            )
        }
