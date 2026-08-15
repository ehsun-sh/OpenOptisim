"""Optical fiber.

Currently the linear model: attenuation and chromatic dispersion. Kerr
nonlinearity and PMD follow in Phase 1.5, with an adaptive-step SSFM built on
the same frequency-domain kernel this module already uses.

Model references: G. P. Agrawal, *Nonlinear Fiber Optics*, ch. 2-3
(NLSE, GVD-induced pulse broadening); ITU-T G.652 for typical parameter values.
"""

from __future__ import annotations

from ..component import Component, Param, PortType
from ..context import SimulationContext
from ..kernels import dispersion_to_beta2, propagate_dispersion
from ..signals import Band, OpticalSignal, Signal
from ..units import db_to_linear


class Fiber(Component):
    """Single-mode fiber: attenuation and chromatic dispersion.

    Attenuation follows ``P_out = P_in * 10**(-alpha_dB_per_km * L_km / 10)``, so
    the field amplitude is scaled by the square root of that factor.

    Dispersion is applied per band, using each band's *own* centre wavelength to
    compute β₂. Two channels a few nanometres apart really do see different
    dispersion, and because every band carries its own centre frequency the model
    gets that right for free — a single-carrier signal model could not express it.

    Attenuation is wavelength-independent here, and the dispersion slope is not
    modelled (β₃ = 0); both are refinements that do not change the interface.
    """

    display_name = "Optical Fiber"
    category = "Fiber"

    length = Param(80.0, unit="km", min=0.0, doc="Fiber span length")
    attenuation = Param(0.2, unit="dB/km", min=0.0, doc="Attenuation coefficient")
    dispersion = Param(
        0.0, unit="ps/nm/km", doc="Dispersion parameter D at the band wavelength (0 disables)"
    )

    inputs = {"in": PortType.OPTICAL}
    outputs = {"out": PortType.OPTICAL}

    def loss_db(self) -> float:
        """Total span loss [dB]."""
        # si() gives dB/m and metres, so their product is dB.
        return self.si("attenuation") * self.si("length")

    def beta2_at(self, wavelength: float) -> float:
        """Group-velocity dispersion β₂ [s²/m] at ``wavelength`` [m]."""
        return dispersion_to_beta2(self.si("dispersion"), wavelength)

    def run(self, ctx: SimulationContext, inputs: dict[str, Signal]) -> dict[str, Signal]:
        signal: OpticalSignal = inputs["in"]
        distance = self.si("length")
        power_factor = db_to_linear(-self.loss_db())
        amplitude_factor = power_factor**0.5

        bands = []
        for band in signal.bands:
            beta2 = self.beta2_at(band.wavelength)
            Ex = propagate_dispersion(band.Ex, band.fs, beta2, distance) * amplitude_factor
            Ey = propagate_dispersion(band.Ey, band.fs, beta2, distance) * amplitude_factor
            bands.append(
                Band(
                    Ex=Ex.astype(ctx.complex_dtype),
                    Ey=Ey.astype(ctx.complex_dtype),
                    f0=band.f0,
                    fs=band.fs,
                )
            )

        return {
            "out": OpticalSignal(
                bands=tuple(bands),
                noise=tuple(n.scale_power(power_factor) for n in signal.noise),
            )
        }
