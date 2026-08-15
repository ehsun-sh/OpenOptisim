"""Optical fiber.

Linear propagation (attenuation and chromatic dispersion) and the Kerr
nonlinearity, solved by adaptive-step split-step Fourier.

Model references: G. P. Agrawal, *Nonlinear Fiber Optics*, ch. 2-4
(NLSE, GVD-induced broadening, SPM, solitons); ITU-T G.652 for typical values.
"""

from __future__ import annotations

from ..component import Component, Param, PortType
from ..context import SimulationContext
from ..kernels import (
    PropagationDiagnostics,
    attenuation_db_per_m_to_alpha,
    dispersion_to_beta2,
    propagate_dispersion,
    propagate_ssfm,
)
from ..signals import Band, OpticalSignal, Signal
from ..units import db_to_linear


class Fiber(Component):
    """Single-mode fiber: attenuation, chromatic dispersion, and Kerr nonlinearity.

    With ``nonlinearity`` left at zero the propagation is linear and is solved
    exactly in one frequency-domain step — no stepping error at all. Give it a
    nonzero value and the same span is solved by split-step Fourier instead,
    which is approximate, so the block reports what it did on its
    ``diagnostics`` port.

    Dispersion is applied per band, using each band's *own* centre wavelength to
    compute beta2. Two channels a few nanometres apart really do see different
    dispersion, and because every band carries its own centre frequency the model
    gets that right for free — a single-carrier signal model could not express it.

    Not yet modelled: the dispersion slope (beta3), polarization-mode dispersion,
    cross-phase modulation and four-wave mixing between bands, and Raman
    scattering. Bands propagate independently, so this is a good model of a
    single channel and an optimistic one of a dense WDM comb.
    """

    display_name = "Optical Fiber"
    category = "Fiber"

    length = Param(80.0, unit="km", min=0.0, doc="Fiber span length")
    attenuation = Param(0.2, unit="dB/km", min=0.0, doc="Attenuation coefficient")
    dispersion = Param(
        0.0, unit="ps/nm/km", doc="Dispersion parameter D at the band wavelength (0 disables)"
    )
    nonlinearity = Param(
        0.0, unit="1/W/km", min=0.0, doc="Kerr coefficient gamma (0 disables, and is exact)"
    )
    max_nonlinear_phase = Param(
        0.005,
        unit="",
        min=1e-6,
        doc="Largest nonlinear phase rotation allowed per split-step [rad]",
    )

    inputs = {"in": PortType.OPTICAL}
    outputs = {"out": PortType.OPTICAL, "diagnostics": PortType.METRIC}

    def loss_db(self) -> float:
        """Total span loss [dB]."""
        # si() gives dB/m and metres, so their product is dB.
        return self.si("attenuation") * self.si("length")

    def beta2_at(self, wavelength: float) -> float:
        """Group-velocity dispersion beta2 [s^2/m] at ``wavelength`` [m]."""
        return dispersion_to_beta2(self.si("dispersion"), wavelength)

    def run(self, ctx: SimulationContext, inputs: dict[str, Signal]) -> dict[str, Signal]:
        signal: OpticalSignal = inputs["in"]
        distance = self.si("length")
        gamma = self.si("nonlinearity")
        power_factor = db_to_linear(-self.loss_db())

        bands = []
        diagnostics = PropagationDiagnostics(0, distance, 0.0, 0.0, 0.0)
        for band in signal.bands:
            beta2 = self.beta2_at(band.wavelength)
            if gamma == 0.0:
                # Linear propagation has a closed-form solution; use it rather
                # than approximating something that need not be approximated.
                amplitude = power_factor**0.5
                ex = propagate_dispersion(band.Ex, band.fs, beta2, distance) * amplitude
                ey = propagate_dispersion(band.Ey, band.fs, beta2, distance) * amplitude
            else:
                alpha = attenuation_db_per_m_to_alpha(self.si("attenuation"))
                ex, diag_x = propagate_ssfm(
                    band.Ex,
                    band.fs,
                    beta2=beta2,
                    gamma=gamma,
                    alpha=alpha,
                    distance=distance,
                    max_nonlinear_phase=self.max_nonlinear_phase,
                )
                ey, _ = propagate_ssfm(
                    band.Ey,
                    band.fs,
                    beta2=beta2,
                    gamma=gamma,
                    alpha=alpha,
                    distance=distance,
                    max_nonlinear_phase=self.max_nonlinear_phase,
                )
                if diag_x.steps > diagnostics.steps:
                    diagnostics = diag_x

            bands.append(
                Band(
                    Ex=ex.astype(ctx.complex_dtype),
                    Ey=ey.astype(ctx.complex_dtype),
                    f0=band.f0,
                    fs=band.fs,
                )
            )

        return {
            "out": OpticalSignal(
                bands=tuple(bands),
                noise=tuple(n.scale_power(power_factor) for n in signal.noise),
            ),
            "diagnostics": diagnostics,
        }
