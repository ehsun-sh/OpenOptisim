"""Optical fiber.

Attenuation, chromatic dispersion, the Kerr nonlinearity, and polarization-mode
dispersion.

Model references: G. P. Agrawal, *Nonlinear Fiber Optics*, ch. 2-4 (NLSE,
GVD-induced broadening, SPM, solitons); ITU-T G.652 for typical values.
"""

from __future__ import annotations

import math
from dataclasses import replace

from ..component import Component, Param, PortType
from ..context import SimulationContext
from ..kernels import (
    PMDSection,
    PropagationDiagnostics,
    apply_pmd,
    attenuation_db_per_m_to_alpha,
    differential_group_delay,
    dispersion_to_beta2,
    propagate_dispersion,
    propagate_ssfm,
    random_pmd_sections,
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

    PMD is drawn as a random realisation, not applied as a fixed impairment,
    because that is what it is: birefringence varies along real fiber and drifts
    with temperature, so the differential group delay is a random variable and a
    link is designed against an outage probability rather than a worst case. The
    realisation is seeded from the run context, so a given run is reproducible
    while a sweep with repeats explores the distribution.

    Not yet modelled: the dispersion slope (beta3), cross-phase modulation and
    four-wave mixing between bands, and Raman scattering. Bands propagate
    independently, so this is a good model of a single channel and an optimistic
    one of a dense WDM comb.
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
    pmd_coefficient = Param(0.0, unit="ps/sqrt(km)", min=0.0, doc="PMD coefficient (0 disables)")
    pmd_sections = Param(60.0, unit="", min=1.0, doc="Waveplates used to build the PMD realisation")

    inputs = {"in": PortType.OPTICAL}
    outputs = {"out": PortType.OPTICAL, "diagnostics": PortType.METRIC}

    def loss_db(self) -> float:
        """Total span loss [dB]."""
        # si() gives dB/m and metres, so their product is dB.
        return self.si("attenuation") * self.si("length")

    def beta2_at(self, wavelength: float) -> float:
        """Group-velocity dispersion beta2 [s^2/m] at ``wavelength`` [m]."""
        return dispersion_to_beta2(self.si("dispersion"), wavelength)

    def mean_dgd(self) -> float:
        """Expected differential group delay over this span [s].

        ``<DGD> = PMD_coefficient * sqrt(L)`` — the square root, not the length,
        because birefringence axes reorient randomly and the delay accumulates
        as a random walk rather than a sum.
        """
        return self.si("pmd_coefficient") * math.sqrt(self.si("length"))

    def run(self, ctx: SimulationContext, inputs: dict[str, Signal]) -> dict[str, Signal]:
        signal: OpticalSignal = inputs["in"]
        distance = self.si("length")
        gamma = self.si("nonlinearity")
        power_factor = db_to_linear(-self.loss_db())

        mean_dgd = self.mean_dgd()
        sections: tuple[PMDSection, ...] = ()
        realised_dgd = 0.0
        if mean_dgd > 0.0:
            sections = random_pmd_sections(
                mean_dgd,
                int(self.pmd_sections),
                ctx.rng("Fiber", self.label, "pmd"),
            )
            realised_dgd = differential_group_delay(sections)

        bands = []
        diagnostics = PropagationDiagnostics(0, distance, 0.0, 0.0, 0.0, realised_dgd)
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
                    diagnostics = replace(diag_x, differential_group_delay=realised_dgd)

            if sections:
                # PMD is applied after dispersion and the Kerr effect rather
                # than interleaved with them. That neglects the interaction
                # between nonlinearity and a rotating polarization state, which
                # matters at high power over long spans and does not at the
                # powers and distances this is usually pointed at.
                ex, ey = apply_pmd(ex, ey, band.fs, sections)

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
