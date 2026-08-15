"""Optical sources."""

from __future__ import annotations

import numpy as np

from ..component import Component, Param, PortType
from ..context import SimulationContext
from ..signals import Band, OpticalSignal, Signal
from ..units import C_LIGHT


class CWLaser(Component):
    """Continuous-wave laser.

    Emits a constant-envelope field in the X polarization. With a non-zero
    linewidth the phase performs a Wiener random walk, which is the standard
    Lorentzian-lineshape model: the phase increment per sample is drawn from
    ``N(0, 2*pi*linewidth*dt)``. Amplitude is untouched, so linewidth changes the
    spectrum without changing the average power — an invariant worth testing.
    """

    display_name = "CW Laser"
    category = "Optical Sources"

    power = Param(0.0, unit="dBm", doc="Average output power")
    wavelength = Param(1550.0, unit="nm", min=1200.0, max=1700.0, doc="Vacuum wavelength")
    linewidth = Param(
        0.0, unit="kHz", min=0.0, doc="Lorentzian FWHM linewidth; 0 disables phase noise"
    )

    outputs = {"out": PortType.OPTICAL}

    def run(self, ctx: SimulationContext, inputs: dict[str, Signal]) -> dict[str, Signal]:
        power_w = self.si("power")
        f0 = C_LIGHT / self.si("wavelength")
        amplitude = np.sqrt(power_w)

        n = ctx.num_samples
        if self.si("linewidth") > 0.0:
            sigma = np.sqrt(2.0 * np.pi * self.si("linewidth") * ctx.time_step)
            rng = ctx.rng("CWLaser", self.label, "phase_noise")
            increments = rng.normal(0.0, sigma, size=n)
            phase = np.cumsum(increments)
            phase -= phase[0]  # start at zero phase so runs are comparable
            Ex = amplitude * np.exp(1j * phase)
        else:
            Ex = np.full(n, amplitude, dtype=np.complex128)

        band = Band(
            Ex=Ex.astype(ctx.complex_dtype),
            Ey=np.zeros(n, dtype=ctx.complex_dtype),
            f0=f0,
            fs=ctx.sample_rate,
        )
        return {"out": OpticalSignal(bands=(band,))}
