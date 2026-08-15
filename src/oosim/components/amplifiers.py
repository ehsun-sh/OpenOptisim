"""Optical amplifiers.

Model reference: G. P. Agrawal, *Fiber-Optic Communication Systems*, ch. 6
(optical amplifiers, ASE and noise figure); ITU-T G.661 for the definitions.
"""

from __future__ import annotations

import numpy as np

from ..component import Component, Param, PortType
from ..context import SimulationContext
from ..signals import Band, NoiseBin, OpticalSignal, Signal
from ..units import C_LIGHT, H_PLANCK, db_to_linear


class EDFA(Component):
    """Erbium-doped fiber amplifier: gain, plus the noise that comes with it.

    Amplification is not free. A phase-insensitive amplifier must add at least
    ``h*nu`` of noise per mode per unit bandwidth, and a real one adds more; the
    noise figure is how much more. The amplified spontaneous emission this block
    generates is what limits how many spans a link can have, so an amplifier
    modelled without it is not a simplification but a fiction.

    ASE is emitted into :class:`~oosim.signals.NoiseBin` rather than into the
    sampled bands. That is the whole reason the noise-bin representation exists:
    ASE covers the amplifier's full bandwidth — terahertz — while the signal
    occupies a few tens of gigahertz of it. Sampling both together would demand
    a sample rate no machine can afford, so the noise is carried as a power
    spectral density and only converted to samples where a detector or a
    nonlinearity actually needs it.

    The spontaneous emission factor follows from the noise figure::

        n_sp = NF * G / (2 * (G - 1))
        S_ASE = n_sp * h * nu * (G - 1)        [W/Hz, per polarization]

    **Saturation is a clamp, not a model.** ``max_output_power`` limits the
    output and the gain is reduced to whatever achieves it; real gain
    compression depends on the inversion level and has dynamics this does not
    capture. Left at its default the amplifier is unsaturated.
    """

    display_name = "EDFA"
    category = "Amplifiers"

    gain = Param(20.0, unit="dB", min=0.0, doc="Small-signal power gain")
    noise_figure = Param(5.0, unit="dB", min=0.0, doc="Noise figure")
    max_output_power = Param(
        30.0, unit="dBm", doc="Output power clamp; the gain is reduced to hold it"
    )
    center_wavelength = Param(
        1550.0, unit="nm", min=1200.0, max=1700.0, doc="Centre of the ASE band"
    )
    bandwidth = Param(4.0, unit="THz", min=0.0, doc="Optical bandwidth over which ASE is emitted")

    inputs = {"in": PortType.OPTICAL}
    outputs = {"out": PortType.OPTICAL}

    def spontaneous_emission_factor(self, gain_linear: float) -> float:
        """``n_sp``, the population inversion factor implied by the noise figure.

        Its floor is 1 — full inversion, a 3 dB noise figure — and a noise figure
        below that would describe an amplifier quieter than quantum mechanics
        allows.
        """
        if gain_linear <= 1.0:
            return 1.0
        noise_figure = db_to_linear(self.noise_figure)
        return noise_figure * gain_linear / (2.0 * (gain_linear - 1.0))

    def ase_psd(self, gain_linear: float) -> float:
        """One-sided ASE power spectral density per polarization [W/Hz]."""
        if gain_linear <= 1.0:
            return 0.0
        frequency = C_LIGHT / self.si("center_wavelength")
        return (
            self.spontaneous_emission_factor(gain_linear)
            * H_PLANCK
            * frequency
            * (gain_linear - 1.0)
        )

    def effective_gain(self, input_power: float) -> float:
        """Linear gain after the output-power clamp."""
        gain = db_to_linear(self.gain)
        ceiling = self.si("max_output_power")
        if input_power > 0.0 and input_power * gain > ceiling:
            gain = max(ceiling / input_power, 1.0)
        return gain

    def run(self, ctx: SimulationContext, inputs: dict[str, Signal]) -> dict[str, Signal]:
        signal: OpticalSignal = inputs["in"]
        gain = self.effective_gain(signal.total_power())
        amplitude = float(np.sqrt(gain))

        bands = tuple(
            Band(
                Ex=(band.Ex.astype(np.complex128) * amplitude).astype(ctx.complex_dtype),
                Ey=(band.Ey.astype(np.complex128) * amplitude).astype(ctx.complex_dtype),
                f0=band.f0,
                fs=band.fs,
            )
            for band in signal.bands
        )

        # Incoming noise is amplified along with the signal — that accumulation
        # across spans is what actually limits a long-haul link.
        noise = [bin_.scale_power(gain) for bin_ in signal.noise]

        bandwidth = self.si("bandwidth")
        psd = self.ase_psd(gain)
        if bandwidth > 0.0 and psd > 0.0:
            centre = C_LIGHT / self.si("center_wavelength")
            noise.append(
                NoiseBin(
                    f_start=centre - bandwidth / 2.0,
                    f_end=centre + bandwidth / 2.0,
                    psd_x=psd,
                    psd_y=psd,
                )
            )

        return {"out": OpticalSignal(bands=bands, noise=tuple(noise))}
