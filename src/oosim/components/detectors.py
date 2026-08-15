"""Photodetectors.

Model reference: G. P. Agrawal, *Fiber-Optic Communication Systems*, ch. 4
(photodetectors, receiver noise).
"""

from __future__ import annotations

import numpy as np

from ..component import BoolParam, Component, Param, PortType
from ..context import SimulationContext
from ..signals import ElectricalSignal, OpticalSignal, Signal
from ..units import K_BOLTZMANN, Q_ELECTRON


class PINPhotodiode(Component):
    """PIN photodiode: square-law detection with shot and thermal noise.

    The mean photocurrent is ``I = R * P + I_dark``, with ``R`` the responsivity
    [A/W]. Two noise sources are added, both white over the simulated bandwidth
    ``B = fs / 2``:

    * **Shot noise**, variance ``2 * q * I * B``. It scales with the
      instantaneous current, so it is generated per sample rather than as a
      single constant — bright samples really are noisier than dark ones, which
      is why an eye diagram's rails have different thicknesses.
    * **Thermal (Johnson) noise**, variance ``4 * k * T * B / R_load``,
      independent of the received power.

    Two simplifications worth stating plainly, both of which change results and
    neither of which is hidden by the interface:

    * Bands are detected incoherently — powers add. Beating between bands lands
      at their frequency separation, far above any realistic receiver bandwidth
      for the channel spacings this is used with, but it is genuinely absent
      rather than merely negligible.
    * Noise bins contribute mean power and its shot noise, but signal-ASE beat
      noise is not modelled. That term only matters once there is an amplifier
      to produce ASE, and it arrives with the EDFA in Phase 1.5.
    """

    display_name = "PIN Photodiode"
    category = "Receivers"

    responsivity = Param(0.8, unit="", min=0.0, doc="Responsivity R [A/W]")
    dark_current = Param(0.0, unit="", min=0.0, doc="Dark current [A]")
    load_resistance = Param(50.0, unit="", min=0.0, doc="Load resistance [ohm]")
    temperature = Param(300.0, unit="", min=0.0, doc="Receiver temperature [K]")
    shot_noise = BoolParam(True, doc="Add shot noise")
    thermal_noise = BoolParam(True, doc="Add thermal (Johnson) noise")

    inputs = {"in": PortType.OPTICAL}
    outputs = {"out": PortType.ELECTRICAL}

    def noise_bandwidth(self, ctx: SimulationContext) -> float:
        """Effective one-sided noise bandwidth [Hz] of the sampled representation."""
        return ctx.sample_rate / 2.0

    def run(self, ctx: SimulationContext, inputs: dict[str, Signal]) -> dict[str, Signal]:
        signal: OpticalSignal = inputs["in"]

        power = np.zeros(ctx.num_samples, dtype=np.float64)
        for band in signal.bands:
            power += np.abs(band.Ex.astype(np.complex128)) ** 2
            power += np.abs(band.Ey.astype(np.complex128)) ** 2
        power += signal.noise_power()

        current = self.si("responsivity") * power + self.si("dark_current")
        bandwidth = self.noise_bandwidth(ctx)

        if self.shot_noise:
            # Variance tracks the instantaneous current, so it is per sample:
            # bright samples really are noisier than dark ones.
            shot_variance = 2.0 * Q_ELECTRON * np.maximum(current, 0.0) * bandwidth
            rng = ctx.rng("PINPhotodiode", self.label, "shot")
            current = current + rng.normal(0.0, np.sqrt(shot_variance))

        if self.thermal_noise and self.si("load_resistance") > 0.0:
            thermal_variance = (
                4.0 * K_BOLTZMANN * self.si("temperature") * bandwidth / self.si("load_resistance")
            )
            rng = ctx.rng("PINPhotodiode", self.label, "thermal")
            current = current + rng.normal(0.0, np.sqrt(thermal_variance), size=ctx.num_samples)

        return {
            "out": ElectricalSignal(
                samples=current.astype(ctx.real_dtype), fs=ctx.sample_rate, unit="A"
            )
        }
