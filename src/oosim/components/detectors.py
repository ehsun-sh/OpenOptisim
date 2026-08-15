"""Photodetectors.

Model reference: G. P. Agrawal, *Fiber-Optic Communication Systems*, ch. 4
(photodetectors, avalanche gain, receiver noise).
"""

from __future__ import annotations

import numpy as np

from ..component import BoolParam, Component, Param, PortType
from ..context import SimulationContext
from ..signals import ElectricalSignal, OpticalSignal, Signal
from ..units import K_BOLTZMANN, Q_ELECTRON


class PINPhotodiode(Component):
    """PIN photodiode: square-law detection with shot and thermal noise.

    The mean photocurrent is ``I = M * (R * P + I_dark)``, with ``R`` the
    responsivity [A/W] and ``M`` the multiplication gain — unity here, and
    overridden by :class:`APDPhotodiode`. Two noise sources are added, both
    white over the simulated bandwidth ``B = fs / 2``:

    * **Shot noise**, variance ``2 * q * I_primary * M**2 * F * B``. It scales
      with the instantaneous current, so it is generated per sample rather than
      as a single constant — bright samples really are noisier than dark ones,
      which is why an eye diagram's rails have different thicknesses.
    * **Thermal (Johnson) noise**, variance ``4 * k * T * B / R_load``,
      independent of the received power.

    Two simplifications worth stating plainly, both of which change results and
    neither of which is hidden by the interface:

    * Bands are detected incoherently — powers add. Beating between bands lands
      at their frequency separation, far above any realistic receiver bandwidth
      for the channel spacings this is used with, but it is genuinely absent
      rather than merely negligible.
    * Noise bins contribute mean power and its shot noise, but signal-ASE beat
      noise is not modelled. On an amplified link that term dominates, so a
      Q-factor computed after an EDFA is optimistic.
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

    def multiplication(self) -> float:
        """Avalanche gain. Unity for a PIN, which has no multiplication region."""
        return 1.0

    def excess_noise_factor(self) -> float:
        """Excess noise from the randomness of multiplication. Unity for a PIN."""
        return 1.0

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

        gain = self.multiplication()
        primary = self.si("responsivity") * power + self.si("dark_current")
        current = gain * primary
        bandwidth = self.noise_bandwidth(ctx)

        if self.shot_noise:
            # Multiplication amplifies the signal by M and the shot noise power
            # by M**2 * F: that is exactly why an avalanche gain cannot be raised
            # without limit, and why an APD has an optimum gain rather than a
            # best one.
            shot_variance = (
                2.0
                * Q_ELECTRON
                * np.maximum(primary, 0.0)
                * gain**2
                * self.excess_noise_factor()
                * bandwidth
            )
            rng = ctx.rng(type(self).__name__, self.label, "shot")
            current = current + rng.normal(0.0, np.sqrt(shot_variance))

        if self.thermal_noise and self.si("load_resistance") > 0.0:
            thermal_variance = (
                4.0 * K_BOLTZMANN * self.si("temperature") * bandwidth / self.si("load_resistance")
            )
            rng = ctx.rng(type(self).__name__, self.label, "thermal")
            current = current + rng.normal(0.0, np.sqrt(thermal_variance), size=ctx.num_samples)

        return {
            "out": ElectricalSignal(
                samples=current.astype(ctx.real_dtype), fs=ctx.sample_rate, unit="A"
            )
        }


class APDPhotodiode(PINPhotodiode):
    """Avalanche photodiode: internal gain, bought with excess noise.

    An APD multiplies the primary photocurrent by ``M`` before any electronics
    see it, which lifts a weak signal above the thermal noise floor of the load.
    The multiplication is a random cascade, though, so it amplifies shot noise
    by more than ``M**2``::

        F(M) = k*M + (2 - 1/M) * (1 - k)

    with ``k`` the ionisation coefficient ratio — a material property, around
    0.02 for silicon and 0.3 to 0.5 for InGaAs at 1550 nm. Lower is better.

    Because the signal grows as ``M`` while shot noise grows as ``M**2 * F`` and
    thermal noise does not grow at all, there is an **optimum gain**, not a best
    one. Below it the receiver is thermal-limited and more gain helps; above it
    the multiplication noise it creates outweighs what it buys.

    At ``M = 1`` the excess noise factor is 1 and this reduces exactly to
    :class:`PINPhotodiode` — asserted in the test suite rather than assumed.
    """

    display_name = "APD Photodiode"
    category = "Receivers"

    gain = Param(10.0, unit="", min=1.0, doc="Avalanche multiplication factor M")
    ionization_ratio = Param(
        0.3, unit="", min=0.0, max=1.0, doc="Ionisation coefficient ratio k; lower is quieter"
    )

    def multiplication(self) -> float:
        return self.gain

    def excess_noise_factor(self) -> float:
        """``F(M) = k*M + (2 - 1/M)(1 - k)``, which is 1 at M = 1 for any k."""
        m = self.gain
        k = self.ionization_ratio
        return k * m + (2.0 - 1.0 / m) * (1.0 - k)
