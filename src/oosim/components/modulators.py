"""Optical modulators.

Model reference: G. P. Agrawal, *Fiber-Optic Communication Systems*, ch. 3
(external modulation, Mach-Zehnder transfer characteristic).
"""

from __future__ import annotations

import numpy as np

from ..component import Component, Param, PortType
from ..context import SimulationContext
from ..signals import Band, ElectricalSignal, OpticalSignal, Signal
from ..units import db_to_linear


class MachZehnderModulator(Component):
    """Push-pull Mach-Zehnder modulator, chirp-free.

    An ideal symmetric MZ interferometer driven push-pull has power transmission

        P_out / P_in = cos**2(pi * V_total / (2 * V_pi))

    with ``V_total = V_drive + V_bias``: full transmission at 0 V, a null at
    ``V_pi``, and half power at the quadrature point ``V_pi / 2``.

    A real device cannot reach a perfect null, so the finite extinction ratio is
    folded in as a floor::

        P_out / P_in = IL * [ (1 - 1/ER) * cos**2(pi*V/(2*V_pi)) + 1/ER ]

    which peaks at ``IL`` and bottoms at ``IL/ER``, so the measured extinction
    ratio equals the declared one exactly — asserted in the test suite.

    Push-pull drive is chirp-free by construction: the field transmission is real
    and non-negative here, and the pi phase jump past the null is carried by the
    sign of the field. Chirped (single-drive) operation is a separate model.

    Modulator bandwidth is not yet limited; the drive is applied sample by
    sample. That refinement lives entirely in this block.
    """

    display_name = "Mach-Zehnder Modulator"
    category = "Modulators"

    v_pi = Param(4.0, unit="V", min=0.0, doc="Voltage for a pi phase shift (drive to null)")
    v_bias = Param(0.0, unit="V", doc="DC bias added to the drive voltage")
    extinction_ratio = Param(30.0, unit="dB", min=0.0, doc="On/off power ratio")
    insertion_loss = Param(0.0, unit="dB", min=0.0, doc="Loss at peak transmission")

    inputs = {"optical_in": PortType.OPTICAL, "electrical_in": PortType.ELECTRICAL}
    outputs = {"out": PortType.OPTICAL}

    def power_transmission(self, drive: np.ndarray) -> np.ndarray:
        """Power transmission for a drive waveform [V], including IL and ER."""
        v_pi = self.si("v_pi")
        if v_pi <= 0.0:
            raise ValueError(f"{self.label}: v_pi must be positive, got {self.v_pi}")

        total = drive.astype(np.float64) + self.si("v_bias")
        ideal = np.cos(np.pi * total / (2.0 * v_pi)) ** 2

        floor = 1.0 / db_to_linear(self.extinction_ratio)
        loss = db_to_linear(-self.insertion_loss)
        return loss * ((1.0 - floor) * ideal + floor)

    def run(self, ctx: SimulationContext, inputs: dict[str, Signal]) -> dict[str, Signal]:
        optical: OpticalSignal = inputs["optical_in"]
        drive: ElectricalSignal = inputs["electrical_in"]

        if drive.num_samples != ctx.num_samples:
            raise ValueError(
                f"{self.label}: drive has {drive.num_samples} samples but the run "
                f"window is {ctx.num_samples} samples"
            )

        amplitude = np.sqrt(self.power_transmission(drive.samples))

        bands = tuple(
            Band(
                Ex=(band.Ex.astype(np.complex128) * amplitude).astype(ctx.complex_dtype),
                Ey=(band.Ey.astype(np.complex128) * amplitude).astype(ctx.complex_dtype),
                f0=band.f0,
                fs=band.fs,
            )
            for band in optical.bands
        )
        # Noise accompanying the carrier is attenuated by the average
        # transmission, not the instantaneous one: it is broadband and does not
        # track the modulation.
        mean_transmission = float(np.mean(self.power_transmission(drive.samples)))
        noise = tuple(n.scale_power(mean_transmission) for n in optical.noise)

        return {"out": OpticalSignal(bands=bands, noise=noise)}
