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


class IQModulator(Component):
    """Dual-parallel MZM: two child modulators combined in quadrature.

    The input field is split, driven through two push-pull MZMs each biased at
    its own null, recombined with one arm delayed by a quarter wave, and emitted::

        E_out = (E_in / 2) * [ t(V_I + bias_I) + exp(j*(pi/2 + err)) * t(V_Q + bias_Q) ]

    with ``t(V) = sin(pi*V / (2*V_pi))`` the *field* transmission of a
    null-biased push-pull arm. This is the block that makes the tool able to
    represent anything past on-off keying: the field, not just its magnitude, is
    now under control, which is what a coherent receiver is built to recover.

    Three properties of the model are worth being explicit about.

    **The 3 dB is real.** With both arms driven to full swing the output power is
    half the input. That is not a modelling loss — it is what a dual-parallel
    structure costs, and it is why a coherent transmitter's launch power budget
    starts 3 dB behind an intensity-modulated one.

    **The sine is real too.** ``t`` is linear only for small drives. At full swing
    it compresses the outer levels of a 16-QAM constellation while leaving QPSK
    exactly alone, because QPSK only ever visits the extremes. Correcting it is
    the transmitter DSP's job; see
    :class:`~oosim.components.electrical.IQDriver`'s ``predistort``.

    **Bias error leaks carrier.** ``bias_i``/``bias_q`` are departures from the
    null. Any offset means ``t(0) != 0``, so an unmodulated component of the
    carrier reaches the output and the constellation acquires a DC offset — which
    is precisely what the residual-carrier spur on a spectrum analyser is, and
    what an automatic bias controller exists to null out.
    """

    display_name = "IQ Modulator"
    category = "Modulators"

    v_pi = Param(4.0, unit="V", min=0.0, doc="Voltage for a pi phase shift in one arm")
    bias_i = Param(0.0, unit="V", doc="I-arm bias error away from the null; leaks carrier")
    bias_q = Param(0.0, unit="V", doc="Q-arm bias error away from the null; leaks carrier")
    quadrature_error = Param(
        0.0, unit="deg", doc="Departure of the I/Q phase relationship from 90 degrees"
    )
    insertion_loss = Param(0.0, unit="dB", min=0.0, doc="Excess loss, on top of the intrinsic 3 dB")

    inputs = {
        "optical_in": PortType.OPTICAL,
        "i": PortType.ELECTRICAL,
        "q": PortType.ELECTRICAL,
    }
    outputs = {"out": PortType.OPTICAL}

    def arm_transmission(self, drive: np.ndarray, bias: float) -> np.ndarray:
        """Field transmission of one null-biased push-pull arm, signed."""
        v_pi = self.si("v_pi")
        if v_pi <= 0.0:
            raise ValueError(f"{self.label}: v_pi must be positive, got {self.v_pi}")
        return np.sin(np.pi * (drive.astype(np.float64) + bias) / (2.0 * v_pi))

    def field_transmission(self, drive_i: np.ndarray, drive_q: np.ndarray) -> np.ndarray:
        """Complex field transmission of the whole modulator, per sample."""
        t_i = self.arm_transmission(drive_i, self.si("bias_i"))
        t_q = self.arm_transmission(drive_q, self.si("bias_q"))
        quadrature = np.exp(1j * (np.pi / 2.0 + self.si("quadrature_error")))
        amplitude = np.sqrt(db_to_linear(-self.insertion_loss))
        return amplitude * 0.5 * (t_i + quadrature * t_q)

    def run(self, ctx: SimulationContext, inputs: dict[str, Signal]) -> dict[str, Signal]:
        optical: OpticalSignal = inputs["optical_in"]
        drive_i: ElectricalSignal = inputs["i"]
        drive_q: ElectricalSignal = inputs["q"]

        for name, drive in (("i", drive_i), ("q", drive_q)):
            if drive.num_samples != ctx.num_samples:
                raise ValueError(
                    f"{self.label}: drive {name!r} has {drive.num_samples} samples but "
                    f"the run window is {ctx.num_samples} samples"
                )

        transmission = self.field_transmission(
            np.asarray(drive_i.samples), np.asarray(drive_q.samples)
        )

        bands = tuple(
            Band(
                Ex=(band.Ex.astype(np.complex128) * transmission).astype(ctx.complex_dtype),
                Ey=(band.Ey.astype(np.complex128) * transmission).astype(ctx.complex_dtype),
                f0=band.f0,
                fs=band.fs,
            )
            for band in optical.bands
        )
        mean_power = float(np.mean(np.abs(transmission) ** 2))
        noise = tuple(n.scale_power(mean_power) for n in optical.noise)

        return {"out": OpticalSignal(bands=bands, noise=noise)}
