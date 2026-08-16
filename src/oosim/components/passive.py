"""Passive optical components."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..component import Component, Param, PortType
from ..context import SimulationContext
from ..signals import Band, NoiseBin, OpticalSignal, Signal
from ..units import db_to_linear


class Combiner(Component):
    """Ideal optical combiner / WDM multiplexer.

    Merges the bands of every input into one signal. Bands stay separately
    sampled, each keeping its own centre frequency — which is the whole point of
    the multi-band signal model, and the reason a 40-channel system does not
    require a physically impossible sample rate.

    Two inputs carrying the *same* centre frequency are rejected rather than
    silently summed: co-located carriers interfere and must be added coherently
    on a common grid, which is a different operation with different physics.
    """

    display_name = "Optical Combiner"
    category = "Passive"

    insertion_loss = Param(0.0, unit="dB", min=0.0, doc="Insertion loss applied to every input")

    outputs = {"out": PortType.OPTICAL}

    def __init__(self, num_inputs: int = 2, *, label: str | None = None, **params: float) -> None:
        if num_inputs < 1:
            raise ValueError(f"num_inputs must be >= 1, got {num_inputs}")
        super().__init__(label=label, **params)
        self.num_inputs = num_inputs
        # Per-instance port set: an N-way combiner has N inputs.
        self.inputs = {f"in{i}": PortType.OPTICAL for i in range(num_inputs)}

    def structural_config(self) -> dict[str, Any]:
        return {"num_inputs": self.num_inputs}

    def run(self, ctx: SimulationContext, inputs: dict[str, Signal]) -> dict[str, Signal]:
        power_factor = db_to_linear(-self.insertion_loss)
        amplitude_factor = power_factor**0.5

        bands = []
        noise: list[NoiseBin] = []
        seen: dict[float, str] = {}
        for name in self.inputs:
            signal: OpticalSignal = inputs[name]
            for band in signal.bands:
                if band.f0 in seen:
                    raise ValueError(
                        f"{self.label}: inputs {seen[band.f0]!r} and {name!r} both carry a band "
                        f"centred at {band.f0 / 1e12:.6f} THz. Co-located carriers must be added "
                        f"coherently on a common grid, not multiplexed."
                    )
                seen[band.f0] = name
                bands.append(band.scale_amplitude(amplitude_factor))
            noise.extend(n.scale_power(power_factor) for n in signal.noise)

        return {"out": OpticalSignal(bands=tuple(bands), noise=tuple(noise))}


class Splitter(Component):
    """Ideal 1-to-N optical power splitter.

    Each output receives ``1/N`` of the input power, which is ``-3.01 dB`` for a
    two-way split. It exists because a graph edge can fan out to as many inputs
    as it likes and no power is lost when it does — which is right for a bit
    stream and wrong for an optical field. A dual-polarization transmitter that
    drives two modulators from one laser really does pay 3 dB, and the split has
    to appear somewhere for the launch power to mean anything.
    """

    display_name = "Optical Splitter"
    category = "Passive"

    excess_loss = Param(0.0, unit="dB", min=0.0, doc="Loss beyond the ideal 10*log10(N) split")

    inputs = {"in": PortType.OPTICAL}

    def __init__(self, num_outputs: int = 2, *, label: str | None = None, **params: float) -> None:
        if num_outputs < 1:
            raise ValueError(f"num_outputs must be >= 1, got {num_outputs}")
        super().__init__(label=label, **params)
        self.num_outputs = num_outputs
        self.outputs = {f"out{i}": PortType.OPTICAL for i in range(num_outputs)}

    def structural_config(self) -> dict[str, Any]:
        return {"num_outputs": self.num_outputs}

    def run(self, ctx: SimulationContext, inputs: dict[str, Signal]) -> dict[str, Signal]:
        signal: OpticalSignal = inputs["in"]
        power_factor = db_to_linear(-self.excess_loss) / self.num_outputs
        amplitude_factor = power_factor**0.5
        split = OpticalSignal(
            bands=tuple(b.scale_amplitude(amplitude_factor) for b in signal.bands),
            noise=tuple(n.scale_power(power_factor) for n in signal.noise),
        )
        return {name: split for name in self.outputs}


class PolarizationCombiner(Component):
    """Polarization beam combiner: two co-frequency signals onto orthogonal states.

    The complement of :class:`Combiner`, and deliberately the opposite rule.
    ``Combiner`` refuses two inputs at the same centre frequency because
    co-located carriers interfere; this block *requires* it, because the whole
    point is to put two carriers at the same wavelength into the same fibre
    without them interfering — which orthogonal polarizations allow and nothing
    else does.

    That is what doubles the capacity of every modern coherent link, and it is
    why the field has been carried as a Jones vector since the first commit
    rather than as a scalar.

    Combining orthogonal states is **lossless**, unlike a power combiner. Only
    the ``x`` component of each input is used: each arm is expected to come from
    a modulator fed by a single-polarization laser, which is what the transmitter
    actually looks like.
    """

    display_name = "Polarization Combiner"
    category = "Passive"

    insertion_loss = Param(0.0, unit="dB", min=0.0, doc="Excess loss; an ideal PBC has none")

    inputs = {"x": PortType.OPTICAL, "y": PortType.OPTICAL}
    outputs = {"out": PortType.OPTICAL}

    def run(self, ctx: SimulationContext, inputs: dict[str, Signal]) -> dict[str, Signal]:
        arm_x: OpticalSignal = inputs["x"]
        arm_y: OpticalSignal = inputs["y"]

        power_factor = db_to_linear(-self.insertion_loss)
        amplitude = power_factor**0.5

        by_frequency: dict[float, Band] = {}
        for band in arm_x.bands:
            by_frequency[band.f0] = Band(
                Ex=band.Ex * amplitude,
                Ey=np.zeros_like(band.Ey),
                f0=band.f0,
                fs=band.fs,
            )
        for band in arm_y.bands:
            existing = by_frequency.get(band.f0)
            if existing is None:
                by_frequency[band.f0] = Band(
                    Ex=np.zeros_like(band.Ex),
                    Ey=band.Ex * amplitude,
                    f0=band.f0,
                    fs=band.fs,
                )
            else:
                by_frequency[band.f0] = Band(
                    Ex=existing.Ex,
                    Ey=band.Ex * amplitude,
                    f0=band.f0,
                    fs=band.fs,
                )

        noise = tuple(
            n.scale_power(power_factor) for signal in (arm_x, arm_y) for n in signal.noise
        )
        return {"out": OpticalSignal(bands=tuple(by_frequency.values()), noise=noise)}


class PolarizationRotator(Component):
    """Rotates the Jones vector by a fixed angle, with an optional phase between axes.

    A deterministic stand-in for what a real fibre does continuously and
    randomly: the state of polarization arriving at a receiver bears no relation
    to the one launched, and drifts on a timescale of milliseconds. A fixed
    rotation is the clean version of that, and it is the impairment that makes
    :class:`~oosim.components.dsp.ButterflyEqualizer` necessary rather than
    optional — at 45 degrees the two tributaries are mixed half and half, and no
    amount of power separates them.

    The rotation is unitary, so it conserves power exactly. Anything that claimed
    to rotate polarization while changing the total power would be wrong, and the
    test suite checks it.
    """

    display_name = "Polarization Rotator"
    category = "Passive"

    angle = Param(0.0, unit="deg", doc="Rotation of the polarization axes")
    phase = Param(0.0, unit="deg", doc="Retardation between the two axes")

    inputs = {"in": PortType.OPTICAL}
    outputs = {"out": PortType.OPTICAL}

    def jones_matrix(self) -> np.ndarray:
        """The 2x2 unitary this block applies."""
        theta = self.si("angle")
        delta = self.si("phase")
        rotation = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=np.complex128
        )
        retarder = np.array([[np.exp(0.5j * delta), 0.0], [0.0, np.exp(-0.5j * delta)]])
        return rotation @ retarder

    def run(self, ctx: SimulationContext, inputs: dict[str, Signal]) -> dict[str, Signal]:
        signal: OpticalSignal = inputs["in"]
        jones = self.jones_matrix()

        bands = []
        for band in signal.bands:
            ex = band.Ex.astype(np.complex128)
            ey = band.Ey.astype(np.complex128)
            bands.append(
                Band(
                    Ex=(jones[0, 0] * ex + jones[0, 1] * ey).astype(ctx.complex_dtype),
                    Ey=(jones[1, 0] * ex + jones[1, 1] * ey).astype(ctx.complex_dtype),
                    f0=band.f0,
                    fs=band.fs,
                )
            )
        return {"out": OpticalSignal(bands=tuple(bands), noise=signal.noise)}


class Attenuator(Component):
    """Ideal fixed optical attenuator."""

    display_name = "Attenuator"
    category = "Passive"

    attenuation = Param(3.0, unit="dB", min=0.0, doc="Attenuation")

    inputs = {"in": PortType.OPTICAL}
    outputs = {"out": PortType.OPTICAL}

    def run(self, ctx: SimulationContext, inputs: dict[str, Signal]) -> dict[str, Signal]:
        signal: OpticalSignal = inputs["in"]
        power_factor = db_to_linear(-self.attenuation)
        amplitude_factor = power_factor**0.5
        return {
            "out": OpticalSignal(
                bands=tuple(b.scale_amplitude(amplitude_factor) for b in signal.bands),
                noise=tuple(n.scale_power(power_factor) for n in signal.noise),
            )
        }
