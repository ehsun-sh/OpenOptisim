"""Passive optical components."""

from __future__ import annotations

from typing import Any

from ..component import Component, Param, PortType
from ..context import SimulationContext
from ..signals import NoiseBin, OpticalSignal, Signal
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
