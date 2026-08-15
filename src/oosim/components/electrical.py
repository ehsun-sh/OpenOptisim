"""Electrical sources and drivers — the transmitter's input side."""

from __future__ import annotations

import numpy as np

from ..component import Component, Param, PortType
from ..context import SimulationContext
from ..signals import BinarySignal, ElectricalSignal, Signal

#: Maximal-length LFSR feedback taps, as (order, tap) exponent pairs.
#: These are the standard polynomials used across optical test equipment;
#: see ITU-T O.150 and IEEE 802.3 for where each order is specified.
PRBS_TAPS: dict[int, tuple[int, int]] = {
    7: (7, 6),
    9: (9, 5),
    11: (11, 9),
    15: (15, 14),
    23: (23, 18),
    31: (31, 28),
}


class PRBSGenerator(Component):
    """Pseudo-random binary sequence from a maximal-length LFSR.

    A PRBS of order n has period ``2**n - 1`` and every n-bit window appears
    exactly once per period except all-zeros. Those properties are what make it a
    test pattern rather than merely random bits, and they are asserted directly
    in the test suite.

    If the requested sequence is longer than one period the pattern repeats,
    which is the intended behaviour — a receiver measuring BER over many periods
    is the normal case.
    """

    display_name = "PRBS Generator"
    category = "Electrical Sources"

    order = Param(7.0, unit="", min=7.0, max=31.0, doc="LFSR order n; period is 2**n - 1")

    outputs = {"out": PortType.BINARY}

    def run(self, ctx: SimulationContext, inputs: dict[str, Signal]) -> dict[str, Signal]:
        order = int(self.order)
        if order not in PRBS_TAPS:
            raise ValueError(
                f"{self.label}: PRBS order {order} is not a standard maximal-length "
                f"polynomial; available orders are {sorted(PRBS_TAPS)}"
            )
        n, tap = PRBS_TAPS[order]

        # Seed the register from the run seed so the pattern is reproducible but
        # not identical across differently-seeded runs. An all-zero state is a
        # fixed point of the LFSR, so it is excluded.
        rng = ctx.rng("PRBSGenerator", self.label, order)
        state = int(rng.integers(1, 1 << n))

        bits = np.empty(ctx.sequence_length, dtype=np.uint8)
        for i in range(ctx.sequence_length):
            feedback = ((state >> (n - 1)) ^ (state >> (tap - 1))) & 1
            bits[i] = state & 1
            state = ((state << 1) | feedback) & ((1 << n) - 1)

        return {"out": BinarySignal(bits=bits, symbol_rate=ctx.bit_rate)}


class NRZDriver(Component):
    """Non-return-to-zero driver: bits in, a rectangular voltage waveform out.

    Each bit is held for a full symbol period. The transition is instantaneous —
    a finite rise time and driver bandwidth are a later refinement, and adding
    them changes only this block.
    """

    display_name = "NRZ Driver"
    category = "Electrical"

    v_low = Param(0.0, unit="V", doc="Voltage representing a 0")
    v_high = Param(1.0, unit="V", doc="Voltage representing a 1")

    inputs = {"in": PortType.BINARY}
    outputs = {"out": PortType.ELECTRICAL}

    def run(self, ctx: SimulationContext, inputs: dict[str, Signal]) -> dict[str, Signal]:
        binary: BinarySignal = inputs["in"]
        if binary.num_bits != ctx.sequence_length:
            raise ValueError(
                f"{self.label}: got {binary.num_bits} bits but the run window holds "
                f"{ctx.sequence_length} symbols"
            )

        levels = np.where(binary.bits.astype(bool), self.si("v_high"), self.si("v_low"))
        samples = np.repeat(levels, ctx.samples_per_symbol)
        return {
            "out": ElectricalSignal(
                samples=samples.astype(ctx.real_dtype), fs=ctx.sample_rate, unit="V"
            )
        }


class DCVoltage(Component):
    """Constant voltage source.

    Exists mainly to characterise a modulator: holding the drive at a fixed
    voltage and sweeping it is how a transfer curve is measured on a bench, and
    it is how the MZM model is validated here.
    """

    display_name = "DC Voltage"
    category = "Electrical Sources"

    voltage = Param(0.0, unit="V", doc="Output voltage")

    outputs = {"out": PortType.ELECTRICAL}

    def run(self, ctx: SimulationContext, inputs: dict[str, Signal]) -> dict[str, Signal]:
        samples = np.full(ctx.num_samples, self.si("voltage"), dtype=ctx.real_dtype)
        return {"out": ElectricalSignal(samples=samples, fs=ctx.sample_rate, unit="V")}
