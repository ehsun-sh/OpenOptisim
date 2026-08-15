"""Bits to symbols and back — the digital edge of a coherent transceiver."""

from __future__ import annotations

import numpy as np

from ..component import Component, Param, PortType
from ..context import SimulationContext
from ..modulation import bits_to_indices, qam_constellation
from ..signals import BinarySignal, Signal, SymbolSignal


class QAMMapper(Component):
    """Groups bits into Gray-coded QAM symbols.

    ``bits_per_symbol`` selects the format: 1 is BPSK, 2 QPSK, 4 16-QAM, 6
    64-QAM, 8 256-QAM. The constellation is normalised to unit mean power, so
    changing format changes the information rate without changing the average
    optical power the laser is asked for.

    The run window holds :attr:`SimulationContext.sequence_length` *symbols*, so
    a source feeding this must supply that many times ``bits_per_symbol`` bits —
    which is what :class:`~oosim.components.electrical.PRBSGenerator`'s own
    ``bits_per_symbol`` is for. Mismatched lengths raise here rather than being
    silently truncated, because a truncated sequence still produces a BER.
    """

    display_name = "QAM Mapper"
    category = "Modulation"

    bits_per_symbol = Param(
        2.0, unit="", min=1.0, max=8.0, doc="1 BPSK, 2 QPSK, 4 16-QAM, 6 64-QAM, 8 256-QAM"
    )

    inputs = {"in": PortType.BINARY}
    outputs = {"out": PortType.SYMBOL}

    def constellation(self) -> np.ndarray:
        return qam_constellation(int(self.bits_per_symbol))

    def run(self, ctx: SimulationContext, inputs: dict[str, Signal]) -> dict[str, Signal]:
        binary: BinarySignal = inputs["in"]
        per_symbol = int(self.bits_per_symbol)

        expected = ctx.sequence_length * per_symbol
        if binary.num_bits != expected:
            raise ValueError(
                f"{self.label}: got {binary.num_bits} bits, but a window of "
                f"{ctx.sequence_length} symbols at {per_symbol} bits/symbol needs {expected}"
            )

        points = self.constellation()
        indices = bits_to_indices(np.asarray(binary.bits), per_symbol)
        return {
            "out": SymbolSignal(
                symbols=points[indices],
                symbol_rate=ctx.bit_rate,
                constellation=points,
            )
        }
