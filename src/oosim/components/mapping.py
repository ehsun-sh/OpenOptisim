"""Bits to symbols and back — the digital edge of a coherent transceiver."""

from __future__ import annotations

import numpy as np

from ..component import BoolParam, Component, Param, PortType
from ..context import SimulationContext
from ..modulation import (
    bits_to_indices,
    differential_decode,
    differential_encode,
    nearest_indices,
    qam_constellation,
    quadrant_constellation,
)
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
    differential = BoolParam(
        False, doc="Encode the quadrant differentially, so a quarter-turn ambiguity costs nothing"
    )

    inputs = {"in": PortType.BINARY}
    outputs = {"out": PortType.SYMBOL}

    def constellation(self) -> np.ndarray:
        """The alphabet, relabelled by quadrant when differential encoding is on."""
        if self.differential:
            return quadrant_constellation(int(self.bits_per_symbol))
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
        if self.differential:
            indices = differential_encode(indices, points.shape[0] // 4)

        return {
            "out": SymbolSignal(
                symbols=points[indices],
                symbol_rate=ctx.bit_rate,
                constellation=points,
            )
        }


class DifferentialDecoder(Component):
    """Undoes :class:`QAMMapper`'s differential quadrant encoding.

    Slices the received symbols against the quadrant-labelled alphabet, differences
    the quadrant back out, and re-emits them under the ordinary Gray labelling — so
    what comes out can be compared directly against a plain mapper fed the same
    bits, with no special handling anywhere downstream.

    **What this buys.** Every blind stage upstream leaves a quarter-turn
    ambiguity: the phase search cannot resolve it, and the butterfly equaliser has
    it too. Absolute labelling turns that into total data loss. Differencing turns
    it into a constant that cancels, so a receiver that settles a quarter turn
    away decodes correctly anyway.

    **What it costs.** A symbol error that crosses a quadrant boundary corrupts
    two consecutive symbols instead of one, because each quadrant is decoded
    relative to its predecessor. Roughly a factor of two in error rate near
    threshold — the standard price of differential coding, and much the better
    trade against losing everything.

    The first symbol has no predecessor and carries no data; discard it, as
    ``ignore_edges`` on the analyser already does.

    **This block emits decisions, so do not measure EVM after it.** Differencing
    the quadrant requires slicing first, and what comes out is ideal constellation
    points — an EVM taken here is exactly zero however bad the link is. Modulation
    quality is a soft measurement and belongs upstream, on the recovered symbols;
    the error count is a hard one and belongs here. A link that wants both needs
    an analyser in each place, which is also how a bench does it.
    """

    display_name = "Differential Decoder"
    category = "DSP"

    inputs = {"in": PortType.SYMBOL}
    outputs = {"out": PortType.SYMBOL}

    def run(self, ctx: SimulationContext, inputs: dict[str, Signal]) -> dict[str, Signal]:
        received: SymbolSignal = inputs["in"]
        encoded = np.asarray(received.constellation)
        quarter = encoded.shape[0] // 4
        if quarter < 1:
            raise ValueError(
                f"{self.label}: a {encoded.shape[0]}-point constellation has no quadrant "
                f"structure to difference"
            )

        indices = nearest_indices(np.asarray(received.symbols), encoded)
        decoded = differential_decode(indices, quarter)

        plain = qam_constellation(received.bits_per_symbol)
        return {
            "out": SymbolSignal(
                symbols=plain[decoded],
                symbol_rate=received.symbol_rate,
                constellation=plain,
            )
        }
