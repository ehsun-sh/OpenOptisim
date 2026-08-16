"""Electrical sources and drivers — the transmitter's input side."""

from __future__ import annotations

import numpy as np

from ..component import BoolParam, Component, Param, PortType
from ..context import SimulationContext
from ..dsp import root_raised_cosine, shape_symbols
from ..signals import BinarySignal, ElectricalSignal, Signal, SymbolSignal

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
    bits_per_symbol = Param(
        1.0,
        unit="",
        min=1.0,
        max=8.0,
        doc="Bits each downstream symbol carries; scales how many bits are emitted",
    )

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

        # The window is measured in symbols, not bits. A binary format consumes
        # one bit per symbol; a higher-order one consumes several, and the
        # generator has to emit enough of them to fill the same window.
        per_symbol = int(self.bits_per_symbol)
        count = ctx.sequence_length * per_symbol

        bits = np.empty(count, dtype=np.uint8)
        for i in range(count):
            feedback = ((state >> (n - 1)) ^ (state >> (tap - 1))) & 1
            bits[i] = state & 1
            state = ((state << 1) | feedback) & ((1 << n) - 1)

        return {"out": BinarySignal(bits=bits, symbol_rate=ctx.bit_rate * per_symbol)}


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


class IQDriver(Component):
    """Turns complex symbols into the two drive waveforms an IQ modulator wants.

    The real and imaginary parts are held for a full symbol, exactly as
    :class:`NRZDriver` holds a bit, and scaled so that the constellation's
    outermost quadrature level reaches ``drive_ratio * v_pi``.

    **Why pre-distortion lives here.** A child MZM biased at its null has *field*
    transmission ``sin(pi*V/(2*V_pi))`` — linear only for small drives. Driven at
    full swing it compresses the outer levels of a 16-QAM constellation while
    leaving QPSK untouched, because QPSK only ever uses the extremes. Real
    transmitters correct this in the DSP by pre-applying the inverse, and so does
    this block: with ``predistort`` on, the drive is
    ``(2*V_pi/pi) * arcsin(drive_ratio * f)`` and the field that emerges is
    proportional to the symbol. Turning it off leaves the compression visible,
    which is the point of being able to turn it off.

    ``v_pi`` is declared here as well as on the modulator because the correction
    genuinely needs it: a transmitter DSP that does not know the modulator's V_pi
    cannot linearise it. Setting the two to different values models exactly that
    mismatch.

    **Pulse shaping raises the peak, and the peak is what clips.** A held symbol
    never leaves the constellation's own levels, so a full-swing drive is exactly
    full swing. A root-raised-cosine waveform overshoots between symbols — that
    is what the tails of the pulse are — so its peak-to-average ratio is higher,
    and driving it at ``drive_ratio = 1`` runs the correction past ``arcsin(1)``
    and clips. The clipping is real and is left visible: at 16-QAM with a 0.2
    roll-off it costs about 7% EVM at full swing and about 1% backed off to 0.4.
    Backing the drive off is what a real transmitter does about it, and the
    residual 1% is the shaping filter's own truncation, set by ``filter_span``.
    """

    display_name = "IQ Driver"
    category = "Electrical"

    v_pi = Param(4.0, unit="V", min=0.0, doc="The modulator's V_pi, as the DSP believes it")
    drive_ratio = Param(
        1.0,
        unit="",
        min=0.01,
        max=1.0,
        doc="Peak drive as a fraction of V_pi; back off to linearise",
    )
    predistort = BoolParam(True, doc="Pre-invert the modulator's sine so the field is linear")
    pulse_shaping = BoolParam(
        False, doc="Root-raised-cosine shaping instead of holding each symbol flat"
    )
    roll_off = Param(0.2, unit="", min=0.0, max=1.0, doc="RRC excess bandwidth factor")
    filter_span = Param(
        16.0,
        unit="",
        min=2.0,
        max=64.0,
        doc="RRC length in symbols; longer leaves less residual ISI",
    )

    inputs = {"in": PortType.SYMBOL}
    outputs = {"i": PortType.ELECTRICAL, "q": PortType.ELECTRICAL}

    def _drive(self, quadrature: np.ndarray, peak: float) -> np.ndarray:
        """Map a normalised quadrature in [-1, 1] to a drive voltage [V]."""
        v_pi = self.si("v_pi")
        ratio = self.drive_ratio
        if self.predistort:
            return (2.0 * v_pi / np.pi) * np.arcsin(np.clip(ratio * quadrature / peak, -1.0, 1.0))
        return ratio * v_pi * quadrature / peak

    def run(self, ctx: SimulationContext, inputs: dict[str, Signal]) -> dict[str, Signal]:
        symbols: SymbolSignal = inputs["in"]
        if symbols.num_symbols != ctx.sequence_length:
            raise ValueError(
                f"{self.label}: got {symbols.num_symbols} symbols but the run window "
                f"holds {ctx.sequence_length}"
            )
        if self.si("v_pi") <= 0.0:
            raise ValueError(f"{self.label}: v_pi must be positive, got {self.v_pi}")

        # Normalise against the *alphabet*, not the symbols that happened to be
        # sent, so a short sequence that never uses an outer point still lands on
        # the same voltages as a long one.
        points = np.asarray(symbols.constellation).astype(np.complex128)
        peak = float(max(np.abs(points.real).max(), np.abs(points.imag).max()))
        if peak <= 0.0:
            raise ValueError(f"{self.label}: the constellation collapses to the origin")

        values = np.asarray(symbols.symbols).astype(np.complex128)

        # Shaping happens on the complex symbol, before the quadratures are taken
        # and before pre-distortion: the pulse shape is what the DSP sends to the
        # converters, and the arcsin correction is applied to the waveform that
        # results. Doing it the other way round would pre-distort a rectangle and
        # then smear the correction.
        if self.pulse_shaping:
            taps = root_raised_cosine(self.roll_off, int(self.filter_span), ctx.samples_per_symbol)
            waveform = shape_symbols(values, ctx.samples_per_symbol, taps)
        else:
            waveform = np.repeat(values, ctx.samples_per_symbol)

        out = {}
        for name, quadrature in (("i", waveform.real), ("q", waveform.imag)):
            samples = self._drive(quadrature, peak)
            out[name] = ElectricalSignal(
                samples=samples.astype(ctx.real_dtype), fs=ctx.sample_rate, unit="V"
            )
        return out


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
