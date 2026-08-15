"""Coherent detection: the 90-degree hybrid, balanced photodiodes, and the sampler.

Model reference: G. P. Agrawal, *Fiber-Optic Communication Systems*, ch. 10
(coherent lightwave systems); K. Kikuchi, "Fundamentals of Coherent Optical Fiber
Communications", JLT 34(1), 2016.
"""

from __future__ import annotations

import numpy as np

from ..component import BoolParam, Component, Param, PortType
from ..context import SimulationContext
from ..signals import Band, ElectricalSignal, OpticalSignal, Signal, SymbolSignal
from ..units import K_BOLTZMANN, Q_ELECTRON


class CoherentReceiver(Component):
    """90-degree hybrid and two balanced photodiode pairs.

    The signal and the local oscillator are mixed in a 2x4 hybrid whose outputs
    are ``(E_s +- E_lo)/2`` and ``(E_s +- j*E_lo)/2``. Differencing each pair
    cancels the direct-detection terms and leaves the beat::

        i_I = R * Re(E_s . conj(E_lo))
        i_Q = R * Im(E_s . conj(E_lo))

    so the two photocurrents are the real and imaginary parts of the optical
    field itself, amplified by the LO. That is the whole point of coherent
    detection: phase survives, and with it every modulation format that lives
    off the real axis.

    **The dot product is over polarization.** ``E_s . conj(E_lo)`` is the Jones
    inner product ``Ex_s*conj(Ex_lo) + Ey_s*conj(Ey_lo)``, so a signal orthogonal
    to the LO produces no photocurrent at all. Polarization fading is not an
    add-on here; it falls out of the model, and a single-polarization coherent
    receiver really does go deaf when the fibre rotates the signal onto the
    orthogonal state.

    **Frequency offset falls out too.** Bands carry their own centre frequency,
    so a signal and an LO that do not sit at the same frequency beat at their
    difference: the mixing term picks up ``exp(2j*pi*(f_s - f_lo)*t)`` and the
    constellation rotates. This is intradyne operation, and the residual rotation
    is what a carrier-recovery stage exists to undo.

    **Noise convention.** The LO dominates: each of the four photodiodes receives
    about ``P_lo/4``, and differencing a pair adds their independent shot noises,
    giving ``var = q * R * P_lo * B`` on each of I and Q. With that convention the
    shot-noise-limited Q-factor of QPSK is ``sqrt(R*P_s / (2*q*B))``, which the
    test suite asserts against a counted error rate rather than assuming.

    Only the signal band nearest the LO is detected. The others beat at their
    frequency separation — hundreds of GHz for any realistic channel grid — which
    is far outside the electrical bandwidth of any receiver. They are genuinely
    rejected here rather than merely negligible, and a WDM demultiplexer in front
    is what makes that true in hardware.
    """

    display_name = "Coherent Receiver"
    category = "Receivers"

    responsivity = Param(0.8, unit="", min=0.0, doc="Responsivity R [A/W]")
    load_resistance = Param(50.0, unit="", min=0.0, doc="Load resistance [ohm]")
    temperature = Param(300.0, unit="", min=0.0, doc="Receiver temperature [K]")
    shot_noise = BoolParam(True, doc="Add LO-dominated shot noise")
    thermal_noise = BoolParam(True, doc="Add thermal (Johnson) noise")

    inputs = {"in": PortType.OPTICAL, "lo": PortType.OPTICAL}
    outputs = {"i": PortType.ELECTRICAL, "q": PortType.ELECTRICAL}

    def noise_bandwidth(self, ctx: SimulationContext) -> float:
        """Effective one-sided noise bandwidth [Hz] of the sampled representation."""
        return ctx.sample_rate / 2.0

    def _nearest_band(self, signal: OpticalSignal, f_lo: float) -> Band | None:
        if not signal.bands:
            return None
        return min(signal.bands, key=lambda b: abs(b.f0 - f_lo))

    def run(self, ctx: SimulationContext, inputs: dict[str, Signal]) -> dict[str, Signal]:
        signal: OpticalSignal = inputs["in"]
        lo: OpticalSignal = inputs["lo"]

        if len(lo.bands) != 1:
            raise ValueError(
                f"{self.label}: the local oscillator must be a single band, got "
                f"{len(lo.bands)}; a coherent receiver mixes against one tone"
            )
        reference = lo.bands[0]

        lo_x = reference.Ex.astype(np.complex128)
        lo_y = reference.Ey.astype(np.complex128)
        lo_power = reference.average_power()

        band = self._nearest_band(signal, reference.f0)
        if band is None:
            mix = np.zeros(ctx.num_samples, dtype=np.complex128)
        else:
            beat = np.exp(2j * np.pi * (band.f0 - reference.f0) * ctx.time_axis())
            mix = (band.Ex.astype(np.complex128) * np.conj(lo_x)) + (
                band.Ey.astype(np.complex128) * np.conj(lo_y)
            )
            mix = mix * beat

        responsivity = self.si("responsivity")
        current_i = responsivity * mix.real
        current_q = responsivity * mix.imag

        bandwidth = self.noise_bandwidth(ctx)
        # Signal-spontaneous beating with accompanying ASE is not modelled; the
        # noise bins contribute no photocurrent here, exactly as they contribute
        # no beat term in the direct-detection receiver.
        if self.shot_noise and lo_power > 0.0:
            variance = Q_ELECTRON * responsivity * lo_power * bandwidth
            rng = ctx.rng(type(self).__name__, self.label, "shot")
            current_i = current_i + rng.normal(0.0, np.sqrt(variance), size=ctx.num_samples)
            current_q = current_q + rng.normal(0.0, np.sqrt(variance), size=ctx.num_samples)

        if self.thermal_noise and self.si("load_resistance") > 0.0:
            variance = (
                4.0 * K_BOLTZMANN * self.si("temperature") * bandwidth / self.si("load_resistance")
            )
            rng = ctx.rng(type(self).__name__, self.label, "thermal")
            current_i = current_i + rng.normal(0.0, np.sqrt(variance), size=ctx.num_samples)
            current_q = current_q + rng.normal(0.0, np.sqrt(variance), size=ctx.num_samples)

        return {
            "i": ElectricalSignal(
                samples=current_i.astype(ctx.real_dtype), fs=ctx.sample_rate, unit="A"
            ),
            "q": ElectricalSignal(
                samples=current_q.astype(ctx.real_dtype), fs=ctx.sample_rate, unit="A"
            ),
        }


class IQSampler(Component):
    """Samples the two photocurrents once per symbol and rebuilds complex symbols.

    This is the boundary between the analogue receiver and the digital domain:
    amperes in, constellation points out. Two things happen here and nothing else.

    **Sampling.** One sample per symbol, taken at ``sample_offset`` into the
    symbol — the middle by default, which is where a rectangular symbol is
    furthest from its own transitions.

    **Automatic gain control.** The photocurrents are scaled by whatever the LO
    power and responsivity happened to be; the constellation they are compared
    against has unit mean power. Normalising here is what a real receiver's AGC
    does, and it is why an EVM does not change when the LO is turned up.

    Carrier phase is deliberately *not* recovered here. A blind estimator has a
    quadrant ambiguity that would silently corrupt an error count, and the
    measurement block already holds the transmitted sequence, so it removes the
    common phase the way a vector signal analyser does — with the reference in
    hand. Blind carrier recovery is a DSP block, and it belongs in the signal
    path rather than hidden inside a sampler.
    """

    display_name = "IQ Sampler"
    category = "Receivers"

    sample_offset = Param(
        -1.0, unit="", doc="Sample instant within the symbol; -1 selects the midpoint"
    )
    agc = BoolParam(True, doc="Normalise the sampled symbols to unit mean power")

    inputs = {"i": PortType.ELECTRICAL, "q": PortType.ELECTRICAL, "reference": PortType.SYMBOL}
    outputs = {"out": PortType.SYMBOL}

    def run(self, ctx: SimulationContext, inputs: dict[str, Signal]) -> dict[str, Signal]:
        current_i: ElectricalSignal = inputs["i"]
        current_q: ElectricalSignal = inputs["q"]
        reference: SymbolSignal = inputs["reference"]

        for name, waveform in (("i", current_i), ("q", current_q)):
            if waveform.num_samples != ctx.num_samples:
                raise ValueError(
                    f"{self.label}: input {name!r} has {waveform.num_samples} samples but "
                    f"the run window is {ctx.num_samples} samples"
                )

        sps = ctx.samples_per_symbol
        offset = sps // 2 if self.sample_offset < 0 else int(self.sample_offset)
        if not 0 <= offset < sps:
            raise ValueError(
                f"{self.label}: sample_offset must be in [0, {sps}), got {offset}"
            )

        grid_i = np.asarray(current_i.samples).astype(np.float64).reshape(-1, sps)
        grid_q = np.asarray(current_q.samples).astype(np.float64).reshape(-1, sps)
        symbols = grid_i[:, offset] + 1j * grid_q[:, offset]

        if self.agc:
            power = float(np.mean(np.abs(symbols) ** 2))
            if power > 0.0:
                symbols = symbols / np.sqrt(power)

        return {
            "out": SymbolSignal(
                symbols=symbols,
                symbol_rate=ctx.bit_rate,
                constellation=np.asarray(reference.constellation),
            )
        }
