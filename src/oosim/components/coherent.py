"""Coherent detection: the 90-degree hybrid, balanced photodiodes, and the sampler.

Model reference: G. P. Agrawal, *Fiber-Optic Communication Systems*, ch. 10
(coherent lightwave systems); K. Kikuchi, "Fundamentals of Coherent Optical Fiber
Communications", JLT 34(1), 2016.
"""

from __future__ import annotations

import math

import numpy as np

from ..component import BoolParam, Component, Param, PortType
from ..context import SimulationContext
from ..dsp import circular_filter, root_raised_cosine
from ..modulation import blind_phase_search
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


class DualPolarizationReceiver(CoherentReceiver):
    """Polarization-diverse coherent receiver: four photocurrents, not two.

    The signal is split by a polarization beam splitter, the LO is split with it,
    and each pair goes to its own 90-degree hybrid. The result is the *complete*
    optical field: two complex quadratures on each of two orthogonal states, from
    which any launched polarization can be reconstructed by linear combination.

    That last clause is the point. A single-polarization receiver goes deaf when
    the fibre rotates the signal onto the orthogonal state; this one cannot,
    because it measures both states and lets the DSP recombine them. It is why a
    coherent link can carry two independent tributaries at one wavelength while a
    direct-detection link carries one.

    The PBS axes are the receiver's own, and they bear no relation to whatever
    the transmitter launched — a fibre rotates the state arbitrarily and drifts.
    So ``x`` and ``y`` here are *not* the two tributaries. Separating those is
    :class:`~oosim.components.dsp.ButterflyEqualizer`'s job, and without it these
    outputs are two mixtures rather than two channels.

    The noise convention is inherited unchanged: each hybrid sees half the LO
    power, so the per-quadrature shot variance is half the single-polarization
    receiver's, and splitting the signal likewise halves what each branch
    receives. The two effects cancel in the SNR, which is why polarization
    multiplexing doubles capacity at the same SNR per tributary rather than
    costing 3 dB.
    """

    display_name = "Dual-Pol Coherent Receiver"
    category = "Receivers"

    inputs = {"in": PortType.OPTICAL, "lo": PortType.OPTICAL}
    outputs = {
        "xi": PortType.ELECTRICAL,
        "xq": PortType.ELECTRICAL,
        "yi": PortType.ELECTRICAL,
        "yq": PortType.ELECTRICAL,
    }

    def run(self, ctx: SimulationContext, inputs: dict[str, Signal]) -> dict[str, Signal]:
        signal: OpticalSignal = inputs["in"]
        lo: OpticalSignal = inputs["lo"]

        if len(lo.bands) != 1:
            raise ValueError(
                f"{self.label}: the local oscillator must be a single band, got "
                f"{len(lo.bands)}; a coherent receiver mixes against one tone"
            )
        reference = lo.bands[0]
        # The LO is launched at 45 degrees so the PBS delivers equal power to both
        # hybrids. A real receiver either does this or uses a polarization-
        # maintaining split; either way each branch gets half.
        lo_amplitude = math.sqrt(reference.average_power() / 2.0)
        lo_phase = np.exp(1j * np.angle(reference.Ex.astype(np.complex128)))
        lo_branch = lo_amplitude * lo_phase

        band = self._nearest_band(signal, reference.f0)
        responsivity = self.si("responsivity")
        bandwidth = self.noise_bandwidth(ctx)
        # Each of the two hybrids sees half the LO.
        shot_variance = Q_ELECTRON * responsivity * (reference.average_power() / 2.0) * bandwidth
        thermal_variance = (
            4.0 * K_BOLTZMANN * self.si("temperature") * bandwidth / self.si("load_resistance")
            if self.si("load_resistance") > 0.0
            else 0.0
        )

        out: dict[str, Signal] = {}
        for axis, field in (("x", "Ex"), ("y", "Ey")):
            if band is None:
                mix = np.zeros(ctx.num_samples, dtype=np.complex128)
            else:
                beat = np.exp(2j * np.pi * (band.f0 - reference.f0) * ctx.time_axis())
                mix = getattr(band, field).astype(np.complex128) * np.conj(lo_branch) * beat

            for quadrature, part in (("i", mix.real), ("q", mix.imag)):
                current = responsivity * part
                if self.shot_noise and shot_variance > 0.0:
                    rng = ctx.rng(type(self).__name__, self.label, "shot", axis, quadrature)
                    current = current + rng.normal(
                        0.0, math.sqrt(shot_variance), size=ctx.num_samples
                    )
                if self.thermal_noise and thermal_variance > 0.0:
                    rng = ctx.rng(type(self).__name__, self.label, "thermal", axis, quadrature)
                    current = current + rng.normal(
                        0.0, math.sqrt(thermal_variance), size=ctx.num_samples
                    )
                out[axis + quadrature] = ElectricalSignal(
                    samples=current.astype(ctx.real_dtype), fs=ctx.sample_rate, unit="A"
                )
        return out


class CarrierRecovery(Component):
    """Blind carrier phase recovery, by phase search.

    A transmitter laser and a local oscillator are independent oscillators, so
    the phase between them performs a random walk whose rate is set by their
    linewidths. Nothing upstream can remove it: it is not a constant, and it is
    not a frequency offset either, so subtracting a line does not help. Left in,
    it puts a **floor** under the measured SNR that no amount of launch power
    lifts — at 100 kHz and 32 GBd, around 18 dB, which is not enough for 16-QAM.

    This block is what a real receiver does about it. See
    :func:`oosim.modulation.blind_phase_search` for the method.

    **The quadrant ambiguity is real and is not hidden.** Every QAM constellation
    here is invariant under a quarter turn, so no blind estimator can tell which
    quadrant it is in; the recovered phase is right modulo pi/2. A deployed link
    resolves this by differentially encoding the quadrant, which costs a little
    sensitivity. That is not implemented yet, so what remains after this block is
    a *constant* rotation by some multiple of pi/2 — and the measurement block
    removes it data-aided, the way a bench analyser does with the pattern in
    hand. The time-varying part, which is the part that actually costs SNR, is
    gone by then.

    A **cycle slip** is the failure mode to know about: if the phase walks faster
    than the window can follow, the estimate latches onto a neighbouring quadrant
    and stays there, and every symbol after the slip is decided in the wrong
    quadrant. It does not degrade gracefully. Widening ``window`` tracks noise
    better but slips sooner; narrowing it does the reverse.
    """

    display_name = "Carrier Recovery"
    category = "DSP"

    test_phases = Param(
        32.0, unit="", min=4.0, max=128.0, doc="Candidate rotations searched across a quarter turn"
    )
    window = Param(
        64.0,
        unit="",
        min=1.0,
        doc="Symbols averaged per estimate; longer is quieter but slips sooner",
    )

    inputs = {"in": PortType.SYMBOL}
    outputs = {"out": PortType.SYMBOL}

    def run(self, ctx: SimulationContext, inputs: dict[str, Signal]) -> dict[str, Signal]:
        received: SymbolSignal = inputs["in"]
        constellation = np.asarray(received.constellation)

        phase = blind_phase_search(
            np.asarray(received.symbols),
            constellation,
            test_phases=int(self.test_phases),
            window=int(self.window),
        )
        recovered = np.asarray(received.symbols).astype(np.complex128) * np.exp(-1j * phase)

        return {
            "out": SymbolSignal(
                symbols=recovered,
                symbol_rate=received.symbol_rate,
                constellation=constellation,
            )
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
    matched_filter = BoolParam(
        False, doc="Apply the receiver's root-raised-cosine half of the pulse shaping"
    )
    roll_off = Param(0.2, unit="", min=0.0, max=1.0, doc="Must match the transmitter's")
    filter_span = Param(16.0, unit="", min=2.0, max=64.0, doc="RRC length in symbols")

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
        # A matched filter puts the pulse's peak on the symbol instant it was
        # launched at, which is sample zero of each symbol; a flat-held symbol has
        # no peak and is best read from the middle, furthest from its own edges.
        default_offset = 0 if self.matched_filter else sps // 2
        offset = default_offset if self.sample_offset < 0 else int(self.sample_offset)
        if not 0 <= offset < sps:
            raise ValueError(
                f"{self.label}: sample_offset must be in [0, {sps}), got {offset}"
            )

        complex_baseband = np.asarray(current_i.samples).astype(np.float64) + 1j * np.asarray(
            current_q.samples
        ).astype(np.float64)

        if self.matched_filter:
            # The receiver's half of the split shaping. Cascaded with the
            # transmitter's root it forms a full raised cosine, which is zero at
            # every symbol instant but its own — so the neighbours contribute
            # nothing to this decision — and it is simultaneously matched to the
            # transmitted pulse, which is what makes it optimal against noise.
            taps = root_raised_cosine(
                self.roll_off, int(self.filter_span), ctx.samples_per_symbol
            )
            complex_baseband = circular_filter(complex_baseband, taps)

        symbols = complex_baseband.reshape(-1, sps)[:, offset]

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
