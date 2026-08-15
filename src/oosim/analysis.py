"""Measurements computed from signals.

These are plain functions rather than components on purpose: the same reductions
are needed by the test suite, by scripts, and later by measurement blocks and the
GUI. Keeping them here means one implementation and one set of tests.

They are also where result data gets *reduced*. A browser must never receive a
million raw samples; it receives what these functions return.
"""

from __future__ import annotations

import math

import numpy as np

from .signals import Band, EyeHistogram, EyeMeasurement, OpticalSignal


def instantaneous_power(band: Band) -> np.ndarray:
    """Instantaneous power [W] per sample, summed over both polarizations."""
    return (np.abs(band.Ex.astype(np.complex128)) ** 2) + (
        np.abs(band.Ey.astype(np.complex128)) ** 2
    )


def rms_time_width(band: Band) -> float:
    """RMS width of the intensity envelope in time [s].

    The second central moment of ``|A(t)|**2``. For a Gaussian intensity profile
    ``exp(-(T/T0)**2)`` this equals ``T0 / sqrt(2)``, so *ratios* of this quantity
    track ``T1/T0`` exactly — which is what the dispersion validation compares
    against the analytical broadening factor.

    The time window is periodic, so this is only meaningful while the pulse stays
    well inside it. A pulse that has spread far enough to wrap around will report
    a width that is wrong rather than merely imprecise.
    """
    power = instantaneous_power(band)
    total = float(power.sum())
    if total <= 0.0:
        raise ValueError("cannot measure the width of a signal carrying no power")

    t = np.arange(band.num_samples, dtype=np.float64) / band.fs
    mean = float((power * t).sum() / total)
    variance = float((power * (t - mean) ** 2).sum() / total)
    return float(np.sqrt(variance))


def peak_power(band: Band) -> float:
    """Highest instantaneous power in the window [W]."""
    return float(instantaneous_power(band).max())


#: Reference bandwidth OSNR is quoted in: 0.1 nm at 1550 nm, the industry
#: convention that makes numbers from different instruments comparable.
OSNR_REFERENCE_BANDWIDTH = 12.5e9


def noise_psd_at(signal: OpticalSignal, frequency: float) -> float:
    """Total noise PSD at ``frequency`` [W/Hz], summed over both polarizations."""
    return sum(
        bin_.psd_x + bin_.psd_y for bin_ in signal.noise if bin_.f_start <= frequency < bin_.f_end
    )


def osnr(signal: OpticalSignal, *, reference_bandwidth: float = OSNR_REFERENCE_BANDWIDTH) -> float:
    """Optical signal-to-noise ratio [dB], in a reference bandwidth.

    Signal power is the total in the sampled bands; noise power is the noise
    PSD *at each band's own centre frequency*, integrated over the reference
    bandwidth. Quoting OSNR in a fixed reference bandwidth rather than the
    signal's own is the convention, and it is the reason a 10 G and a 100 G
    channel with the same OSNR do not have the same margin.
    """
    if reference_bandwidth <= 0.0:
        raise ValueError(f"reference_bandwidth must be positive, got {reference_bandwidth}")

    signal_power = signal.signal_power()
    if signal_power <= 0.0:
        return -math.inf

    noise_power = sum(noise_psd_at(signal, band.f0) * reference_bandwidth for band in signal.bands)
    if noise_power <= 0.0:
        return math.inf
    return 10.0 * math.log10(signal_power / noise_power)


# --------------------------------------------------------------------------
# Decision-circuit analysis
# --------------------------------------------------------------------------


def ber_from_q(q: float) -> float:
    """Bit error rate for a Q-factor under Gaussian noise.

    ``BER = 0.5 * erfc(Q / sqrt(2))``. Q = 6 gives 9.87e-10, the value behind
    the industry's "Q of 6 is error-free" shorthand.

    This is an approximation in exactly one respect: it assumes the decision
    variable is Gaussian on both rails. That holds for thermal noise and for
    shot noise at realistic photon counts, and stops holding for a strongly
    amplified, ASE-dominated link — where the true distribution is
    non-central chi-squared and this formula is optimistic.
    """
    if q <= 0.0:
        return 0.5
    return 0.5 * math.erfc(q / math.sqrt(2.0))


def q_factor(mean_one: float, std_one: float, mean_zero: float, std_zero: float) -> float:
    """``Q = (mu1 - mu0) / (sigma1 + sigma0)``."""
    spread = std_one + std_zero
    if spread <= 0.0:
        return math.inf if mean_one > mean_zero else 0.0
    return (mean_one - mean_zero) / spread


def optimal_threshold(mean_one: float, std_one: float, mean_zero: float, std_zero: float) -> float:
    """Decision level that equalises the two error probabilities.

    ``V_th = (sigma0 * mu1 + sigma1 * mu0) / (sigma0 + sigma1)``. With unequal
    rail noise this sits nearer the quieter rail, which is why a real receiver's
    threshold is not simply the midpoint between the levels.
    """
    spread = std_one + std_zero
    if spread <= 0.0:
        return 0.5 * (mean_one + mean_zero)
    return (std_zero * mean_one + std_one * mean_zero) / spread


def rail_statistics(samples: np.ndarray, bits: np.ndarray) -> tuple[float, float, float, float]:
    """Mean and standard deviation of the mark and space rails.

    Returns ``(mean_one, std_one, mean_zero, std_zero)``.
    """
    marks = samples[bits.astype(bool)]
    spaces = samples[~bits.astype(bool)]
    if marks.size == 0 or spaces.size == 0:
        raise ValueError(
            "the reference sequence contains only one symbol value; a Q-factor needs both rails"
        )
    return (
        float(marks.mean()),
        float(marks.std()),
        float(spaces.mean()),
        float(spaces.std()),
    )


def measure_eye(
    samples: np.ndarray,
    bits: np.ndarray,
    samples_per_symbol: int,
    *,
    sample_offset: int | None = None,
    ignore_edges: int = 0,
) -> EyeMeasurement:
    """Take decision-circuit measurements on a received waveform.

    ``sample_offset`` selects the instant within each symbol at which the
    decision is made. Left as ``None`` it is chosen to maximise Q, which is what
    a receiver's clock recovery converges to; a fixed offset can be passed to
    study a mis-timed sampling instant.

    ``ignore_edges`` drops that many symbols from each end of the window. The
    filtering upstream is circular, so the first and last symbols carry a wrap
    from the far end of the sequence and are not representative.
    """
    if samples_per_symbol < 1:
        raise ValueError(f"samples_per_symbol must be >= 1, got {samples_per_symbol}")

    num_symbols = bits.shape[0]
    expected = num_symbols * samples_per_symbol
    if samples.shape[0] != expected:
        raise ValueError(
            f"waveform has {samples.shape[0]} samples but {num_symbols} bits at "
            f"{samples_per_symbol} samples/symbol needs {expected}"
        )

    grid = samples.astype(np.float64).reshape(num_symbols, samples_per_symbol)
    if ignore_edges > 0:
        if 2 * ignore_edges >= num_symbols:
            raise ValueError(
                f"ignore_edges={ignore_edges} would discard the whole sequence "
                f"of {num_symbols} symbols"
            )
        grid = grid[ignore_edges:-ignore_edges]
        bits = bits[ignore_edges:-ignore_edges]

    offsets = range(samples_per_symbol) if sample_offset is None else [sample_offset]
    q = -math.inf
    offset = 0
    mean_one = std_one = mean_zero = std_zero = 0.0
    for candidate in offsets:
        if not 0 <= candidate < samples_per_symbol:
            raise ValueError(f"sample_offset must be in [0, {samples_per_symbol}), got {candidate}")
        stats = rail_statistics(grid[:, candidate], bits)
        candidate_q = q_factor(*stats)
        if candidate_q > q:
            q, offset = candidate_q, candidate
            mean_one, std_one, mean_zero, std_zero = stats

    threshold = optimal_threshold(mean_one, std_one, mean_zero, std_zero)
    decided = (grid[:, offset] > threshold).astype(np.uint8)
    errors = int(np.count_nonzero(decided != bits))

    return EyeMeasurement(
        q_factor=q,
        mean_one=mean_one,
        mean_zero=mean_zero,
        std_one=std_one,
        std_zero=std_zero,
        threshold=threshold,
        sample_offset=offset,
        bits_evaluated=int(bits.shape[0]),
        errors=errors,
    )


def eye_histogram(
    samples: np.ndarray,
    samples_per_symbol: int,
    symbol_rate: float,
    *,
    span_symbols: int = 2,
    time_bins: int = 128,
    amplitude_bins: int = 128,
    unit: str = "",
) -> EyeHistogram:
    """Bin a waveform into an eye diagram.

    The output size is set by ``time_bins`` and ``amplitude_bins`` alone. A run
    with a million samples and one with ten million produce the same-sized
    result, which is the point: this is where the data reduction happens, so the
    UI never receives a raw sample buffer.

    Time resolution is capped at one column per sample. A trace holds exactly
    ``span_symbols * samples_per_symbol`` samples, all landing on that many
    distinct instants, so asking for more columns than that does not reveal more
    detail — it interleaves empty columns between populated ones and renders as
    vertical banding rather than an eye. Oversampling is what buys horizontal
    resolution here, not the bin count.
    """
    if span_symbols < 1:
        raise ValueError(f"span_symbols must be >= 1, got {span_symbols}")

    trace_length = span_symbols * samples_per_symbol
    num_traces = samples.shape[0] // trace_length
    if num_traces == 0:
        raise ValueError(
            f"waveform is shorter than one {span_symbols}-symbol trace "
            f"({samples.shape[0]} < {trace_length} samples)"
        )

    time_bins = min(time_bins, trace_length)

    traces = samples[: num_traces * trace_length].astype(np.float64).reshape(-1, trace_length)
    time_within_trace = np.arange(trace_length) / (samples_per_symbol * symbol_rate)

    counts, amplitude_edges, time_edges = np.histogram2d(
        traces.ravel(),
        np.tile(time_within_trace, num_traces),
        bins=(amplitude_bins, time_bins),
    )
    return EyeHistogram(
        counts=counts,
        time_edges=time_edges,
        amplitude_edges=amplitude_edges,
        unit=unit,
    )
