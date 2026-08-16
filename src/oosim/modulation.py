"""Constellations, bit mapping, and the analytical error rates they are checked against.

These are plain functions rather than component methods for the same reason the
measurements in :mod:`oosim.analysis` are: the transmitter builds a constellation,
the receiver slices against it, the test suite compares both to theory, and one
implementation is the only way those three can agree.

Model reference: J. G. Proakis, *Digital Communications*, ch. 4 (signal space and
optimum receivers) and ch. 5 (error probability for QAM).
"""

from __future__ import annotations

import math

import numpy as np


def gray_pam_levels(bits_per_axis: int) -> np.ndarray:
    """Gray-coded PAM levels, indexed by bit pattern.

    ``levels[b]`` is the amplitude that the ``bits_per_axis``-bit pattern ``b``
    maps to, drawn from the odd integers centred on zero. Ordering them by Gray
    code rather than by magnitude is what makes adjacent levels differ in exactly
    one bit, which in turn is why a symbol error at high SNR usually costs one bit
    error and not ``bits_per_axis`` of them.
    """
    if bits_per_axis < 1:
        raise ValueError(f"bits_per_axis must be >= 1, got {bits_per_axis}")

    count = 1 << bits_per_axis
    levels = np.empty(count, dtype=np.float64)
    for index in range(count):
        levels[index ^ (index >> 1)] = 2.0 * index - (count - 1)
    return levels


def qam_constellation(bits_per_symbol: int) -> np.ndarray:
    """Gray-coded QAM constellation of unit mean power, indexed by bit pattern.

    ``bits_per_symbol = 1`` gives BPSK; even values give square QAM — QPSK, 16-QAM,
    64-QAM, 256-QAM — built as independent Gray-coded PAM on each quadrature, with
    the more significant half of the bits on I.

    Normalising to unit mean power is what lets a launch power be set once, on the
    laser, and mean the same thing for every format. Without it, switching from
    QPSK to 16-QAM would silently change the average optical power.

    Odd orders above 1 (32-QAM, 128-QAM) are cross constellations rather than
    products of two PAMs and are not implemented; the error says so rather than
    quietly returning a rectangle nobody uses.
    """
    if bits_per_symbol < 1:
        raise ValueError(f"bits_per_symbol must be >= 1, got {bits_per_symbol}")
    if bits_per_symbol == 1:
        points = np.array([-1.0, 1.0], dtype=np.complex128)
        return points / math.sqrt(float(np.mean(np.abs(points) ** 2)))
    if bits_per_symbol % 2:
        raise NotImplementedError(
            f"{1 << bits_per_symbol}-QAM is a cross constellation, not a square one, "
            f"and is not implemented; use an even bits_per_symbol (or 1 for BPSK)"
        )

    per_axis = bits_per_symbol // 2
    levels = gray_pam_levels(per_axis)
    patterns = np.arange(1 << bits_per_symbol)
    points = levels[patterns >> per_axis] + 1j * levels[patterns & ((1 << per_axis) - 1)]
    return points / math.sqrt(float(np.mean(np.abs(points) ** 2)))


def bits_to_indices(bits: np.ndarray, bits_per_symbol: int) -> np.ndarray:
    """Pack a bit stream into symbol indices, most significant bit first."""
    if bits_per_symbol < 1:
        raise ValueError(f"bits_per_symbol must be >= 1, got {bits_per_symbol}")
    if bits.shape[0] % bits_per_symbol:
        raise ValueError(
            f"{bits.shape[0]} bits does not divide into whole symbols of "
            f"{bits_per_symbol} bits"
        )

    grouped = bits.astype(np.int64).reshape(-1, bits_per_symbol)
    weights = 1 << np.arange(bits_per_symbol - 1, -1, -1, dtype=np.int64)
    return (grouped * weights).sum(axis=1)


def indices_to_bits(indices: np.ndarray, bits_per_symbol: int) -> np.ndarray:
    """Unpack symbol indices back into a bit stream, most significant bit first.

    The exact inverse of :func:`bits_to_indices`, which the test suite asserts
    directly rather than trusting the two to have been written consistently.
    """
    if bits_per_symbol < 1:
        raise ValueError(f"bits_per_symbol must be >= 1, got {bits_per_symbol}")

    shifts = np.arange(bits_per_symbol - 1, -1, -1, dtype=np.int64)
    bits = (indices.astype(np.int64)[:, None] >> shifts) & 1
    return bits.reshape(-1).astype(np.uint8)


def nearest_indices(symbols: np.ndarray, constellation: np.ndarray) -> np.ndarray:
    """Hard-decide each symbol to the nearest constellation point.

    This is the maximum-likelihood decision for additive Gaussian noise of equal
    variance on both quadratures, which is what a coherent receiver's noise is
    once the LO shot noise dominates.
    """
    if constellation.shape[0] == 0:
        raise ValueError("cannot decide against an empty constellation")

    distances = np.abs(symbols.astype(np.complex128)[:, None] - constellation[None, :])
    return np.argmin(distances, axis=1)


def quadrant_constellation(bits_per_symbol: int) -> np.ndarray:
    """A QAM alphabet relabelled as ``(quadrant, position within it)``.

    ``points[q * size + r]`` is the point ``r`` of the first quadrant turned
    through ``q`` quarter turns, where ``size`` is a quarter of the alphabet.

    The relabelling is the whole trick behind differential encoding. Under the
    plain Gray labelling a quarter turn permutes the bits in a way that depends
    on the point, because rotating swaps the roles of the I and Q magnitudes.
    Under this one it does exactly one thing: it adds a constant to ``q`` and
    leaves ``r`` alone. That reduces an ambiguity nothing blind can resolve to a
    single unknown offset — which is precisely the kind of thing differential
    encoding removes.

    Within-quadrant order follows the plain constellation's own, so the alphabet
    is the same set of points with the same minimum distance; only the labels
    move. Gray adjacency across quadrant boundaries is given up in exchange, and
    that costs a little BER at a given SNR — the price of not losing every bit
    after the receiver's phase estimator settles a quarter turn away.
    """
    points = qam_constellation(bits_per_symbol)
    if bits_per_symbol < 2:
        raise ValueError(
            f"differential quadrant encoding needs at least 2 bits per symbol, "
            f"got {bits_per_symbol}"
        )

    first = np.array(
        sorted(
            (p for p in points if p.real > 0 and p.imag > 0),
            key=lambda p: (round(p.real, 9), round(p.imag, 9)),
        ),
        dtype=np.complex128,
    )
    expected = points.shape[0] // 4
    if first.shape[0] != expected:
        raise ValueError(
            f"expected {expected} points in the first quadrant, found {first.shape[0]}"
        )
    turns = np.exp(0.5j * math.pi * np.arange(4))
    return (turns[:, None] * first[None, :]).reshape(-1)


def differential_encode(indices: np.ndarray, quarter_size: int) -> np.ndarray:
    """Accumulate the quadrant part of each symbol index; leave the rest absolute.

    The transmitted quadrant is the running sum of the requested *changes*, so a
    receiver that recovers the constellation a quarter turn away still sees the
    right differences and decodes the right data. Only the very first symbol is
    lost, because it has no predecessor to be different from.
    """
    values = indices.astype(np.int64)
    deltas, residues = np.divmod(values, quarter_size)
    quadrants = np.cumsum(deltas) % 4
    return quadrants * quarter_size + residues


def differential_decode(indices: np.ndarray, quarter_size: int) -> np.ndarray:
    """The inverse of :func:`differential_encode`: difference the quadrant back out."""
    values = indices.astype(np.int64)
    quadrants, residues = np.divmod(values, quarter_size)
    deltas = np.diff(quadrants, prepend=quadrants[:1] * 0) % 4
    return deltas * quarter_size + residues


def blind_phase_search(
    symbols: np.ndarray,
    constellation: np.ndarray,
    *,
    test_phases: int = 32,
    window: int = 64,
) -> np.ndarray:
    """Estimate the carrier phase per symbol, without knowing what was sent.

    The blind phase search of Pfau, Hoffmann and Noe (JLT 27(8), 2009), which is
    what a real coherent receiver runs. For each of ``test_phases`` candidate
    rotations it de-rotates the symbol, measures the distance to the nearest
    constellation point, and sums that over a sliding window; the candidate with
    the smallest sum wins. Averaging over a window is the whole trick — a single
    symbol cannot distinguish phase noise from additive noise, and a run of them
    can.

    Only ``[0, pi/2)`` is searched, because every QAM constellation here is
    invariant under a quarter turn. That symmetry is also the method's cost: the
    result is correct **modulo pi/2**, and nothing blind can do better. A real
    link resolves the ambiguity by differentially encoding the quadrant; see
    :class:`~oosim.components.coherent.CarrierRecovery` for how it is resolved
    here.

    ``window`` is the one real trade. Too short and the estimate is noisy, which
    shows up as extra EVM; too long and it cannot follow a fast-drifting laser,
    which shows up as an SNR ceiling that no amount of power lifts. The default
    of 64 symbols suits a linewidth-times-symbol-period around 1e-5, which is a
    100 kHz laser at 32 GBd.
    """
    if test_phases < 2:
        raise ValueError(f"test_phases must be >= 2, got {test_phases}")
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")

    values = symbols.astype(np.complex128)
    points = np.asarray(constellation).astype(np.complex128)
    count = values.shape[0]
    if count == 0:
        return np.zeros(0, dtype=np.float64)

    candidates = np.arange(test_phases, dtype=np.float64) * (math.pi / 2.0) / test_phases
    rotations = np.exp(-1j * candidates)

    # The full cost tensor is (symbols x phases x points), which for a long run
    # at 256-QAM is tens of gigabytes. It is built a slab at a time instead.
    cost = np.empty((count, test_phases), dtype=np.float64)
    stride = max(1, int(4e6 // (test_phases * max(points.size, 1))))
    for start in range(0, count, stride):
        block = values[start : start + stride, None] * rotations[None, :]
        distance = np.abs(block[:, :, None] - points[None, None, :])
        cost[start : start + stride] = distance.min(axis=2) ** 2

    # Centred moving sum, with the ends held rather than tapered: a window that
    # shrinks at the edges is noisier exactly where there is no data to check it
    # against, and the first and last few symbols are discarded by the analyser
    # anyway.
    lead = window // 2
    padded = np.pad(cost, ((lead, window - lead), (0, 0)), mode="edge")
    cumulative = np.concatenate([np.zeros((1, test_phases)), np.cumsum(padded, axis=0)])
    summed = cumulative[window : window + count] - cumulative[:count]

    phase = candidates[np.argmin(summed, axis=1)]

    # Unwrap: consecutive estimates that differ by about a quarter turn are the
    # ambiguity re-latching, not the laser jumping.
    quarter = math.pi / 2.0
    steps = np.diff(phase)
    steps -= quarter * np.round(steps / quarter)
    return np.concatenate([[phase[0]], phase[0] + np.cumsum(steps)])


def error_vector_magnitude(received: np.ndarray, reference: np.ndarray) -> float:
    """RMS EVM, as a fraction of the reference's RMS amplitude.

    ``sqrt(mean|r - s|**2 / mean|s|**2)``. Normalising by the *reference* power
    rather than by the peak or by the received power is the convention that makes
    ``SNR = 1 / EVM**2`` hold exactly, which is the identity
    :func:`snr_from_evm` relies on and the test suite checks.
    """
    if received.shape != reference.shape:
        raise ValueError(f"shape mismatch: received {received.shape}, reference {reference.shape}")
    if received.size == 0:
        raise ValueError("cannot measure EVM over an empty sequence")

    r = received.astype(np.complex128)
    s = reference.astype(np.complex128)
    reference_power = float(np.mean(np.abs(s) ** 2))
    if reference_power <= 0.0:
        raise ValueError("the reference sequence carries no power")
    return float(math.sqrt(float(np.mean(np.abs(r - s) ** 2)) / reference_power))


def snr_from_evm(evm: float) -> float:
    """Linear SNR implied by an RMS EVM: ``1 / EVM**2``."""
    if evm <= 0.0:
        return math.inf
    return 1.0 / (evm * evm)


def _q(x: float) -> float:
    """Gaussian tail probability ``Q(x) = 0.5 * erfc(x / sqrt(2))``."""
    return 0.5 * math.erfc(x / math.sqrt(2.0))


def ser_qam(snr_symbol: float, bits_per_symbol: int) -> float:
    """Analytical symbol error rate for square QAM in Gaussian noise.

    ``SER = 4*(1 - 1/sqrt(M))*Q(sqrt(3*SNR/(M-1))) - 4*(1 - 1/sqrt(M))**2 * Q(...)**2``

    with ``SNR`` the *symbol* signal-to-noise ratio in linear units. The squared
    term is the correction for the two quadratures failing together; dropping it
    is the usual approximation and is optimistic by a few percent near threshold,
    which is exactly the region a sensitivity curve is read in.

    This exists to be compared against counted errors. A model checked only
    against itself is not checked.
    """
    if bits_per_symbol == 1:
        return _q(math.sqrt(2.0 * max(snr_symbol, 0.0)))
    if bits_per_symbol % 2:
        raise NotImplementedError("analytical SER is implemented for square QAM and BPSK only")
    if snr_symbol <= 0.0:
        return 1.0 - 1.0 / (1 << bits_per_symbol)

    order = 1 << bits_per_symbol
    edge = 1.0 - 1.0 / math.sqrt(order)
    tail = _q(math.sqrt(3.0 * snr_symbol / (order - 1)))
    return 4.0 * edge * tail - 4.0 * edge * edge * tail * tail


def ber_qam(snr_symbol: float, bits_per_symbol: int) -> float:
    """Approximate BER for Gray-coded square QAM: ``SER / bits_per_symbol``.

    Valid where symbol errors land on a nearest neighbour, which Gray coding makes
    a single bit error. It is optimistic at low SNR, where errors reach past the
    neighbours and cost more than one bit each.
    """
    return ser_qam(snr_symbol, bits_per_symbol) / bits_per_symbol
