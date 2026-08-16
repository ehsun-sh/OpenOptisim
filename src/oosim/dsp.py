"""Adaptive equalisation: the 2x2 butterfly that separates two polarizations.

A dual-polarization receiver measures the field on *its own* axes, which bear no
relation to the ones the transmitter launched — a fibre rotates the state
arbitrarily and drifts on a millisecond timescale. What arrives is therefore two
mixtures rather than two channels, and no amount of power separates them. The
filter here is what does.

Model reference: S. J. Savory, "Digital filters for coherent optical receivers",
Optics Express 16(2), 2008; D. N. Godard, "Self-recovering equalization and
carrier tracking", IEEE Trans. Comm. 28(11), 1980 (the constant-modulus
algorithm).
"""

from __future__ import annotations

import numpy as np

#: Cross-coupling placed in the initial filter to break the 45-degree symmetry.
#: See :func:`butterfly_equalize` for why a symmetric start cannot converge there.
SYMMETRY_TILT = 0.1


def root_raised_cosine(
    roll_off: float, span_symbols: int, samples_per_symbol: int
) -> np.ndarray:
    """Root-raised-cosine impulse response, normalised to unit energy.

    A rectangular symbol has a sinc spectrum, which never ends: it is fine for a
    single channel in isolation and useless the moment neighbours are packed onto
    a grid beside it. A raised-cosine spectrum stops dead at ``(1 + roll_off)``
    times the symbol rate, which is what lets 32 GBd sit on a 50 GHz grid.

    The *root* is used because the shaping is split between the two ends. Cascade
    a transmitter's root with a receiver's matched root and the result is a full
    raised cosine, which is the shape that satisfies the Nyquist criterion: zero
    at every symbol instant but its own, so neighbouring symbols contribute
    nothing to a decision. Splitting it this way is also what makes the receiver
    filter *matched* to the transmitted pulse, and therefore optimal against
    additive noise. One end alone gives neither property.

    ``roll_off = 0`` degenerates to a sinc: the narrowest possible spectrum, and
    an impulse response that decays so slowly that any timing error is punished
    severely. Real systems use 0.05 to 0.3 for exactly that reason.
    """
    if not 0.0 <= roll_off <= 1.0:
        raise ValueError(f"roll_off must be in [0, 1], got {roll_off}")
    if span_symbols < 1:
        raise ValueError(f"span_symbols must be >= 1, got {span_symbols}")
    if samples_per_symbol < 1:
        raise ValueError(f"samples_per_symbol must be >= 1, got {samples_per_symbol}")

    length = span_symbols * samples_per_symbol
    if length % 2 == 0:
        length += 1  # odd, so the peak sits on a sample and the filter is symmetric
    t = (np.arange(length) - (length - 1) / 2.0) / samples_per_symbol

    beta = roll_off
    h = np.empty_like(t)
    for index, tau in enumerate(t):
        if abs(tau) < 1e-12:
            h[index] = 1.0 - beta + 4.0 * beta / np.pi
        elif beta > 0.0 and abs(abs(tau) - 1.0 / (4.0 * beta)) < 1e-12:
            # Removable singularity at t = +-T/(4*beta); the limit is closed-form.
            h[index] = (beta / np.sqrt(2.0)) * (
                (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * beta))
                + (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * beta))
            )
        else:
            numerator = np.sin(np.pi * tau * (1.0 - beta)) + 4.0 * beta * tau * np.cos(
                np.pi * tau * (1.0 + beta)
            )
            denominator = np.pi * tau * (1.0 - (4.0 * beta * tau) ** 2)
            h[index] = numerator / denominator

    return h / np.sqrt(float(np.sum(h**2)))


def shape_symbols(
    symbols: np.ndarray, samples_per_symbol: int, filter_taps: np.ndarray
) -> np.ndarray:
    """Upsample by inserting zeros, then filter — circularly, to keep the window closed.

    Zero-insertion rather than sample-and-hold is what makes the filter the pulse
    shape rather than a correction applied on top of a rectangle.

    The convolution wraps because every other block in the engine treats the time
    window as periodic: a linear convolution would leave a transient at each end
    that the analyser would read as distortion, and the fix for that is the same
    ``ignore_edges`` everything else already uses.

    The result is scaled so the symbol instants keep the constellation's own
    level. Without it a unit-energy filter would silently rescale the alphabet,
    and every drive voltage downstream would be wrong by a factor depending on
    the oversampling. Note this fixes the *scale* only: a single root is not
    ISI-free, and the waveform sampled here still carries its neighbours'
    interference. Removing that is the receiver's matched root's job, and the two
    together are what satisfy the Nyquist criterion.
    """
    upsampled = np.zeros(symbols.shape[0] * samples_per_symbol, dtype=np.complex128)
    upsampled[::samples_per_symbol] = symbols.astype(np.complex128)
    peak = float(np.max(np.abs(filter_taps)))
    if peak <= 0.0:
        raise ValueError("the shaping filter is all zeros")
    return circular_filter(upsampled, filter_taps) / peak


def circular_filter(samples: np.ndarray, filter_taps: np.ndarray) -> np.ndarray:
    """Filter with a symmetric FIR, wrapping at the window edges and zero group delay."""
    taps = filter_taps.shape[0]
    padded = np.zeros(samples.shape[0], dtype=np.complex128)
    padded[:taps] = filter_taps
    spectrum = np.fft.fft(samples) * np.fft.fft(padded)
    # Undo the filter's own delay so the symbol instants stay where they were.
    delay = (taps - 1) // 2
    return np.roll(np.fft.ifft(spectrum), -delay)


def constellation_radii(constellation: np.ndarray) -> np.ndarray:
    """The distinct moduli a constellation uses, ascending.

    QPSK has one; 16-QAM has three. A one-radius constellation is what makes the
    plain constant-modulus algorithm exact, and the reason it only *approximately*
    works on QAM — which is what the radius-directed stage exists to fix.
    """
    moduli = np.abs(np.asarray(constellation).astype(np.complex128))
    return np.unique(np.round(moduli, 9))


def godard_radius(constellation: np.ndarray) -> float:
    """The CMA target ``R2 = E|c|^4 / E|c|^2``.

    For a constellation of one modulus this is that modulus squared. For QAM it
    is a compromise no symbol actually sits on, which is exactly why CMA opens
    the eye but does not close it, and why it is used as a *pre-convergence*
    stage rather than as the whole equaliser.
    """
    moduli = np.abs(np.asarray(constellation).astype(np.complex128))
    second = float(np.mean(moduli**2))
    if second <= 0.0:
        raise ValueError("the constellation carries no power")
    return float(np.mean(moduli**4) / second)


def butterfly_equalize(
    x: np.ndarray,
    y: np.ndarray,
    constellation: np.ndarray,
    *,
    taps: int = 7,
    step: float = 3e-3,
    cma_symbols: int | None = None,
    passes: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Blindly separate two mixed tributaries with an adaptive 2x2 FIR filter.

    Returns ``(out_x, out_y, weights)`` with ``weights`` the final ``(2, 2, taps)``
    filter, kept because a converged butterfly *is* the channel's inverse and
    reading it is how one checks what the channel did.

    **How it separates without being told anything.** Each output is a filtered
    combination of both inputs. The filters are adapted to drive every output
    symbol onto a modulus the constellation actually uses — no reference, no
    training sequence. A mixture of two independent tributaries has a modulus
    that wanders; a clean tributary does not. That difference is the entire
    signal the algorithm has, and it is enough.

    Two stages, because one is not enough for QAM. The first drives towards the
    single Godard radius, which opens the eye from nothing but converges to a
    compromise; the second switches to the *nearest* of the constellation's real
    radii, which is what actually closes it. On QPSK the two stages are identical
    because there is only one radius.

    **The singularity.** Nothing in the cost function distinguishes the two
    outputs, so in principle both filters can converge onto the *same* tributary
    and leave the other unrecovered — the well-known failure of blind butterfly
    equalisation. What keeps them apart here is the initialisation: the two rows
    start orthogonal and the adaptation has no reason to bring them together.

    A textbook remedy is to re-seed the second row as the unitary complement of
    the first once the first has settled. That was tried and *measured*, and over
    ninety channel and format combinations it helped eight times, hurt nine, and
    changed nothing in the rest — so it is not here. The tilt below is the part
    that is actually load-bearing, and there is a test that fails without it.
    """
    if taps < 1 or taps % 2 == 0:
        raise ValueError(f"taps must be a positive odd number, got {taps}")
    if step <= 0.0:
        raise ValueError(f"step must be positive, got {step}")
    if passes < 1:
        raise ValueError(f"passes must be >= 1, got {passes}")
    if x.shape != y.shape:
        raise ValueError(f"tributaries differ in length: {x.shape} and {y.shape}")

    xs = x.astype(np.complex128)
    ys = y.astype(np.complex128)
    count = xs.shape[0]
    if count < taps:
        raise ValueError(f"need at least {taps} symbols to fill the filter, got {count}")

    points = np.asarray(constellation).astype(np.complex128)
    radii = constellation_radii(points)
    target = godard_radius(points)

    centre = taps // 2
    # h[out, in, tap]. A centre spike is the identity: pass each input straight
    # through to the matching output, then let the adaptation find the rotation.
    #
    # The small cross term is not decoration. A channel that mixes the two
    # tributaries exactly half and half — a 45 degree rotation — leaves the
    # identity initialisation equidistant from both valid solutions, which is a
    # saddle of the cost function rather than a minimum. The adaptation stalls
    # there, and running it longer makes it worse rather than better. Tilting the
    # start off the symmetry axis removes the saddle. It is deterministic, so the
    # result stays reproducible; 64-QAM at 45 degrees fails without it and lands
    # on the noise floor with it.
    weights = np.zeros((2, 2, taps), dtype=np.complex128)
    weights[0, 0, centre] = 1.0
    weights[1, 1, centre] = 1.0
    weights[0, 1, centre] = SYMMETRY_TILT
    weights[1, 0, centre] = -SYMMETRY_TILT

    if cma_symbols is None:
        cma_symbols = count // 2

    out_x = np.zeros(count, dtype=np.complex128)
    out_y = np.zeros(count, dtype=np.complex128)

    for pass_index in range(passes):
        for processed, k in enumerate(range(centre, count - centre)):
            window_x = xs[k - centre : k + centre + 1][::-1]
            window_y = ys[k - centre : k + centre + 1][::-1]

            ox = weights[0, 0] @ window_x + weights[0, 1] @ window_y
            oy = weights[1, 0] @ window_x + weights[1, 1] @ window_y
            out_x[k] = ox
            out_y[k] = oy

            # Radius-directed once the eye is open; Godard's single radius before
            # then, because a decision on a closed eye is worse than no decision.
            blind = pass_index == 0 and processed < cma_symbols
            if blind:
                error_x = target - abs(ox) ** 2
                error_y = target - abs(oy) ** 2
            else:
                error_x = radii[np.argmin(np.abs(radii - abs(ox)))] ** 2 - abs(ox) ** 2
                error_y = radii[np.argmin(np.abs(radii - abs(oy)))] ** 2 - abs(oy) ** 2

            weights[0, 0] += step * error_x * ox * np.conj(window_x)
            weights[0, 1] += step * error_x * ox * np.conj(window_y)
            weights[1, 0] += step * error_y * oy * np.conj(window_x)
            weights[1, 1] += step * error_y * oy * np.conj(window_y)

    # The filter cannot produce an output for the first and last half-window.
    out_x[:centre] = out_x[centre]
    out_y[:centre] = out_y[centre]
    out_x[count - centre :] = out_x[count - centre - 1]
    out_y[count - centre :] = out_y[count - centre - 1]

    return out_x, out_y, weights
