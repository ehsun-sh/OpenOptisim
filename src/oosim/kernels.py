"""Numerical kernels.

Everything computationally heavy goes through this module. It is deliberately
narrow — a handful of array-in/array-out functions with no knowledge of
components, graphs, or units — so that the back-end can change without touching
any physics code above it. Today that back-end is NumPy; CuPy (`cupy.fft` is a
drop-in for `numpy.fft`) and a native module are the intended next options.

**FFT library.** `numpy.fft` uses pocketfft (BSD). FFTW is *not* used and must
not be introduced: it is GPL-2.0-or-later, and linking it — directly or through
`pyFFTW` — would relicense the whole project.
"""

from __future__ import annotations

import numpy as np

from .units import C_LIGHT


def angular_frequency_grid(num_samples: int, sample_rate: float) -> np.ndarray:
    """Angular frequency offsets from the band centre [rad/s], in FFT order.

    Returned in `numpy.fft` output order (positive frequencies first, then
    negative), so it multiplies an un-shifted spectrum directly.
    """
    return 2.0 * np.pi * np.fft.fftfreq(num_samples, d=1.0 / sample_rate)


def dispersion_to_beta2(dispersion: float, wavelength: float) -> float:
    """Convert the dispersion parameter D [s/m²] to the GVD parameter β₂ [s²/m].

    ``beta2 = -D * lambda**2 / (2*pi*c)``

    The sign matters: standard single-mode fiber has D > 0 at 1550 nm and
    therefore β₂ < 0 (anomalous dispersion).
    """
    return -dispersion * wavelength**2 / (2.0 * np.pi * C_LIGHT)


def propagate_dispersion(
    field: np.ndarray, sample_rate: float, beta2: float, distance: float
) -> np.ndarray:
    """Propagate a complex envelope through pure group-velocity dispersion.

    Solves the linear part of the NLSE, ``dA/dz = -(i*beta2/2) * d2A/dT2``, which
    in the frequency domain is an exact all-pass phase rotation::

        A(z, w) = A(0, w) * exp(i * beta2 * w**2 * z / 2)

    Because the transfer function has unit magnitude, this conserves energy
    exactly (up to floating-point) and is exactly invertible by propagating
    ``-distance`` — both of which are asserted in the test suite.

    Only β₂ is modelled, so the result is insensitive to the sign convention of
    the Fourier transform (ω appears squared). That stops being true as soon as
    β₃ or a group-delay term is added, so this note is worth keeping.

    The sign of β₂ itself, however, is *not* free — and the unchirped broadening
    formula cannot detect an error in it, being even in β₂. Only the chirped case
    can, which is why ``test_chirped_pulse_compresses_before_broadening`` exists.

    The phase argument reaches thousands of radians over a realistic span, so the
    transform runs in double precision regardless of the storage precision and
    the caller casts the result back. Correctness first; if profiling later shows
    this matters, the precision policy belongs here, in one place.
    """
    if distance == 0.0 or beta2 == 0.0:
        return field.astype(np.complex128, copy=True)

    omega = angular_frequency_grid(field.shape[0], sample_rate)
    transfer = np.exp(0.5j * beta2 * omega**2 * distance)
    return np.fft.ifft(np.fft.fft(field.astype(np.complex128)) * transfer)


def gaussian_lowpass_response(frequency: np.ndarray, bandwidth: float) -> np.ndarray:
    """Amplitude response of a Gaussian low-pass with 3 dB power bandwidth ``bandwidth``.

    ``|H(f)|**2 = exp(-ln2 * (f/B)**2)``, which is exactly 1/2 at ``f = B`` — that
    identity is the definition of the 3 dB point and is asserted in the tests.
    """
    if bandwidth <= 0.0:
        raise ValueError(f"bandwidth must be positive, got {bandwidth}")
    return np.exp(-0.5 * np.log(2.0) * (frequency / bandwidth) ** 2)


def gaussian_noise_bandwidth(bandwidth: float) -> float:
    """Equivalent noise bandwidth of :func:`gaussian_lowpass_response` [Hz].

    ``B_n = integral of |H(f)|**2 df = B * sqrt(pi / (4 ln2)) ~ 1.0645 * B``.

    Having this in closed form is what makes the filter's effect on noise
    checkable by arithmetic rather than by eyeballing a variance.
    """
    return bandwidth * float(np.sqrt(np.pi / (4.0 * np.log(2.0))))


def lowpass_filter(samples: np.ndarray, sample_rate: float, bandwidth: float) -> np.ndarray:
    """Zero-phase Gaussian low-pass filter of a real waveform.

    The response is real and even, so the filter has no phase and no group delay:
    the output stays aligned with the input and nothing downstream has to
    compensate a delay. That is not causal, which a physical receiver is — a
    causal Bessel model, with the group delay that comes with it, is a later
    refinement and belongs in this same function.

    Filtering is circular, since the window is treated as periodic. The impulse
    response spans a couple of symbols, so a few symbols at each end of the
    window are contaminated by the wrap; analysis blocks drop them.
    """
    n = samples.shape[0]
    spectrum = np.fft.rfft(samples.astype(np.float64))
    response = gaussian_lowpass_response(np.fft.rfftfreq(n, d=1.0 / sample_rate), bandwidth)
    return np.fft.irfft(spectrum * response, n)
