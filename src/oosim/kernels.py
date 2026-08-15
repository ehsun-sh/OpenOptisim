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
