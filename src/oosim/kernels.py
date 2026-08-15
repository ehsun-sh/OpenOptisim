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

from dataclasses import dataclass

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


def soliton_peak_power(beta2: float, gamma: float, width: float, order: int = 1) -> float:
    """Peak power of a soliton of the given order [W].

    A sech pulse of width ``T0`` is a soliton of order N when
    ``N**2 = gamma * P0 * T0**2 / |beta2|``. The fundamental (N = 1) is the case
    where the chirp Kerr imposes exactly cancels the chirp dispersion imposes,
    so the pulse propagates unchanged — which makes it the sharpest available
    check that both effects are implemented correctly and with the right signs
    relative to each other.

    Requires anomalous dispersion (``beta2 < 0``); in normal dispersion the two
    effects add instead of cancelling and no bright soliton exists.
    """
    if beta2 >= 0.0:
        raise ValueError(f"bright solitons need anomalous dispersion (beta2 < 0), got {beta2}")
    if gamma <= 0.0:
        raise ValueError(f"gamma must be positive, got {gamma}")
    return order**2 * abs(beta2) / (gamma * width**2)


def soliton_period(beta2: float, width: float) -> float:
    """Soliton period ``z0 = (pi/2) * T0**2 / |beta2|`` [m]."""
    return 0.5 * np.pi * width**2 / abs(beta2)


def attenuation_db_per_m_to_alpha(attenuation_db_per_m: float) -> float:
    """Convert a loss coefficient in dB/m to the power attenuation ``alpha`` [1/m].

    ``P(z) = P(0) * exp(-alpha * z)``, so ``alpha = ln(10)/10 * dB per metre``.
    """
    return attenuation_db_per_m * np.log(10.0) / 10.0


@dataclass(frozen=True)
class PropagationDiagnostics:
    """What the propagator actually did, so accuracy can be audited.

    A split-step result is only as good as its step size, and a fixed-step run
    produces answers that look plausible and are wrong. Reporting the step count
    and the largest nonlinear phase per step means the number can be checked
    rather than trusted.
    """

    steps: int
    distance: float
    shortest_step: float
    longest_step: float
    peak_nonlinear_phase: float
    """Largest nonlinear phase rotation applied in any single step [rad]."""

    def __repr__(self) -> str:
        return (
            f"PropagationDiagnostics({self.steps} steps over {self.distance / 1e3:.1f} km, "
            f"max phase {self.peak_nonlinear_phase:.4f} rad)"
        )


def propagate_ssfm(
    field: np.ndarray,
    sample_rate: float,
    *,
    beta2: float,
    gamma: float,
    alpha: float,
    distance: float,
    max_nonlinear_phase: float = 0.005,
    max_step: float | None = None,
) -> tuple[np.ndarray, PropagationDiagnostics]:
    """Solve the nonlinear Schrödinger equation by symmetric split-step Fourier.

    ``dA/dz = -(alpha/2) A - (i beta2 / 2) d2A/dT2 + i gamma |A|**2 A``

    Each step applies half the linear operator in the frequency domain, the full
    nonlinear phase in the time domain, then the other half linear operator. The
    symmetric ordering makes the local error third order in the step size rather
    than second.

    **The step size is adaptive, and that is not optional.** The nonlinear term
    is a phase rotation proportional to instantaneous power, so a step long
    enough to rotate the peak by an appreciable angle stops commuting with
    dispersion in a way that quietly changes the answer. Steps here are bounded
    so the largest nonlinear rotation per step stays under
    ``max_nonlinear_phase``; the default of 5 mrad is conservative. Because the
    bound is recomputed from the current peak power, steps lengthen naturally as
    the pulse loses power to attenuation.

    Returns the propagated field and :class:`PropagationDiagnostics`.
    """
    if distance < 0.0:
        raise ValueError(f"distance must be non-negative, got {distance}")
    if max_nonlinear_phase <= 0.0:
        raise ValueError(f"max_nonlinear_phase must be positive, got {max_nonlinear_phase}")

    a = field.astype(np.complex128, copy=True)
    if distance == 0.0:
        return a, PropagationDiagnostics(0, 0.0, 0.0, 0.0, 0.0)

    omega = angular_frequency_grid(field.shape[0], sample_rate)
    dispersion_operator = 0.5j * beta2 * omega**2
    ceiling = max_step if max_step is not None else distance

    travelled = 0.0
    steps = 0
    shortest = np.inf
    longest = 0.0
    peak_phase = 0.0

    while travelled < distance:
        remaining = distance - travelled
        step = min(ceiling, remaining)
        peak_power = float(np.max(np.abs(a) ** 2))
        if gamma != 0.0 and peak_power > 0.0:
            step = min(step, max_nonlinear_phase / (abs(gamma) * peak_power))
        # A step can only be shortened to the point where it still advances;
        # without this an extreme peak power would stall the loop.
        step = max(min(step, remaining), remaining * 1e-9)

        half = np.exp(-alpha * step / 4.0 + dispersion_operator * (step / 2.0))
        a = np.fft.ifft(np.fft.fft(a) * half)
        phase = gamma * np.abs(a) ** 2 * step
        a = a * np.exp(1j * phase)
        a = np.fft.ifft(np.fft.fft(a) * half)

        travelled += step
        steps += 1
        shortest = min(shortest, step)
        longest = max(longest, step)
        peak_phase = max(peak_phase, float(np.max(np.abs(phase))))

    return a, PropagationDiagnostics(
        steps=steps,
        distance=distance,
        shortest_step=float(shortest),
        longest_step=longest,
        peak_nonlinear_phase=peak_phase,
    )


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
