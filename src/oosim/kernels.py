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

    differential_group_delay: float = 0.0
    """DGD realised on this run [s]. PMD is random, so this differs run to run;
    it is reported because the value a result depends on should be visible."""

    def __repr__(self) -> str:
        pmd = (
            f", DGD {self.differential_group_delay * 1e12:.2f} ps"
            if self.differential_group_delay
            else ""
        )
        return (
            f"PropagationDiagnostics({self.steps} steps over {self.distance / 1e3:.1f} km, "
            f"max phase {self.peak_nonlinear_phase:.4f} rad{pmd})"
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


# --------------------------------------------------------------------------
# Polarization-mode dispersion
# --------------------------------------------------------------------------

#: Ratio of the mean square DGD to the squared mean, for a Maxwellian
#: distribution. It is a pure number, so measuring it is a shape test that does
#: not depend on getting the scale right.
MAXWELLIAN_MOMENT_RATIO = 3.0 * np.pi / 8.0


@dataclass(frozen=True)
class PMDSection:
    """One birefringent waveplate: a fixed delay between two random axes."""

    unitary: np.ndarray
    """2x2 unitary rotating into this section's principal states."""

    dgd: float
    """Differential group delay of this section [s]."""


def random_unitary_2x2(rng: np.random.Generator) -> np.ndarray:
    """A Haar-uniform SU(2) matrix, built from a random unit quaternion.

    Uniformity matters: birefringence axes in real fiber have no preferred
    orientation, and sampling them non-uniformly would bias the DGD statistics
    the whole model exists to reproduce.
    """
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    return np.array(
        [[q[0] + 1j * q[1], q[2] + 1j * q[3]], [-q[2] + 1j * q[3], q[0] - 1j * q[1]]],
        dtype=np.complex128,
    )


def random_pmd_sections(
    mean_dgd: float, sections: int, rng: np.random.Generator
) -> tuple[PMDSection, ...]:
    """Build a waveplate chain whose mean DGD is ``mean_dgd``.

    PMD is not a fixed impairment. Birefringence varies randomly along real
    fiber and drifts with temperature, so the differential group delay is a
    *random variable* with a Maxwellian distribution — which is why PMD is
    quoted as a coefficient in ps/sqrt(km) and why outage probability, rather
    than a worst case, is what gets designed against.

    Concatenating ``N`` randomly oriented waveplates of equal delay reproduces
    that distribution. Each section's delay follows from the second moment
    adding while the mean does not::

        <DGD> = section_dgd * sqrt(8N / (3*pi))

    so a target mean fixes the per-section delay. Around 50 sections is enough
    for the statistics to converge.
    """
    if sections < 1:
        raise ValueError(f"sections must be >= 1, got {sections}")
    if mean_dgd < 0.0:
        raise ValueError(f"mean_dgd must be non-negative, got {mean_dgd}")

    section_dgd = mean_dgd * np.sqrt(3.0 * np.pi / (8.0 * sections))
    return tuple(
        PMDSection(unitary=random_unitary_2x2(rng), dgd=float(section_dgd)) for _ in range(sections)
    )


def pmd_jones_matrix(sections: tuple[PMDSection, ...], omega: float) -> np.ndarray:
    """Total Jones transfer matrix of a waveplate chain at angular frequency ``omega``."""
    total = np.eye(2, dtype=np.complex128)
    for section in sections:
        phase = np.exp(0.5j * omega * section.dgd)
        delay = np.array([[phase, 0.0], [0.0, np.conj(phase)]], dtype=np.complex128)
        total = section.unitary @ delay @ total
    return total


def differential_group_delay(
    sections: tuple[PMDSection, ...], *, probe_spacing: float = 2.0 * np.pi * 1e9
) -> float:
    """Measure the chain's DGD by Jones matrix eigenanalysis [s].

    Takes the transfer matrix at two nearby frequencies, forms
    ``J(w2) @ inv(J(w1))``, and reads the delay off the argument of its
    eigenvalue ratio. This is the method a bench PMD analyser uses, which is a
    good reason to prefer it over reading the number back out of the parameters
    that generated it: it measures the chain rather than trusting it.

    ``probe_spacing`` must be small enough that ``probe_spacing * DGD`` stays
    below pi, or the phase wraps and the answer folds.
    """
    if not sections:
        return 0.0

    j1 = pmd_jones_matrix(sections, -probe_spacing / 2.0)
    j2 = pmd_jones_matrix(sections, probe_spacing / 2.0)
    eigenvalues = np.linalg.eigvals(j2 @ np.linalg.inv(j1))
    difference = np.angle(eigenvalues[0] / eigenvalues[1])
    return float(abs(difference) / probe_spacing)


def apply_pmd(
    ex: np.ndarray, ey: np.ndarray, sample_rate: float, sections: tuple[PMDSection, ...]
) -> tuple[np.ndarray, np.ndarray]:
    """Propagate a Jones vector through a waveplate chain.

    Applied in the frequency domain, section by section, in the same order
    :func:`pmd_jones_matrix` multiplies them — so what a signal experiences and
    what the DGD measurement reports are the same chain.
    """
    if not sections:
        return ex.astype(np.complex128, copy=True), ey.astype(np.complex128, copy=True)

    omega = angular_frequency_grid(ex.shape[0], sample_rate)
    spectrum_x = np.fft.fft(ex.astype(np.complex128))
    spectrum_y = np.fft.fft(ey.astype(np.complex128))

    for section in sections:
        phase = np.exp(0.5j * omega * section.dgd)
        delayed_x = spectrum_x * phase
        delayed_y = spectrum_y * np.conj(phase)
        u = section.unitary
        spectrum_x = u[0, 0] * delayed_x + u[0, 1] * delayed_y
        spectrum_y = u[1, 0] * delayed_x + u[1, 1] * delayed_y

    return np.fft.ifft(spectrum_x), np.fft.ifft(spectrum_y)


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
