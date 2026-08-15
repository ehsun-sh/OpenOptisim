"""Validation of the split-step Fourier propagator and the Kerr nonlinearity.

The soliton test below is the one that matters. Dispersion and self-phase
modulation can each be individually correct while having the wrong sign relative
to each other, and every linear test still passes. A fundamental soliton only
propagates unchanged when the two chirps cancel, so it is sensitive to exactly
the error nothing else here can see.

Reference: G. P. Agrawal, *Nonlinear Fiber Optics*, ch. 4-5.
"""

from __future__ import annotations

import numpy as np
import pytest

from oosim import Band, Graph, SimulationContext
from oosim.analysis import instantaneous_power, peak_power, rms_time_width
from oosim.components import Fiber, GaussianPulse, PowerMeter, SechPulse
from oosim.kernels import (
    PropagationDiagnostics,
    dispersion_to_beta2,
    propagate_dispersion,
    propagate_ssfm,
    soliton_peak_power,
    soliton_period,
)
from oosim.units import w_to_dbm

# A pulse-scale window: 0.39 ps per sample over 400 ps.
PULSE_CTX = SimulationContext(bit_rate=10e9, samples_per_symbol=256, sequence_length=4)

T0_PS = 10.0
D_SMF = 17.0  # ps/(nm*km)
GAMMA_SMF = 1.3  # 1/(W*km)

BETA2_SI = dispersion_to_beta2(D_SMF * 1e-6, 1550e-9)  # negative
GAMMA_SI = GAMMA_SMF * 1e-3
T0_SI = T0_PS * 1e-12


def _propagate(
    source: SechPulse | GaussianPulse,
    *,
    length_km: float,
    dispersion: float = D_SMF,
    nonlinearity: float = GAMMA_SMF,
    attenuation: float = 0.0,
    max_phase: float = 0.005,
) -> tuple[Band, Band, PropagationDiagnostics]:
    """Run one pulse through one fiber. Returns (launched, received, diagnostics)."""
    g = Graph(PULSE_CTX)
    src = g.add(source)
    fiber = g.add(
        Fiber(
            length=length_km,
            attenuation=attenuation,
            dispersion=dispersion,
            nonlinearity=nonlinearity,
            max_nonlinear_phase=max_phase,
        )
    )
    meter = g.add(PowerMeter())
    g.chain(src, fiber, meter)

    results = g.run(keep=[src, fiber])
    return (
        results.port(src, "out").bands[0],
        results.port(fiber, "out").bands[0],
        results.port(fiber, "diagnostics"),
    )


# --------------------------------------------------------------------------
# The split-step machinery, before any nonlinearity
# --------------------------------------------------------------------------


def test_ssfm_without_nonlinearity_equals_the_exact_linear_solution() -> None:
    """With gamma = 0 the two half-steps compose into the full linear operator,
    so the split-step answer is not an approximation — it is the same arithmetic.
    Any discrepancy here is a bug in the stepping, not stepping error.
    """
    rng = np.random.default_rng(0)
    field = rng.normal(size=512) + 1j * rng.normal(size=512)

    exact = propagate_dispersion(field, 2.56e12, BETA2_SI, 40e3)
    stepped, diagnostics = propagate_ssfm(
        field, 2.56e12, beta2=BETA2_SI, gamma=0.0, alpha=0.0, distance=40e3
    )

    np.testing.assert_allclose(stepped, exact, rtol=1e-12, atol=1e-12)
    assert diagnostics.steps == 1


def test_zero_distance_is_a_no_op() -> None:
    field = np.ones(64, dtype=np.complex128)
    out, diagnostics = propagate_ssfm(
        field, 1e12, beta2=BETA2_SI, gamma=GAMMA_SI, alpha=0.0, distance=0.0
    )
    np.testing.assert_array_equal(out, field)
    assert diagnostics.steps == 0


def test_lossless_propagation_conserves_energy() -> None:
    """Kerr is a phase rotation and dispersion is all-pass, so with no loss the
    pulse energy is untouched. A leak here means the FFT normalisation is wrong.
    """
    source = SechPulse(peak_power=w_to_dbm(0.2), width=T0_PS)
    launched, received, _ = _propagate(source, length_km=20.0, attenuation=0.0)

    assert received.average_power() == pytest.approx(launched.average_power(), rel=1e-6)


def test_loss_still_applies_under_nonlinear_propagation() -> None:
    source = SechPulse(peak_power=w_to_dbm(0.05), width=T0_PS)
    launched, received, _ = _propagate(source, length_km=20.0, attenuation=0.2)

    expected = launched.average_power() * 10 ** (-0.2 * 20.0 / 10)
    assert received.average_power() == pytest.approx(expected, rel=1e-4)


# --------------------------------------------------------------------------
# Self-phase modulation on its own
# --------------------------------------------------------------------------


def test_spm_alone_leaves_the_time_domain_intensity_untouched() -> None:
    """Without dispersion, the Kerr effect multiplies the field by a unit-modulus
    phase. It reshapes the spectrum and cannot touch |A(T)| at all — an exact
    property, so it is asserted exactly rather than approximately.
    """
    source = GaussianPulse(peak_power=w_to_dbm(0.5), width=T0_PS)
    launched, received, _ = _propagate(source, length_km=20.0, dispersion=0.0)

    # atol is set relative to the 0.5 W peak: in the far tails the launched
    # pulse underflows to exactly zero while the round trip through the FFT
    # leaves values around 1e-29, and a relative tolerance there is meaningless.
    np.testing.assert_allclose(
        instantaneous_power(received), instantaneous_power(launched), rtol=1e-6, atol=1e-15
    )


def test_spm_broadens_the_spectrum() -> None:
    """The complement of the test above: the intensity is unchanged in time, so
    the effect has to appear in frequency."""
    source = GaussianPulse(peak_power=w_to_dbm(0.5), width=T0_PS)
    launched, received, _ = _propagate(source, length_km=40.0, dispersion=0.0)

    def rms_bandwidth(band: Band) -> float:
        spectrum = np.abs(np.fft.fft(np.asarray(band.Ex, dtype=np.complex128))) ** 2
        f = np.fft.fftfreq(spectrum.size, d=1.0 / band.fs)
        total = spectrum.sum()
        mean = (spectrum * f).sum() / total
        return float(np.sqrt((spectrum * (f - mean) ** 2).sum() / total))

    assert rms_bandwidth(received) > 3 * rms_bandwidth(launched)


def test_nonlinear_phase_grows_with_power_and_distance() -> None:
    """phi_max = gamma * P0 * L, checked through the reported diagnostics."""
    for power_w, length_km in ((0.1, 10.0), (0.2, 10.0), (0.1, 20.0)):
        source = SechPulse(peak_power=w_to_dbm(power_w), width=T0_PS)
        _, _, diagnostics = _propagate(source, length_km=length_km, dispersion=0.0)
        total_phase = diagnostics.peak_nonlinear_phase * diagnostics.steps
        assert total_phase == pytest.approx(GAMMA_SI * power_w * length_km * 1e3, rel=0.02)


# --------------------------------------------------------------------------
# The fundamental soliton
# --------------------------------------------------------------------------


def test_soliton_peak_power_matches_the_defining_relation() -> None:
    p0 = soliton_peak_power(BETA2_SI, GAMMA_SI, T0_SI)
    assert GAMMA_SI * p0 * T0_SI**2 / abs(BETA2_SI) == pytest.approx(1.0)
    # ~167 mW for standard fiber and a 10 ps pulse.
    assert p0 == pytest.approx(0.167, rel=0.02)


def test_normal_dispersion_has_no_bright_soliton() -> None:
    with pytest.raises(ValueError, match="anomalous dispersion"):
        soliton_peak_power(abs(BETA2_SI), GAMMA_SI, T0_SI)


def test_fundamental_soliton_propagates_unchanged() -> None:
    """The central test of this module.

    At N = 1 the chirp Kerr imposes exactly cancels the chirp dispersion
    imposes, so |A(T)| is invariant over any distance. Get either sign wrong and
    the two add instead of cancelling: the pulse broadens or collapses at once.
    Every linear test in the suite still passes under that error.
    """
    p0 = soliton_peak_power(BETA2_SI, GAMMA_SI, T0_SI)
    distance_km = soliton_period(BETA2_SI, T0_SI) / 1e3

    source = SechPulse(peak_power=w_to_dbm(p0), width=T0_PS)
    launched, received, diagnostics = _propagate(source, length_km=distance_km)

    assert diagnostics.steps > 100, "step control produced a suspiciously coarse run"
    assert rms_time_width(received) == pytest.approx(rms_time_width(launched), rel=0.01)
    assert peak_power(received) == pytest.approx(peak_power(launched), rel=0.01)
    np.testing.assert_allclose(
        instantaneous_power(received), instantaneous_power(launched), rtol=0.02, atol=1e-4
    )


def test_soliton_survives_several_periods() -> None:
    """Stepping error accumulates, so holding shape over one period is weaker
    evidence than holding it over four."""
    p0 = soliton_peak_power(BETA2_SI, GAMMA_SI, T0_SI)
    distance_km = 4 * soliton_period(BETA2_SI, T0_SI) / 1e3

    source = SechPulse(peak_power=w_to_dbm(p0), width=T0_PS)
    launched, received, _ = _propagate(source, length_km=distance_km)

    assert rms_time_width(received) == pytest.approx(rms_time_width(launched), rel=0.02)


def test_the_same_pulse_without_nonlinearity_broadens() -> None:
    """A control for the test above: the soliton's invariance is the nonlinearity
    doing work, not the propagator failing to do anything.

    The comparison that matters is between the two runs. Over one soliton period
    the identical pulse spreads by about 40% on dispersion alone, and holds its
    width to within 1% once the Kerr effect is switched on.
    """
    p0 = soliton_peak_power(BETA2_SI, GAMMA_SI, T0_SI)
    distance_km = soliton_period(BETA2_SI, T0_SI) / 1e3

    launched, linear, _ = _propagate(
        SechPulse(peak_power=w_to_dbm(p0), width=T0_PS),
        length_km=distance_km,
        nonlinearity=0.0,
    )
    _, soliton, _ = _propagate(
        SechPulse(peak_power=w_to_dbm(p0), width=T0_PS), length_km=distance_km
    )

    assert rms_time_width(linear) > 1.3 * rms_time_width(launched)
    assert rms_time_width(linear) > 1.3 * rms_time_width(soliton)


def test_the_same_pulse_without_dispersion_keeps_its_shape_for_another_reason() -> None:
    """SPM alone also preserves |A(T)|, so shape invariance by itself proves
    nothing. It is invariance *with both effects active* that is the result."""
    p0 = soliton_peak_power(BETA2_SI, GAMMA_SI, T0_SI)
    source = SechPulse(peak_power=w_to_dbm(p0), width=T0_PS)
    launched, received, _ = _propagate(source, length_km=7.25, dispersion=0.0)

    assert rms_time_width(received) == pytest.approx(rms_time_width(launched), rel=1e-6)


def test_a_higher_order_soliton_returns_to_its_shape_after_one_period() -> None:
    """An N = 2 soliton compresses and splits within a period and recovers at the
    end of it. It exercises the propagator far harder than N = 1, where the
    solution is stationary and a sluggish integrator could still look right.
    """
    p0 = soliton_peak_power(BETA2_SI, GAMMA_SI, T0_SI, order=2)
    period_km = soliton_period(BETA2_SI, T0_SI) / 1e3
    source = SechPulse(peak_power=w_to_dbm(p0), width=T0_PS)

    launched, mid, _ = _propagate(source, length_km=period_km / 2, max_phase=0.002)
    _, full, _ = _propagate(
        SechPulse(peak_power=w_to_dbm(p0), width=T0_PS),
        length_km=period_km,
        max_phase=0.002,
    )

    # Compressed at half a period, recovered at a full one.
    assert peak_power(mid) > 3 * peak_power(launched)
    assert peak_power(full) == pytest.approx(peak_power(launched), rel=0.05)


# --------------------------------------------------------------------------
# Step control
# --------------------------------------------------------------------------


def test_the_answer_converges_as_the_step_bound_tightens() -> None:
    """Halving the allowed nonlinear phase per step must barely move the result.
    If it does move, the default is too coarse and every nonlinear number the
    tool reports is suspect.
    """
    p0 = soliton_peak_power(BETA2_SI, GAMMA_SI, T0_SI, order=2)
    distance_km = soliton_period(BETA2_SI, T0_SI) / 1e3 / 2

    coarse = _propagate(
        SechPulse(peak_power=w_to_dbm(p0), width=T0_PS),
        length_km=distance_km,
        max_phase=0.005,
    )[1]
    fine = _propagate(
        SechPulse(peak_power=w_to_dbm(p0), width=T0_PS),
        length_km=distance_km,
        max_phase=0.0005,
    )[1]

    np.testing.assert_allclose(
        instantaneous_power(coarse), instantaneous_power(fine), rtol=0.01, atol=1e-3
    )


def test_more_power_buys_more_steps() -> None:
    """The step bound is a nonlinear-phase budget, so a stronger pulse must be
    integrated more finely for the same span."""
    counts = []
    for power_w in (0.05, 0.2):
        _, _, diagnostics = _propagate(
            SechPulse(peak_power=w_to_dbm(power_w), width=T0_PS), length_km=10.0
        )
        counts.append(diagnostics.steps)
    assert counts[1] > 3 * counts[0]


def test_attenuation_lets_the_steps_lengthen() -> None:
    """As the pulse loses power the nonlinear phase per metre falls, so a lossy
    span needs fewer steps than a lossless one of the same length."""
    _, _, lossless = _propagate(
        SechPulse(peak_power=w_to_dbm(0.2), width=T0_PS), length_km=60.0, attenuation=0.0
    )
    _, _, lossy = _propagate(
        SechPulse(peak_power=w_to_dbm(0.2), width=T0_PS), length_km=60.0, attenuation=0.25
    )

    assert lossy.steps < lossless.steps
    assert lossy.longest_step > lossy.shortest_step


def test_diagnostics_report_what_the_propagator_did() -> None:
    source = SechPulse(peak_power=w_to_dbm(0.1), width=T0_PS)
    _, _, diagnostics = _propagate(source, length_km=25.0)

    assert diagnostics.distance == pytest.approx(25e3)
    assert diagnostics.peak_nonlinear_phase <= 0.005 + 1e-12
    assert diagnostics.shortest_step > 0


def test_a_nonsensical_step_bound_is_rejected() -> None:
    with pytest.raises(ValueError, match="max_nonlinear_phase must be positive"):
        propagate_ssfm(
            np.ones(16, dtype=np.complex128),
            1e12,
            beta2=BETA2_SI,
            gamma=GAMMA_SI,
            alpha=0.0,
            distance=1e3,
            max_nonlinear_phase=0.0,
        )


def test_linear_fiber_still_takes_the_exact_path() -> None:
    """Leaving nonlinearity at zero must not silently switch to an approximate
    solver: the closed-form answer is available and is what should be used."""
    g = Graph(PULSE_CTX)
    source = g.add(GaussianPulse(peak_power=0.0, width=T0_PS))
    fiber = g.add(Fiber(length=80.0, attenuation=0.0, dispersion=D_SMF, nonlinearity=0.0))
    meter = g.add(PowerMeter())
    g.chain(source, fiber, meter)

    assert g.run()[fiber].steps == 0
