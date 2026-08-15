"""Validation of polarization-mode dispersion.

PMD is a statistical impairment, so most of these are statistical tests: a
single realisation says nothing, and the thing to check is that the ensemble has
the right distribution. The shape test is the sharper of the two — the ratio
``<DGD^2>/<DGD>^2 = 3*pi/8`` is a pure number, so it holds or fails
independently of whether the scale is right.

Reference: G. P. Agrawal, *Fiber-Optic Communication Systems*, §2.3.5.
"""

from __future__ import annotations

import math
from functools import cache

import numpy as np
import pytest

from oosim import Graph, SimulationContext
from oosim.analysis import instantaneous_power
from oosim.components import Fiber, GaussianPulse, PowerMeter
from oosim.kernels import (
    MAXWELLIAN_MOMENT_RATIO,
    PMDSection,
    apply_pmd,
    differential_group_delay,
    random_pmd_sections,
    random_unitary_2x2,
)

REALISATIONS = 3000


@cache
def _dgd_ensemble(
    mean_dgd: float, sections: int = 60, realisations: int = REALISATIONS
) -> np.ndarray:
    """Measured DGD over many independent realisations [s].

    No signal is propagated: the chain's Jones matrix is enough, which is what
    makes a three-thousand-sample ensemble affordable at all. Cached because
    several tests interrogate different moments of the same ensemble, and
    rebuilding it each time is the dominant cost of this module.
    """
    rng = np.random.default_rng(20260815)
    return np.array(
        [
            differential_group_delay(random_pmd_sections(mean_dgd, sections, rng))
            for _ in range(realisations)
        ]
    )


# --------------------------------------------------------------------------
# The waveplate chain
# --------------------------------------------------------------------------


def test_random_rotations_are_unitary() -> None:
    """Birefringence rotates the polarization state; it cannot create or destroy
    power. A non-unitary rotation would leak energy on every section."""
    rng = np.random.default_rng(1)
    for _ in range(100):
        u = random_unitary_2x2(rng)
        np.testing.assert_allclose(u.conj().T @ u, np.eye(2), atol=1e-12)


def test_a_single_section_has_exactly_its_own_dgd() -> None:
    """With one waveplate the measurement is deterministic, so the eigenanalysis
    can be checked against an exact answer before being trusted on a chain."""
    for dgd_ps in (1.0, 5.0, 20.0):
        section = PMDSection(unitary=np.eye(2, dtype=np.complex128), dgd=dgd_ps * 1e-12)
        assert differential_group_delay((section,)) == pytest.approx(dgd_ps * 1e-12, rel=1e-9)


def test_an_empty_chain_has_no_dgd() -> None:
    assert differential_group_delay(()) == 0.0


def test_section_count_must_be_positive() -> None:
    with pytest.raises(ValueError, match="sections must be"):
        random_pmd_sections(1e-12, 0, np.random.default_rng(0))


# --------------------------------------------------------------------------
# Maxwellian statistics
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mean_dgd_ps", [1.0, 5.0, 20.0])
def test_the_ensemble_mean_dgd_is_what_was_asked_for(mean_dgd_ps: float) -> None:
    """The per-section delay is derived from the target mean, so this checks the
    ``sqrt(8N/3pi)`` factor that relates them."""
    ensemble = _dgd_ensemble(mean_dgd_ps * 1e-12)
    assert ensemble.mean() == pytest.approx(mean_dgd_ps * 1e-12, rel=0.04)


def test_the_dgd_distribution_is_maxwellian() -> None:
    """``<DGD^2> / <DGD>^2 = 3*pi/8``, a pure number.

    This is the sharper of the two statistical tests: being dimensionless, it
    holds or fails without reference to the scale, so it catches a chain that has
    the right average delay for the wrong reason.
    """
    ensemble = _dgd_ensemble(10e-12)
    ratio = float((ensemble**2).mean() / ensemble.mean() ** 2)
    assert ratio == pytest.approx(MAXWELLIAN_MOMENT_RATIO, rel=0.03)
    assert pytest.approx(1.178, rel=1e-3) == MAXWELLIAN_MOMENT_RATIO


def test_the_distribution_has_the_maxwellian_spread() -> None:
    """std/mean = sqrt(3pi/8 - 1) ~ 0.42 — a wide distribution, which is the
    whole reason PMD is designed against an outage probability."""
    ensemble = _dgd_ensemble(10e-12)
    expected = math.sqrt(MAXWELLIAN_MOMENT_RATIO - 1.0)
    assert ensemble.std() / ensemble.mean() == pytest.approx(expected, rel=0.05)


def test_dgd_can_greatly_exceed_its_mean() -> None:
    """The tail is what causes outages: a link designed for the average DGD will
    fail some of the time."""
    ensemble = _dgd_ensemble(10e-12)
    assert ensemble.max() > 2.5 * ensemble.mean()


def test_the_statistics_do_not_depend_on_the_section_count() -> None:
    """The section count is a numerical choice, not a physical one, so the
    distribution it produces must not move with it."""
    ratios = []
    for sections in (30, 60, 120):
        ensemble = _dgd_ensemble(10e-12, sections=sections, realisations=1500)
        ratios.append(float((ensemble**2).mean() / ensemble.mean() ** 2))
    for ratio in ratios:
        assert ratio == pytest.approx(MAXWELLIAN_MOMENT_RATIO, rel=0.05)


# --------------------------------------------------------------------------
# In the fiber
# --------------------------------------------------------------------------


def _span_dgd(pmd_coefficient: float, length_km: float, seed: int) -> float:
    ctx = SimulationContext(bit_rate=10e9, samples_per_symbol=8, sequence_length=16, seed=seed)
    g = Graph(ctx)
    source = g.add(GaussianPulse(peak_power=0.0, width=20.0, label="src"))
    fiber = g.add(
        Fiber(
            length=length_km,
            attenuation=0.0,
            pmd_coefficient=pmd_coefficient,
            label="fiber",
        )
    )
    meter = g.add(PowerMeter(label="meter"))
    g.chain(source, fiber, meter)
    return float(g.run()[fiber].differential_group_delay)


def test_mean_dgd_grows_as_the_square_root_of_length() -> None:
    """A random walk, not a sum: four times the fiber is twice the delay.

    Getting this wrong — accumulating DGD linearly — would make PMD look like a
    hard reach limit instead of the statistical one it is.
    """
    fiber = Fiber(length=400.0, attenuation=0.0, pmd_coefficient=0.1)
    assert fiber.mean_dgd() == pytest.approx(0.1e-12 / math.sqrt(1e3) * math.sqrt(400e3))

    short = Fiber(length=100.0, pmd_coefficient=0.1).mean_dgd()
    long = Fiber(length=400.0, pmd_coefficient=0.1).mean_dgd()
    assert long / short == pytest.approx(2.0)


def test_the_span_realisation_varies_with_the_seed_and_repeats_with_it() -> None:
    """PMD is drawn, not fixed. Two seeds must differ; one seed must not."""
    first = _span_dgd(0.5, 100.0, seed=1)
    again = _span_dgd(0.5, 100.0, seed=1)
    other = _span_dgd(0.5, 100.0, seed=2)

    assert first == again
    assert first != other
    assert first > 0.0


def test_a_fiber_without_pmd_reports_no_dgd() -> None:
    assert _span_dgd(0.0, 100.0, seed=1) == 0.0


def test_pmd_conserves_power() -> None:
    """Birefringence redistributes light between polarizations; it cannot lose any."""
    ctx = SimulationContext(bit_rate=10e9, samples_per_symbol=64, sequence_length=8, seed=3)
    g = Graph(ctx)
    source = g.add(GaussianPulse(peak_power=0.0, width=20.0, label="src"))
    fiber = g.add(Fiber(length=100.0, attenuation=0.0, pmd_coefficient=2.0, label="fiber"))
    meter = g.add(PowerMeter(label="meter"))
    g.chain(source, fiber, meter)

    results = g.run(keep=[source])
    launched = results.port(source, "out").signal_power()
    assert results[meter].power_w == pytest.approx(launched, rel=1e-5)


def test_pmd_moves_power_into_the_other_polarization() -> None:
    """The launched pulse is entirely in X. Birefringence must not leave it there."""
    ctx = SimulationContext(bit_rate=10e9, samples_per_symbol=64, sequence_length=8, seed=5)
    g = Graph(ctx)
    source = g.add(GaussianPulse(peak_power=0.0, width=20.0, label="src"))
    fiber = g.add(Fiber(length=100.0, attenuation=0.0, pmd_coefficient=2.0, label="fiber"))
    meter = g.add(PowerMeter(label="meter"))
    g.chain(source, fiber, meter)

    band = g.run(keep=[fiber]).port(fiber, "out").bands[0]
    power_y = float(np.mean(np.abs(np.asarray(band.Ey)) ** 2))
    assert power_y > 0.0


def test_a_single_waveplate_splits_a_diagonal_pulse_by_exactly_its_dgd() -> None:
    """The deterministic case, where the effect can be measured rather than
    inferred: launch equally into both principal states and the two copies
    arrive separated by the DGD.
    """
    n, fs = 4096, 2.56e12
    dgd = 40e-12
    t = np.arange(n) / fs
    envelope = np.exp(-(((t - t.mean()) / 5e-12) ** 2) / 2)

    ex, ey = apply_pmd(
        envelope.astype(np.complex128) / math.sqrt(2),
        envelope.astype(np.complex128) / math.sqrt(2),
        fs,
        (PMDSection(unitary=np.eye(2, dtype=np.complex128), dgd=dgd),),
    )

    peak_x = float(t[np.argmax(np.abs(ex) ** 2)])
    peak_y = float(t[np.argmax(np.abs(ey) ** 2)])
    assert abs(peak_x - peak_y) == pytest.approx(dgd, rel=0.02)


def test_dgd_spreads_the_detected_pulse() -> None:
    """The consequence that matters at the receiver: the two polarization
    components arrive at different times, so the total intensity is broader."""
    ctx = SimulationContext(bit_rate=10e9, samples_per_symbol=256, sequence_length=4, seed=11)

    widths = []
    for pmd in (0.0, 20.0):
        g = Graph(ctx)
        source = g.add(GaussianPulse(peak_power=0.0, width=5.0, label="src"))
        fiber = g.add(Fiber(length=100.0, attenuation=0.0, pmd_coefficient=pmd, label="fiber"))
        meter = g.add(PowerMeter(label="meter"))
        g.chain(source, fiber, meter)

        band = g.run(keep=[fiber]).port(fiber, "out").bands[0]
        power = instantaneous_power(band)
        t = np.arange(band.num_samples) / band.fs
        mean = (power * t).sum() / power.sum()
        widths.append(math.sqrt(float((power * (t - mean) ** 2).sum() / power.sum())))

    assert widths[1] > 1.2 * widths[0]
