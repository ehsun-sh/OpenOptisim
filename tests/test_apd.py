"""Validation of the avalanche photodiode.

The result worth having here is that the optimum gain exists and is interior.
An APD is a trade: multiplication lifts the signal above the load's thermal
noise, but the multiplication is random, so it amplifies shot noise faster than
it amplifies the signal. Nothing in the code searches for that optimum — it
falls out of ``F(M)`` and the noise arithmetic, which is why finding it is a
test of the model rather than of a formula.

Reference: G. P. Agrawal, *Fiber-Optic Communication Systems*, §4.4.
"""

from __future__ import annotations

import numpy as np
import pytest

from oosim import Graph, SimulationContext
from oosim.components import APDPhotodiode, CWLaser, PINPhotodiode
from oosim.units import K_BOLTZMANN, Q_ELECTRON, dbm_to_w

NOISE_CTX = SimulationContext(bit_rate=10e9, samples_per_symbol=16, sequence_length=4096, seed=99)


def _detect(
    power_dbm: float,
    *,
    gain: float = 10.0,
    ionization_ratio: float = 0.3,
    responsivity: float = 0.8,
    shot_noise: bool = False,
    thermal_noise: bool = False,
    load_resistance: float = 50.0,
    ctx: SimulationContext | None = None,
) -> np.ndarray:
    ctx = ctx or SimulationContext(bit_rate=10e9, samples_per_symbol=8, sequence_length=64)
    g = Graph(ctx)
    laser = g.add(CWLaser(power=power_dbm, wavelength=1550.0, label="laser"))
    apd = g.add(
        APDPhotodiode(
            gain=gain,
            ionization_ratio=ionization_ratio,
            responsivity=responsivity,
            load_resistance=load_resistance,
            shot_noise=shot_noise,
            thermal_noise=thermal_noise,
            label="apd",
        )
    )
    g.chain(laser, apd)
    return np.asarray(g.run()[apd].samples, dtype=np.float64)


# --------------------------------------------------------------------------
# Multiplication
# --------------------------------------------------------------------------


@pytest.mark.parametrize("gain", [1.0, 5.0, 10.0, 50.0])
def test_photocurrent_is_multiplied_by_the_avalanche_gain(gain: float) -> None:
    current = _detect(-20.0, gain=gain)
    assert current.mean() == pytest.approx(gain * 0.8 * dbm_to_w(-20.0), rel=1e-4)


def test_excess_noise_factor_matches_the_definition() -> None:
    """``F(M) = k*M + (2 - 1/M)(1 - k)``."""
    for gain, k in ((10.0, 0.3), (20.0, 0.02), (5.0, 0.5)):
        apd = APDPhotodiode(gain=gain, ionization_ratio=k)
        expected = k * gain + (2.0 - 1.0 / gain) * (1.0 - k)
        assert apd.excess_noise_factor() == pytest.approx(expected, rel=1e-12)


def test_excess_noise_is_unity_at_unit_gain_for_any_material() -> None:
    """F(1) = 1 identically: with no multiplication there is nothing random about
    it, whatever the material's ionisation ratio."""
    for k in (0.0, 0.02, 0.3, 0.5, 1.0):
        assert APDPhotodiode(gain=1.0, ionization_ratio=k).excess_noise_factor() == pytest.approx(
            1.0
        )


def test_a_quieter_material_has_a_lower_excess_noise_factor() -> None:
    """Silicon (k ~ 0.02) multiplies far more quietly than InGaAs (k ~ 0.3), which
    is why silicon APDs are used where the wavelength allows it."""
    silicon = APDPhotodiode(gain=20.0, ionization_ratio=0.02).excess_noise_factor()
    ingaas = APDPhotodiode(gain=20.0, ionization_ratio=0.3).excess_noise_factor()
    assert silicon < ingaas


# --------------------------------------------------------------------------
# An APD at unit gain is a PIN
# --------------------------------------------------------------------------


def _detector_current(detector: PINPhotodiode, seed: int = 4) -> np.ndarray:
    ctx = SimulationContext(bit_rate=10e9, samples_per_symbol=8, sequence_length=4096, seed=seed)
    g = Graph(ctx)
    laser = g.add(CWLaser(power=-15.0, label="laser"))
    g.add(detector)
    g.chain(laser, detector)
    return np.asarray(g.run()[detector].samples, dtype=np.float64)


def test_at_unit_gain_the_apd_is_bit_for_bit_a_pin() -> None:
    """With the noise off, the two detectors compute identical samples.

    Shared code makes this true by construction, which is the point: the two
    cannot drift apart, and the PIN's validated relations carry over unchanged.
    """
    pin = _detector_current(
        PINPhotodiode(responsivity=0.8, shot_noise=False, thermal_noise=False, label="d")
    )
    apd = _detector_current(
        APDPhotodiode(gain=1.0, responsivity=0.8, shot_noise=False, thermal_noise=False, label="d")
    )
    np.testing.assert_array_equal(apd, pin)


def test_at_unit_gain_the_apd_has_the_same_noise_as_a_pin() -> None:
    """The realisations differ — the two classes seed their streams under their
    own names, which is correct — so this compares the statistics, not the
    samples."""
    pin = _detector_current(PINPhotodiode(responsivity=0.8, label="d"))
    apd = _detector_current(APDPhotodiode(gain=1.0, responsivity=0.8, label="d"))

    assert apd.var() == pytest.approx(pin.var(), rel=0.05)
    assert apd.mean() == pytest.approx(pin.mean(), rel=1e-3)


# --------------------------------------------------------------------------
# Noise
# --------------------------------------------------------------------------


@pytest.mark.parametrize("gain", [1.0, 5.0, 20.0])
def test_shot_noise_variance_scales_as_gain_squared_times_excess_noise(gain: float) -> None:
    """``sigma^2 = 2 q I_primary M^2 F B`` — the ``M**2 F`` is what makes the
    avalanche a trade rather than a free win."""
    current = _detect(-20.0, gain=gain, shot_noise=True, ctx=NOISE_CTX)

    primary = 0.8 * dbm_to_w(-20.0)
    apd = APDPhotodiode(gain=gain, ionization_ratio=0.3)
    expected = (
        2.0
        * Q_ELECTRON
        * primary
        * gain**2
        * apd.excess_noise_factor()
        * (NOISE_CTX.sample_rate / 2.0)
    )
    assert current.var() == pytest.approx(expected, rel=0.05)


def test_thermal_noise_is_untouched_by_the_avalanche() -> None:
    """Thermal noise arises in the load, after the multiplication, so gain cannot
    change it. That asymmetry is precisely what an APD exploits."""
    variances = [
        _detect(-30.0, gain=gain, thermal_noise=True, ctx=NOISE_CTX).var()
        for gain in (1.0, 10.0, 50.0)
    ]
    expected = 4.0 * K_BOLTZMANN * 300.0 * (NOISE_CTX.sample_rate / 2.0) / 50.0
    for variance in variances:
        assert variance == pytest.approx(expected, rel=0.05)


# --------------------------------------------------------------------------
# The optimum gain
# --------------------------------------------------------------------------


def _snr(gain: float, power_dbm: float = -30.0, ionization_ratio: float = 0.3) -> float:
    """Signal-to-noise ratio at the detector output, in power terms."""
    current = _detect(
        power_dbm,
        gain=gain,
        ionization_ratio=ionization_ratio,
        shot_noise=True,
        thermal_noise=True,
        load_resistance=1000.0,
        ctx=NOISE_CTX,
    )
    return float(current.mean() ** 2 / current.var())


def test_snr_has_an_interior_optimum_in_the_avalanche_gain() -> None:
    """The central result.

    Signal power grows as M^2, shot noise as M^2*F, and thermal noise not at all.
    Below the optimum the receiver is thermal-limited and more gain helps; above
    it, multiplication noise costs more than it buys. Nothing in the code looks
    for this — it falls out of F(M) and the noise arithmetic.
    """
    gains = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0]
    snrs = [_snr(gain) for gain in gains]

    best = int(np.argmax(snrs))
    assert 0 < best < len(gains) - 1, (
        f"optimum is at an endpoint: {list(zip(gains, snrs, strict=True))}"
    )
    assert snrs[best] > 2 * snrs[0], "the avalanche should buy a real improvement"
    assert snrs[-1] < snrs[best], "excess noise should eventually dominate"


def test_a_quieter_material_tolerates_more_gain() -> None:
    """Lower k means F rises more slowly, so the optimum moves to higher M — the
    reason silicon APDs run at gains an InGaAs device could not."""

    def optimum(k: float) -> float:
        gains = [2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0]
        return gains[int(np.argmax([_snr(g, ionization_ratio=k) for g in gains]))]

    assert optimum(0.02) > optimum(0.5)


def test_gain_below_unity_is_rejected() -> None:
    """An avalanche multiplies; a value below one would be attenuation wearing
    the wrong name."""
    with pytest.raises(ValueError, match="below the minimum"):
        APDPhotodiode(gain=0.5)


def test_the_ionization_ratio_stays_physical() -> None:
    with pytest.raises(ValueError, match="above the maximum"):
        APDPhotodiode(ionization_ratio=1.5)
