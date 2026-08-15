"""Validation of chromatic dispersion against closed-form results.

A Gaussian input is the reason this can be checked exactly: propagating one
through pure GVD has an analytical solution, so the model is compared with
arithmetic rather than with another simulator.

Reference: G. P. Agrawal, *Nonlinear Fiber Optics*, §3.2 (eq. 3.2.9-3.2.10).
"""

from __future__ import annotations

import numpy as np
import pytest

from oosim import Graph, SimulationContext
from oosim.analysis import instantaneous_power, rms_time_width
from oosim.components import Combiner, Fiber, GaussianPulse, PowerMeter
from oosim.kernels import dispersion_to_beta2

# A pulse-scale window: 0.39 ps per sample, 400 ps long. Wide enough that a pulse
# broadened several-fold stays clear of the periodic wrap that would corrupt the
# width measurement.
PULSE_CTX = SimulationContext(bit_rate=10e9, samples_per_symbol=256, sequence_length=4)

T0_PS = 10.0
D_SMF = 17.0  # ps/(nm*km), standard single-mode fiber near 1550 nm


def _beta2_ps2_per_km(dispersion: float = D_SMF, wavelength_nm: float = 1550.0) -> float:
    """β₂ in ps²/km, the unit the textbook formulae are usually quoted in."""
    beta2_si = dispersion_to_beta2(dispersion * 1e-6, wavelength_nm * 1e-9)
    return beta2_si * 1e24 * 1e3


def _propagate(
    length_km: float, *, chirp: float = 0.0, dispersion: float = D_SMF, t0_ps: float = T0_PS
) -> tuple[float, float]:
    """Run a Gaussian pulse through a lossless dispersive fiber.

    Returns the input and output RMS widths [s].
    """
    g = Graph(PULSE_CTX)
    source = g.add(GaussianPulse(peak_power=0.0, width=t0_ps, chirp=chirp, wavelength=1550.0))
    fiber = g.add(Fiber(length=length_km, attenuation=0.0, dispersion=dispersion))
    meter = g.add(PowerMeter())
    g.chain(source, fiber, meter)

    results = g.run(keep=[source, fiber])
    launched = results.port(source, "out").bands[0]
    received = results.port(fiber, "out").bands[0]
    return rms_time_width(launched), rms_time_width(received)


# --------------------------------------------------------------------------
# Unchirped broadening:  T1/T0 = sqrt(1 + (z / L_D)**2),  L_D = T0**2 / |beta2|
# --------------------------------------------------------------------------


def test_beta2_matches_the_textbook_value_for_standard_fiber() -> None:
    """D = 17 ps/(nm*km) at 1550 nm corresponds to beta2 ~ -21.7 ps^2/km.

    Positive D gives negative beta2 (anomalous dispersion). Getting this backwards
    is a classic error and every result below depends on it.
    """
    assert _beta2_ps2_per_km() == pytest.approx(-21.7, abs=0.1)
    assert _beta2_ps2_per_km(dispersion=-17.0) == pytest.approx(+21.7, abs=0.1)


@pytest.mark.parametrize("z_over_ld", [0.5, 1.0, 2.0, 3.0])
def test_unchirped_gaussian_broadens_analytically(z_over_ld: float) -> None:
    ld_km = T0_PS**2 / abs(_beta2_ps2_per_km())  # dispersion length [km]
    length_km = z_over_ld * ld_km

    width_in, width_out = _propagate(length_km)

    expected = np.sqrt(1.0 + z_over_ld**2)
    assert width_out / width_in == pytest.approx(expected, rel=1e-4)


def test_dispersion_length_is_where_the_pulse_broadens_by_root_two() -> None:
    """The definition of L_D, stated as its own test because it is the anchor
    every other dispersion result is scaled against."""
    ld_km = T0_PS**2 / abs(_beta2_ps2_per_km())
    width_in, width_out = _propagate(ld_km)
    assert width_out / width_in == pytest.approx(np.sqrt(2.0), rel=1e-4)


def test_broadening_scales_with_the_square_of_the_pulse_width() -> None:
    """L_D goes as T0**2, so a pulse twice as wide broadens four times more slowly.

    This catches an error in the T0 convention that a single-width test cannot.
    """
    length_km = 5.0
    ratios = []
    for t0 in (T0_PS, 2 * T0_PS):
        width_in, width_out = _propagate(length_km, t0_ps=t0)
        ld_km = t0**2 / abs(_beta2_ps2_per_km())
        ratios.append((width_out / width_in, np.sqrt(1.0 + (length_km / ld_km) ** 2)))

    for measured, expected in ratios:
        assert measured == pytest.approx(expected, rel=1e-4)


def test_zero_dispersion_leaves_the_pulse_untouched() -> None:
    width_in, width_out = _propagate(80.0, dispersion=0.0)
    assert width_out == pytest.approx(width_in, rel=1e-9)


# --------------------------------------------------------------------------
# Chirped pulse — this is what pins the *sign* of beta2
# --------------------------------------------------------------------------


def test_chirped_pulse_compresses_before_broadening() -> None:
    """T1/T0 = sqrt((1 + C*beta2*z/T0**2)**2 + (beta2*z/T0**2)**2).

    The unchirped formula is even in beta2, so it passes whichever sign the
    implementation uses. With chirp the two signs give opposite behaviour: for
    C = +2 in anomalous fiber the pulse compresses to ~45% of its input width,
    where a flipped sign would broaden it to ~184%. Nothing else in the suite
    distinguishes those.
    """
    chirp = 2.0
    beta2 = _beta2_ps2_per_km()  # negative for standard fiber
    ld_km = T0_PS**2 / abs(beta2)
    # Minimum width occurs at z/L_D = C / (1 + C**2) for anomalous dispersion.
    length_km = ld_km * chirp / (1.0 + chirp**2)

    width_in, width_out = _propagate(length_km, chirp=chirp)

    x = beta2 * length_km / T0_PS**2  # dimensionless beta2*z/T0^2
    expected = np.sqrt((1.0 + chirp * x) ** 2 + x**2)

    assert expected < 0.5, "test setup no longer exercises compression"
    assert width_out / width_in == pytest.approx(expected, rel=1e-4)
    assert width_out < width_in


@pytest.mark.parametrize("chirp", [-2.0, -0.5, 0.5, 2.0])
def test_chirped_broadening_matches_the_analytical_factor(chirp: float) -> None:
    beta2 = _beta2_ps2_per_km()
    length_km = 3.0
    width_in, width_out = _propagate(length_km, chirp=chirp)

    x = beta2 * length_km / T0_PS**2
    expected = np.sqrt((1.0 + chirp * x) ** 2 + x**2)
    assert width_out / width_in == pytest.approx(expected, rel=1e-4)


def test_opposite_chirp_signs_are_not_symmetric() -> None:
    """Guards the test above: if the model ignored the sign of the chirp, every
    parametrised case would still pass. These two must differ."""
    _, out_positive = _propagate(3.0, chirp=+2.0)
    _, out_negative = _propagate(3.0, chirp=-2.0)
    assert out_positive != pytest.approx(out_negative, rel=1e-3)


# --------------------------------------------------------------------------
# Structural properties of the propagator
# --------------------------------------------------------------------------


def test_dispersion_conserves_energy() -> None:
    """The transfer function has unit magnitude, so this holds exactly (Parseval).

    Any leak here means the FFT normalisation is wrong, which would otherwise
    surface much later as a mysterious power offset.
    """
    g = Graph(PULSE_CTX)
    source = g.add(GaussianPulse(peak_power=0.0, width=T0_PS))
    fiber = g.add(Fiber(length=50.0, attenuation=0.0, dispersion=D_SMF))
    meter = g.add(PowerMeter())
    g.chain(source, fiber, meter)

    results = g.run(keep=[source])
    launched = results.port(source, "out").signal_power()
    received = results[meter].power_w

    assert received == pytest.approx(launched, rel=1e-6)


def test_dispersion_is_exactly_reversible() -> None:
    """Propagating +D then -D over the same length restores the input.

    This is the basis of dispersion compensation, and it is a strong check that
    the propagator is a true all-pass phase rotation rather than an approximation.
    """
    g = Graph(PULSE_CTX)
    source = g.add(GaussianPulse(peak_power=0.0, width=T0_PS, chirp=1.0))
    span = g.add(Fiber(length=80.0, attenuation=0.0, dispersion=D_SMF, label="span"))
    dcf = g.add(Fiber(length=80.0, attenuation=0.0, dispersion=-D_SMF, label="dcf"))
    meter = g.add(PowerMeter())
    g.chain(source, span, dcf, meter)

    results = g.run(keep=[source, dcf])
    launched = results.port(source, "out").bands[0]
    recovered = results.port(dcf, "out").bands[0]

    assert rms_time_width(recovered) == pytest.approx(rms_time_width(launched), rel=1e-5)
    np.testing.assert_allclose(
        instantaneous_power(recovered), instantaneous_power(launched), rtol=1e-4, atol=1e-12
    )


def test_dispersion_does_not_change_average_power() -> None:
    """Broadening redistributes power in time; it does not create or destroy it."""
    g = Graph(PULSE_CTX)
    source = g.add(GaussianPulse(peak_power=0.0, width=T0_PS))
    fiber = g.add(Fiber(length=30.0, attenuation=0.2, dispersion=D_SMF))
    meter = g.add(PowerMeter())
    g.chain(source, fiber, meter)

    results = g.run(keep=[source])
    launched = results.port(source, "out").signal_power()
    expected = launched * 10 ** (-0.2 * 30.0 / 10)

    assert results[meter].power_w == pytest.approx(expected, rel=1e-5)


# --------------------------------------------------------------------------
# Per-band dispersion — more work the multi-band model does for free
# --------------------------------------------------------------------------


def test_each_band_is_dispersed_at_its_own_wavelength() -> None:
    """beta2 depends on lambda**2, so widely separated channels broaden differently.

    A single-carrier signal model has one centre wavelength and physically cannot
    represent this; here it falls out of the band list with no special handling.
    """
    ctx = PULSE_CTX
    g = Graph(ctx)
    short = g.add(GaussianPulse(width=T0_PS, wavelength=1300.0, label="short"))
    long = g.add(GaussianPulse(width=T0_PS, wavelength=1600.0, label="long"))
    mux = g.add(Combiner(2))
    fiber = g.add(Fiber(length=6.0, attenuation=0.0, dispersion=D_SMF))
    meter = g.add(PowerMeter())

    g.connect(short, mux["in0"])
    g.connect(long, mux["in1"])
    g.chain(mux, fiber, meter)

    results = g.run(keep=[fiber])
    out = results.port(fiber, "out")

    widths = {round(band.wavelength * 1e9): rms_time_width(band) for band in out.bands}
    assert set(widths) == {1300, 1600}

    # |beta2| goes as lambda**2, so the 1600 nm channel disperses more.
    assert widths[1600] > widths[1300]

    for wavelength_nm, width in widths.items():
        beta2 = _beta2_ps2_per_km(wavelength_nm=wavelength_nm)
        ld_km = T0_PS**2 / abs(beta2)
        expected = np.sqrt(1.0 + (6.0 / ld_km) ** 2)
        assert width / (T0_PS * 1e-12 / np.sqrt(2)) == pytest.approx(expected, rel=1e-3)
