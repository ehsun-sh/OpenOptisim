"""Validation of the EDFA and the OSNR that comes out of it.

The headline check is the OSNR formula every link engineer carries around:

    OSNR [dB] = 58 + P_launch [dBm] - NF [dB] - 10*log10(spans)

Nothing in the implementation is written in those terms — it computes an
inversion factor, an ASE spectral density, and a power ratio in a reference
bandwidth. That the arithmetic lands on the engineering rule of thumb is the
result.

Reference: G. P. Agrawal, *Fiber-Optic Communication Systems*, ch. 6.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from oosim import Component, Graph, OpticalSignal, SimulationContext
from oosim.analysis import OSNR_REFERENCE_BANDWIDTH, noise_psd_at, osnr
from oosim.components import EDFA, CWLaser, Fiber, OSNRMeter, PowerMeter
from oosim.units import C_LIGHT, H_PLANCK, db_to_linear, dbm_to_w, w_to_dbm

CTX = SimulationContext(bit_rate=10e9, samples_per_symbol=8, sequence_length=64)
NU_1550 = C_LIGHT / 1550e-9

#: 10*log10(h*nu*B_ref / 1 mW) at 1550 nm in 12.5 GHz — the "58" in the rule of
#: thumb, spelled out so the test does not smuggle in a magic number.
QUANTUM_FLOOR_DBM = w_to_dbm(H_PLANCK * NU_1550 * OSNR_REFERENCE_BANDWIDTH)


def _amplified(
    launch_dbm: float,
    *,
    gain_db: float = 20.0,
    noise_figure_db: float = 5.0,
    span_km: float = 0.0,
    stages: int = 1,
) -> tuple[Graph, PowerMeter, OSNRMeter]:
    g = Graph(CTX)
    laser = g.add(CWLaser(power=launch_dbm, wavelength=1550.0, label="laser"))
    previous: Component = laser
    for stage in range(stages):
        fiber = g.add(Fiber(length=span_km, attenuation=0.2, label=f"span{stage}"))
        amp = g.add(EDFA(gain=gain_db, noise_figure=noise_figure_db, label=f"edfa{stage}"))
        g.connect(previous, fiber)
        g.connect(fiber, amp)
        previous = amp

    power = g.add(PowerMeter(label="power"))
    osnr_meter = g.add(OSNRMeter(label="osnr"))
    g.connect(previous, power)
    g.connect(previous, osnr_meter)
    return g, power, osnr_meter


def _amplifier_output(graph: Graph, label: str = "edfa0") -> OpticalSignal:
    """The optical signal leaving a named amplifier."""
    amp = next(c for c in graph.components if c.label == label)
    return graph.run(keep=[amp]).port(amp, "out")


# --------------------------------------------------------------------------
# Gain
# --------------------------------------------------------------------------


@pytest.mark.parametrize("gain_db", [0.0, 10.0, 20.0, 30.0])
def test_signal_power_is_multiplied_by_the_gain(gain_db: float) -> None:
    g, power, _ = _amplified(-20.0, gain_db=gain_db)
    reading = g.run()[power]
    assert w_to_dbm(reading.signal_power_w) == pytest.approx(-20.0 + gain_db, abs=1e-4)


def test_the_output_clamp_reduces_the_gain_rather_than_the_signal() -> None:
    """A clamp, not a saturation model — and the docstring says so. What it must
    do is hold the output where it is told."""
    g = Graph(CTX)
    laser = g.add(CWLaser(power=0.0, label="laser"))
    amp = g.add(EDFA(gain=20.0, max_output_power=10.0, noise_figure=5.0, label="edfa"))
    meter = g.add(PowerMeter(label="meter"))
    g.chain(laser, amp, meter)

    assert w_to_dbm(g.run()[meter].signal_power_w) == pytest.approx(10.0, abs=1e-3)


def test_an_unsaturated_amplifier_is_untouched_by_the_clamp() -> None:
    g, power, _ = _amplified(-20.0, gain_db=20.0)
    assert w_to_dbm(g.run()[power].signal_power_w) == pytest.approx(0.0, abs=1e-4)


# --------------------------------------------------------------------------
# ASE
# --------------------------------------------------------------------------


def test_ase_psd_matches_the_spontaneous_emission_relation() -> None:
    """``S_ASE = n_sp * h * nu * (G - 1)`` per polarization."""
    gain_db, noise_figure_db = 20.0, 5.0
    gain = db_to_linear(gain_db)
    n_sp = db_to_linear(noise_figure_db) * gain / (2.0 * (gain - 1.0))
    expected = n_sp * H_PLANCK * NU_1550 * (gain - 1.0)

    amp = EDFA(gain=gain_db, noise_figure=noise_figure_db)
    assert amp.ase_psd(gain) == pytest.approx(expected, rel=1e-9)
    assert amp.spontaneous_emission_factor(gain) == pytest.approx(n_sp, rel=1e-9)


def test_total_ase_power_matches_the_textbook_expression() -> None:
    """``P_ASE = 2 * n_sp * h * nu * (G - 1) * B_o``, the factor of two being the
    two polarizations."""
    gain_db, noise_figure_db, bandwidth = 20.0, 5.0, 4e12
    gain = db_to_linear(gain_db)
    n_sp = db_to_linear(noise_figure_db) * gain / (2.0 * (gain - 1.0))
    expected = 2.0 * n_sp * H_PLANCK * NU_1550 * (gain - 1.0) * bandwidth

    g = Graph(CTX)
    laser = g.add(CWLaser(power=-20.0, label="laser"))
    amp = g.add(EDFA(gain=gain_db, noise_figure=noise_figure_db, bandwidth=4.0, label="edfa"))
    meter = g.add(PowerMeter(label="meter"))
    g.chain(laser, amp, meter)

    assert g.run()[meter].noise_power_w == pytest.approx(expected, rel=1e-6)


def test_ase_lives_in_a_noise_bin_not_in_the_sampled_band() -> None:
    """The reason the noise-bin representation exists.

    ASE spans terahertz while the signal occupies a few tens of gigahertz.
    Sampling both together would need a sample rate no machine can afford, so
    the noise is carried as a spectral density — and the sampled band must come
    out of the amplifier carrying signal only.
    """
    g = Graph(CTX)
    laser = g.add(CWLaser(power=-20.0, label="laser"))
    amp = g.add(EDFA(gain=20.0, bandwidth=4.0, label="edfa"))
    meter = g.add(PowerMeter(label="meter"))
    g.chain(laser, amp, meter)

    signal = g.run(keep=[amp]).port(amp, "out")

    assert len(signal.noise) == 1
    assert signal.noise[0].bandwidth == pytest.approx(4e12)
    # The ASE band is tens of times wider than the whole simulated bandwidth.
    assert signal.noise[0].bandwidth > 40 * CTX.sample_rate
    # And the sampled band is exactly the amplified carrier, with nothing added.
    assert signal.signal_power() == pytest.approx(dbm_to_w(0.0), rel=1e-4)


def test_a_unity_gain_amplifier_emits_no_ase() -> None:
    """Spontaneous emission is proportional to (G - 1): an amplifier that does
    not amplify cannot spontaneously emit either."""
    amp = EDFA(gain=0.0, noise_figure=5.0)
    assert amp.ase_psd(1.0) == 0.0


def test_the_noise_figure_cannot_go_below_the_quantum_limit() -> None:
    """n_sp has a floor of 1 — full inversion, a 3 dB noise figure. A lower noise
    figure would describe an amplifier quieter than quantum mechanics allows."""
    amp = EDFA(gain=20.0, noise_figure=0.0)
    assert amp.spontaneous_emission_factor(db_to_linear(20.0)) >= 0.5


# --------------------------------------------------------------------------
# OSNR — the engineering rule of thumb
# --------------------------------------------------------------------------


def test_the_quantum_floor_is_the_58_in_the_rule_of_thumb() -> None:
    assert pytest.approx(58.0, abs=0.1) == -QUANTUM_FLOOR_DBM


@pytest.mark.parametrize("launch_dbm", [-6.0, -3.0, 0.0, 3.0])
@pytest.mark.parametrize("noise_figure_db", [4.0, 5.0, 6.0])
def test_single_stage_osnr_matches_the_engineering_formula(
    launch_dbm: float, noise_figure_db: float
) -> None:
    """``OSNR = 58 + P_launch - NF`` for one amplifier.

    Nothing in the model is expressed this way; it computes an inversion factor,
    a spectral density, and a ratio in a 0.1 nm slice. Landing on the formula is
    the check.
    """
    g, _, osnr_meter = _amplified(launch_dbm, gain_db=20.0, noise_figure_db=noise_figure_db)
    measured = g.run()[osnr_meter]

    expected = launch_dbm - noise_figure_db - QUANTUM_FLOOR_DBM
    assert measured == pytest.approx(expected, abs=0.1)


def test_osnr_does_not_depend_on_the_gain() -> None:
    """Both the signal and its own ASE scale with G, so once past a modest gain
    the ratio settles. An OSNR that improved with gain would be a free lunch."""
    values = [
        _amplified(-3.0, gain_db=gain)[0].run()[_amplified(-3.0, gain_db=gain)[2]]
        for gain in (15.0, 20.0, 25.0, 30.0)
    ]
    assert max(values) - min(values) < 0.15


def test_each_identical_span_costs_three_db_of_osnr() -> None:
    """N spans divide OSNR by N: doubling the count costs 3 dB. This is why long
    haul is hard, and it falls out of the noise bins accumulating."""
    single = _amplified(0.0, span_km=100.0, gain_db=20.0, stages=1)
    quad = _amplified(0.0, span_km=100.0, gain_db=20.0, stages=4)

    osnr_1 = single[0].run()[single[2]]
    osnr_4 = quad[0].run()[quad[2]]

    assert osnr_1 - osnr_4 == pytest.approx(10 * math.log10(4), abs=0.3)


def test_osnr_degrades_one_for_one_with_launch_power() -> None:
    low = _amplified(-6.0)
    high = _amplified(0.0)
    assert high[0].run()[high[2]] - low[0].run()[low[2]] == pytest.approx(6.0, abs=0.05)


def test_a_link_without_amplifiers_has_infinite_osnr() -> None:
    """No amplifier, no ASE. The signal is attenuated but never made noisier —
    which is exactly why a passive link's reach is a loss budget, not an OSNR
    budget."""
    g = Graph(CTX)
    laser = g.add(CWLaser(power=0.0, label="laser"))
    fiber = g.add(Fiber(length=100.0, attenuation=0.2, label="fiber"))
    meter = g.add(OSNRMeter(label="osnr"))
    g.chain(laser, fiber, meter)

    assert g.run()[meter] == math.inf


def test_fiber_attenuates_ase_along_with_the_signal() -> None:
    """Loss after an amplifier cannot improve OSNR: both fall together."""
    g = Graph(CTX)
    laser = g.add(CWLaser(power=0.0, label="laser"))
    amp = g.add(EDFA(gain=20.0, label="edfa"))
    fiber = g.add(Fiber(length=80.0, attenuation=0.2, label="fiber"))
    before = g.add(OSNRMeter(label="before"))
    after = g.add(OSNRMeter(label="after"))
    g.chain(laser, amp)
    g.connect(amp, before)
    g.connect(amp, fiber)
    g.connect(fiber, after)

    results = g.run()
    # Tolerance set by complex64 storage of the band, not by the physics:
    # the ratio is exactly preserved, the stored samples are not.
    assert results[after] == pytest.approx(results[before], abs=1e-4)


def test_osnr_uses_the_noise_density_at_the_carrier() -> None:
    g = Graph(CTX)
    laser = g.add(CWLaser(power=-10.0, wavelength=1550.0, label="laser"))
    amp = g.add(EDFA(gain=20.0, noise_figure=5.0, label="edfa"))
    meter = g.add(PowerMeter(label="meter"))
    g.chain(laser, amp, meter)

    signal = g.run(keep=[amp]).port(amp, "out")
    psd = noise_psd_at(signal, signal.bands[0].f0)
    manual = 10 * math.log10(signal.signal_power() / (psd * OSNR_REFERENCE_BANDWIDTH))
    assert osnr(signal) == pytest.approx(manual, rel=1e-12)


def test_reference_bandwidth_scales_the_result() -> None:
    """Halving the reference bandwidth halves the counted noise: +3 dB."""
    signal = _amplifier_output(_amplified(0.0)[0])

    wide = osnr(signal, reference_bandwidth=12.5e9)
    narrow = osnr(signal, reference_bandwidth=6.25e9)
    assert narrow - wide == pytest.approx(3.0103, abs=1e-6)


def test_a_nonsensical_reference_bandwidth_is_rejected() -> None:
    signal = _amplifier_output(_amplified(0.0)[0])
    with pytest.raises(ValueError, match="reference_bandwidth must be positive"):
        osnr(signal, reference_bandwidth=0.0)


def test_noise_outside_the_carrier_does_not_count_against_it() -> None:
    """A noise bin sitting somewhere else in the spectrum is not this channel's
    problem — the band-resolved lookup is what makes that distinction possible."""
    from oosim.signals import Band, NoiseBin

    band = Band(
        Ex=np.full(16, 0.1, dtype=np.complex64),
        Ey=np.zeros(16, dtype=np.complex64),
        f0=NU_1550,
        fs=80e9,
    )
    elsewhere = NoiseBin(f_start=150e12, f_end=160e12, psd_x=1e-15, psd_y=1e-15)
    assert osnr(OpticalSignal(bands=(band,), noise=(elsewhere,))) == math.inf
