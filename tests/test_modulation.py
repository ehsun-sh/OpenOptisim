"""Constellations, Gray coding, and the analytical error rates.

These are the pieces the coherent chain is measured *against*, so they are
checked on their own first. A constellation that is quietly not Gray-coded still
produces a believable BER — about twice the right one — and nothing downstream
would notice.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from oosim.modulation import (
    ber_qam,
    bits_to_indices,
    blind_phase_search,
    error_vector_magnitude,
    gray_pam_levels,
    indices_to_bits,
    nearest_indices,
    qam_constellation,
    ser_qam,
    snr_from_evm,
)

SQUARE_ORDERS = [2, 4, 6, 8]
ALL_ORDERS = [1, *SQUARE_ORDERS]


@pytest.mark.parametrize("bits_per_symbol", ALL_ORDERS)
def test_constellation_has_unit_mean_power(bits_per_symbol: int) -> None:
    """Every format is normalised, so launch power means the same thing for all of them.

    Without this, switching a link from QPSK to 16-QAM would silently change the
    average optical power and every sensitivity number would move for the wrong
    reason.
    """
    points = qam_constellation(bits_per_symbol)
    assert float(np.mean(np.abs(points) ** 2)) == pytest.approx(1.0, rel=1e-12)


@pytest.mark.parametrize("bits_per_symbol", ALL_ORDERS)
def test_constellation_has_the_expected_size(bits_per_symbol: int) -> None:
    assert qam_constellation(bits_per_symbol).shape == (1 << bits_per_symbol,)


@pytest.mark.parametrize("bits_per_symbol", ALL_ORDERS)
def test_nearest_neighbours_differ_in_exactly_one_bit(bits_per_symbol: int) -> None:
    """The defining property of Gray coding, checked on the geometry itself.

    A symbol error at usable SNR lands on a nearest neighbour. If that neighbour
    differs in one bit, the error costs one bit; if the labelling is wrong it
    costs more, and the BER-from-SER relation every estimate here uses breaks.
    """
    points = qam_constellation(bits_per_symbol)
    distance = np.abs(points[:, None] - points[None, :])
    np.fill_diagonal(distance, np.inf)
    smallest = distance.min()

    for i, j in zip(*np.where(np.abs(distance - smallest) < 1e-9), strict=True):
        assert bin(int(i) ^ int(j)).count("1") == 1, (
            f"points {i} and {j} are nearest neighbours but differ in "
            f"{bin(int(i) ^ int(j)).count('1')} bits"
        )


def test_gray_pam_levels_are_the_odd_integers() -> None:
    levels = gray_pam_levels(2)
    assert sorted(levels) == [-3.0, -1.0, 1.0, 3.0]
    # Ordering is by Gray code, not by magnitude: 00, 01, 11, 10.
    assert list(levels) == [-3.0, -1.0, 3.0, 1.0]


@pytest.mark.parametrize("bits_per_symbol", ALL_ORDERS)
def test_bit_packing_round_trips(bits_per_symbol: int) -> None:
    rng = np.random.default_rng(4)
    bits = rng.integers(0, 2, size=bits_per_symbol * 500).astype(np.uint8)
    indices = bits_to_indices(bits, bits_per_symbol)
    assert np.array_equal(indices_to_bits(indices, bits_per_symbol), bits)


def test_bit_packing_rejects_a_partial_symbol() -> None:
    with pytest.raises(ValueError, match="whole symbols"):
        bits_to_indices(np.zeros(7, dtype=np.uint8), 4)


@pytest.mark.parametrize("bits_per_symbol", ALL_ORDERS)
def test_noiseless_symbols_demap_exactly(bits_per_symbol: int) -> None:
    points = qam_constellation(bits_per_symbol)
    indices = np.arange(points.shape[0])
    assert np.array_equal(nearest_indices(points[indices], points), indices)


def test_cross_constellations_are_refused_rather_than_faked() -> None:
    """32-QAM is a cross, not a rectangle. Returning a rectangle would be wrong."""
    with pytest.raises(NotImplementedError, match="cross constellation"):
        qam_constellation(5)


def test_evm_of_a_perfect_sequence_is_zero() -> None:
    points = qam_constellation(4)
    assert error_vector_magnitude(points, points) == pytest.approx(0.0, abs=1e-15)


def test_snr_is_the_reciprocal_of_evm_squared() -> None:
    """The identity that makes an EVM reading interchangeable with an SNR."""
    rng = np.random.default_rng(9)
    points = qam_constellation(4)
    symbols = points[rng.integers(0, 16, size=40_000)]

    target_snr = 20.0
    sigma = math.sqrt(1.0 / (2.0 * target_snr))  # per quadrature, unit signal power
    noisy = symbols + rng.normal(0, sigma, symbols.shape) + 1j * rng.normal(0, sigma, symbols.shape)

    measured = snr_from_evm(error_vector_magnitude(noisy, symbols))
    assert measured == pytest.approx(target_snr, rel=0.03)


#: (bits_per_symbol, SNR) pairs that put each format in the countable range.
#: A higher-order constellation needs more SNR to reach the same error rate, so
#: a single SNR list would be either too noisy to be interesting for BPSK or too
#: quiet to count for 64-QAM.
COUNTABLE_POINTS = [
    (1, 4.0),
    (1, 7.0),
    (2, 7.0),
    (2, 10.0),
    (4, 13.0),
    (4, 16.0),
    (6, 18.0),
    (6, 21.0),
]


@pytest.mark.parametrize(("bits_per_symbol", "snr_db"), COUNTABLE_POINTS)
def test_analytical_ser_matches_a_monte_carlo_channel(bits_per_symbol: int, snr_db: float) -> None:
    """The closed form is checked against counted errors, not against itself.

    This is the reference the whole coherent chain is later compared to, so it
    has to stand on its own.

    The tolerance is Poisson rather than a fixed fraction. Counting N errors
    carries a standard error of sqrt(N), so at 16 dB — where a few dozen errors
    land in the whole run — a 25% band is *tighter* than the statistics allow and
    a passing model would fail it about as often as a broken one. Four standard
    errors is the band that means something at every point on the curve.
    """
    trials = 400_000
    rng = np.random.default_rng(17)
    points = qam_constellation(bits_per_symbol)
    indices = rng.integers(0, points.shape[0], size=trials)
    symbols = points[indices]

    snr = 10.0 ** (snr_db / 10.0)
    sigma = math.sqrt(1.0 / (2.0 * snr))
    noisy = symbols + rng.normal(0, sigma, symbols.shape) + 1j * rng.normal(0, sigma, symbols.shape)

    counted = int(np.count_nonzero(nearest_indices(noisy, points) != indices))
    expected = ser_qam(snr, bits_per_symbol) * trials

    assert expected > 10.0, "this operating point is too quiet to count; pick a lower SNR"
    assert abs(counted - expected) < 4.0 * math.sqrt(expected)


def test_ber_is_ser_shared_over_the_bits_a_symbol_carries() -> None:
    assert ber_qam(100.0, 4) == pytest.approx(ser_qam(100.0, 4) / 4)


def test_a_higher_order_format_needs_more_snr_for_the_same_ser() -> None:
    """The whole trade the tool exists to let someone explore."""
    snr = 100.0
    assert ser_qam(snr, 2) < ser_qam(snr, 4) < ser_qam(snr, 6)


# --------------------------------------------------------------------------
# Blind phase search
# --------------------------------------------------------------------------


def _impaired(
    bits_per_symbol: int, phase: np.ndarray, sigma: float = 0.04, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A rotated, noisy symbol sequence and the alphabet it was drawn from."""
    rng = np.random.default_rng(seed)
    points = qam_constellation(bits_per_symbol)
    indices = rng.integers(0, points.shape[0], size=phase.shape[0])
    clean = points[indices]
    noise = rng.normal(0, sigma, phase.shape[0]) + 1j * rng.normal(0, sigma, phase.shape[0])
    return clean * np.exp(1j * phase) + noise, clean, points


def _residual(recovered: np.ndarray, clean: np.ndarray) -> float:
    """The constant rotation left over, which the quadrant ambiguity permits."""
    return float(np.angle(np.mean(recovered * np.conj(clean))))


@pytest.mark.parametrize("bits_per_symbol", [2, 4, 6])
@pytest.mark.parametrize(
    ("name", "make"),
    [
        ("static", lambda n: np.full(n, 0.3)),
        ("frequency offset", lambda n: np.linspace(0.0, 2.0, n)),
        ("random walk", lambda n: np.cumsum(np.random.default_rng(5).normal(0, 0.01, n))),
    ],
)
def test_phase_search_tracks_every_kind_of_rotation(
    bits_per_symbol: int, name: str, make: Any
) -> None:
    """A constant, a ramp, and a walk — the three things a carrier actually does.

    The walk is the one that matters: it is not removable by subtracting a
    constant or a line, which is exactly why a blind *search* is needed rather
    than a closed-form estimate.
    """
    count = 4096
    received, clean, points = _impaired(bits_per_symbol, make(count))

    phase = blind_phase_search(received, points)
    recovered = received * np.exp(-1j * phase)
    # What is left may be a constant quarter turn; remove it the way the
    # measurement block does, then the symbols must decide correctly.
    aligned = recovered * np.exp(-1j * _residual(recovered, clean))

    errors = np.count_nonzero(nearest_indices(aligned, points) != nearest_indices(clean, points))
    assert errors <= count // 200, f"{name}: {errors} symbol errors after recovery"


def test_what_remains_is_a_quarter_turn_at_most() -> None:
    """The ambiguity is a real limit of blind estimation, not a bug to fix.

    Asserted rather than glossed: no blind method can resolve it, and a link that
    needs it resolved has to encode the quadrant differentially.
    """
    count = 2048
    received, clean, points = _impaired(4, np.full(count, 1.1))
    recovered = received * np.exp(-1j * blind_phase_search(received, points))

    residual = _residual(recovered, clean)
    nearest_quarter = (math.pi / 2.0) * round(residual / (math.pi / 2.0))
    assert abs(residual - nearest_quarter) < 0.05


def test_the_window_is_a_trade_and_not_a_free_parameter() -> None:
    """Too long a window cannot follow a fast laser. That is the documented cost.

    A test that only ever used the default would let the window silently stop
    mattering, and the parameter would become decoration.
    """
    count = 4096
    fast = np.cumsum(np.random.default_rng(7).normal(0, 0.06, count))
    received, clean, points = _impaired(4, fast)
    reference = nearest_indices(clean, points)

    def errors(window: int) -> int:
        recovered = received * np.exp(-1j * blind_phase_search(received, points, window=window))
        aligned = recovered * np.exp(-1j * _residual(recovered, clean))
        return int(np.count_nonzero(nearest_indices(aligned, points) != reference))

    assert errors(16) < errors(512)


@pytest.mark.parametrize(
    ("kwargs", "match"), [({"test_phases": 1}, "test_phases"), ({"window": 0}, "window")]
)
def test_phase_search_rejects_impossible_settings(kwargs: dict[str, int], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        blind_phase_search(qam_constellation(2), qam_constellation(2), **kwargs)


def test_phase_search_handles_an_empty_sequence() -> None:
    assert blind_phase_search(np.zeros(0, dtype=np.complex128), qam_constellation(2)).shape == (0,)
