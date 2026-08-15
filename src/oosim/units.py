"""Unit conversion.

Every quantity is stored internally in **SI base units** — watts, hertz, metres,
seconds. Component parameters are declared in whatever unit an engineer actually
uses (dBm, nm, km, dB/km) and converted exactly once, at the component boundary.

Unit confusion is the most common source of wrong answers in optical simulation,
so conversion lives in one place with tests rather than being open-coded per block.
"""

from __future__ import annotations

import math
from collections.abc import Callable

#: Speed of light in vacuum [m/s] (CODATA, exact by definition).
C_LIGHT = 299_792_458.0

#: Planck constant [J·s] (CODATA, exact by definition).
H_PLANCK = 6.626_070_15e-34

#: Elementary charge [C] (CODATA, exact by definition).
Q_ELECTRON = 1.602_176_634e-19

#: Boltzmann constant [J/K] (CODATA, exact by definition).
K_BOLTZMANN = 1.380_649e-23


def dbm_to_w(p_dbm: float) -> float:
    """Convert power in dBm to watts."""
    return 10.0 ** (p_dbm / 10.0) * 1e-3


def w_to_dbm(p_w: float) -> float:
    """Convert power in watts to dBm. Zero power maps to -inf, not an error."""
    if p_w <= 0.0:
        return -math.inf
    return 10.0 * math.log10(p_w / 1e-3)


def db_to_linear(x_db: float) -> float:
    """Convert a power ratio in dB to a linear power ratio."""
    return 10.0 ** (x_db / 10.0)


def linear_to_db(x: float) -> float:
    """Convert a linear power ratio to dB."""
    if x <= 0.0:
        return -math.inf
    return 10.0 * math.log10(x)


def wavelength_to_frequency(lambda_m: float) -> float:
    """Convert vacuum wavelength [m] to optical frequency [Hz]."""
    if lambda_m <= 0.0:
        raise ValueError(f"wavelength must be positive, got {lambda_m}")
    return C_LIGHT / lambda_m


def frequency_to_wavelength(f_hz: float) -> float:
    """Convert optical frequency [Hz] to vacuum wavelength [m]."""
    if f_hz <= 0.0:
        raise ValueError(f"frequency must be positive, got {f_hz}")
    return C_LIGHT / f_hz


# Conversions to/from SI, keyed by the unit string used in a Param declaration.
_TO_SI: dict[str, Callable[[float], float]] = {
    "": lambda x: x,
    # Power
    "W": lambda x: x,
    "mW": lambda x: x * 1e-3,
    "dBm": dbm_to_w,
    # Ratios
    "dB": db_to_linear,
    "dB/km": lambda x: x * 1e-3,  # -> dB/m (still logarithmic, per metre)
    # Dispersion: ps/(nm*km) -> s/m^2
    "ps/nm/km": lambda x: x * 1e-6,
    # Nonlinearity: 1/(W*km) -> 1/(W*m)
    "1/W/km": lambda x: x * 1e-3,
    # PMD coefficient: ps/sqrt(km) -> s/sqrt(m)
    "ps/sqrt(km)": lambda x: x * 1e-12 / math.sqrt(1e3),
    # Length
    "m": lambda x: x,
    "km": lambda x: x * 1e3,
    "nm": lambda x: x * 1e-9,
    # Frequency
    "Hz": lambda x: x,
    "kHz": lambda x: x * 1e3,
    "MHz": lambda x: x * 1e6,
    "GHz": lambda x: x * 1e9,
    "THz": lambda x: x * 1e12,
    # Time
    "s": lambda x: x,
    "ps": lambda x: x * 1e-12,
    # Voltage
    "V": lambda x: x,
}

_FROM_SI: dict[str, Callable[[float], float]] = {
    "": lambda x: x,
    "W": lambda x: x,
    "mW": lambda x: x * 1e3,
    "dBm": w_to_dbm,
    "dB": linear_to_db,
    "dB/km": lambda x: x * 1e3,
    "ps/nm/km": lambda x: x * 1e6,
    "1/W/km": lambda x: x * 1e3,
    "ps/sqrt(km)": lambda x: x * 1e12 * math.sqrt(1e3),
    "m": lambda x: x,
    "km": lambda x: x * 1e-3,
    "nm": lambda x: x * 1e9,
    "Hz": lambda x: x,
    "kHz": lambda x: x * 1e-3,
    "MHz": lambda x: x * 1e-6,
    "GHz": lambda x: x * 1e-9,
    "THz": lambda x: x * 1e-12,
    "s": lambda x: x,
    "ps": lambda x: x * 1e12,
    "V": lambda x: x,
}


def known_units() -> frozenset[str]:
    """Every unit string accepted by :func:`to_si` and :func:`from_si`."""
    return frozenset(_TO_SI)


def to_si(value: float, unit: str) -> float:
    """Convert ``value`` expressed in ``unit`` to its SI base unit."""
    try:
        return _TO_SI[unit](value)
    except KeyError:
        raise ValueError(f"unknown unit {unit!r}; known units: {sorted(_TO_SI)}") from None


def from_si(value: float, unit: str) -> float:
    """Convert ``value`` in SI base units to ``unit``."""
    try:
        return _FROM_SI[unit](value)
    except KeyError:
        raise ValueError(f"unknown unit {unit!r}; known units: {sorted(_FROM_SI)}") from None
