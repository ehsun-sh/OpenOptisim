"""Signal types carried between components.

The optical signal model is the single most consequential decision in the whole
engine, so the reasoning is spelled out here rather than left to the design doc.

**Why a list of bands.** A scalar centre frequency can represent exactly one
carrier. A 40-channel DWDM system spans several THz; sampling that as one band
would need a sample rate no machine can afford. Real system simulators therefore
carry a *set* of independently sampled bands, each with its own centre frequency,
and only merge them onto a common grid where physics forces it. Building this in
from the start costs almost nothing; retrofitting it means rewriting every block.

**Why noise is separate.** Amplifier ASE covers the whole amplifier bandwidth
while the signal occupies a small slice of it. Represented as samples, the noise
alone would dictate the sample rate. Carried as spectral bins with a power
spectral density, it costs a handful of floats and is only converted to samples
where a detector or a nonlinearity actually needs signal-noise beating.

**Field convention.** ``Ex`` and ``Ey`` are complex envelope amplitudes scaled so
that instantaneous power is ``|Ex|**2 + |Ey|**2`` in watts — i.e. the fields are
in units of sqrt(W). Average power is the mean of that over the time window.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from .units import frequency_to_wavelength


def freeze(a: np.ndarray) -> np.ndarray:
    """Return a read-only view of ``a``.

    Signals are immutable: a block receives read-only inputs and returns new
    outputs. A WDM link holds hundreds of megabytes, so blocks that only change
    metadata share buffers instead of copying, and immutability is what makes
    that sharing safe.
    """
    view = a.view()
    view.flags.writeable = False
    return view


@dataclass(frozen=True)
class Band:
    """One sampled band: the complex envelope in two orthogonal polarizations.

    ``Ex``/``Ey`` form the Jones vector. Sample ``k`` of the underlying optical
    field is ``Ex[k] * exp(2j*pi*f0*t_k)``; the envelope is what we store.
    """

    Ex: np.ndarray
    """X-polarization complex envelope [sqrt(W)], shape (N,)."""

    Ey: np.ndarray
    """Y-polarization complex envelope [sqrt(W)], shape (N,)."""

    f0: float
    """Band centre frequency [Hz]."""

    fs: float
    """Band sample rate [Hz]."""

    def __post_init__(self) -> None:
        if self.Ex.ndim != 1 or self.Ey.ndim != 1:
            raise ValueError(f"Ex and Ey must be 1-D, got {self.Ex.ndim}-D and {self.Ey.ndim}-D")
        if self.Ex.shape != self.Ey.shape:
            raise ValueError(
                f"Ex and Ey must have the same length, got {self.Ex.shape} and {self.Ey.shape}"
            )
        if not np.issubdtype(self.Ex.dtype, np.complexfloating):
            raise TypeError(f"Ex must be complex, got dtype {self.Ex.dtype}")
        if not np.issubdtype(self.Ey.dtype, np.complexfloating):
            raise TypeError(f"Ey must be complex, got dtype {self.Ey.dtype}")
        if self.f0 <= 0:
            raise ValueError(f"f0 must be positive, got {self.f0}")
        if self.fs <= 0:
            raise ValueError(f"fs must be positive, got {self.fs}")
        object.__setattr__(self, "Ex", freeze(self.Ex))
        object.__setattr__(self, "Ey", freeze(self.Ey))

    @property
    def num_samples(self) -> int:
        return int(self.Ex.shape[0])

    @property
    def wavelength(self) -> float:
        """Centre wavelength in vacuum [m]."""
        return frequency_to_wavelength(self.f0)

    @property
    def bandwidth(self) -> float:
        """Sampled bandwidth [Hz], i.e. the width of the represented spectrum."""
        return self.fs

    def average_power(self) -> float:
        """Mean power over the time window [W], summed over both polarizations."""
        px = float(np.mean(np.abs(self.Ex) ** 2))
        py = float(np.mean(np.abs(self.Ey) ** 2))
        return px + py

    def scale_amplitude(self, factor: float) -> Band:
        """Return a copy with both polarizations scaled by an *amplitude* factor.

        Power scales by ``factor**2``.
        """
        return replace(self, Ex=self.Ex * factor, Ey=self.Ey * factor)


@dataclass(frozen=True)
class NoiseBin:
    """Spectrally-resolved noise power, carried outside the sampled bands.

    ``psd_x``/``psd_y`` are one-sided power spectral densities per polarization
    [W/Hz], assumed flat across ``[f_start, f_end)``.
    """

    f_start: float
    f_end: float
    psd_x: float
    psd_y: float

    def __post_init__(self) -> None:
        if self.f_end <= self.f_start:
            raise ValueError(f"f_end must exceed f_start, got [{self.f_start}, {self.f_end})")
        if self.psd_x < 0 or self.psd_y < 0:
            raise ValueError(f"PSD must be non-negative, got ({self.psd_x}, {self.psd_y})")

    @property
    def bandwidth(self) -> float:
        return self.f_end - self.f_start

    def total_power(self) -> float:
        """Integrated noise power in this bin [W], both polarizations."""
        return (self.psd_x + self.psd_y) * self.bandwidth

    def scale_power(self, factor: float) -> NoiseBin:
        """Return a copy with the PSD scaled by a *power* factor."""
        return replace(self, psd_x=self.psd_x * factor, psd_y=self.psd_y * factor)


@dataclass(frozen=True)
class OpticalSignal:
    """A set of sampled bands plus the noise accompanying them."""

    bands: tuple[Band, ...] = ()
    noise: tuple[NoiseBin, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "bands", tuple(self.bands))
        object.__setattr__(self, "noise", tuple(self.noise))
        centres = [b.f0 for b in self.bands]
        if len(set(centres)) != len(centres):
            raise ValueError(
                "two bands share a centre frequency; combining co-located carriers "
                "requires coherent addition on a common grid, which the caller must "
                "do explicitly"
            )

    @property
    def num_bands(self) -> int:
        return len(self.bands)

    def band_at(self, f0: float, rtol: float = 1e-9) -> Band:
        """The band whose centre frequency matches ``f0``."""
        for b in self.bands:
            if abs(b.f0 - f0) <= rtol * f0:
                return b
        raise KeyError(f"no band centred at {f0:.6e} Hz; have {[b.f0 for b in self.bands]}")

    def signal_power(self) -> float:
        """Total power in the sampled bands [W], excluding noise bins."""
        return sum(b.average_power() for b in self.bands)

    def noise_power(self) -> float:
        """Total power in the noise bins [W]."""
        return sum(n.total_power() for n in self.noise)

    def total_power(self) -> float:
        """Signal plus noise power [W]."""
        return self.signal_power() + self.noise_power()


@dataclass(frozen=True)
class ElectricalSignal:
    """A real-valued electrical waveform.

    ``unit`` records what the samples physically are — volts out of a driver,
    amperes out of a photodiode. It is carried rather than assumed so that a
    block receiving a waveform can tell whether it is being handed the right
    quantity, and so plots can label their axes without guessing.
    """

    samples: np.ndarray
    fs: float
    unit: str = "V"

    def __post_init__(self) -> None:
        if self.samples.ndim != 1:
            raise ValueError(f"samples must be 1-D, got {self.samples.ndim}-D")
        if np.issubdtype(self.samples.dtype, np.complexfloating):
            raise TypeError("an electrical waveform is real-valued, got a complex array")
        if self.fs <= 0:
            raise ValueError(f"fs must be positive, got {self.fs}")
        object.__setattr__(self, "samples", freeze(self.samples))

    @property
    def num_samples(self) -> int:
        return int(self.samples.shape[0])

    def mean(self) -> float:
        return float(np.mean(self.samples.astype(np.float64)))

    def variance(self) -> float:
        return float(np.var(self.samples.astype(np.float64)))


@dataclass(frozen=True)
class BinarySignal:
    """A sequence of bits, one per symbol.

    Bits are stored unsampled — one entry per symbol, not per sample. Upsampling
    to a waveform is a driver's job, and keeping the two apart means a receiver
    can compare decided bits against transmitted ones without having to undo a
    pulse shape first.
    """

    bits: np.ndarray
    symbol_rate: float

    def __post_init__(self) -> None:
        if self.bits.ndim != 1:
            raise ValueError(f"bits must be 1-D, got {self.bits.ndim}-D")
        if self.bits.dtype != np.uint8:
            raise TypeError(f"bits must be uint8 (0 or 1), got dtype {self.bits.dtype}")
        if self.bits.size and int(self.bits.max()) > 1:
            raise ValueError("bits must contain only 0 and 1")
        if self.symbol_rate <= 0:
            raise ValueError(f"symbol_rate must be positive, got {self.symbol_rate}")
        object.__setattr__(self, "bits", freeze(self.bits))

    @property
    def num_bits(self) -> int:
        return int(self.bits.shape[0])

    def ones_fraction(self) -> float:
        """Fraction of the sequence that is 1 — the mark density."""
        if self.num_bits == 0:
            return 0.0
        return float(np.mean(self.bits.astype(np.float64)))


@dataclass(frozen=True)
class BandPower:
    """Measured power of a single band."""

    f0: float
    wavelength_nm: float
    power_w: float

    @property
    def power_dbm(self) -> float:
        from .units import w_to_dbm

        return w_to_dbm(self.power_w)


@dataclass(frozen=True)
class PowerReading:
    """Result of a power measurement."""

    signal_power_w: float
    noise_power_w: float
    bands: tuple[BandPower, ...] = field(default=())

    @property
    def power_w(self) -> float:
        """Total measured power [W], signal plus noise."""
        return self.signal_power_w + self.noise_power_w

    @property
    def power_dbm(self) -> float:
        """Total measured power [dBm]."""
        from .units import w_to_dbm

        return w_to_dbm(self.power_w)

    def __repr__(self) -> str:
        per_band = ", ".join(f"{b.wavelength_nm:.2f}nm={b.power_dbm:.3f}dBm" for b in self.bands)
        return f"PowerReading({self.power_dbm:.3f} dBm" + (f"; {per_band})" if per_band else ")")


@dataclass(frozen=True)
class EyeHistogram:
    """A binned eye diagram — the reduced form of a waveform, not the waveform.

    Rendering an eye means binning a few million samples into a picture. Doing
    that binning here, in the engine, is what keeps raw sample buffers out of the
    UI: ``counts`` is a fixed-size array whose size depends on the requested
    resolution and not at all on how long the simulation ran.
    """

    counts: np.ndarray
    """2-D histogram, shape (amplitude_bins, time_bins)."""

    time_edges: np.ndarray
    """Bin edges along the time axis [s], spanning ``span_symbols`` symbols."""

    amplitude_edges: np.ndarray
    """Bin edges along the amplitude axis, in the waveform's own unit."""

    unit: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "counts", freeze(self.counts))
        object.__setattr__(self, "time_edges", freeze(self.time_edges))
        object.__setattr__(self, "amplitude_edges", freeze(self.amplitude_edges))

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.counts.shape[0]), int(self.counts.shape[1]))


@dataclass(frozen=True)
class EyeMeasurement:
    """Decision-circuit measurements taken on a received waveform."""

    q_factor: float
    """(mu1 - mu0) / (sigma1 + sigma0) at the sampling instant."""

    mean_one: float
    mean_zero: float
    std_one: float
    std_zero: float

    threshold: float
    """Decision level used, chosen for equal error probability."""

    sample_offset: int
    """Samples into the symbol at which the decision was taken."""

    bits_evaluated: int
    errors: int

    @property
    def ber_gaussian(self) -> float:
        """BER predicted from Q under the Gaussian-noise approximation."""
        from .analysis import ber_from_q

        return ber_from_q(self.q_factor)

    @property
    def ber_counted(self) -> float:
        """BER measured by comparing decided bits against the transmitted ones.

        Meaningful only when enough errors occurred to count: at a BER of 1e-9 a
        window of ten thousand bits will show zero errors and report 0.0, which
        is a statement about the window, not about the link.
        """
        if self.bits_evaluated == 0:
            return 0.0
        return self.errors / self.bits_evaluated

    @property
    def q_db(self) -> float:
        """Q expressed in dB (20*log10 Q), as it is usually quoted."""
        import math

        if self.q_factor <= 0:
            return -math.inf
        return 20.0 * math.log10(self.q_factor)

    def __repr__(self) -> str:
        return (
            f"EyeMeasurement(Q={self.q_factor:.3f}, BER={self.ber_gaussian:.3e}, "
            f"counted={self.errors}/{self.bits_evaluated})"
        )


#: Anything that may travel along an edge of the graph.
Signal = Any
