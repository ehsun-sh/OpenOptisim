"""Global simulation parameters shared by every block in a run."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal

import numpy as np

Precision = Literal["single", "double"]


@dataclass(frozen=True)
class SimulationContext:
    """Parameters that every component in a run must agree on.

    Sample rate, sample count and time origin belong to the *run*, not to
    individual signals. If each signal carried its own, two blocks could
    silently disagree about the time window and the simulation would be
    meaningless while still producing plausible-looking numbers.
    """

    bit_rate: float
    """Symbol (or bit) rate [Bd or b/s]."""

    samples_per_symbol: int
    """Oversampling factor. Must be at least 2 to satisfy Nyquist."""

    sequence_length: int
    """Number of symbols in the time window."""

    seed: int = 0
    """Master seed. Every stochastic block derives its generator from this."""

    precision: Precision = "single"
    """Sampled-field precision. ``single`` (complex64) is the default: double
    precision doubles memory traffic and FFT cost for accuracy that
    system-level optical simulation rarely needs."""

    def __post_init__(self) -> None:
        if self.bit_rate <= 0:
            raise ValueError(f"bit_rate must be positive, got {self.bit_rate}")
        if self.samples_per_symbol < 2:
            raise ValueError(
                f"samples_per_symbol must be >= 2 (Nyquist), got {self.samples_per_symbol}"
            )
        if self.sequence_length < 1:
            raise ValueError(f"sequence_length must be >= 1, got {self.sequence_length}")

    @property
    def sample_rate(self) -> float:
        """Fs [Hz]."""
        return self.bit_rate * self.samples_per_symbol

    @property
    def num_samples(self) -> int:
        """N, the number of samples in the time window."""
        return self.sequence_length * self.samples_per_symbol

    @property
    def time_step(self) -> float:
        """dt [s]."""
        return 1.0 / self.sample_rate

    @property
    def time_window(self) -> float:
        """T [s], the total simulated duration."""
        return self.num_samples / self.sample_rate

    @property
    def complex_dtype(self) -> np.dtype[np.complexfloating]:
        """NumPy dtype for sampled complex fields."""
        return np.dtype(np.complex64 if self.precision == "single" else np.complex128)

    @property
    def real_dtype(self) -> np.dtype[np.floating]:
        """NumPy dtype for sampled real waveforms."""
        return np.dtype(np.float32 if self.precision == "single" else np.float64)

    def time_axis(self) -> np.ndarray:
        """Sample times [s], starting at zero."""
        return np.arange(self.num_samples, dtype=np.float64) * self.time_step

    def rng(self, *key: object) -> np.random.Generator:
        """A generator derived deterministically from the seed and ``key``.

        Stochastic blocks must draw from this, never from a global generator.
        Because each block's stream depends only on its own identity, adding or
        removing a block does not perturb the noise realisation of any other —
        which is what makes regression tests over noisy results possible at all.
        """
        digest = hashlib.blake2b(repr(key).encode("utf-8"), digest_size=8).digest()
        stream = int.from_bytes(digest, "big")
        return np.random.default_rng(np.random.SeedSequence([self.seed, stream]))
