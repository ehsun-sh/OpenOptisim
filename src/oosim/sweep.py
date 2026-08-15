"""Parameter sweeps and repeated runs.

A single simulation is rarely the answer to an engineering question. "What is
the sensitivity?" and "how far can this reach?" are both curves, and producing
them by hand-writing a loop that mutates a graph is exactly where a stray value
gets left behind and quietly contaminates every later point. Sweeping is
first-class for that reason, and overrides are applied per run and rolled back.

Points are independent, which makes running them in parallel straightforward.
That is not implemented yet: execution here is sequential.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from typing import Any, TypeVar

import numpy as np

from .component import Component
from .graph import Graph, Results

T = TypeVar("T")

#: An axis is identified by the component (or its label) and a parameter name.
AxisKey = tuple[Component | str, str]


def _label(target: Component | str) -> str:
    return target if isinstance(target, str) else target.label


def derive_run_seed(base_seed: int, run_index: int) -> int:
    """A seed for repeated run ``run_index``, derived from ``base_seed``.

    Hashed rather than incremented. Adjacent seeds are not a problem for a
    modern generator, but deriving by hash keeps the same guarantee the
    per-block streams rely on — that seeds which look related are not — and it
    costs nothing.
    """
    digest = hashlib.blake2b(f"{base_seed}:{run_index}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big")


@dataclass(frozen=True)
class SweepPoint:
    """One combination of swept parameter values, with the runs taken there."""

    index: int
    values: dict[str, float | bool]
    """Swept values at this point, keyed ``"label.parameter"``."""

    runs: tuple[Results, ...]

    def __repr__(self) -> str:
        settings = ", ".join(f"{k}={v}" for k, v in self.values.items())
        return f"SweepPoint({settings}; {len(self.runs)} run(s))"


@dataclass(frozen=True)
class SweepResult:
    """Every point of a sweep, in the order the axes were given."""

    points: tuple[SweepPoint, ...]
    axes: dict[str, tuple[float | bool, ...]]

    def __len__(self) -> int:
        return len(self.points)

    def __iter__(self) -> Any:
        return iter(self.points)

    def axis(self, target: Component | str, parameter: str) -> np.ndarray:
        """The values one axis took, in sweep order."""
        key = f"{_label(target)}.{parameter}"
        if key not in self.axes:
            raise KeyError(f"{key!r} was not swept; axes were {sorted(self.axes)}")
        return np.array([point.values[key] for point in self.points], dtype=np.float64)

    def metric(self, component: Component, extract: Callable[[Any], float]) -> np.ndarray:
        """Pull one number out of every run, shaped ``(points, runs)``.

        ``extract`` receives whatever the component put on its output port::

            q = result.metric(analyzer, lambda m: m.q_factor)
            mean_q = q.mean(axis=1)
        """
        return np.array(
            [[float(extract(run[component])) for run in point.runs] for point in self.points],
            dtype=np.float64,
        )


def sweep(
    graph: Graph,
    axes: Mapping[AxisKey, Sequence[float | bool]],
    *,
    runs: int = 1,
    keep: list[Component] | None = None,
    fixed: Mapping[AxisKey, float | bool] | None = None,
) -> SweepResult:
    """Run ``graph`` once for every combination of the given parameter values.

    ``axes`` maps ``(component, parameter)`` to the values it should take; the
    sweep covers their cartesian product, varying the last axis fastest.

    ``runs`` repeats each point with independently derived seeds. That is what
    turns a noisy measurement into a distribution — a single BER estimate at a
    marginal operating point is one sample, not an answer.

    ``fixed`` applies the same override at every point, which saves editing the
    graph just to hold something constant for one study.

    The graph is left exactly as it was found, including if a run raises.
    """
    if runs < 1:
        raise ValueError(f"runs must be >= 1, got {runs}")
    if not axes:
        raise ValueError("a sweep needs at least one axis")

    keys = list(axes)
    named_axes = {f"{_label(t)}.{p}": tuple(axes[(t, p)]) for (t, p) in keys}
    for name, axis_values in named_axes.items():
        if not axis_values:
            raise ValueError(f"axis {name!r} has no values")

    points: list[SweepPoint] = []
    for index, combination in enumerate(product(*(axes[key] for key in keys))):
        overrides: dict[AxisKey, float | bool] = dict(fixed or {})
        overrides.update(dict(zip(keys, combination, strict=True)))

        results = tuple(
            graph.run(
                keep,
                overrides=overrides,
                seed=derive_run_seed(graph.ctx.seed, run_index) if runs > 1 else None,
            )
            for run_index in range(runs)
        )
        values = {
            f"{_label(target)}.{parameter}": value
            for (target, parameter), value in zip(keys, combination, strict=True)
        }
        points.append(SweepPoint(index=index, values=values, runs=results))

    return SweepResult(points=tuple(points), axes=named_axes)
