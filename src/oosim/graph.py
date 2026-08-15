"""The simulation graph and its execution engine.

Execution is **block-mode**: each component is invoked exactly once per run and
processes the entire time window in a single call. This is what commercial
system simulators do, and it makes every block a pure function of its inputs,
which in turn is what allows vectorised (and later GPU) implementations without
any change to the scheduler.

It is deliberately *not* a streaming, sample-at-a-time scheduler like GNU Radio.
Optical system simulation runs a fixed-length sequence and analyses the result;
streaming would buy nothing and complicate every block.
"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import replace
from itertools import pairwise
from typing import TypeVar

from .component import Component, Port, PortType
from .context import SimulationContext
from .signals import Signal

C = TypeVar("C", bound=Component)


class GraphError(Exception):
    """Invalid graph structure, detected before any component runs."""


class CycleError(GraphError):
    """The graph contains a feedback loop with no loop-control component.

    Cycles are not forbidden in principle — recirculating loops and optical
    feedback are real — but they need an explicit component that declares how
    many times the loop runs. Leaving the semantics implicit would make results
    depend on scheduler internals.
    """


class Results:
    """Signals produced by a run, addressed by component and port."""

    def __init__(self, values: dict[tuple[str, str], Signal]) -> None:
        self._values = values

    def port(self, component: Component, name: str) -> Signal:
        """The signal on one named output port."""
        key = (component.label, name)
        if key not in self._values:
            raise KeyError(
                f"no retained result for {component.label}.{name}; "
                f"intermediate signals are released once consumed — "
                f"pass keep=[{component.label!r}] to run() to retain it"
            )
        return self._values[key]

    def __getitem__(self, component: Component) -> Signal:
        """The signal on the component's only output port."""
        retained = [name for (label, name) in self._values if label == component.label]
        if not retained:
            raise KeyError(
                f"no retained result for {component.label}; "
                f"pass keep=[...] to run() to retain intermediate signals"
            )
        if len(retained) > 1:
            raise KeyError(
                f"{component.label} has {len(retained)} retained outputs "
                f"({sorted(retained)}); use results.port(component, name)"
            )
        return self._values[(component.label, retained[0])]

    def __contains__(self, component: Component) -> bool:
        return any(label == component.label for label, _ in self._values)

    def __repr__(self) -> str:
        keys = ", ".join(f"{label}.{name}" for label, name in sorted(self._values))
        return f"Results({keys})"


class Graph:
    """A directed graph of components, executed by :meth:`run`."""

    def __init__(self, ctx: SimulationContext) -> None:
        self.ctx = ctx
        self._components: list[Component] = []
        # (dst component label, dst port name) -> source Port
        self._edges: dict[tuple[str, str], Port] = {}

    def add(self, component: C) -> C:
        """Add a component and return it, so it can be assigned in one line.

        Components without an explicit label get one derived from their class and
        their position among components of that class in *this graph*. That makes
        labels a function of the graph alone, which matters because a component's
        label seeds its random stream: a process-global counter would make the
        same script produce different noise on every run.
        """
        if not component.has_explicit_label:
            same_kind = sum(1 for c in self._components if type(c) is type(component))
            component.label = f"{type(component).__name__}{same_kind + 1}"
        if any(c.label == component.label for c in self._components):
            raise GraphError(f"duplicate component label {component.label!r}")
        self._components.append(component)
        return component

    @property
    def components(self) -> tuple[Component, ...]:
        return tuple(self._components)

    @property
    def edges(self) -> dict[tuple[str, str], Port]:
        """Connections, keyed by the destination ``(label, port name)``."""
        return dict(self._edges)

    def connect(self, src: Port | Component, dst: Port | Component) -> None:
        """Wire an output port to an input port.

        Components may be passed directly when they have exactly one output or
        one input respectively, which covers most of a linear link.
        """
        src_port = src.sole_output() if isinstance(src, Component) else src
        dst_port = dst.sole_input() if isinstance(dst, Component) else dst

        if src_port.name not in src_port.component.outputs:
            raise GraphError(f"{src_port} is not an output port")
        if dst_port.name not in dst_port.component.inputs:
            raise GraphError(f"{dst_port} is not an input port")
        if src_port.type is not dst_port.type:
            raise GraphError(
                f"cannot connect {src_port} ({src_port.type}) to "
                f"{dst_port} ({dst_port.type}): port types differ"
            )

        key = (dst_port.component.label, dst_port.name)
        if key in self._edges:
            raise GraphError(
                f"{dst_port} is already driven by {self._edges[key]}; "
                f"an input port takes exactly one connection"
            )
        for component in (src_port.component, dst_port.component):
            if component not in self._components:
                raise GraphError(f"{component.label} was not added to the graph")
        self._edges[key] = src_port

    def chain(self, *components: Component) -> None:
        """Connect a sequence of single-input/single-output components in order."""
        for upstream, downstream in pairwise(components):
            self.connect(upstream, downstream)

    # -- validation and scheduling ----------------------------------------

    def _validate(self) -> None:
        for component in self._components:
            for name in component.inputs:
                if (component.label, name) not in self._edges:
                    raise GraphError(f"{component.label}.{name} is not connected")

    def _topological_order(self) -> list[Component]:
        by_label = {c.label: c for c in self._components}
        successors: dict[str, list[str]] = defaultdict(list)
        in_degree: dict[str, int] = {c.label: 0 for c in self._components}

        for (dst_label, _), src_port in self._edges.items():
            successors[src_port.component.label].append(dst_label)
            in_degree[dst_label] += 1

        ready = deque(sorted(label for label, deg in in_degree.items() if deg == 0))
        order: list[Component] = []
        while ready:
            label = ready.popleft()
            order.append(by_label[label])
            for successor in successors[label]:
                in_degree[successor] -= 1
                if in_degree[successor] == 0:
                    ready.append(successor)

        if len(order) != len(self._components):
            unresolved = sorted(label for label, deg in in_degree.items() if deg > 0)
            raise CycleError(
                f"the graph contains a feedback loop involving {unresolved}. "
                f"Break it with an explicit loop-control component that declares "
                f"an iteration count."
            )
        return order

    # -- execution --------------------------------------------------------

    def _resolve_label(self, component: Component | str) -> str:
        if isinstance(component, str):
            return component
        return component.label

    @contextmanager
    def _applied(
        self, overrides: Mapping[tuple[Component | str, str], float | bool] | None
    ) -> Iterator[None]:
        """Temporarily apply parameter overrides, restoring them afterwards.

        A sweep must not leave the graph altered: the same graph object is reused
        for every point, so a leaked value would silently contaminate every later
        run. Restoration happens even if the run raises.
        """
        if not overrides:
            yield
            return

        by_label = {c.label: c for c in self._components}
        saved: list[tuple[Component, dict[str, float | bool]]] = []
        try:
            for (target, param), value in overrides.items():
                label = self._resolve_label(target)
                component = by_label.get(label)
                if component is None:
                    raise GraphError(f"no component labelled {label!r} in this graph")
                spec = component.param_specs().get(param)
                if spec is None:
                    raise GraphError(f"{label} has no parameter {param!r}")
                saved.append((component, dict(component._values)))
                component._values = {**component._values, param: spec.validate(value)}
            yield
        finally:
            for component, values in reversed(saved):
                component._values = values

    def run(
        self,
        keep: list[Component] | None = None,
        *,
        overrides: Mapping[tuple[Component | str, str], float | bool] | None = None,
        seed: int | None = None,
    ) -> Results:
        """Execute every component once, in dependency order.

        Intermediate signals are released as soon as their last consumer has run,
        so peak memory is the width of the graph cut rather than the whole graph.
        Metric outputs, outputs of sink components, and anything named in ``keep``
        are retained and returned.

        ``overrides`` sets parameters for this run only, keyed by
        ``(component_or_label, parameter_name)``; the graph is left unchanged.
        ``seed`` replaces the context seed, which is how repeated runs draw
        independent noise from the same graph.
        """
        with self._applied(overrides):
            return self._run(keep, seed)

    def _run(self, keep: list[Component] | None, seed: int | None) -> Results:
        ctx = self.ctx if seed is None else replace(self.ctx, seed=seed)
        self._validate()
        order = self._topological_order()

        consumers_remaining: dict[tuple[str, str], int] = defaultdict(int)
        for src_port in self._edges.values():
            consumers_remaining[(src_port.component.label, src_port.name)] += 1

        keep_labels = {c.label for c in (keep or [])}
        live: dict[tuple[str, str], Signal] = {}
        retained: dict[tuple[str, str], Signal] = {}

        for component in order:
            inputs = {}
            for name in component.inputs:
                src = self._edges[(component.label, name)]
                inputs[name] = live[(src.component.label, src.name)]

            produced = component.run(ctx, inputs)

            missing = set(component.outputs) - set(produced)
            if missing:
                raise GraphError(
                    f"{component.label}.run() did not return output(s) {sorted(missing)}"
                )
            extra = set(produced) - set(component.outputs)
            if extra:
                raise GraphError(
                    f"{component.label}.run() returned undeclared output(s) {sorted(extra)}"
                )

            # No downstream consumer means this is a sink: its results are what
            # the caller actually asked for, so they are always retained.
            is_sink = all(
                consumers_remaining[(component.label, name)] == 0 for name in component.outputs
            )
            for name, value in produced.items():
                key = (component.label, name)
                live[key] = value
                if (
                    component.outputs[name] is PortType.METRIC
                    or is_sink
                    or component.label in keep_labels
                ):
                    retained[key] = value

            # Release inputs whose last consumer has now run.
            for name in component.inputs:
                src_port = self._edges[(component.label, name)]
                src_key = (src_port.component.label, src_port.name)
                consumers_remaining[src_key] -= 1
                if consumers_remaining[src_key] == 0 and src_key not in retained:
                    live.pop(src_key, None)

        return Results(retained)

    def __repr__(self) -> str:
        return f"Graph({len(self._components)} components, {len(self._edges)} edges)"
