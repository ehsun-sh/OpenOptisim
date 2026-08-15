"""Component base class, typed ports, and the parameter/unit system."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, overload

from .context import SimulationContext
from .signals import Signal
from .units import from_si, known_units, to_si


class PortType(Enum):
    """What kind of signal a port carries.

    Typing ports is what lets the editor reject invalid wiring at edit time
    instead of failing halfway through a run. An MZM has an electrical input as
    well as an optical one; without port types that distinction cannot be
    expressed at all.
    """

    OPTICAL = "optical"
    ELECTRICAL = "electrical"
    BINARY = "binary"
    SYMBOL = "symbol"
    METRIC = "metric"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class Port:
    """A reference to one named port on one component instance."""

    component: Component
    name: str
    type: PortType

    def __repr__(self) -> str:
        return f"{self.component.label}.{self.name}"


class Param:
    """A component parameter, declared once, with its unit and valid range.

    The unit is part of the declaration rather than a comment, so that
    :meth:`Component.si` can convert consistently and the GUI manifest can be
    generated from the same source. Declaring parameters twice — once in code
    and once in a manifest file — guarantees they drift apart.
    """

    def __init__(
        self,
        default: float,
        *,
        unit: str = "",
        min: float | None = None,
        max: float | None = None,
        doc: str = "",
    ) -> None:
        if unit not in known_units():
            raise ValueError(f"unknown unit {unit!r}; known units: {sorted(known_units())}")
        self.default = default
        self.unit = unit
        self.min = min
        self.max = max
        self.doc = doc
        self.name = "<unbound>"

    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name

    @overload
    def __get__(self, obj: None, owner: type | None = None) -> Param: ...

    @overload
    def __get__(self, obj: Component, owner: type | None = None) -> float: ...

    def __get__(self, obj: Component | None, owner: type | None = None) -> Param | float:
        if obj is None:
            return self
        return obj._values.get(self.name, self.default)

    def validate(self, value: float) -> float:
        if not isinstance(value, int | float) or isinstance(value, bool):
            raise TypeError(f"{self.name} must be a number, got {value!r}")
        value = float(value)
        if math.isnan(value):
            raise ValueError(f"{self.name} must not be NaN")
        if self.min is not None and value < self.min:
            raise ValueError(f"{self.name}={value} {self.unit} is below the minimum {self.min}")
        if self.max is not None and value > self.max:
            raise ValueError(f"{self.name}={value} {self.unit} is above the maximum {self.max}")
        return value

    def to_dict(self) -> dict[str, Any]:
        """The JSON-serialisable description used to generate GUI manifests."""
        d: dict[str, Any] = {"type": "float", "default": self.default, "unit": self.unit}
        if self.min is not None:
            d["min"] = self.min
        if self.max is not None:
            d["max"] = self.max
        if self.doc:
            d["doc"] = self.doc
        return d


class BoolParam(Param):
    """A component parameter that is a flag rather than a quantity.

    A separate class rather than a mode on :class:`Param` so that a float
    parameter still types as ``float`` at every use site. Flags are read as plain
    attributes; :meth:`Component.si` refuses them, because converting a switch to
    an SI unit is meaningless and asking for it means something is confused.
    """

    def __init__(self, default: bool, *, doc: str = "") -> None:
        super().__init__(0.0, doc=doc)
        self.default = default

    @overload
    def __get__(self, obj: None, owner: type | None = None) -> BoolParam: ...

    @overload
    def __get__(self, obj: Component, owner: type | None = None) -> bool: ...

    def __get__(self, obj: Component | None, owner: type | None = None) -> BoolParam | bool:
        if obj is None:
            return self
        return bool(obj._values.get(self.name, self.default))

    def validate(self, value: object) -> bool:
        if not isinstance(value, bool):
            raise TypeError(f"{self.name} must be True or False, got {value!r}")
        return value

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"type": "bool", "default": self.default}
        if self.doc:
            d["doc"] = self.doc
        return d


class Component:
    """Base class for every block in a simulation graph.

    A component is a pure function of its inputs and its parameters: it receives
    read-only signals and returns new ones. Nothing is mutated in place, so the
    scheduler is free to order, cache, or parallelise execution.
    """

    #: Human-readable name, shown in the GUI palette.
    display_name: ClassVar[str] = ""

    #: Palette grouping.
    category: ClassVar[str] = "Uncategorised"

    #: Model version, bumped when numerical behaviour changes.
    version: ClassVar[str] = "0.1.0"

    # Port declarations. These are class-level defaults that `__init__` copies
    # onto the instance, so a component with a configurable port count (an N-way
    # combiner, say) can rebind its own without touching the class.
    inputs: dict[str, PortType] = {}
    outputs: dict[str, PortType] = {}

    def __init__(self, *, label: str | None = None, **params: float | bool) -> None:
        self.inputs = dict(type(self).inputs)
        self.outputs = dict(type(self).outputs)
        declared = self.param_specs()
        unknown = set(params) - set(declared)
        if unknown:
            raise TypeError(
                f"{type(self).__name__} has no parameter(s) {sorted(unknown)}; "
                f"declared: {sorted(declared)}"
            )
        self._values: dict[str, float | bool] = {
            name: declared[name].validate(value) for name, value in params.items()
        }

        # A label identifies the component to the RNG, so it must depend only on
        # the graph, never on how many components happened to be constructed
        # earlier in the process. Graph.add() assigns the automatic form.
        self.has_explicit_label = label is not None
        self.label = label if label is not None else type(self).__name__

    # -- introspection ----------------------------------------------------

    @classmethod
    def param_specs(cls) -> dict[str, Param]:
        """Every :class:`Param` declared on this class or a base class."""
        specs: dict[str, Param] = {}
        for klass in reversed(cls.__mro__):
            for name, value in vars(klass).items():
                if isinstance(value, Param):
                    specs[name] = value
        return specs

    @classmethod
    def manifest(cls) -> dict[str, Any]:
        """The component description consumed by the GUI.

        Generated from the class, never hand-written, so the schema cannot drift
        away from the implementation.
        """
        return {
            "name": cls.display_name or cls.__name__,
            "class": f"{cls.__module__}.{cls.__qualname__}",
            "category": cls.category,
            "version": cls.version,
            "parameters": {name: spec.to_dict() for name, spec in cls.param_specs().items()},
            "ports": {
                "inputs": {name: str(t) for name, t in cls.inputs.items()},
                "outputs": {name: str(t) for name, t in cls.outputs.items()},
            },
        }

    def si(self, name: str) -> float:
        """The value of parameter ``name`` converted to its SI base unit."""
        spec = self.param_specs().get(name)
        if spec is None:
            raise KeyError(f"{type(self).__name__} has no parameter {name!r}")
        if isinstance(spec, BoolParam):
            raise TypeError(f"{name} is a flag, not a quantity; read it as an attribute")
        return to_si(getattr(self, name), spec.unit)

    def display(self, name: str) -> tuple[float, str]:
        """The value of parameter ``name`` in its declared unit, with the unit."""
        spec = self.param_specs().get(name)
        if spec is None:
            raise KeyError(f"{type(self).__name__} has no parameter {name!r}")
        return from_si(self.si(name), spec.unit), spec.unit

    # -- ports ------------------------------------------------------------

    def __getitem__(self, port_name: str) -> Port:
        if port_name in self.outputs:
            return Port(self, port_name, self.outputs[port_name])
        if port_name in self.inputs:
            return Port(self, port_name, self.inputs[port_name])
        raise KeyError(
            f"{type(self).__name__} has no port {port_name!r}; "
            f"inputs={sorted(self.inputs)} outputs={sorted(self.outputs)}"
        )

    def sole_output(self) -> Port:
        """The only output port, for chaining. Raises if there is not exactly one."""
        if len(self.outputs) != 1:
            raise ValueError(
                f"{self.label} has {len(self.outputs)} outputs; name one explicitly, "
                f"e.g. {self.label}['{next(iter(self.outputs), 'out')}']"
            )
        return self[next(iter(self.outputs))]

    def sole_input(self) -> Port:
        """The only input port, for chaining. Raises if there is not exactly one."""
        if len(self.inputs) != 1:
            raise ValueError(
                f"{self.label} has {len(self.inputs)} inputs; name one explicitly, "
                f"e.g. {self.label}['{next(iter(self.inputs), 'in')}']"
            )
        return self[next(iter(self.inputs))]

    # -- execution --------------------------------------------------------

    def run(self, ctx: SimulationContext, inputs: dict[str, Signal]) -> dict[str, Signal]:
        """Process the whole time window and return one signal per output port.

        Called exactly once per run. Implementations must not mutate ``inputs``
        and must not hold state between calls.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement run()")

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v!r}" for k, v in sorted(self._values.items()))
        return f"{type(self).__name__}({params})" if params else f"{type(self).__name__}()"
