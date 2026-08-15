"""OpenOptiSim — simulation of optical communication links and photonic systems.

The public surface is deliberately small. Everything the GUI will eventually do
goes through it: if a feature is not reachable from here, it does not exist.

    >>> from oosim import SimulationContext, Graph
    >>> from oosim.components import CWLaser, Fiber, PowerMeter
    >>> ctx = SimulationContext(bit_rate=10e9, samples_per_symbol=16, sequence_length=64)
    >>> g = Graph(ctx)
    >>> laser = g.add(CWLaser(power=0.0, wavelength=1550.0))
    >>> fiber = g.add(Fiber(length=80.0, attenuation=0.2))
    >>> meter = g.add(PowerMeter())
    >>> g.chain(laser, fiber, meter)
    >>> round(g.run()[meter].power_dbm, 3)
    -16.0
"""

from __future__ import annotations

from .component import BoolParam, Component, Param, Port, PortType
from .context import SimulationContext
from .graph import CycleError, Graph, GraphError, Results
from .signals import (
    Band,
    BandPower,
    BinarySignal,
    ElectricalSignal,
    EyeHistogram,
    EyeMeasurement,
    NoiseBin,
    OpticalSignal,
    PowerReading,
)

__version__ = "0.0.1.dev0"

__all__ = [
    "Band",
    "BandPower",
    "BinarySignal",
    "BoolParam",
    "Component",
    "CycleError",
    "ElectricalSignal",
    "EyeHistogram",
    "EyeMeasurement",
    "Graph",
    "GraphError",
    "NoiseBin",
    "OpticalSignal",
    "Param",
    "Port",
    "PortType",
    "PowerReading",
    "Results",
    "SimulationContext",
    "__version__",
]
