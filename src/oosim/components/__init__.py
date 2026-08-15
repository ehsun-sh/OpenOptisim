"""Built-in component library.

Components are ordinary Python classes. Third-party components install as normal
Python packages — no build step and no ABI to match, which is the difference
between a plugin system researchers will actually use and one they will not.
"""

from __future__ import annotations

from .analyzers import BERAnalyzer, EyeDiagram
from .detectors import PINPhotodiode
from .electrical import DCVoltage, NRZDriver, PRBSGenerator
from .fiber import Fiber
from .filters import ElectricalFilter
from .meters import PowerMeter
from .modulators import MachZehnderModulator
from .passive import Attenuator, Combiner
from .sources import CWLaser, GaussianPulse

__all__ = [
    "Attenuator",
    "BERAnalyzer",
    "CWLaser",
    "Combiner",
    "DCVoltage",
    "ElectricalFilter",
    "EyeDiagram",
    "Fiber",
    "GaussianPulse",
    "MachZehnderModulator",
    "NRZDriver",
    "PINPhotodiode",
    "PRBSGenerator",
    "PowerMeter",
]
