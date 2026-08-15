"""Built-in component library.

Components are ordinary Python classes. Third-party components install as normal
Python packages — no build step and no ABI to match, which is the difference
between a plugin system researchers will actually use and one they will not.
"""

from __future__ import annotations

from .amplifiers import EDFA
from .analyzers import BERAnalyzer, EyeDiagram
from .detectors import PINPhotodiode
from .electrical import DCVoltage, NRZDriver, PRBSGenerator
from .fiber import Fiber
from .filters import ElectricalFilter
from .meters import OSNRMeter, PowerMeter
from .modulators import MachZehnderModulator
from .passive import Attenuator, Combiner
from .sources import CWLaser, GaussianPulse, SechPulse

__all__ = [
    "EDFA",
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
    "OSNRMeter",
    "PINPhotodiode",
    "PRBSGenerator",
    "PowerMeter",
    "SechPulse",
]
