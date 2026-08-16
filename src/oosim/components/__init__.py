"""Built-in component library.

Components are ordinary Python classes. Third-party components install as normal
Python packages — no build step and no ABI to match, which is the difference
between a plugin system researchers will actually use and one they will not.
"""

from __future__ import annotations

from .amplifiers import EDFA
from .analyzers import BERAnalyzer, ConstellationAnalyzer, ConstellationDiagram, EyeDiagram
from .coherent import (
    CarrierRecovery,
    CoherentReceiver,
    DualPolarizationReceiver,
    IQSampler,
)
from .detectors import APDPhotodiode, PINPhotodiode
from .dsp import ButterflyEqualizer
from .electrical import DCVoltage, IQDriver, NRZDriver, PRBSGenerator
from .fiber import Fiber
from .filters import ElectricalFilter
from .mapping import DifferentialDecoder, QAMMapper
from .meters import OSNRMeter, PowerMeter
from .modulators import IQModulator, MachZehnderModulator
from .passive import (
    Attenuator,
    Combiner,
    PolarizationCombiner,
    PolarizationRotator,
    Splitter,
)
from .sources import CWLaser, GaussianPulse, SechPulse

__all__ = [
    "EDFA",
    "APDPhotodiode",
    "Attenuator",
    "BERAnalyzer",
    "ButterflyEqualizer",
    "CWLaser",
    "CarrierRecovery",
    "CoherentReceiver",
    "Combiner",
    "ConstellationAnalyzer",
    "ConstellationDiagram",
    "DCVoltage",
    "DifferentialDecoder",
    "DualPolarizationReceiver",
    "ElectricalFilter",
    "EyeDiagram",
    "Fiber",
    "GaussianPulse",
    "IQDriver",
    "IQModulator",
    "IQSampler",
    "MachZehnderModulator",
    "NRZDriver",
    "OSNRMeter",
    "PINPhotodiode",
    "PRBSGenerator",
    "PolarizationCombiner",
    "PolarizationRotator",
    "PowerMeter",
    "QAMMapper",
    "SechPulse",
    "Splitter",
]
