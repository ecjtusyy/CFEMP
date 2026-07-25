"""CFEMP 论文基准的可复现实装。"""

from .plate_impact import (
    CFEMPPlateImpact1D,
    PlateImpactConfig,
    SimulationHistory,
    analytical_contact_stress,
    analytical_separation_time,
)

__all__ = [
    "CFEMPPlateImpact1D",
    "PlateImpactConfig",
    "SimulationHistory",
    "analytical_contact_stress",
    "analytical_separation_time",
]
