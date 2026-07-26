"""CFEMP 板碰撞基准。"""

from .plate_impact import (
    CFEMPPlateImpact2D,
    PlateImpactConfig,
    SimulationHistory,
    analytical_contact_stress,
    analytical_separation_time,
)
from .plate_impact_3d import (
    PlateImpact3DConfig,
    SimulationHistory3D,
    TaichiCFEMPPlateImpact3D,
    analytical_contact_stress_3d,
    analytical_separation_time_3d,
)

__all__ = [
    "CFEMPPlateImpact2D",
    "PlateImpactConfig",
    "SimulationHistory",
    "analytical_contact_stress",
    "analytical_separation_time",
    "PlateImpact3DConfig",
    "SimulationHistory3D",
    "TaichiCFEMPPlateImpact3D",
    "analytical_contact_stress_3d",
    "analytical_separation_time_3d",
]
