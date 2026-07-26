"""独立的 MPM-FEM 惩罚接触示例。

该包用于展示惩罚函数接触，不冒充论文的拉格朗日乘子 CFEMP。
"""

from .contact import PenaltyContact, solve_contact_equilibrium
from .fem import FEMWallBeam
from .mpm import SmallStrainMPM2D

__all__ = [
    "FEMWallBeam",
    "PenaltyContact",
    "SmallStrainMPM2D",
    "solve_contact_equilibrium",
]
