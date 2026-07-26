"""二维无摩擦接触投影。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ProjectionResult:
    active: bool
    closing_speed: float
    impulse_on_mpm: np.ndarray

    @property
    def magnitude(self) -> float:
        return float(np.linalg.norm(self.impulse_on_mpm))


def project_normal_momentum(
    fem_mass: float,
    fem_momentum: np.ndarray,
    mpm_mass: float,
    mpm_momentum: np.ndarray,
    normal: np.ndarray,
    *,
    tolerance: float = 0.0,
) -> ProjectionResult:
    """把一对二维动量投影到非穿透速度集合。"""

    if fem_mass <= 0.0 or mpm_mass <= 0.0:
        raise ValueError("接触质量必须为正")

    fem_momentum = np.asarray(fem_momentum, dtype=float)
    mpm_momentum = np.asarray(mpm_momentum, dtype=float)
    normal = np.asarray(normal, dtype=float)
    if fem_momentum.shape != (2,) or mpm_momentum.shape != (2,):
        raise ValueError("二维动量必须是长度为 2 的向量")
    if normal.shape != (2,):
        raise ValueError("二维法向必须是长度为 2 的向量")

    norm = float(np.linalg.norm(normal))
    if norm <= 0.0:
        raise ValueError("接触法向不能为零")
    normal = normal / norm

    fem_velocity = fem_momentum / fem_mass
    mpm_velocity = mpm_momentum / mpm_mass
    closing = float(np.dot(fem_velocity - mpm_velocity, normal))
    if closing <= tolerance:
        return ProjectionResult(False, closing, np.zeros(2))

    reduced_mass = fem_mass * mpm_mass / (fem_mass + mpm_mass)
    return ProjectionResult(True, closing, reduced_mass * closing * normal)
