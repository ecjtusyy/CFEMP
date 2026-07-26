"""MPM 粒子与 FEM 梁之间的法向惩罚接触。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fem import FEMWallBeam
from .mpm import SmallStrainMPM2D


@dataclass(frozen=True)
class ContactResult:
    active_particles: int
    max_penetration: float
    particle_resultant: np.ndarray
    wall_resultant: np.ndarray

    @property
    def balance_error(self) -> float:
        return float(
            np.linalg.norm(self.particle_resultant + self.wall_resultant)
        )


class PenaltyContact:
    """墙在左、粒子在右；墙对粒子的法向为 ``(+1, 0)``。"""

    def __init__(
        self,
        stiffness: float,
        particle_radius: float,
        *,
        damping: float = 0.0,
        max_penetration: float | None = None,
    ):
        if stiffness <= 0.0 or particle_radius <= 0.0:
            raise ValueError("惩罚刚度和粒子半径必须为正")
        self.stiffness = float(stiffness)
        self.particle_radius = float(particle_radius)
        self.damping = float(damping)
        self.penetration_cap = (
            float(max_penetration)
            if max_penetration is not None
            else particle_radius
        )

    def evaluate(
        self,
        mpm: SmallStrainMPM2D,
        wall: FEMWallBeam,
    ) -> tuple[np.ndarray, ContactResult]:
        particle_force = np.zeros_like(mpm.x)
        wall.clear_loads()
        wall_nodes = wall.nodes
        active = 0
        max_penetration = 0.0

        for p, (position, velocity) in enumerate(zip(mpm.x, mpm.v)):
            local = (position[1] - wall.y0) / wall.dy
            segment = int(np.floor(local))
            if segment < 0 or segment >= wall.segments:
                continue
            eta = float(np.clip(local - segment, 0.0, 1.0))
            wall_x = (
                (1.0 - eta) * wall_nodes[segment, 0]
                + eta * wall_nodes[segment + 1, 0]
            )
            gap = position[0] - self.particle_radius - wall_x
            if gap >= 0.0:
                continue

            penetration = min(-gap, self.penetration_cap)
            # v_x < 0 表示粒子向墙接近，阻尼力必须指向 +x。
            magnitude = (
                self.stiffness * penetration
                + self.damping * max(-float(velocity[0]), 0.0)
            )
            force = np.array([magnitude, 0.0])
            particle_force[p] = force
            wall.add_horizontal_force(segment, -(1.0 - eta) * magnitude)
            wall.add_horizontal_force(segment + 1, -eta * magnitude)
            active += 1
            max_penetration = max(max_penetration, penetration)

        particle_resultant = np.sum(particle_force, axis=0)
        wall_resultant = np.array([np.sum(wall.force[0::2]), 0.0])
        result = ContactResult(
            active_particles=active,
            max_penetration=max_penetration,
            particle_resultant=particle_resultant,
            wall_resultant=wall_resultant,
        )
        return particle_force, result


def solve_contact_equilibrium(
    mpm: SmallStrainMPM2D,
    wall: FEMWallBeam,
    contact: PenaltyContact,
    *,
    iterations: int = 3,
) -> ContactResult:
    """反复更新墙位移，但只把最终一轮接触力写给粒子一次。"""

    if iterations < 1:
        raise ValueError("iterations 至少为 1")
    final_force = np.zeros_like(mpm.x)
    result: ContactResult | None = None
    for _ in range(iterations):
        final_force, result = contact.evaluate(mpm, wall)
        wall.solve_static()
    mpm.contact_force[:] = final_force
    assert result is not None
    return result
