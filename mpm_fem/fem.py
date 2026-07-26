"""竖直 Euler-Bernoulli 梁，用作惩罚接触示例中的 FEM 墙。"""

from __future__ import annotations

import numpy as np


class FEMWallBeam:
    def __init__(
        self,
        x: float,
        y: float,
        height: float,
        segments: int,
        *,
        young: float,
        inertia: float,
    ):
        self.x0 = float(x)
        self.y0 = float(y)
        self.height = float(height)
        self.segments = int(segments)
        self.young = float(young)
        self.inertia = float(inertia)
        ys = np.linspace(y, y + height, segments + 1)
        self.nodes0 = np.column_stack([np.full_like(ys, x), ys])
        self.dof = 2 * (segments + 1)
        self.q = np.zeros(self.dof)
        self.force = np.zeros(self.dof)
        self.fixed = np.zeros(self.dof, dtype=bool)
        self.fixed[:2] = True
        self.dy = height / segments

    @property
    def nodes(self) -> np.ndarray:
        result = self.nodes0.copy()
        result[:, 0] += self.q[0::2]
        return result

    def clear_loads(self) -> None:
        self.force.fill(0.0)

    def add_horizontal_force(self, node: int, force: float) -> None:
        self.force[2 * node] += force

    def stiffness(self) -> np.ndarray:
        length = self.dy
        ei = self.young * self.inertia
        local = ei / length**3 * np.array(
            [
                [12.0, 6.0 * length, -12.0, 6.0 * length],
                [6.0 * length, 4.0 * length**2, -6.0 * length, 2.0 * length**2],
                [-12.0, -6.0 * length, 12.0, -6.0 * length],
                [6.0 * length, 2.0 * length**2, -6.0 * length, 4.0 * length**2],
            ]
        )
        matrix = np.zeros((self.dof, self.dof))
        for element in range(self.segments):
            ids = np.array(
                [
                    2 * element,
                    2 * element + 1,
                    2 * element + 2,
                    2 * element + 3,
                ]
            )
            matrix[np.ix_(ids, ids)] += local
        return matrix

    def solve_static(self) -> None:
        free = ~self.fixed
        matrix = self.stiffness()
        self.q[free] = np.linalg.solve(
            matrix[np.ix_(free, free)], self.force[free]
        )
        self.q[self.fixed] = 0.0
