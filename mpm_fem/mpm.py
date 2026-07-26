"""小应变二维 MPM，供惩罚接触示例使用。"""

from __future__ import annotations

import numpy as np

from .common import hat_weights_and_grads, node_id


class SmallStrainMPM2D:
    def __init__(
        self,
        origin: tuple[float, float],
        dx: float,
        nx: int,
        ny: int,
        *,
        density: float,
        young: float,
        poisson: float,
        gravity: float = -9.81,
    ):
        self.origin = np.asarray(origin, dtype=float)
        self.dx = float(dx)
        self.nx = int(nx)
        self.ny = int(ny)
        self.node_count = self.nx * self.ny
        self.density = float(density)
        self.gravity = float(gravity)
        self.mu = young / (2.0 * (1.0 + poisson))
        self.lam = young * poisson / (
            (1.0 + poisson) * (1.0 - 2.0 * poisson)
        )

        self.x = np.zeros((0, 2), dtype=float)
        self.v = np.zeros((0, 2), dtype=float)
        self.mass = np.zeros(0, dtype=float)
        self.volume = np.zeros(0, dtype=float)
        self.strain = np.zeros((0, 2, 2), dtype=float)
        self.stress = np.zeros((0, 2, 2), dtype=float)
        self.contact_force = np.zeros((0, 2), dtype=float)

        self.grid_mass = np.zeros(self.node_count, dtype=float)
        self.grid_momentum = np.zeros((self.node_count, 2), dtype=float)
        self.grid_internal_force = np.zeros((self.node_count, 2), dtype=float)
        self.grid_external_force = np.zeros((self.node_count, 2), dtype=float)
        self.fixed = np.zeros((self.node_count, 2), dtype=bool)

    def add_rect_particles(
        self,
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
        *,
        particles_per_cell: int = 2,
        initial_velocity: tuple[float, float] = (0.0, 0.0),
    ) -> float:
        spacing = self.dx / particles_per_cell
        radius = 0.49 * spacing
        xs = np.arange(xmin, xmax, spacing, dtype=float)
        ys = np.arange(ymin, ymax, spacing, dtype=float)
        xx, yy = np.meshgrid(xs, ys, indexing="xy")
        points = np.column_stack([xx.ravel(), yy.ravel()])
        count = len(points)
        volume = spacing**2

        self.x = np.concatenate([self.x, points])
        self.v = np.concatenate(
            [
                self.v,
                np.tile(np.asarray(initial_velocity, dtype=float), (count, 1)),
            ]
        )
        self.mass = np.concatenate(
            [self.mass, np.full(count, self.density * volume)]
        )
        self.volume = np.concatenate(
            [self.volume, np.full(count, volume)]
        )
        self.strain = np.concatenate(
            [self.strain, np.zeros((count, 2, 2))], axis=0
        )
        self.stress = np.concatenate(
            [self.stress, np.zeros((count, 2, 2))], axis=0
        )
        self.contact_force = np.concatenate(
            [self.contact_force, np.zeros((count, 2))], axis=0
        )
        return radius

    def set_bottom_roller(self) -> None:
        y0 = self.origin[1]
        for i in range(self.nx):
            self.fixed[node_id(i, 0, self.nx), 1] = True
        # 去除水平方向刚体漂移。
        self.fixed[node_id(0, 0, self.nx), 0] = True

    def clear_grid(self) -> None:
        self.grid_mass.fill(0.0)
        self.grid_momentum.fill(0.0)
        self.grid_internal_force.fill(0.0)
        self.grid_external_force.fill(0.0)

    def p2g(self, gravity_scale: float) -> None:
        self.clear_grid()
        for p, xp in enumerate(self.x):
            nodes, weights, grads = hat_weights_and_grads(
                xp, self.origin, self.dx, self.nx, self.ny
            )
            for node, weight, grad in zip(nodes, weights, grads):
                index = node_id(*node, self.nx)
                self.grid_mass[index] += weight * self.mass[p]
                self.grid_momentum[index] += (
                    weight * self.mass[p] * self.v[p]
                )
                # 这里已经包含内力的负号，后面必须直接相加。
                self.grid_internal_force[index] += (
                    -self.volume[p] * (self.stress[p] @ grad)
                )
                self.grid_external_force[index] += (
                    weight * self.contact_force[p]
                )

        gravity = np.array([0.0, self.gravity * gravity_scale])
        self.grid_external_force += self.grid_mass[:, None] * gravity

    def advance_grid(self, dt: float, damping: float) -> None:
        active = self.grid_mass > 0.0
        velocity = np.zeros_like(self.grid_momentum)
        velocity[active] = (
            self.grid_momentum[active] / self.grid_mass[active, None]
        )
        total_force = self.grid_internal_force + self.grid_external_force
        velocity[active] += (
            dt * total_force[active] / self.grid_mass[active, None]
        )
        velocity[active] *= 1.0 - float(np.clip(damping, 0.0, 0.98))
        velocity[self.fixed] = 0.0
        self.grid_momentum[active] = (
            velocity[active] * self.grid_mass[active, None]
        )

    def g2p(self, dt: float) -> None:
        velocity_grid = np.zeros_like(self.grid_momentum)
        active = self.grid_mass > 0.0
        velocity_grid[active] = (
            self.grid_momentum[active] / self.grid_mass[active, None]
        )
        for p, xp in enumerate(self.x.copy()):
            nodes, weights, grads = hat_weights_and_grads(
                xp, self.origin, self.dx, self.nx, self.ny
            )
            vp = np.zeros(2)
            grad_v = np.zeros((2, 2))
            for node, weight, grad in zip(nodes, weights, grads):
                index = node_id(*node, self.nx)
                vp += weight * velocity_grid[index]
                grad_v += np.outer(velocity_grid[index], grad)

            self.v[p] = vp
            self.x[p] += dt * vp
            deps = 0.5 * (grad_v + grad_v.T) * dt
            self.strain[p] += deps
            trace = np.trace(self.strain[p])
            self.stress[p] = (
                2.0 * self.mu * self.strain[p]
                + self.lam * trace * np.eye(2)
            )

        lower = self.origin + 1.0e-10
        upper = self.origin + np.array(
            [(self.nx - 1) * self.dx, (self.ny - 1) * self.dx]
        ) - 1.0e-10
        self.x[:] = np.clip(self.x, lower, upper)

    def step(
        self,
        dt: float,
        *,
        damping: float = 0.05,
        gravity_scale: float = 1.0,
    ) -> None:
        if dt <= 0.0:
            raise ValueError("dt 必须为正")
        self.p2g(gravity_scale)
        self.advance_grid(dt, damping)
        self.g2p(dt)
        self.contact_force.fill(0.0)
