"""Lian et al. (2011) 对称板碰撞的二维平面应变实现。"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from typing import Iterable

import numpy as np

from .contact import project_normal_momentum


@dataclass(frozen=True)
class PlateImpactConfig:
    length: float = 21.0e-3
    width: float = 3.0e-3
    thickness: float = 3.0e-3
    young: float = 65.0e9
    poisson: float = 0.0
    density: float = 2.75e3
    impact_speed: float = 100.0
    fem_element_size: float = 0.5e-3
    mpm_cell_size: float = 0.5e-3
    particle_spacing: float = 0.25e-3
    dt: float = 2.0e-8
    end_time: float = 15.0e-6
    snapshot_times: tuple[float, ...] = (3.0e-6, 8.7e-6, 15.0e-6)
    contact_tolerance: float = 1.0e-8
    flip_ratio: float = 0.98

    @property
    def area(self) -> float:
        return self.width * self.thickness

    @property
    def wave_speed(self) -> float:
        modulus = (
            self.young
            * (1.0 - self.poisson)
            / ((1.0 + self.poisson) * (1.0 - 2.0 * self.poisson))
        )
        return sqrt(modulus / self.density)

    @property
    def stable_dt(self) -> float:
        return min(self.fem_element_size, self.mpm_cell_size) / self.wave_speed

    @property
    def constitutive_matrix(self) -> np.ndarray:
        lam = (
            self.young
            * self.poisson
            / ((1.0 + self.poisson) * (1.0 - 2.0 * self.poisson))
        )
        mu = self.young / (2.0 * (1.0 + self.poisson))
        return np.array(
            [
                [lam + 2.0 * mu, lam, 0.0],
                [lam, lam + 2.0 * mu, 0.0],
                [0.0, 0.0, mu],
            ]
        )

    def validate(self) -> None:
        positive = {
            "length": self.length,
            "width": self.width,
            "thickness": self.thickness,
            "young": self.young,
            "density": self.density,
            "fem_element_size": self.fem_element_size,
            "mpm_cell_size": self.mpm_cell_size,
            "particle_spacing": self.particle_spacing,
            "dt": self.dt,
            "end_time": self.end_time,
        }
        for name, value in positive.items():
            if value <= 0.0:
                raise ValueError(f"{name} 必须为正")
        if not -1.0 < self.poisson < 0.5:
            raise ValueError("poisson 必须位于 (-1, 0.5)")
        if self.dt > 0.5 * self.stable_dt:
            raise ValueError("时间步超过 0.5 h/c")
        if not 0.0 <= self.flip_ratio <= 1.0:
            raise ValueError("flip_ratio 必须位于 [0, 1]")

        for size in (self.fem_element_size, self.particle_spacing):
            for extent in (self.length, self.width):
                if not np.isclose(extent / size, round(extent / size)):
                    raise ValueError("板尺寸必须是离散尺寸的整数倍")
        if not np.isclose(
            self.width / self.mpm_cell_size,
            round(self.width / self.mpm_cell_size),
        ):
            raise ValueError("板宽必须是 MPM 网格尺寸的整数倍")
        ratio = self.fem_element_size / self.mpm_cell_size
        if not 0.5 <= ratio <= 3.0:
            raise ValueError("当前尺寸比 R 只验证到 [0.5, 3.0]")


def analytical_contact_stress(config: PlateImpactConfig) -> float:
    return -config.density * config.wave_speed * config.impact_speed


def analytical_separation_time(config: PlateImpactConfig) -> float:
    return 2.0 * config.length / config.wave_speed


def _q4_gradients(
    coordinates: np.ndarray, xi: float, eta: float
) -> tuple[np.ndarray, float, np.ndarray]:
    d_nat = 0.25 * np.array(
        [
            [-(1.0 - eta), -(1.0 - xi)],
            [1.0 - eta, -(1.0 + xi)],
            [1.0 + eta, 1.0 + xi],
            [-(1.0 + eta), 1.0 - xi],
        ]
    )
    jacobian = coordinates.T @ d_nat
    det_j = float(np.linalg.det(jacobian))
    if det_j <= 0.0:
        raise ValueError("Q4 单元 Jacobian 非正")
    gradients = d_nat @ np.linalg.inv(jacobian)

    b = np.zeros((3, 8))
    for a, (dx, dy) in enumerate(gradients):
        b[0, 2 * a] = dx
        b[1, 2 * a + 1] = dy
        b[2, 2 * a] = dy
        b[2, 2 * a + 1] = dx
    return gradients, det_j, b


class Quad4Plate:
    """左板：二维 Q4、团块质量、显式中心差分。"""

    def __init__(self, config: PlateImpactConfig):
        self.config = config
        h = config.fem_element_size
        self.nx = int(round(config.length / h))
        self.ny = int(round(config.width / h))

        xs = np.linspace(-config.length, 0.0, self.nx + 1)
        ys = np.linspace(0.0, config.width, self.ny + 1)
        self.x0 = np.array([[x, y] for x in xs for y in ys])
        self.x = self.x0.copy()
        self.v = np.zeros_like(self.x)
        self.v[:, 0] = config.impact_speed

        def node(i: int, j: int) -> int:
            return i * (self.ny + 1) + j

        self.elements = np.array(
            [
                [
                    node(i, j),
                    node(i + 1, j),
                    node(i + 1, j + 1),
                    node(i, j + 1),
                ]
                for i in range(self.nx)
                for j in range(self.ny)
            ],
            dtype=int,
        )
        self.element_columns = np.repeat(np.arange(self.nx), self.ny)
        self.surface_nodes = np.array(
            [node(self.nx, j) for j in range(self.ny + 1)], dtype=int
        )
        self.surface_normals = np.tile(
            np.array([1.0, 0.0]), (len(self.surface_nodes), 1)
        )
        self.side_nodes = np.flatnonzero(
            np.isclose(self.x0[:, 1], 0.0)
            | np.isclose(self.x0[:, 1], config.width)
        )

        gauss = 1.0 / sqrt(3.0)
        points = (
            (-gauss, -gauss),
            (gauss, -gauss),
            (gauss, gauss),
            (-gauss, gauss),
        )
        ne = len(self.elements)
        self.b_matrix = np.zeros((ne, 4, 3, 8))
        self.det_j = np.zeros((ne, 4))
        self.mass = np.zeros(len(self.x0))
        self.volume = np.zeros(ne)

        for e, conn in enumerate(self.elements):
            coordinates = self.x0[conn]
            for q, (xi, eta) in enumerate(points):
                _, det_j, b = _q4_gradients(coordinates, xi, eta)
                self.b_matrix[e, q] = b
                self.det_j[e, q] = det_j
            volume = float(np.sum(self.det_j[e]) * config.thickness)
            self.volume[e] = volume
            self.mass[conn] += 0.25 * config.density * volume

        self.strain = np.zeros((ne, 4, 3))
        self.stress = np.zeros((ne, 4, 3))
        self.apply_side_constraint(self.v)

    def apply_side_constraint(self, velocity: np.ndarray) -> None:
        velocity[self.side_nodes, 1] = 0.0

    def update_stress(self, velocity: np.ndarray, dt: float) -> None:
        d = self.config.constitutive_matrix
        element_velocity = velocity[self.elements].reshape(-1, 8)
        rate = np.einsum(
            "eqij,ej->eqi", self.b_matrix, element_velocity
        )
        self.strain += dt * rate
        self.stress = self.strain @ d.T

    def nodal_force(self) -> np.ndarray:
        force = np.zeros_like(self.v)
        local = -np.einsum(
            "eqkd,eqk,eq->ed",
            self.b_matrix,
            self.stress,
            self.det_j * self.config.thickness,
        )
        np.add.at(
            force,
            self.elements.ravel(),
            local.reshape(-1, 2),
        )
        force[self.side_nodes, 1] = 0.0
        return force

    def energy(self) -> tuple[float, float]:
        kinetic = 0.5 * np.sum(self.mass[:, None] * self.v**2)
        strain = 0.5 * np.sum(
            np.einsum("eqi,eqi->eq", self.stress, self.strain)
            * self.det_j
            * self.config.thickness
        )
        return float(kinetic), float(strain)

    def momentum(self) -> np.ndarray:
        return np.sum(self.mass[:, None] * self.v, axis=0)

    def stress_field(self) -> tuple[np.ndarray, np.ndarray]:
        centers = np.mean(self.x[self.elements], axis=1)
        stress = np.mean(self.stress, axis=1)
        return centers, stress


class MaterialPointPlate:
    """右板：二维 USF-MPM，双线性背景网格。"""

    def __init__(self, config: PlateImpactConfig):
        self.config = config
        dp = config.particle_spacing
        self.npx = int(round(config.length / dp))
        self.npy = int(round(config.width / dp))
        self.x = np.array(
            [
                [(i + 0.5) * dp, (j + 0.5) * dp]
                for i in range(self.npx)
                for j in range(self.npy)
            ]
        )
        self.column = np.repeat(np.arange(self.npx), self.npy)
        self.v = np.zeros_like(self.x)
        self.v[:, 0] = -config.impact_speed

        particle_volume = dp * dp * config.thickness
        self.volume = np.full(len(self.x), particle_volume)
        self.mass = np.full(len(self.x), config.density * particle_volume)
        self.strain = np.zeros((len(self.x), 3))
        self.stress = np.zeros((len(self.x), 3))

        h = config.mpm_cell_size
        self.grid_x = np.arange(-h, config.length + 2.5 * h, h)
        self.grid_y = np.arange(0.0, config.width + 0.5 * h, h)
        self.grid_shape = (len(self.grid_x), len(self.grid_y))
        self.grid_size = self.grid_shape[0] * self.grid_shape[1]
        self.side_grid = np.array(
            [
                self._flat(i, j)
                for i in range(self.grid_shape[0])
                for j in (0, self.grid_shape[1] - 1)
            ],
            dtype=int,
        )

    def _flat(self, i: int, j: int) -> int:
        return i * self.grid_shape[1] + j

    def _stencil(
        self, position: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ids, weights, gradients = self._stencils(
            np.asarray(position, dtype=float)[None, :]
        )
        return ids[0], weights[0], gradients[0]

    def _stencils(
        self, position: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        h = self.config.mpm_cell_size
        position = np.asarray(position, dtype=float)
        i = np.floor((position[:, 0] - self.grid_x[0]) / h).astype(int)
        j = np.floor((position[:, 1] - self.grid_y[0]) / h).astype(int)
        i = np.clip(i, 0, self.grid_shape[0] - 2)
        j = np.clip(j, 0, self.grid_shape[1] - 2)
        rx = (position[:, 0] - self.grid_x[i]) / h
        ry = (position[:, 1] - self.grid_y[j]) / h

        ids = np.column_stack(
            (
                i * self.grid_shape[1] + j,
                (i + 1) * self.grid_shape[1] + j,
                (i + 1) * self.grid_shape[1] + j + 1,
                i * self.grid_shape[1] + j + 1,
            )
        )
        weights = np.column_stack(
            (
                (1.0 - rx) * (1.0 - ry),
                rx * (1.0 - ry),
                rx * ry,
                (1.0 - rx) * ry,
            )
        )
        gradients = np.stack(
            (
                np.column_stack((-(1.0 - ry) / h, -(1.0 - rx) / h)),
                np.column_stack(((1.0 - ry) / h, -rx / h)),
                np.column_stack((ry / h, rx / h)),
                np.column_stack((-ry / h, (1.0 - rx) / h)),
            ),
            axis=1,
        )
        return ids, weights, gradients

    def apply_side_constraint(self, velocity: np.ndarray) -> None:
        velocity[self.side_grid, 1] = 0.0

    def p2g(self) -> tuple[np.ndarray, np.ndarray]:
        grid_mass = np.zeros(self.grid_size)
        grid_momentum = np.zeros((self.grid_size, 2))
        ids, weights, _ = self._stencils(self.x)
        np.add.at(
            grid_mass,
            ids.ravel(),
            (weights * self.mass[:, None]).ravel(),
        )
        contribution = (
            weights[:, :, None]
            * self.mass[:, None, None]
            * self.v[:, None, :]
        )
        np.add.at(
            grid_momentum,
            ids.ravel(),
            contribution.reshape(-1, 2),
        )
        return grid_mass, grid_momentum

    def grid_velocity(
        self, grid_mass: np.ndarray, grid_momentum: np.ndarray
    ) -> np.ndarray:
        velocity = np.zeros_like(grid_momentum)
        active = grid_mass > 0.0
        velocity[active] = grid_momentum[active] / grid_mass[active, None]
        self.apply_side_constraint(velocity)
        return velocity

    def update_stress(self, grid_velocity: np.ndarray, dt: float) -> None:
        d = self.config.constitutive_matrix
        ids, _, gradients = self._stencils(self.x)
        velocity_gradient = np.einsum(
            "pai,paj->pij", grid_velocity[ids], gradients
        )
        rate = np.column_stack(
            (
                velocity_gradient[:, 0, 0],
                velocity_gradient[:, 1, 1],
                velocity_gradient[:, 0, 1]
                + velocity_gradient[:, 1, 0],
            )
        )
        self.strain += dt * rate
        self.stress = self.strain @ d.T
        jacobian_rate = 1.0 + dt * (
            velocity_gradient[:, 0, 0]
            + velocity_gradient[:, 1, 1]
        )
        self.volume *= np.maximum(0.9, jacobian_rate)

    def grid_force(self) -> np.ndarray:
        force = np.zeros((self.grid_size, 2))
        ids, _, gradients = self._stencils(self.x)
        tensor = np.empty((len(self.x), 2, 2))
        tensor[:, 0, 0] = self.stress[:, 0]
        tensor[:, 1, 1] = self.stress[:, 1]
        tensor[:, 0, 1] = self.stress[:, 2]
        tensor[:, 1, 0] = self.stress[:, 2]
        contribution = -self.volume[:, None, None] * np.einsum(
            "paj,pij->pai", gradients, tensor
        )
        np.add.at(
            force,
            ids.ravel(),
            contribution.reshape(-1, 2),
        )
        force[self.side_grid, 1] = 0.0
        return force

    def g2p(
        self,
        velocity_before: np.ndarray,
        velocity_after: np.ndarray,
        dt: float,
    ) -> None:
        ids, weights, _ = self._stencils(self.x)
        delta = np.sum(
            weights[:, :, None]
            * (velocity_after[ids] - velocity_before[ids]),
            axis=1,
        )
        pic = np.sum(
            weights[:, :, None] * velocity_after[ids], axis=1
        )
        flip = self.v + delta
        self.v = (
            self.config.flip_ratio * flip
            + (1.0 - self.config.flip_ratio) * pic
        )
        self.x += dt * pic

    def surface_position(self) -> float:
        return float(
            np.min(self.x[:, 0]) - 0.5 * self.config.particle_spacing
        )

    def energy(self) -> tuple[float, float]:
        kinetic = 0.5 * np.sum(self.mass[:, None] * self.v**2)
        strain = np.sum(
            0.5 * np.einsum("ij,ij->i", self.stress, self.strain)
            * self.volume
        )
        return float(kinetic), float(strain)

    def momentum(self) -> np.ndarray:
        return np.sum(self.mass[:, None] * self.v, axis=0)

    def stress_field(self) -> tuple[np.ndarray, np.ndarray]:
        return self.x.copy(), self.stress.copy()


@dataclass
class ContactStep:
    active_nodes: int = 0
    gap: float = np.inf
    impulse_1: float = 0.0
    impulse_2: float = 0.0
    balance_residual: float = 0.0

    @property
    def active(self) -> bool:
        return self.active_nodes > 0

    @property
    def impulse(self) -> float:
        return self.impulse_1 + self.impulse_2


@dataclass
class SimulationHistory:
    time: list[float] = field(default_factory=list)
    kinetic: list[float] = field(default_factory=list)
    strain: list[float] = field(default_factory=list)
    total_energy: list[float] = field(default_factory=list)
    momentum_x: list[float] = field(default_factory=list)
    momentum_y: list[float] = field(default_factory=list)
    gap: list[float] = field(default_factory=list)
    contact_impulse: list[float] = field(default_factory=list)
    contact_nodes: list[int] = field(default_factory=list)
    contact_balance: list[float] = field(default_factory=list)
    transverse_velocity: list[float] = field(default_factory=list)
    snapshots: dict[float, dict[str, np.ndarray]] = field(
        default_factory=dict
    )
    separation_time: float | None = None

    def as_arrays(self) -> dict[str, np.ndarray]:
        return {
            "time": np.asarray(self.time),
            "kinetic": np.asarray(self.kinetic),
            "strain": np.asarray(self.strain),
            "total_energy": np.asarray(self.total_energy),
            "momentum_x": np.asarray(self.momentum_x),
            "momentum_y": np.asarray(self.momentum_y),
            "gap": np.asarray(self.gap),
            "contact_impulse": np.asarray(self.contact_impulse),
            "contact_nodes": np.asarray(self.contact_nodes),
            "contact_balance": np.asarray(self.contact_balance),
            "transverse_velocity": np.asarray(self.transverse_velocity),
        }


class CFEMPPlateImpact2D:
    """二维 Q4-FEM / MPM 局部多网格接触。"""

    def __init__(self, config: PlateImpactConfig | None = None):
        self.config = config or PlateImpactConfig()
        self.config.validate()
        self.fem = Quad4Plate(self.config)
        self.mpm = MaterialPointPlate(self.config)
        self.time = 0.0
        self.has_contacted = False

    def _gap(self) -> float:
        fem_surface = np.max(self.fem.x[self.fem.surface_nodes, 0])
        return self.mpm.surface_position() - float(fem_surface)

    def _map_fem_surface(
        self, velocity: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mass = np.zeros(self.mpm.grid_size)
        momentum = np.zeros((self.mpm.grid_size, 2))
        normal_sum = np.zeros_like(momentum)
        for node, normal in zip(
            self.fem.surface_nodes, self.fem.surface_normals
        ):
            ids, weights, _ = self.mpm._stencil(self.fem.x[node])
            nodal_mass = self.fem.mass[node]
            np.add.at(mass, ids, weights * nodal_mass)
            np.add.at(
                momentum,
                ids,
                weights[:, None] * nodal_mass * velocity[node],
            )
            np.add.at(
                normal_sum,
                ids,
                weights[:, None] * nodal_mass * normal,
            )
        active = mass > 0.0
        normal_sum[active] /= mass[active, None]
        lengths = np.linalg.norm(normal_sum, axis=1)
        valid = lengths > 0.0
        normal_sum[valid] /= lengths[valid, None]
        return mass, momentum, normal_sum

    def _scatter_surface_impulse(
        self, grid_impulse: np.ndarray, grid_fem_mass: np.ndarray
    ) -> tuple[np.ndarray, float]:
        nodal_impulse = np.zeros_like(self.fem.v)
        for node in self.fem.surface_nodes:
            ids, weights, _ = self.mpm._stencil(self.fem.x[node])
            active = grid_fem_mass[ids] > 0.0
            nodal_impulse[node] += np.sum(
                (
                    self.fem.mass[node]
                    * weights[active]
                    / grid_fem_mass[ids[active]]
                )[:, None]
                * grid_impulse[ids[active]],
                axis=0,
            )
        residual = np.linalg.norm(
            np.sum(grid_impulse, axis=0)
            - np.sum(nodal_impulse, axis=0)
        )
        return nodal_impulse, float(residual)

    def _project_grid(
        self,
        fem_mass: np.ndarray,
        fem_momentum: np.ndarray,
        mpm_mass: np.ndarray,
        mpm_momentum: np.ndarray,
        normal: np.ndarray,
        *,
        enabled: bool,
    ) -> tuple[np.ndarray, int]:
        impulse = np.zeros_like(mpm_momentum)
        if not enabled:
            return impulse, 0
        candidates = np.flatnonzero(
            (fem_mass > 1.0e-18) & (mpm_mass > 1.0e-18)
        )
        active = 0
        for index in candidates:
            result = project_normal_momentum(
                float(fem_mass[index]),
                fem_momentum[index],
                float(mpm_mass[index]),
                mpm_momentum[index],
                normal[index],
            )
            if result.active:
                impulse[index] = result.impulse_on_mpm
                active += 1
        return impulse, active

    def step(self) -> ContactStep:
        dt = self.config.dt
        gap = self._gap()
        close = gap <= self.config.contact_tolerance

        mpm_mass, mpm_momentum = self.mpm.p2g()
        velocity_before = self.mpm.grid_velocity(
            mpm_mass, mpm_momentum
        )
        fem_mass, fem_momentum, normal = self._map_fem_surface(
            self.fem.v
        )

        impulse_1, active_1 = self._project_grid(
            fem_mass,
            fem_momentum,
            mpm_mass,
            mpm_momentum,
            normal,
            enabled=close,
        )
        fem_impulse_1, residual_1 = self._scatter_surface_impulse(
            -impulse_1, fem_mass
        )
        self.fem.v += fem_impulse_1 / self.fem.mass[:, None]
        self.fem.apply_side_constraint(self.fem.v)
        mpm_momentum += impulse_1
        adjusted_grid_velocity = self.mpm.grid_velocity(
            mpm_mass, mpm_momentum
        )

        self.fem.update_stress(self.fem.v, dt)
        self.mpm.update_stress(adjusted_grid_velocity, dt)
        fem_force = self.fem.nodal_force()
        mpm_force = self.mpm.grid_force()

        fem_trial_velocity = (
            self.fem.v + dt * fem_force / self.fem.mass[:, None]
        )
        self.fem.apply_side_constraint(fem_trial_velocity)
        mpm_trial_momentum = mpm_momentum + dt * mpm_force
        fem_trial_mass, fem_trial_momentum, normal = (
            self._map_fem_surface(fem_trial_velocity)
        )
        impulse_2, active_2 = self._project_grid(
            fem_trial_mass,
            fem_trial_momentum,
            mpm_mass,
            mpm_trial_momentum,
            normal,
            enabled=close,
        )
        fem_impulse_2, residual_2 = self._scatter_surface_impulse(
            -impulse_2, fem_trial_mass
        )
        fem_trial_velocity += fem_impulse_2 / self.fem.mass[:, None]
        self.fem.apply_side_constraint(fem_trial_velocity)
        mpm_trial_momentum += impulse_2

        final_grid_velocity = self.mpm.grid_velocity(
            mpm_mass, mpm_trial_momentum
        )
        self.fem.v = fem_trial_velocity
        self.fem.x += dt * self.fem.v
        self.mpm.g2p(velocity_before, final_grid_velocity, dt)
        self.time += dt

        active_nodes = max(active_1, active_2)
        if active_nodes:
            self.has_contacted = True
        return ContactStep(
            active_nodes=active_nodes,
            gap=gap,
            impulse_1=float(np.sum(np.linalg.norm(impulse_1, axis=1))),
            impulse_2=float(np.sum(np.linalg.norm(impulse_2, axis=1))),
            balance_residual=max(residual_1, residual_2),
        )

    def total_momentum(self) -> np.ndarray:
        return self.fem.momentum() + self.mpm.momentum()

    def total_energy(self) -> tuple[float, float, float]:
        fem_ke, fem_se = self.fem.energy()
        mpm_ke, mpm_se = self.mpm.energy()
        kinetic = fem_ke + mpm_ke
        strain = fem_se + mpm_se
        return kinetic, strain, kinetic + strain

    def combined_stress_profile(self) -> tuple[np.ndarray, np.ndarray]:
        fem_position, fem_stress = self.fem.stress_field()
        fem_x = np.array(
            [
                np.mean(fem_position[self.fem.element_columns == i, 0])
                for i in range(self.fem.nx)
            ]
        )
        fem_sxx = np.array(
            [
                np.mean(fem_stress[self.fem.element_columns == i, 0])
                for i in range(self.fem.nx)
            ]
        )
        mpm_x = np.array(
            [
                np.mean(self.mpm.x[self.mpm.column == i, 0])
                for i in range(self.mpm.npx)
            ]
        )
        mpm_sxx = np.array(
            [
                np.mean(self.mpm.stress[self.mpm.column == i, 0])
                for i in range(self.mpm.npx)
            ]
        )
        return np.concatenate([fem_x, mpm_x]), np.concatenate(
            [fem_sxx, mpm_sxx]
        )

    def snapshot(self) -> dict[str, np.ndarray]:
        fem_position, fem_stress = self.fem.stress_field()
        mpm_position, mpm_stress = self.mpm.stress_field()
        profile_x, profile_stress = self.combined_stress_profile()
        return {
            "fem_position": fem_position,
            "fem_stress": fem_stress,
            "mpm_position": mpm_position,
            "mpm_stress": mpm_stress,
            "profile_x": profile_x,
            "profile_stress": profile_stress,
        }

    def run(
        self,
        *,
        end_time: float | None = None,
        snapshot_times: Iterable[float] | None = None,
    ) -> SimulationHistory:
        target = self.config.end_time if end_time is None else end_time
        wanted = sorted(
            self.config.snapshot_times
            if snapshot_times is None
            else snapshot_times
        )
        history = SimulationHistory()
        snapshot_index = 0
        last_contact_time: float | None = None
        self._record(history, ContactStep(gap=self._gap()))

        while self.time < target - 0.5 * self.config.dt:
            contact = self.step()
            self._record(history, contact)
            if contact.active:
                last_contact_time = self.time
            while (
                snapshot_index < len(wanted)
                and self.time >= wanted[snapshot_index] - 0.5 * self.config.dt
            ):
                history.snapshots[float(wanted[snapshot_index])] = (
                    self.snapshot()
                )
                snapshot_index += 1

        if last_contact_time is not None:
            history.separation_time = last_contact_time + self.config.dt
        return history

    def _record(
        self, history: SimulationHistory, contact: ContactStep
    ) -> None:
        kinetic, strain, total = self.total_energy()
        momentum = self.total_momentum()
        history.time.append(self.time)
        history.kinetic.append(kinetic)
        history.strain.append(strain)
        history.total_energy.append(total)
        history.momentum_x.append(float(momentum[0]))
        history.momentum_y.append(float(momentum[1]))
        history.gap.append(self._gap())
        history.contact_impulse.append(contact.impulse)
        history.contact_nodes.append(contact.active_nodes)
        history.contact_balance.append(contact.balance_residual)
        history.transverse_velocity.append(
            float(
                max(
                    np.max(np.abs(self.fem.v[:, 1])),
                    np.max(np.abs(self.mpm.v[:, 1])),
                )
            )
        )
