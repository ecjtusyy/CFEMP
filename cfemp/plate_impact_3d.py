"""Taichi 三维 CFEMP 对称板碰撞。"""

import tempfile
from dataclasses import dataclass, field
from math import sqrt
from pathlib import Path
from typing import Iterable

import numpy as np
import taichi as ti


def _init_taichi() -> None:
    runtime = ti.lang.impl.get_runtime()
    if runtime.prog is None:
        cache = Path(tempfile.gettempdir()) / "cfemp-taichi-cache"
        ti.init(
            arch=ti.cpu,
            default_fp=ti.f64,
            default_ip=ti.i32,
            offline_cache=False,
            offline_cache_file_path=str(cache),
        )


@dataclass(frozen=True)
class PlateImpact3DConfig:
    length: float = 21.0e-3
    width: float = 3.0e-3
    depth: float = 3.0e-3
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
        return self.width * self.depth

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
    def lame(self) -> tuple[float, float]:
        lam = (
            self.young
            * self.poisson
            / ((1.0 + self.poisson) * (1.0 - 2.0 * self.poisson))
        )
        mu = self.young / (2.0 * (1.0 + self.poisson))
        return lam, mu

    def validate(self) -> None:
        positive = {
            "length": self.length,
            "width": self.width,
            "depth": self.depth,
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
            for extent in (self.length, self.width, self.depth):
                cells = extent / size
                if not np.isclose(cells, round(cells)):
                    raise ValueError("实体尺寸必须是离散尺寸的整数倍")
        for extent in (self.width, self.depth):
            cells = extent / self.mpm_cell_size
            if not np.isclose(cells, round(cells)):
                raise ValueError("横向尺寸必须是 MPM 网格尺寸的整数倍")
        if not np.isclose(self.fem_element_size, self.mpm_cell_size):
            raise ValueError("三维基准使用匹配的 FEM/MPM 网格")


def analytical_contact_stress_3d(config: PlateImpact3DConfig) -> float:
    return -config.density * config.wave_speed * config.impact_speed


def analytical_separation_time_3d(config: PlateImpact3DConfig) -> float:
    return 2.0 * config.length / config.wave_speed


def _hex8_mesh(
    config: PlateImpact3DConfig,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    h = config.fem_element_size
    nx = int(round(config.length / h))
    ny = int(round(config.width / h))
    nz = int(round(config.depth / h))
    xs = np.linspace(-config.length, 0.0, nx + 1)
    ys = np.linspace(0.0, config.width, ny + 1)
    zs = np.linspace(0.0, config.depth, nz + 1)
    nodes = np.array(
        [[x, y, z] for x in xs for y in ys for z in zs],
        dtype=float,
    )

    def node(i: int, j: int, k: int) -> int:
        return (i * (ny + 1) + j) * (nz + 1) + k

    elements = np.array(
        [
            [
                node(i, j, k),
                node(i + 1, j, k),
                node(i + 1, j + 1, k),
                node(i, j + 1, k),
                node(i, j, k + 1),
                node(i + 1, j, k + 1),
                node(i + 1, j + 1, k + 1),
                node(i, j + 1, k + 1),
            ]
            for i in range(nx)
            for j in range(ny)
            for k in range(nz)
        ],
        dtype=np.int32,
    )
    surface = np.array(
        [node(nx, j, k) for j in range(ny + 1) for k in range(nz + 1)],
        dtype=np.int32,
    )
    lock_y = (
        np.isclose(nodes[:, 1], 0.0)
        | np.isclose(nodes[:, 1], config.width)
    ).astype(np.int32)
    lock_z = (
        np.isclose(nodes[:, 2], 0.0)
        | np.isclose(nodes[:, 2], config.depth)
    ).astype(np.int32)
    columns = np.repeat(np.arange(nx), ny * nz)
    return nodes, elements, surface, lock_y, lock_z, columns


def _hex8_quadrature(
    nodes: np.ndarray,
    elements: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    natural = np.array(
        [
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ]
    )
    gauss = 1.0 / sqrt(3.0)
    points = np.array(
        [
            [xi, eta, zeta]
            for xi in (-gauss, gauss)
            for eta in (-gauss, gauss)
            for zeta in (-gauss, gauss)
        ]
    )
    gradients = np.empty((len(elements), 8, 8, 3))
    det_j = np.empty((len(elements), 8))
    for e, conn in enumerate(elements):
        coordinates = nodes[conn]
        for q, (xi, eta, zeta) in enumerate(points):
            d_nat = np.empty((8, 3))
            for a, (sx, sy, sz) in enumerate(natural):
                d_nat[a, 0] = 0.125 * sx * (1.0 + sy * eta) * (
                    1.0 + sz * zeta
                )
                d_nat[a, 1] = 0.125 * sy * (1.0 + sx * xi) * (
                    1.0 + sz * zeta
                )
                d_nat[a, 2] = 0.125 * sz * (1.0 + sx * xi) * (
                    1.0 + sy * eta
                )
            jacobian = coordinates.T @ d_nat
            determinant = float(np.linalg.det(jacobian))
            if determinant <= 0.0:
                raise ValueError("HEX8 单元 Jacobian 非正")
            gradients[e, q] = d_nat @ np.linalg.inv(jacobian)
            det_j[e, q] = determinant
    return gradients, det_j


@dataclass
class ContactStep3D:
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
class SimulationHistory3D:
    time: list[float] = field(default_factory=list)
    kinetic: list[float] = field(default_factory=list)
    strain: list[float] = field(default_factory=list)
    total_energy: list[float] = field(default_factory=list)
    momentum_x: list[float] = field(default_factory=list)
    momentum_y: list[float] = field(default_factory=list)
    momentum_z: list[float] = field(default_factory=list)
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
            "momentum_z": np.asarray(self.momentum_z),
            "gap": np.asarray(self.gap),
            "contact_impulse": np.asarray(self.contact_impulse),
            "contact_nodes": np.asarray(self.contact_nodes),
            "contact_balance": np.asarray(self.contact_balance),
            "transverse_velocity": np.asarray(self.transverse_velocity),
        }


@ti.data_oriented
class TaichiCFEMPPlateImpact3D:
    """HEX8 FEM 与三维 MPM 的局部多网格接触。"""

    def __init__(self, config: PlateImpact3DConfig | None = None):
        _init_taichi()
        self.config = config or PlateImpact3DConfig()
        self.config.validate()
        self.time = 0.0

        (
            fem_nodes,
            elements,
            surface,
            lock_y,
            lock_z,
            self.fem_columns,
        ) = _hex8_mesh(self.config)
        gradients, det_j = _hex8_quadrature(fem_nodes, elements)
        self.fem_nodes0 = fem_nodes
        self.elements_np = elements
        self.fem_nx = int(round(self.config.length / self.config.fem_element_size))
        self.fem_ny = int(round(self.config.width / self.config.fem_element_size))
        self.fem_nz = int(round(self.config.depth / self.config.fem_element_size))
        self.n_fem_nodes = len(fem_nodes)
        self.n_elements = len(elements)

        volume = np.sum(det_j, axis=1)
        fem_mass = np.zeros(self.n_fem_nodes)
        for e, conn in enumerate(elements):
            fem_mass[conn] += (
                self.config.density * volume[e] / 8.0
            )
        self.fem_x = ti.Vector.field(3, ti.f64, shape=self.n_fem_nodes)
        self.fem_v = ti.Vector.field(3, ti.f64, shape=self.n_fem_nodes)
        self.fem_trial_v = ti.Vector.field(3, ti.f64, shape=self.n_fem_nodes)
        self.fem_mass = ti.field(ti.f64, shape=self.n_fem_nodes)
        self.fem_force = ti.Vector.field(3, ti.f64, shape=self.n_fem_nodes)
        self.fem_lock_y = ti.field(ti.i32, shape=self.n_fem_nodes)
        self.fem_lock_z = ti.field(ti.i32, shape=self.n_fem_nodes)
        self.elements = ti.Vector.field(8, ti.i32, shape=self.n_elements)
        self.fem_grad = ti.Vector.field(
            3, ti.f64, shape=(self.n_elements, 8, 8)
        )
        self.fem_det_j = ti.field(
            ti.f64, shape=(self.n_elements, 8)
        )
        self.fem_strain = ti.Vector.field(
            6, ti.f64, shape=(self.n_elements, 8)
        )
        self.fem_stress = ti.Vector.field(
            6, ti.f64, shape=(self.n_elements, 8)
        )
        self.surface_field = ti.field(ti.i32, shape=len(surface))

        fem_velocity = np.zeros_like(fem_nodes)
        fem_velocity[:, 0] = self.config.impact_speed
        self.fem_x.from_numpy(fem_nodes)
        self.fem_v.from_numpy(fem_velocity)
        self.fem_mass.from_numpy(fem_mass)
        self.fem_lock_y.from_numpy(lock_y)
        self.fem_lock_z.from_numpy(lock_z)
        self.elements.from_numpy(elements)
        self.fem_grad.from_numpy(gradients)
        self.fem_det_j.from_numpy(det_j)
        self.surface_field.from_numpy(surface)

        dp = self.config.particle_spacing
        self.mpm_nx = int(round(self.config.length / dp))
        self.mpm_ny = int(round(self.config.width / dp))
        self.mpm_nz = int(round(self.config.depth / dp))
        particles = np.array(
            [
                [(i + 0.5) * dp, (j + 0.5) * dp, (k + 0.5) * dp]
                for i in range(self.mpm_nx)
                for j in range(self.mpm_ny)
                for k in range(self.mpm_nz)
            ]
        )
        self.mpm_columns = np.repeat(
            np.arange(self.mpm_nx), self.mpm_ny * self.mpm_nz
        )
        self.n_particles = len(particles)
        particle_volume = dp**3
        particle_mass = self.config.density * particle_volume
        self.mpm_x = ti.Vector.field(3, ti.f64, shape=self.n_particles)
        self.mpm_v = ti.Vector.field(3, ti.f64, shape=self.n_particles)
        self.mpm_mass = ti.field(ti.f64, shape=self.n_particles)
        self.mpm_volume = ti.field(ti.f64, shape=self.n_particles)
        self.mpm_strain = ti.Vector.field(6, ti.f64, shape=self.n_particles)
        self.mpm_stress = ti.Vector.field(6, ti.f64, shape=self.n_particles)
        particle_velocity = np.zeros_like(particles)
        particle_velocity[:, 0] = -self.config.impact_speed
        self.mpm_x.from_numpy(particles)
        self.mpm_v.from_numpy(particle_velocity)
        self.mpm_mass.from_numpy(
            np.full(self.n_particles, particle_mass)
        )
        self.mpm_volume.from_numpy(
            np.full(self.n_particles, particle_volume)
        )

        h = self.config.mpm_cell_size
        self.grid_origin = np.array([-h, 0.0, 0.0])
        grid_x = np.arange(-h, self.config.length + 2.5 * h, h)
        grid_y = np.arange(0.0, self.config.width + 0.5 * h, h)
        grid_z = np.arange(0.0, self.config.depth + 0.5 * h, h)
        self.grid_shape = (len(grid_x), len(grid_y), len(grid_z))
        self.grid_mass = ti.field(ti.f64, shape=self.grid_shape)
        self.grid_momentum = ti.Vector.field(3, ti.f64, shape=self.grid_shape)
        self.grid_force = ti.Vector.field(3, ti.f64, shape=self.grid_shape)
        self.grid_velocity = ti.Vector.field(3, ti.f64, shape=self.grid_shape)
        self.grid_velocity_before = ti.Vector.field(
            3, ti.f64, shape=self.grid_shape
        )
        self.contact_fem_mass = ti.field(ti.f64, shape=self.grid_shape)
        self.contact_fem_momentum = ti.Vector.field(
            3, ti.f64, shape=self.grid_shape
        )
        self.contact_normal_sum = ti.Vector.field(
            3, ti.f64, shape=self.grid_shape
        )
        self.contact_impulse = ti.Vector.field(
            3, ti.f64, shape=self.grid_shape
        )
        self.contact_active = ti.field(ti.i32, shape=())
        self.contact_magnitude = ti.field(ti.f64, shape=())
        self.contact_grid_total = ti.Vector.field(3, ti.f64, shape=())
        self.contact_fem_total = ti.Vector.field(3, ti.f64, shape=())

        self.metric_kinetic = ti.field(ti.f64, shape=())
        self.metric_strain = ti.field(ti.f64, shape=())
        self.metric_momentum = ti.Vector.field(3, ti.f64, shape=())
        self.metric_transverse = ti.field(ti.f64, shape=())
        self.gap_fem_surface = ti.field(ti.f64, shape=())
        self.gap_mpm_surface = ti.field(ti.f64, shape=())

        self.lam, self.mu = self.config.lame
        self.dt = self.config.dt
        self.inv_h = 1.0 / self.config.mpm_cell_size
        self.flip_ratio = self.config.flip_ratio
        self.origin_x = float(self.grid_origin[0])
        self.origin_y = float(self.grid_origin[1])
        self.origin_z = float(self.grid_origin[2])

    @ti.func
    def _mpm_base(self, x):
        base = ti.cast(
            ti.floor(
                (x - ti.Vector([
                    self.origin_x,
                    self.origin_y,
                    self.origin_z,
                ]))
                * self.inv_h
            ),
            ti.i32,
        )
        base[0] = ti.max(0, ti.min(base[0], self.grid_shape[0] - 2))
        base[1] = ti.max(0, ti.min(base[1], self.grid_shape[1] - 2))
        base[2] = ti.max(0, ti.min(base[2], self.grid_shape[2] - 2))
        return base

    @ti.func
    def _mpm_shape(self, fx, offset):
        wx = (1.0 - fx[0]) * (1 - offset[0]) + fx[0] * offset[0]
        wy = (1.0 - fx[1]) * (1 - offset[1]) + fx[1] * offset[1]
        wz = (1.0 - fx[2]) * (1 - offset[2]) + fx[2] * offset[2]
        weight = wx * wy * wz
        gradient = ti.Vector(
            [
                (2 * offset[0] - 1) * self.inv_h * wy * wz,
                wx * (2 * offset[1] - 1) * self.inv_h * wz,
                wx * wy * (2 * offset[2] - 1) * self.inv_h,
            ]
        )
        return weight, gradient

    @ti.func
    def _stress_from_strain(self, strain):
        trace = strain[0] + strain[1] + strain[2]
        stress = ti.Vector.zero(ti.f64, 6)
        stress[0] = self.lam * trace + 2.0 * self.mu * strain[0]
        stress[1] = self.lam * trace + 2.0 * self.mu * strain[1]
        stress[2] = self.lam * trace + 2.0 * self.mu * strain[2]
        stress[3] = self.mu * strain[3]
        stress[4] = self.mu * strain[4]
        stress[5] = self.mu * strain[5]
        return stress

    @ti.func
    def _stress_tensor(self, stress):
        return ti.Matrix(
            [
                [stress[0], stress[3], stress[5]],
                [stress[3], stress[1], stress[4]],
                [stress[5], stress[4], stress[2]],
            ]
        )

    @ti.kernel
    def _reset_grid(self):
        for I in ti.grouped(self.grid_mass):
            self.grid_mass[I] = 0.0
            self.grid_momentum[I] = ti.Vector.zero(ti.f64, 3)
            self.grid_force[I] = ti.Vector.zero(ti.f64, 3)
            self.grid_velocity[I] = ti.Vector.zero(ti.f64, 3)
            self.grid_velocity_before[I] = ti.Vector.zero(ti.f64, 3)

    @ti.kernel
    def _p2g(self):
        origin = ti.Vector([self.origin_x, self.origin_y, self.origin_z])
        for p in range(self.n_particles):
            base = self._mpm_base(self.mpm_x[p])
            fx = (self.mpm_x[p] - (origin + base * (1.0 / self.inv_h))) * (
                self.inv_h
            )
            for offset in ti.static(ti.grouped(ti.ndrange(2, 2, 2))):
                weight, _ = self._mpm_shape(fx, offset)
                I = base + offset
                ti.atomic_add(
                    self.grid_mass[I],
                    weight * self.mpm_mass[p],
                )
                for d in ti.static(range(3)):
                    ti.atomic_add(
                        self.grid_momentum[I][d],
                        weight * self.mpm_mass[p] * self.mpm_v[p][d],
                    )

    @ti.kernel
    def _compute_grid_velocity(self, copy_before: ti.i32):
        for I in ti.grouped(self.grid_mass):
            velocity = ti.Vector.zero(ti.f64, 3)
            if self.grid_mass[I] > 0.0:
                velocity = self.grid_momentum[I] / self.grid_mass[I]
            if I[1] == 0 or I[1] == self.grid_shape[1] - 1:
                velocity[1] = 0.0
            if I[2] == 0 or I[2] == self.grid_shape[2] - 1:
                velocity[2] = 0.0
            self.grid_velocity[I] = velocity
            if copy_before == 1:
                self.grid_velocity_before[I] = velocity

    @ti.kernel
    def _reset_contact(self):
        for I in ti.grouped(self.contact_fem_mass):
            self.contact_fem_mass[I] = 0.0
            self.contact_fem_momentum[I] = ti.Vector.zero(ti.f64, 3)
            self.contact_normal_sum[I] = ti.Vector.zero(ti.f64, 3)
            self.contact_impulse[I] = ti.Vector.zero(ti.f64, 3)
        self.contact_active[None] = 0
        self.contact_magnitude[None] = 0.0
        self.contact_grid_total[None] = ti.Vector.zero(ti.f64, 3)
        self.contact_fem_total[None] = ti.Vector.zero(ti.f64, 3)

    @ti.kernel
    def _map_surface(self, use_trial: ti.i32):
        origin = ti.Vector([self.origin_x, self.origin_y, self.origin_z])
        normal = ti.Vector([1.0, 0.0, 0.0])
        for a in range(self.surface_field.shape[0]):
            node = self.surface_field[a]
            position = self.fem_x[node]
            velocity = self.fem_v[node]
            if use_trial == 1:
                velocity = self.fem_trial_v[node]
            base = self._mpm_base(position)
            fx = (position - (origin + base * (1.0 / self.inv_h))) * (
                self.inv_h
            )
            nodal_mass = self.fem_mass[node]
            for offset in ti.static(ti.grouped(ti.ndrange(2, 2, 2))):
                weight, _ = self._mpm_shape(fx, offset)
                I = base + offset
                mapped_mass = weight * nodal_mass
                ti.atomic_add(self.contact_fem_mass[I], mapped_mass)
                for d in ti.static(range(3)):
                    ti.atomic_add(
                        self.contact_fem_momentum[I][d],
                        mapped_mass * velocity[d],
                    )
                    ti.atomic_add(
                        self.contact_normal_sum[I][d],
                        mapped_mass * normal[d],
                    )

    @ti.kernel
    def _project_contact(self, enabled: ti.i32):
        for I in ti.grouped(self.contact_fem_mass):
            fem_mass = self.contact_fem_mass[I]
            mpm_mass = self.grid_mass[I]
            if enabled == 1 and fem_mass > 1.0e-18 and mpm_mass > 1.0e-18:
                normal = self.contact_normal_sum[I]
                normal_length = normal.norm()
                if normal_length > 0.0:
                    normal /= normal_length
                    fem_velocity = self.contact_fem_momentum[I] / fem_mass
                    mpm_velocity = self.grid_momentum[I] / mpm_mass
                    closing = (fem_velocity - mpm_velocity).dot(normal)
                    if closing > 0.0:
                        reduced_mass = (
                            fem_mass * mpm_mass / (fem_mass + mpm_mass)
                        )
                        impulse = reduced_mass * closing * normal
                        self.contact_impulse[I] = impulse
                        self.grid_momentum[I] += impulse
                        ti.atomic_add(self.contact_active[None], 1)
                        ti.atomic_add(
                            self.contact_magnitude[None], impulse.norm()
                        )
                        for d in ti.static(range(3)):
                            ti.atomic_add(
                                self.contact_grid_total[None][d],
                                impulse[d],
                            )

    @ti.kernel
    def _scatter_contact(self, use_trial: ti.i32):
        origin = ti.Vector([self.origin_x, self.origin_y, self.origin_z])
        for a in range(self.surface_field.shape[0]):
            node = self.surface_field[a]
            position = self.fem_x[node]
            base = self._mpm_base(position)
            fx = (position - (origin + base * (1.0 / self.inv_h))) * (
                self.inv_h
            )
            nodal_impulse = ti.Vector.zero(ti.f64, 3)
            for offset in ti.static(ti.grouped(ti.ndrange(2, 2, 2))):
                weight, _ = self._mpm_shape(fx, offset)
                I = base + offset
                if self.contact_fem_mass[I] > 0.0:
                    nodal_impulse -= (
                        self.fem_mass[node]
                        * weight
                        / self.contact_fem_mass[I]
                        * self.contact_impulse[I]
                    )
            if use_trial == 1:
                self.fem_trial_v[node] += (
                    nodal_impulse / self.fem_mass[node]
                )
                if self.fem_lock_y[node] == 1:
                    self.fem_trial_v[node][1] = 0.0
                if self.fem_lock_z[node] == 1:
                    self.fem_trial_v[node][2] = 0.0
            else:
                self.fem_v[node] += nodal_impulse / self.fem_mass[node]
                if self.fem_lock_y[node] == 1:
                    self.fem_v[node][1] = 0.0
                if self.fem_lock_z[node] == 1:
                    self.fem_v[node][2] = 0.0
            for d in ti.static(range(3)):
                ti.atomic_add(
                    self.contact_fem_total[None][d], nodal_impulse[d]
                )

    @ti.kernel
    def _update_fem_stress(self):
        for e, q in ti.ndrange(self.n_elements, 8):
            velocity_gradient = ti.Matrix.zero(ti.f64, 3, 3)
            for a in range(8):
                node = self.elements[e][a]
                velocity_gradient += self.fem_v[node].outer_product(
                    self.fem_grad[e, q, a]
                )
            rate = ti.Vector(
                [
                    velocity_gradient[0, 0],
                    velocity_gradient[1, 1],
                    velocity_gradient[2, 2],
                    velocity_gradient[0, 1] + velocity_gradient[1, 0],
                    velocity_gradient[1, 2] + velocity_gradient[2, 1],
                    velocity_gradient[2, 0] + velocity_gradient[0, 2],
                ]
            )
            self.fem_strain[e, q] += self.dt * rate
            self.fem_stress[e, q] = self._stress_from_strain(
                self.fem_strain[e, q]
            )

    @ti.kernel
    def _update_mpm_stress(self):
        origin = ti.Vector([self.origin_x, self.origin_y, self.origin_z])
        for p in range(self.n_particles):
            base = self._mpm_base(self.mpm_x[p])
            fx = (self.mpm_x[p] - (origin + base * (1.0 / self.inv_h))) * (
                self.inv_h
            )
            velocity_gradient = ti.Matrix.zero(ti.f64, 3, 3)
            for offset in ti.static(ti.grouped(ti.ndrange(2, 2, 2))):
                _, gradient = self._mpm_shape(fx, offset)
                I = base + offset
                velocity_gradient += self.grid_velocity[I].outer_product(
                    gradient
                )
            rate = ti.Vector(
                [
                    velocity_gradient[0, 0],
                    velocity_gradient[1, 1],
                    velocity_gradient[2, 2],
                    velocity_gradient[0, 1] + velocity_gradient[1, 0],
                    velocity_gradient[1, 2] + velocity_gradient[2, 1],
                    velocity_gradient[2, 0] + velocity_gradient[0, 2],
                ]
            )
            self.mpm_strain[p] += self.dt * rate
            self.mpm_stress[p] = self._stress_from_strain(
                self.mpm_strain[p]
            )
            jacobian_rate = 1.0 + self.dt * (
                velocity_gradient[0, 0]
                + velocity_gradient[1, 1]
                + velocity_gradient[2, 2]
            )
            self.mpm_volume[p] *= ti.max(0.9, jacobian_rate)

    @ti.kernel
    def _clear_forces(self):
        for i in range(self.n_fem_nodes):
            self.fem_force[i] = ti.Vector.zero(ti.f64, 3)
        for I in ti.grouped(self.grid_force):
            self.grid_force[I] = ti.Vector.zero(ti.f64, 3)

    @ti.kernel
    def _fem_internal_force(self):
        for e, q in ti.ndrange(self.n_elements, 8):
            tensor = self._stress_tensor(self.fem_stress[e, q])
            measure = self.fem_det_j[e, q]
            for a in range(8):
                node = self.elements[e][a]
                contribution = (
                    -measure * tensor @ self.fem_grad[e, q, a]
                )
                for d in ti.static(range(3)):
                    ti.atomic_add(
                        self.fem_force[node][d], contribution[d]
                    )

    @ti.kernel
    def _mpm_internal_force(self):
        origin = ti.Vector([self.origin_x, self.origin_y, self.origin_z])
        for p in range(self.n_particles):
            base = self._mpm_base(self.mpm_x[p])
            fx = (self.mpm_x[p] - (origin + base * (1.0 / self.inv_h))) * (
                self.inv_h
            )
            tensor = self._stress_tensor(self.mpm_stress[p])
            for offset in ti.static(ti.grouped(ti.ndrange(2, 2, 2))):
                _, gradient = self._mpm_shape(fx, offset)
                I = base + offset
                contribution = (
                    -self.mpm_volume[p] * tensor @ gradient
                )
                for d in ti.static(range(3)):
                    ti.atomic_add(
                        self.grid_force[I][d], contribution[d]
                    )

    @ti.kernel
    def _make_trial_state(self):
        for i in range(self.n_fem_nodes):
            velocity = self.fem_v[i] + (
                self.dt * self.fem_force[i] / self.fem_mass[i]
            )
            if self.fem_lock_y[i] == 1:
                velocity[1] = 0.0
            if self.fem_lock_z[i] == 1:
                velocity[2] = 0.0
            self.fem_trial_v[i] = velocity
        for I in ti.grouped(self.grid_momentum):
            self.grid_momentum[I] += self.dt * self.grid_force[I]

    @ti.kernel
    def _finish_step(self):
        for i in range(self.n_fem_nodes):
            self.fem_v[i] = self.fem_trial_v[i]
            self.fem_x[i] += self.dt * self.fem_v[i]

    @ti.kernel
    def _g2p(self):
        origin = ti.Vector([self.origin_x, self.origin_y, self.origin_z])
        for p in range(self.n_particles):
            base = self._mpm_base(self.mpm_x[p])
            fx = (self.mpm_x[p] - (origin + base * (1.0 / self.inv_h))) * (
                self.inv_h
            )
            delta = ti.Vector.zero(ti.f64, 3)
            pic = ti.Vector.zero(ti.f64, 3)
            for offset in ti.static(ti.grouped(ti.ndrange(2, 2, 2))):
                weight, _ = self._mpm_shape(fx, offset)
                I = base + offset
                delta += weight * (
                    self.grid_velocity[I] - self.grid_velocity_before[I]
                )
                pic += weight * self.grid_velocity[I]
            flip = self.mpm_v[p] + delta
            self.mpm_v[p] = (
                self.flip_ratio * flip + (1.0 - self.flip_ratio) * pic
            )
            self.mpm_x[p] += self.dt * pic

    @ti.kernel
    def _reset_metrics(self):
        self.metric_kinetic[None] = 0.0
        self.metric_strain[None] = 0.0
        self.metric_momentum[None] = ti.Vector.zero(ti.f64, 3)
        self.metric_transverse[None] = 0.0

    @ti.kernel
    def _accumulate_metrics(self):
        for i in range(self.n_fem_nodes):
            ti.atomic_add(
                self.metric_kinetic[None],
                0.5 * self.fem_mass[i] * self.fem_v[i].dot(self.fem_v[i]),
            )
            for d in ti.static(range(3)):
                ti.atomic_add(
                    self.metric_momentum[None][d],
                    self.fem_mass[i] * self.fem_v[i][d],
                )
            ti.atomic_max(
                self.metric_transverse[None],
                ti.abs(self.fem_v[i][1]),
            )
            ti.atomic_max(
                self.metric_transverse[None],
                ti.abs(self.fem_v[i][2]),
            )
        for e, q in ti.ndrange(self.n_elements, 8):
            ti.atomic_add(
                self.metric_strain[None],
                0.5
                * self.fem_stress[e, q].dot(self.fem_strain[e, q])
                * self.fem_det_j[e, q],
            )
        for p in range(self.n_particles):
            ti.atomic_add(
                self.metric_kinetic[None],
                0.5 * self.mpm_mass[p] * self.mpm_v[p].dot(self.mpm_v[p]),
            )
            ti.atomic_add(
                self.metric_strain[None],
                0.5
                * self.mpm_stress[p].dot(self.mpm_strain[p])
                * self.mpm_volume[p],
            )
            for d in ti.static(range(3)):
                ti.atomic_add(
                    self.metric_momentum[None][d],
                    self.mpm_mass[p] * self.mpm_v[p][d],
                )
            ti.atomic_max(
                self.metric_transverse[None],
                ti.abs(self.mpm_v[p][1]),
            )
            ti.atomic_max(
                self.metric_transverse[None],
                ti.abs(self.mpm_v[p][2]),
            )

    @ti.kernel
    def _reset_gap(self):
        self.gap_fem_surface[None] = -1.0e30
        self.gap_mpm_surface[None] = 1.0e30

    @ti.kernel
    def _accumulate_gap(self):
        for a in range(self.surface_field.shape[0]):
            node = self.surface_field[a]
            ti.atomic_max(
                self.gap_fem_surface[None], self.fem_x[node][0]
            )
        for p in range(self.n_particles):
            ti.atomic_min(
                self.gap_mpm_surface[None], self.mpm_x[p][0]
            )

    def _gap(self) -> float:
        self._reset_gap()
        self._accumulate_gap()
        return (
            float(self.gap_mpm_surface[None])
            - 0.5 * self.config.particle_spacing
            - float(self.gap_fem_surface[None])
        )

    def step(self) -> ContactStep3D:
        gap = self._gap()
        close = int(gap <= self.config.contact_tolerance)

        self._reset_grid()
        self._p2g()
        self._compute_grid_velocity(1)
        self._reset_contact()
        self._map_surface(0)
        self._project_contact(close)
        self._scatter_contact(0)
        active_1 = int(self.contact_active[None])
        impulse_1 = float(self.contact_magnitude[None])
        residual_1 = float(
            np.linalg.norm(
                np.asarray(self.contact_grid_total[None])
                + np.asarray(self.contact_fem_total[None])
            )
        )
        self._compute_grid_velocity(0)

        self._update_fem_stress()
        self._update_mpm_stress()
        self._clear_forces()
        self._fem_internal_force()
        self._mpm_internal_force()
        self._make_trial_state()
        self._reset_contact()
        self._map_surface(1)
        self._project_contact(close)
        self._scatter_contact(1)
        active_2 = int(self.contact_active[None])
        impulse_2 = float(self.contact_magnitude[None])
        residual_2 = float(
            np.linalg.norm(
                np.asarray(self.contact_grid_total[None])
                + np.asarray(self.contact_fem_total[None])
            )
        )
        self._compute_grid_velocity(0)
        self._finish_step()
        self._g2p()
        self.time += self.config.dt

        return ContactStep3D(
            active_nodes=max(active_1, active_2),
            gap=gap,
            impulse_1=impulse_1,
            impulse_2=impulse_2,
            balance_residual=max(residual_1, residual_2),
        )

    def total_state(self) -> tuple[float, float, np.ndarray, float]:
        self._reset_metrics()
        self._accumulate_metrics()
        return (
            float(self.metric_kinetic[None]),
            float(self.metric_strain[None]),
            np.asarray(self.metric_momentum[None], dtype=float),
            float(self.metric_transverse[None]),
        )

    def combined_stress_profile(self) -> tuple[np.ndarray, np.ndarray]:
        fem_position = self.fem_x.to_numpy()
        fem_stress = self.fem_stress.to_numpy()
        element_center = np.mean(
            fem_position[self.elements_np], axis=1
        )
        element_sxx = np.mean(fem_stress[:, :, 0], axis=1)
        fem_x = np.array(
            [
                np.mean(element_center[self.fem_columns == i, 0])
                for i in range(self.fem_nx)
            ]
        )
        fem_sxx = np.array(
            [
                np.mean(element_sxx[self.fem_columns == i])
                for i in range(self.fem_nx)
            ]
        )

        mpm_position = self.mpm_x.to_numpy()
        mpm_stress = self.mpm_stress.to_numpy()
        mpm_x = np.array(
            [
                np.mean(mpm_position[self.mpm_columns == i, 0])
                for i in range(self.mpm_nx)
            ]
        )
        mpm_sxx = np.array(
            [
                np.mean(mpm_stress[self.mpm_columns == i, 0])
                for i in range(self.mpm_nx)
            ]
        )
        return np.concatenate((fem_x, mpm_x)), np.concatenate(
            (fem_sxx, mpm_sxx)
        )

    def snapshot(self) -> dict[str, np.ndarray]:
        fem_nodes = self.fem_x.to_numpy()
        fem_stress = self.fem_stress.to_numpy()
        mpm_position = self.mpm_x.to_numpy()
        mpm_stress = self.mpm_stress.to_numpy()
        profile_x, profile_stress = self.combined_stress_profile()
        return {
            "fem_nodes": fem_nodes,
            "fem_elements": self.elements_np.copy(),
            "fem_position": np.mean(
                fem_nodes[self.elements_np], axis=1
            ),
            "fem_stress": np.mean(fem_stress, axis=1),
            "mpm_position": mpm_position,
            "mpm_stress": mpm_stress,
            "profile_x": profile_x,
            "profile_stress": profile_stress,
        }

    def _record(
        self,
        history: SimulationHistory3D,
        contact: ContactStep3D,
    ) -> None:
        kinetic, strain, momentum, transverse = self.total_state()
        history.time.append(self.time)
        history.kinetic.append(kinetic)
        history.strain.append(strain)
        history.total_energy.append(kinetic + strain)
        history.momentum_x.append(float(momentum[0]))
        history.momentum_y.append(float(momentum[1]))
        history.momentum_z.append(float(momentum[2]))
        history.gap.append(self._gap())
        history.contact_impulse.append(contact.impulse)
        history.contact_nodes.append(contact.active_nodes)
        history.contact_balance.append(contact.balance_residual)
        history.transverse_velocity.append(transverse)

    def run(
        self,
        *,
        end_time: float | None = None,
        snapshot_times: Iterable[float] | None = None,
    ) -> SimulationHistory3D:
        target = self.config.end_time if end_time is None else end_time
        wanted = sorted(
            self.config.snapshot_times
            if snapshot_times is None
            else snapshot_times
        )
        history = SimulationHistory3D()
        snapshot_index = 0
        last_contact_time: float | None = None
        self._record(history, ContactStep3D(gap=self._gap()))

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
