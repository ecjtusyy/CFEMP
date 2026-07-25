"""论文对称弹性板碰撞算例的一维平面应变约化。

左板采用线性有限元，右板采用线性形函数 MPM。耦合点分别保存 FEM 与
MPM 两套质量和动量，通过论文中的两段法向接触冲量交换动量：

J1 = μ(v_fem - v_mpm)
J2 = dt (m_mpm f_fem - m_fem f_mpm) / (m_fem + m_mpm)

接触冲量对两侧等大反向，因此不改变系统总动量。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class PlateImpactConfig:
    """Lian 等（2011）对称弹性板碰撞的 SI 制参数。"""

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
    dt: float = 1.0e-8
    end_time: float = 15.0e-6
    snapshot_times: tuple[float, ...] = (3.0e-6, 8.7e-6, 15.0e-6)
    contact_tolerance: float = 1.0e-10
    flip_ratio: float = 0.95

    @property
    def area(self) -> float:
        return self.width * self.thickness

    @property
    def wave_speed(self) -> float:
        # 论文 ν=0，此时一维与平面应变波速一致。
        constrained_modulus = (
            self.young
            * (1.0 - self.poisson)
            / ((1.0 + self.poisson) * (1.0 - 2.0 * self.poisson))
        )
        return sqrt(constrained_modulus / self.density)

    @property
    def stable_dt(self) -> float:
        return min(self.fem_element_size, self.mpm_cell_size) / self.wave_speed

    def validate(self) -> None:
        if self.dt <= 0.0 or self.end_time <= 0.0:
            raise ValueError("时间步和终止时间必须为正")
        if self.dt > 0.5 * self.stable_dt:
            raise ValueError(
                f"时间步过大：dt={self.dt:.3e}, 建议不超过 "
                f"0.5*h/c={0.5*self.stable_dt:.3e}"
            )
        for name, value in (
            ("length", self.length),
            ("area", self.area),
            ("young", self.young),
            ("density", self.density),
            ("fem_element_size", self.fem_element_size),
            ("mpm_cell_size", self.mpm_cell_size),
            ("particle_spacing", self.particle_spacing),
        ):
            if value <= 0.0:
                raise ValueError(f"{name} 必须为正")
        if not np.isclose(
            self.length / self.fem_element_size,
            round(self.length / self.fem_element_size),
        ):
            raise ValueError("FEM 板长必须是单元尺寸的整数倍")
        if not np.isclose(
            self.length / self.particle_spacing,
            round(self.length / self.particle_spacing),
        ):
            raise ValueError("MPM 板长必须是粒子间距的整数倍")
        if not 0.0 <= self.flip_ratio <= 1.0:
            raise ValueError("flip_ratio 必须位于 [0, 1]")


def analytical_contact_stress(config: PlateImpactConfig) -> float:
    """相同材料板对称碰撞产生的压应力，压缩取负。"""

    return -config.density * config.wave_speed * config.impact_speed


def analytical_separation_time(config: PlateImpactConfig) -> float:
    """压缩波到自由端反射并返回接触面的时间。"""

    return 2.0 * config.length / config.wave_speed


class LinearFEMBar:
    """总拉格朗日一维线性有限元杆，使用团块质量。"""

    def __init__(self, config: PlateImpactConfig):
        self.config = config
        ne = int(round(config.length / config.fem_element_size))
        self.x0 = np.linspace(-config.length, 0.0, ne + 1)
        self.x = self.x0.copy()
        self.v = np.full(ne + 1, config.impact_speed, dtype=float)
        self.mass = np.zeros(ne + 1, dtype=float)
        element_mass = (
            config.density * config.area * config.fem_element_size
        )
        self.mass[:-1] += 0.5 * element_mass
        self.mass[1:] += 0.5 * element_mass
        self.strain = np.zeros(ne, dtype=float)
        self.stress = np.zeros(ne, dtype=float)

    def update_stress(self, velocity: np.ndarray, dt: float) -> None:
        grad_v = np.diff(velocity) / self.config.fem_element_size
        self.strain += dt * grad_v
        self.stress = self.config.young * self.strain

    def nodal_force(self) -> np.ndarray:
        force = np.zeros_like(self.v)
        traction = self.config.area * self.stress
        force[:-1] += traction
        force[1:] -= traction
        return force

    def energy(self) -> tuple[float, float]:
        kinetic = 0.5 * np.sum(self.mass * self.v**2)
        volume = self.config.area * self.config.fem_element_size
        strain = np.sum(0.5 * self.stress * self.strain * volume)
        return float(kinetic), float(strain)

    def momentum(self) -> float:
        return float(np.sum(self.mass * self.v))

    def stress_points(self) -> tuple[np.ndarray, np.ndarray]:
        return 0.5 * (self.x[:-1] + self.x[1:]), self.stress.copy()


class MaterialPointBar:
    """一维 MPM 杆；粒子保存状态，背景网格每步重置。"""

    def __init__(self, config: PlateImpactConfig):
        self.config = config
        nparticles = int(round(config.length / config.particle_spacing))
        self.x = (
            np.arange(nparticles, dtype=float) + 0.5
        ) * config.particle_spacing
        self.v = np.full(nparticles, -config.impact_speed, dtype=float)
        particle_volume = config.area * config.particle_spacing
        self.volume = np.full(nparticles, particle_volume, dtype=float)
        self.mass = np.full(
            nparticles, config.density * particle_volume, dtype=float
        )
        self.strain = np.zeros(nparticles, dtype=float)
        self.stress = np.zeros(nparticles, dtype=float)

        # 两侧留出背景格，避免自由端附近丢失形函数支撑域。
        h = config.mpm_cell_size
        self.grid_x = np.arange(-2.0 * h, config.length + 2.5 * h, h)

    def _stencil(self, x: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        h = self.config.mpm_cell_size
        cell = int(np.floor((x - self.grid_x[0]) / h))
        cell = int(np.clip(cell, 0, len(self.grid_x) - 2))
        left = self.grid_x[cell]
        xi = (x - left) / h
        ids = np.array([cell, cell + 1], dtype=int)
        weights = np.array([1.0 - xi, xi], dtype=float)
        grads = np.array([-1.0 / h, 1.0 / h], dtype=float)
        return ids, weights, grads

    def p2g(self) -> tuple[np.ndarray, np.ndarray]:
        grid_mass = np.zeros_like(self.grid_x)
        grid_momentum = np.zeros_like(self.grid_x)
        for xp, vp, mp in zip(self.x, self.v, self.mass):
            ids, weights, _ = self._stencil(float(xp))
            grid_mass[ids] += weights * mp
            grid_momentum[ids] += weights * mp * vp
        return grid_mass, grid_momentum

    @staticmethod
    def grid_velocity(
        grid_mass: np.ndarray, grid_momentum: np.ndarray
    ) -> np.ndarray:
        velocity = np.zeros_like(grid_momentum)
        active = grid_mass > 0.0
        velocity[active] = grid_momentum[active] / grid_mass[active]
        return velocity

    def update_stress(self, grid_velocity: np.ndarray, dt: float) -> None:
        for p, xp in enumerate(self.x):
            ids, _, grads = self._stencil(float(xp))
            grad_v = float(np.dot(grid_velocity[ids], grads))
            self.strain[p] += dt * grad_v
            self.stress[p] = self.config.young * self.strain[p]

    def grid_force(self) -> np.ndarray:
        force = np.zeros_like(self.grid_x)
        for xp, volume, stress in zip(self.x, self.volume, self.stress):
            ids, _, grads = self._stencil(float(xp))
            force[ids] += -volume * stress * grads
        return force

    def g2p(
        self,
        grid_velocity_before: np.ndarray,
        grid_velocity_after: np.ndarray,
        dt: float,
    ) -> None:
        for p, xp in enumerate(self.x.copy()):
            ids, weights, _ = self._stencil(float(xp))
            delta_v = np.dot(
                weights,
                grid_velocity_after[ids] - grid_velocity_before[ids],
            )
            pic_velocity = float(np.dot(weights, grid_velocity_after[ids]))
            flip_velocity = self.v[p] + delta_v
            self.v[p] = (
                self.config.flip_ratio * flip_velocity
                + (1.0 - self.config.flip_ratio) * pic_velocity
            )
            self.x[p] += dt * pic_velocity

    def surface_position(self) -> float:
        # 初始粒子位于子域中心，左表面在最左粒子中心左侧半个间距。
        return float(np.min(self.x) - 0.5 * self.config.particle_spacing)

    def contact_grid_index(self, x_contact: float) -> int:
        return int(np.argmin(np.abs(self.grid_x - x_contact)))

    def energy(self) -> tuple[float, float]:
        kinetic = 0.5 * np.sum(self.mass * self.v**2)
        strain = np.sum(0.5 * self.stress * self.strain * self.volume)
        return float(kinetic), float(strain)

    def momentum(self) -> float:
        return float(np.sum(self.mass * self.v))

    def stress_points(self) -> tuple[np.ndarray, np.ndarray]:
        return self.x.copy(), self.stress.copy()


@dataclass
class ContactStep:
    active: bool = False
    grid_index: int = -1
    gap: float = np.inf
    impulse_1: float = 0.0
    impulse_2: float = 0.0
    fem_mass: float = 0.0
    mpm_mass: float = 0.0

    @property
    def impulse(self) -> float:
        return self.impulse_1 + self.impulse_2


@dataclass
class SimulationHistory:
    time: list[float] = field(default_factory=list)
    kinetic: list[float] = field(default_factory=list)
    strain: list[float] = field(default_factory=list)
    total_energy: list[float] = field(default_factory=list)
    momentum: list[float] = field(default_factory=list)
    gap: list[float] = field(default_factory=list)
    contact_impulse: list[float] = field(default_factory=list)
    snapshots: dict[float, tuple[np.ndarray, np.ndarray]] = field(
        default_factory=dict
    )
    separation_time: float | None = None

    def as_arrays(self) -> dict[str, np.ndarray]:
        return {
            "time": np.asarray(self.time),
            "kinetic": np.asarray(self.kinetic),
            "strain": np.asarray(self.strain),
            "total_energy": np.asarray(self.total_energy),
            "momentum": np.asarray(self.momentum),
            "gap": np.asarray(self.gap),
            "contact_impulse": np.asarray(self.contact_impulse),
        }


class CFEMPPlateImpact1D:
    """论文网格接触法的一维约化求解器。"""

    def __init__(self, config: PlateImpactConfig | None = None):
        self.config = config or PlateImpactConfig()
        self.config.validate()
        self.fem = LinearFEMBar(self.config)
        self.mpm = MaterialPointBar(self.config)
        self.time = 0.0
        self.was_in_contact = False
        self.has_contacted = False
        self.last_contact = ContactStep()

    def _gap(self) -> float:
        return self.mpm.surface_position() - float(self.fem.x[-1])

    def _precontact(
        self,
        grid_mass: np.ndarray,
        grid_momentum: np.ndarray,
    ) -> tuple[ContactStep, np.ndarray]:
        contact = ContactStep(gap=self._gap())
        grid_velocity_before = self.mpm.grid_velocity(
            grid_mass, grid_momentum
        )
        index = self.mpm.contact_grid_index(float(self.fem.x[-1]))
        ms = float(grid_mass[index])
        mr = float(self.fem.mass[-1])
        if ms <= 0.0:
            return contact, grid_velocity_before

        vr = float(self.fem.v[-1])
        vs = float(grid_velocity_before[index])
        closing_velocity = vr - vs
        geometrically_close = contact.gap <= self.config.contact_tolerance
        contact.active = geometrically_close and (
            closing_velocity > 0.0 or self.was_in_contact
        )
        if not contact.active:
            return contact, grid_velocity_before

        reduced_mass = mr * ms / (mr + ms)
        impulse_1 = max(0.0, reduced_mass * closing_velocity)
        self.fem.v[-1] -= impulse_1 / mr
        grid_momentum[index] += impulse_1

        contact.grid_index = index
        contact.impulse_1 = impulse_1
        contact.fem_mass = mr
        contact.mpm_mass = ms
        self.has_contacted = True
        return contact, grid_velocity_before

    def _finish_contact(
        self,
        contact: ContactStep,
        fem_force: np.ndarray,
        mpm_force: np.ndarray,
        grid_momentum: np.ndarray,
    ) -> None:
        if not contact.active:
            return
        index = contact.grid_index
        mr = contact.fem_mass
        ms = contact.mpm_mass
        impulse_2 = self.config.dt * (
            ms * fem_force[-1] - mr * mpm_force[index]
        ) / (mr + ms)

        # 接触只能传递压力，不能传递拉力。
        impulse_2 = max(impulse_2, -contact.impulse_1)
        contact.impulse_2 = impulse_2
        self.fem.v[-1] -= impulse_2 / mr
        grid_momentum[index] += impulse_2

        if contact.impulse <= 1.0e-18:
            contact.active = False

    def step(self) -> ContactStep:
        dt = self.config.dt
        grid_mass, grid_momentum = self.mpm.p2g()
        contact, grid_velocity_before = self._precontact(
            grid_mass, grid_momentum
        )
        adjusted_grid_velocity = self.mpm.grid_velocity(
            grid_mass, grid_momentum
        )

        # 论文 USF 顺序：用预调整后的半步速度更新应力。
        self.fem.update_stress(self.fem.v, dt)
        self.mpm.update_stress(adjusted_grid_velocity, dt)
        fem_force = self.fem.nodal_force()
        mpm_force = self.mpm.grid_force()

        self.fem.v += dt * fem_force / self.fem.mass
        grid_momentum += dt * mpm_force
        self._finish_contact(
            contact, fem_force, mpm_force, grid_momentum
        )

        final_grid_velocity = self.mpm.grid_velocity(
            grid_mass, grid_momentum
        )
        self.fem.x += dt * self.fem.v
        self.mpm.g2p(
            grid_velocity_before,
            final_grid_velocity,
            dt,
        )
        self.time += dt
        self.was_in_contact = contact.active
        self.last_contact = contact
        return contact

    def total_momentum(self) -> float:
        return self.fem.momentum() + self.mpm.momentum()

    def total_energy(self) -> tuple[float, float, float]:
        fem_ke, fem_se = self.fem.energy()
        mpm_ke, mpm_se = self.mpm.energy()
        kinetic = fem_ke + mpm_ke
        strain = fem_se + mpm_se
        return kinetic, strain, kinetic + strain

    def combined_stress_profile(self) -> tuple[np.ndarray, np.ndarray]:
        xf, sf = self.fem.stress_points()
        xm, sm = self.mpm.stress_points()
        order = np.argsort(np.concatenate([xf, xm]))
        x = np.concatenate([xf, xm])[order]
        stress = np.concatenate([sf, sm])[order]
        return x, stress

    def run(
        self,
        *,
        end_time: float | None = None,
        snapshot_times: Iterable[float] | None = None,
    ) -> SimulationHistory:
        target_time = end_time or self.config.end_time
        wanted = sorted(
            self.config.snapshot_times
            if snapshot_times is None
            else snapshot_times
        )
        history = SimulationHistory()
        snapshot_index = 0
        last_contact_time: float | None = None

        self._record(history, 0.0)
        while self.time < target_time - 0.5 * self.config.dt:
            contact = self.step()
            self._record(history, contact.impulse)
            if contact.active:
                last_contact_time = self.time

            while (
                snapshot_index < len(wanted)
                and self.time >= wanted[snapshot_index] - 0.5 * self.config.dt
            ):
                key = float(wanted[snapshot_index])
                history.snapshots[key] = self.combined_stress_profile()
                snapshot_index += 1

        # 线性 MPM 的自由表面位置会在一个时间步量级内轻微抖动，可能造成
        # 单步“离开后重新接触”。最终分离时间取最后一次压缩接触之后的时刻，
        # 避免把这种网格穿越抖动误报为物理分离。
        if last_contact_time is not None:
            history.separation_time = last_contact_time + self.config.dt

        return history

    def _record(
        self, history: SimulationHistory, contact_impulse: float
    ) -> None:
        kinetic, strain, total = self.total_energy()
        history.time.append(self.time)
        history.kinetic.append(kinetic)
        history.strain.append(strain)
        history.total_energy.append(total)
        history.momentum.append(self.total_momentum())
        history.gap.append(self._gap())
        history.contact_impulse.append(contact_impulse)
