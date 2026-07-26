"""运行独立的惩罚接触演示。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib
import numpy as np

from .contact import PenaltyContact, solve_contact_equilibrium
from .fem import FEMWallBeam
from .mpm import SmallStrainMPM2D

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def build_demo() -> tuple[
    SmallStrainMPM2D, FEMWallBeam, PenaltyContact
]:
    dx = 0.1
    mpm = SmallStrainMPM2D(
        (0.0, 0.0),
        dx,
        31,
        16,
        density=1800.0,
        young=2.0e6,
        poisson=0.3,
    )
    radius = mpm.add_rect_particles(
        0.523,
        2.0,
        0.5 * dx / 2,
        1.0,
        particles_per_cell=2,
    )
    mpm.set_bottom_roller()
    wall = FEMWallBeam(
        0.5,
        0.0,
        1.0,
        10,
        young=30.0e9,
        inertia=0.3**3 / 12.0,
    )
    contact = PenaltyContact(
        stiffness=2.0e6,
        particle_radius=radius,
        damping=2.0e3,
        max_penetration=0.25 * dx,
    )
    return mpm, wall, contact


def run_demo(
    output_dir: str | os.PathLike[str],
    *,
    steps: int = 300,
    dt: float = 2.0e-5,
) -> dict[str, float]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    mpm, wall, contact = build_demo()
    max_balance_error = 0.0
    max_penetration = 0.0
    contact_count = 0

    for step in range(steps):
        result = solve_contact_equilibrium(
            mpm, wall, contact, iterations=3
        )
        max_balance_error = max(max_balance_error, result.balance_error)
        max_penetration = max(max_penetration, result.max_penetration)
        contact_count = max(contact_count, result.active_particles)
        gravity_scale = min((step + 1) / max(1, steps // 2), 1.0)
        mpm.step(
            dt,
            damping=0.08,
            gravity_scale=gravity_scale,
        )

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    particles = ax.scatter(
        mpm.x[:, 0],
        mpm.x[:, 1],
        s=6,
        c=mpm.stress[:, 0, 0] / 1e3,
    )
    wall_nodes = wall.nodes
    ax.plot(wall_nodes[:, 0], wall_nodes[:, 1], "k-", lw=2.0)
    fig.colorbar(particles, ax=ax, label=r"$\sigma_{xx}$ (kPa)")
    ax.set_aspect("equal")
    ax.set(
        xlabel="x (m)",
        ylabel="y (m)",
        title="Penalty-contact MPM soil and FEM wall",
    )
    fig.tight_layout()
    fig.savefig(output / "penalty_contact.png", dpi=180)
    plt.close(fig)

    metrics = {
        "steps": steps,
        "max_active_particles": contact_count,
        "max_penetration_m": max_penetration,
        "max_action_reaction_error_n": max_balance_error,
    }
    (output / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 MPM-FEM 惩罚接触演示")
    parser.add_argument(
        "--output",
        default="results/penalty_contact",
        help="图片与指标输出目录",
    )
    parser.add_argument("--steps", type=int, default=300)
    args = parser.parse_args()
    print(
        json.dumps(
            run_demo(args.output, steps=args.steps),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
