"""生成 Taichi 三维对称板碰撞结果。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib
import numpy as np

from .plate_impact_3d import (
    PlateImpact3DConfig,
    SimulationHistory3D,
    TaichiCFEMPPlateImpact3D,
    analytical_contact_stress_3d,
    analytical_separation_time_3d,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402


def _plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 180,
            "font.size": 9.5,
            "axes.grid": True,
            "grid.alpha": 0.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def analytical_profile_3d(
    x: np.ndarray,
    time_value: float,
    config: PlateImpact3DConfig,
) -> np.ndarray:
    stress = np.zeros_like(x)
    front = min(config.wave_speed * time_value, config.length)
    if time_value <= analytical_separation_time_3d(config):
        stress[np.abs(x) <= front] = analytical_contact_stress_3d(config)
    return stress


def _draw_discretization(
    solver: TaichiCFEMPPlateImpact3D,
    output: Path,
) -> None:
    nodes = solver.fem_nodes0
    particles = solver.mpm_x.to_numpy()
    fig = plt.figure(figsize=(11.0, 4.6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_proj_type("ortho")
    ax.scatter(
        nodes[:, 0] * 1e3,
        nodes[:, 1] * 1e3,
        nodes[:, 2] * 1e3,
        s=1.2,
        c="#28536B",
        alpha=0.55,
        linewidths=0,
        label="HEX8 nodes",
    )
    sample = np.arange(0, len(particles), 4)
    ax.scatter(
        particles[sample, 0] * 1e3,
        particles[sample, 1] * 1e3,
        particles[sample, 2] * 1e3,
        s=0.7,
        c="#C44536",
        alpha=0.45,
        linewidths=0,
        label="MPM particles",
    )
    ax.set_xlabel("x (mm)", labelpad=5)
    ax.set_title("3D HEX8 FEM / MPM discretization", pad=10)
    ax.set_yticks((0.0, 1.5, 3.0))
    ax.set_zticks((0.0, 1.5, 3.0))
    ax.tick_params(axis="both", labelsize=8, pad=1)
    ax.set_box_aspect((8.0, 1.8, 1.8))
    ax.view_init(elev=24, azim=-62)
    ax.legend(loc="upper center", ncol=2)
    ax.text2D(
        0.69,
        0.04,
        "cross-section: 3 x 3 mm",
        transform=ax.transAxes,
        fontsize=9,
    )
    fig.subplots_adjust(left=0.02, right=0.95, bottom=0.02, top=0.88)
    fig.savefig(output / "discretization_3d.png", bbox_inches="tight")
    plt.close(fig)


def _draw_stress_slice(
    snapshot: dict[str, np.ndarray],
    config: PlateImpact3DConfig,
    output: Path,
) -> None:
    center = 0.5 * config.depth
    half_band = 0.26e-3
    fem_position = snapshot["fem_position"]
    fem_stress = snapshot["fem_stress"][:, 0] / 1e6
    mpm_position = snapshot["mpm_position"]
    mpm_stress = snapshot["mpm_stress"][:, 0] / 1e6
    fem_slice = np.abs(fem_position[:, 2] - center) <= half_band
    mpm_slice = np.abs(mpm_position[:, 2] - center) <= half_band
    scale = abs(analytical_contact_stress_3d(config)) / 1e6
    norm = Normalize(vmin=-1.2 * scale, vmax=0.1 * scale)

    fig, ax = plt.subplots(figsize=(10.0, 2.8))
    artist = ax.scatter(
        fem_position[fem_slice, 0] * 1e3,
        fem_position[fem_slice, 1] * 1e3,
        c=fem_stress[fem_slice],
        cmap="coolwarm",
        norm=norm,
        marker="s",
        s=24,
        linewidths=0,
    )
    ax.scatter(
        mpm_position[mpm_slice, 0] * 1e3,
        mpm_position[mpm_slice, 1] * 1e3,
        c=mpm_stress[mpm_slice],
        cmap="coolwarm",
        norm=norm,
        marker="s",
        s=4,
        linewidths=0,
    )
    ax.axvline(0.0, color="black", lw=0.6)
    ax.set(
        xlabel="x (mm)",
        ylabel="y (mm)",
        title=r"mid-plane $\sigma_{xx}$ at 3.0 $\mu$s",
        ylim=(-0.15, config.width * 1e3 + 0.15),
    )
    ax.set_aspect("equal")
    colorbar = fig.colorbar(artist, ax=ax, pad=0.015)
    colorbar.set_label(r"$\sigma_{xx}$ (MPa)")
    fig.tight_layout()
    fig.savefig(output / "stress_slice_3us.png")
    plt.close(fig)


def _draw_histories(
    history: SimulationHistory3D,
    config: PlateImpact3DConfig,
    output: Path,
) -> None:
    arrays = history.as_arrays()
    snapshot = history.snapshots[3.0e-6]
    x = snapshot["profile_x"]
    stress = snapshot["profile_stress"]

    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    ax.plot(
        x * 1e3,
        analytical_profile_3d(x, 3.0e-6, config) / 1e6,
        "k--",
        lw=1.4,
        label="analytical",
    )
    ax.plot(
        x * 1e3,
        stress / 1e6,
        color="#1D6A96",
        lw=1.1,
        label="Taichi 3D CFEMP",
    )
    ax.axvline(0.0, color="#B23A48", lw=0.8)
    ax.set(
        xlabel="x (mm)",
        ylabel=r"$\sigma_{xx}$ (MPa)",
        title=r"centerline stress at 3.0 $\mu$s",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "stress_profile_3us.png")
    plt.close(fig)

    time_us = arrays["time"] * 1e6
    initial_energy = arrays["total_energy"][0]
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    ax.plot(time_us, arrays["kinetic"] / initial_energy, label="kinetic")
    ax.plot(time_us, arrays["strain"] / initial_energy, label="strain")
    ax.plot(time_us, arrays["total_energy"] / initial_energy, label="total")
    ax.set(xlabel=r"time ($\mu$s)", ylabel="normalized energy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "energy_evolution.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    ax.plot(time_us, arrays["gap"] * 1e6, color="#287271")
    if history.separation_time is not None:
        ax.axvline(
            history.separation_time * 1e6,
            color="#B23A48",
            ls="--",
            label=f"Taichi 3D {history.separation_time*1e6:.2f}",
        )
    exact = analytical_separation_time_3d(config) * 1e6
    ax.axvline(exact, color="black", ls=":", label=f"analytical {exact:.2f}")
    ax.set(xlabel=r"time ($\mu$s)", ylabel=r"gap ($\mu$m)")
    ax.legend(title=r"separation time ($\mu$s)")
    fig.tight_layout()
    fig.savefig(output / "contact_separation.png")
    plt.close(fig)


def _stress_metric(
    snapshot: dict[str, np.ndarray],
    config: PlateImpact3DConfig,
) -> float:
    x = snapshot["profile_x"]
    stress = snapshot["profile_stress"]
    plateau = np.abs(x) <= 0.5 * config.wave_speed * 3.0e-6
    return float(np.median(stress[plateau]))


def run_benchmark_3d(
    output_dir: str | os.PathLike[str],
) -> dict[str, float | int | str]:
    _plot_style()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = PlateImpact3DConfig()
    solver = TaichiCFEMPPlateImpact3D(config)
    _draw_discretization(solver, output)
    history = solver.run()
    arrays = history.as_arrays()
    snapshot = history.snapshots[3.0e-6]
    _draw_stress_slice(snapshot, config, output)
    _draw_histories(history, config, output)

    exact_stress = analytical_contact_stress_3d(config)
    numerical_stress = _stress_metric(snapshot, config)
    separation = history.separation_time or float("nan")
    initial_energy = arrays["total_energy"][0]
    momentum_scale = (
        config.density
        * config.area
        * config.length
        * config.impact_speed
    )
    momentum = np.sqrt(
        (arrays["momentum_x"] - arrays["momentum_x"][0]) ** 2
        + (arrays["momentum_y"] - arrays["momentum_y"][0]) ** 2
        + (arrays["momentum_z"] - arrays["momentum_z"][0]) ** 2
    )

    fem_sxx = snapshot["fem_stress"][:, 0].reshape(
        solver.fem_nx, solver.fem_ny * solver.fem_nz
    )
    mpm_sxx = snapshot["mpm_stress"][:, 0].reshape(
        solver.mpm_nx, solver.mpm_ny * solver.mpm_nz
    )
    transverse_stress_spread = max(
        float(np.max(np.ptp(fem_sxx, axis=1))),
        float(np.max(np.ptp(mpm_sxx, axis=1))),
    ) / abs(exact_stress)

    metrics: dict[str, float | int | str] = {
        "dimensions": 3,
        "backend": "taichi-cpu",
        "fem_nodes": solver.n_fem_nodes,
        "fem_hex8_elements": solver.n_elements,
        "mpm_particles": solver.n_particles,
        "grid_nodes": int(np.prod(solver.grid_shape)),
        "velocity_components": 3,
        "stress_components": 6,
        "element_nodes": 8,
        "analytical_contact_stress_mpa": exact_stress / 1e6,
        "numerical_contact_stress_mpa": numerical_stress / 1e6,
        "stress_relative_error": abs(numerical_stress - exact_stress)
        / abs(exact_stress),
        "analytical_separation_time_us": analytical_separation_time_3d(config)
        * 1e6,
        "numerical_separation_time_us": separation * 1e6,
        "separation_relative_error": abs(
            separation - analytical_separation_time_3d(config)
        )
        / analytical_separation_time_3d(config),
        "max_relative_energy_error": float(
            np.max(np.abs(arrays["total_energy"] - initial_energy))
            / initial_energy
        ),
        "max_normalized_momentum_error": float(
            np.max(momentum) / momentum_scale
        ),
        "max_contact_impulse_balance_kg_m_s": float(
            np.max(arrays["contact_balance"])
        ),
        "max_transverse_velocity_m_s": float(
            np.max(arrays["transverse_velocity"])
        ),
        "max_transverse_stress_spread": transverse_stress_spread,
        "max_penetration_um": float(max(0.0, -np.min(arrays["gap"])) * 1e6),
        "steps": int(len(arrays["time"]) - 1),
    }
    (output / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(output / "history.npz", **arrays)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="results/symmetric_plate_impact_3d",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            run_benchmark_3d(args.output),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
