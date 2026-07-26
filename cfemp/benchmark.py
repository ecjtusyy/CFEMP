"""生成二维对称板碰撞基准结果。"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import replace
from pathlib import Path

import matplotlib
import numpy as np

from .plate_impact import (
    CFEMPPlateImpact2D,
    PlateImpactConfig,
    SimulationHistory,
    analytical_contact_stress,
    analytical_separation_time,
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


def analytical_profile(
    x: np.ndarray,
    time: float,
    config: PlateImpactConfig,
) -> np.ndarray:
    stress = np.zeros_like(x)
    front = min(config.wave_speed * time, config.length)
    if time <= analytical_separation_time(config):
        stress[np.abs(x) <= front] = analytical_contact_stress(config)
    return stress


def _stress_metric(
    snapshot: dict[str, np.ndarray],
    config: PlateImpactConfig,
    time: float,
) -> float:
    x = snapshot["profile_x"]
    stress = snapshot["profile_stress"]
    plateau = np.abs(x) <= 0.5 * config.wave_speed * time
    return float(np.median(stress[plateau]))


def _draw_mesh(solver: CFEMPPlateImpact2D, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.0, 2.6))
    for conn in solver.fem.elements:
        points = solver.fem.x0[conn[[0, 1, 2, 3, 0]]]
        ax.plot(points[:, 0] * 1e3, points[:, 1] * 1e3, "k-", lw=0.25)
    ax.scatter(
        solver.mpm.x[:, 0] * 1e3,
        solver.mpm.x[:, 1] * 1e3,
        s=2.0,
        c="#C44536",
        linewidths=0,
    )
    ax.axvline(0.0, color="#2D6A8A", lw=1.0)
    ax.set(
        xlabel="x (mm)",
        ylabel="y (mm)",
        title="Q4 FEM / MPM discretization",
        xlim=(-solver.config.length * 1e3, solver.config.length * 1e3),
        ylim=(-0.15, solver.config.width * 1e3 + 0.15),
    )
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(output / "discretization_2d.png")
    plt.close(fig)


def _draw_stress_field(
    snapshot: dict[str, np.ndarray],
    config: PlateImpactConfig,
    output: Path,
) -> None:
    fem_position = snapshot["fem_position"]
    fem_stress = snapshot["fem_stress"][:, 0] / 1e6
    mpm_position = snapshot["mpm_position"]
    mpm_stress = snapshot["mpm_stress"][:, 0] / 1e6
    scale = abs(analytical_contact_stress(config)) / 1e6
    norm = Normalize(vmin=-1.15 * scale, vmax=0.1 * scale)

    fig, ax = plt.subplots(figsize=(10.0, 2.8))
    fem_artist = ax.scatter(
        fem_position[:, 0] * 1e3,
        fem_position[:, 1] * 1e3,
        c=fem_stress,
        norm=norm,
        cmap="coolwarm",
        marker="s",
        s=28,
        linewidths=0,
    )
    ax.scatter(
        mpm_position[:, 0] * 1e3,
        mpm_position[:, 1] * 1e3,
        c=mpm_stress,
        norm=norm,
        cmap="coolwarm",
        marker="s",
        s=5,
        linewidths=0,
    )
    ax.axvline(0.0, color="black", lw=0.6, alpha=0.7)
    ax.set(
        xlabel="x (mm)",
        ylabel="y (mm)",
        title=r"$\sigma_{xx}$ at 3.0 $\mu$s",
        ylim=(-0.15, config.width * 1e3 + 0.15),
    )
    ax.set_aspect("equal")
    colorbar = fig.colorbar(fem_artist, ax=ax, pad=0.015)
    colorbar.set_label(r"$\sigma_{xx}$ (MPa)")
    fig.tight_layout()
    fig.savefig(output / "stress_field_3us.png")
    plt.close(fig)


def _draw_histories(
    history: SimulationHistory,
    config: PlateImpactConfig,
    output: Path,
) -> None:
    arrays = history.as_arrays()
    snapshot = history.snapshots[3.0e-6]
    x = snapshot["profile_x"]
    stress = snapshot["profile_stress"]

    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    ax.plot(
        x * 1e3,
        analytical_profile(x, 3.0e-6, config) / 1e6,
        "k--",
        lw=1.5,
        label="analytical",
    )
    ax.plot(x * 1e3, stress / 1e6, color="#1D6A96", lw=1.2, label="2D CFEMP")
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

    initial_energy = arrays["total_energy"][0]
    time_us = arrays["time"] * 1e6
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
            label=f"2D CFEMP {history.separation_time*1e6:.2f}",
        )
    exact = analytical_separation_time(config) * 1e6
    ax.axvline(exact, color="black", ls=":", label=f"analytical {exact:.2f}")
    ax.set(xlabel=r"time ($\mu$s)", ylabel=r"gap ($\mu$m)")
    ax.legend(title=r"separation time ($\mu$s)")
    fig.tight_layout()
    fig.savefig(output / "contact_separation.png")
    plt.close(fig)


def run_time_refinement(
    output: Path,
    base_config: PlateImpactConfig,
    base_history: SimulationHistory,
) -> float:
    end_time = 3.0e-6
    time_steps = np.array([4.0e-8, 2.0e-8, 1.0e-8, 5.0e-9])
    profiles: dict[float, tuple[np.ndarray, np.ndarray]] = {}

    base = base_history.snapshots[end_time]
    profiles[base_config.dt] = (base["profile_x"], base["profile_stress"])
    for dt in time_steps:
        if np.isclose(dt, base_config.dt, rtol=0.0, atol=1.0e-20):
            continue
        config = replace(
            base_config,
            dt=float(dt),
            end_time=end_time,
            snapshot_times=(end_time,),
        )
        history = CFEMPPlateImpact2D(config).run()
        snapshot = history.snapshots[end_time]
        profiles[float(dt)] = (
            snapshot["profile_x"],
            snapshot["profile_stress"],
        )

    reference_dt = float(time_steps[-1])
    reference_x, reference_stress = profiles[reference_dt]
    rows: list[tuple[float, int, float]] = []
    for dt in time_steps[:-1]:
        x, stress = profiles[float(dt)]
        reference = np.interp(x, reference_x, reference_stress)
        error = float(
            np.linalg.norm(stress - reference) / np.linalg.norm(reference)
        )
        rows.append((float(dt), int(round(end_time / dt)), error))

    slope = float(
        np.polyfit(
            np.log([row[0] for row in rows]),
            np.log([row[2] for row in rows]),
            1,
        )[0]
    )
    with (output / "time_refinement.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("dt_s", "steps_to_3us", "relative_l2_error"))
        writer.writerows(rows)
        writer.writerow((reference_dt, int(end_time / reference_dt), 0.0))

    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    dt_values = np.array([row[0] for row in rows])
    errors = np.array([row[2] for row in rows])
    ax.loglog(dt_values, errors, "o-", color="#1D6A96")
    guide = errors[-1] * (dt_values / dt_values[-1]) ** slope
    ax.loglog(dt_values, guide, "k--", label=f"slope {slope:.2f}")
    ax.set(xlabel="time step (s)", ylabel=r"relative $L_2$ error")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "time_refinement.png")
    plt.close(fig)
    return slope


def run_aspect_ratio_study(
    output: Path,
    base_config: PlateImpactConfig,
    base_history: SimulationHistory,
) -> dict[str, float]:
    end_time = 3.0e-6
    ratios = (0.5, 1.0, 2.0, 3.0)
    histories = {1.0: base_history}
    for ratio in ratios:
        if ratio == 1.0:
            continue
        config = replace(
            base_config,
            fem_element_size=ratio * base_config.mpm_cell_size,
            end_time=end_time,
            snapshot_times=(end_time,),
        )
        histories[ratio] = CFEMPPlateImpact2D(config).run()

    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    rows: list[tuple[float, float, int, float, float, float, float]] = []
    metrics: dict[str, float] = {}
    for ratio in ratios:
        config = replace(
            base_config,
            fem_element_size=ratio * base_config.mpm_cell_size,
        )
        snapshot = histories[ratio].snapshots[end_time]
        x = snapshot["profile_x"]
        stress = snapshot["profile_stress"]
        exact = analytical_profile(x, end_time, config)
        relative_error = float(
            np.linalg.norm(stress - exact) / np.linalg.norm(exact)
        )
        peak_ratio = float(np.min(stress) / analytical_contact_stress(config))
        arrays = histories[ratio].as_arrays()
        until_snapshot = arrays["time"] <= end_time + 0.5 * config.dt
        momentum_scale = (
            config.density
            * config.area
            * config.length
            * config.impact_speed
        )
        momentum_error = np.hypot(
            arrays["momentum_x"] - arrays["momentum_x"][0],
            arrays["momentum_y"] - arrays["momentum_y"][0],
        )
        normalized_momentum_error = float(
            np.max(momentum_error[until_snapshot]) / momentum_scale
        )
        impulse_balance = float(
            np.max(arrays["contact_balance"][until_snapshot])
        )
        elements = int(
            round(config.length / config.fem_element_size)
            * round(config.width / config.fem_element_size)
        )
        rows.append(
            (
                ratio,
                config.fem_element_size * 1e3,
                elements,
                relative_error,
                peak_ratio,
                normalized_momentum_error,
                impulse_balance,
            )
        )
        metrics[f"R{ratio:g}_relative_l2_error"] = relative_error
        metrics[f"R{ratio:g}_compressive_peak_ratio"] = peak_ratio
        metrics[
            f"R{ratio:g}_normalized_momentum_error"
        ] = normalized_momentum_error
        metrics[
            f"R{ratio:g}_impulse_balance_kg_m_s"
        ] = impulse_balance
        ax.plot(
            x * 1e3,
            stress / 1e6,
            lw=1.0,
            label=f"R={ratio:g}",
        )

    reference_x = histories[1.0].snapshots[end_time]["profile_x"]
    ax.plot(
        reference_x * 1e3,
        analytical_profile(reference_x, end_time, base_config) / 1e6,
        "k--",
        lw=1.4,
        label="analytical",
    )
    ax.set(
        xlabel="x (mm)",
        ylabel=r"$\sigma_{xx}$ (MPa)",
        title=r"mesh ratio study at 3.0 $\mu$s",
    )
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(output / "mesh_ratio_study.png")
    plt.close(fig)

    with (output / "mesh_ratio_study.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            (
                "R",
                "fem_element_size_mm",
                "fem_q4_elements",
                "relative_l2_error",
                "compressive_peak_ratio",
                "normalized_momentum_error",
                "impulse_balance_kg_m_s",
            )
        )
        writer.writerows(rows)
    return metrics


def run_benchmark(output_dir: str | os.PathLike[str]) -> dict[str, float]:
    _plot_style()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    config = PlateImpactConfig()
    solver = CFEMPPlateImpact2D(config)
    _draw_mesh(solver, output)
    history = solver.run()
    arrays = history.as_arrays()
    snapshot = history.snapshots[3.0e-6]
    _draw_stress_field(snapshot, config, output)
    _draw_histories(history, config, output)
    refinement_order = run_time_refinement(
        output, config, history
    )
    ratio_metrics = run_aspect_ratio_study(output, config, history)

    numerical_stress = _stress_metric(snapshot, config, 3.0e-6)
    exact_stress = analytical_contact_stress(config)
    separation = history.separation_time or float("nan")
    initial_energy = arrays["total_energy"][0]
    momentum_scale = (
        config.density
        * config.area
        * config.length
        * config.impact_speed
    )
    momentum = np.hypot(
        arrays["momentum_x"] - arrays["momentum_x"][0],
        arrays["momentum_y"] - arrays["momentum_y"][0],
    )

    fem_sxx = snapshot["fem_stress"][:, 0].reshape(
        solver.fem.nx, solver.fem.ny
    )
    mpm_sxx = snapshot["mpm_stress"][:, 0].reshape(
        solver.mpm.npx, solver.mpm.npy
    )
    transverse_stress_spread = max(
        float(np.max(np.ptp(fem_sxx, axis=1))),
        float(np.max(np.ptp(mpm_sxx, axis=1))),
    ) / abs(exact_stress)

    metrics = {
        "dimensions": 2,
        "fem_nodes": int(len(solver.fem.x)),
        "fem_q4_elements": int(len(solver.fem.elements)),
        "mpm_particles": int(len(solver.mpm.x)),
        "analytical_contact_stress_mpa": exact_stress / 1e6,
        "numerical_contact_stress_mpa": numerical_stress / 1e6,
        "stress_relative_error": abs(numerical_stress - exact_stress)
        / abs(exact_stress),
        "analytical_separation_time_us": analytical_separation_time(config)
        * 1e6,
        "numerical_separation_time_us": separation * 1e6,
        "separation_relative_error": abs(
            separation - analytical_separation_time(config)
        )
        / analytical_separation_time(config),
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
        "time_refinement_order": refinement_order,
        "steps": int(len(arrays["time"]) - 1),
        **ratio_metrics,
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
        default="results/symmetric_plate_impact",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            run_benchmark(args.output),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
