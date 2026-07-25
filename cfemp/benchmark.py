"""运行论文对称板碰撞基准并生成可复现图表。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib
import numpy as np

from .plate_impact import (
    CFEMPPlateImpact1D,
    PlateImpactConfig,
    analytical_contact_stress,
    analytical_separation_time,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _configure_plotting() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 180,
            "font.size": 10,
            "axes.grid": True,
            "grid.alpha": 0.25,
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
    wave_distance = min(config.wave_speed * time, config.length)
    in_wave = np.abs(x) <= wave_distance
    if time <= analytical_separation_time(config):
        stress[in_wave] = analytical_contact_stress(config)
    return stress


def run_benchmark(output_dir: str | os.PathLike[str]) -> dict[str, float]:
    _configure_plotting()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = PlateImpactConfig()
    solver = CFEMPPlateImpact1D(config)
    history = solver.run()
    arrays = history.as_arrays()

    snapshot_time = 3.0e-6
    x, stress = history.snapshots[snapshot_time]
    analytical = analytical_profile(x, snapshot_time, config)

    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    ax.plot(x * 1e3, analytical / 1e6, "k--", lw=1.8, label="1D analytical")
    ax.plot(x * 1e3, stress / 1e6, color="#1768AC", lw=1.4, label="CFEMP")
    ax.axvline(0.0, color="#D1495B", lw=0.9, alpha=0.8)
    ax.set(
        xlabel="Position x (mm)",
        ylabel="Axial stress (MPa)",
        title="Symmetric plate impact: stress profile at 3.0 μs",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "stress_profile_3us.png")
    plt.close(fig)

    initial_energy = arrays["total_energy"][0]
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    time_us = arrays["time"] * 1e6
    ax.plot(time_us, arrays["kinetic"] / initial_energy, label="Kinetic")
    ax.plot(time_us, arrays["strain"] / initial_energy, label="Strain")
    ax.plot(time_us, arrays["total_energy"] / initial_energy, label="Total")
    ax.set(
        xlabel="Time (μs)",
        ylabel="Normalized energy",
        title="CFEMP energy evolution",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "energy_evolution.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    ax.plot(time_us, arrays["gap"] * 1e6, color="#2A9D8F")
    if history.separation_time is not None:
        ax.axvline(
            history.separation_time * 1e6,
            color="#D1495B",
            ls="--",
            label=f"Numerical separation: {history.separation_time*1e6:.2f} μs",
        )
    ax.axvline(
        analytical_separation_time(config) * 1e6,
        color="black",
        ls=":",
        label=f"Analytical: {analytical_separation_time(config)*1e6:.2f} μs",
    )
    ax.set(
        xlabel="Time (μs)",
        ylabel="Interface gap (μm)",
        title="Contact and separation history",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "contact_separation.png")
    plt.close(fig)

    plateau = stress[np.abs(x) <= 0.5 * config.wave_speed * snapshot_time]
    numerical_stress = float(np.median(plateau))
    exact_stress = analytical_contact_stress(config)
    separation = history.separation_time or float("nan")
    momentum_scale = (
        config.density
        * config.area
        * config.length
        * config.impact_speed
    )
    metrics = {
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
            np.max(np.abs(arrays["momentum"] - arrays["momentum"][0]))
            / momentum_scale
        ),
        "steps": int(len(arrays["time"]) - 1),
    }
    (output / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(output / "history.npz", **arrays)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="复现 Lian et al. (2011) CFEMP 对称弹性板碰撞"
    )
    parser.add_argument(
        "--output",
        default="results/symmetric_plate_impact",
        help="图片与指标输出目录",
    )
    args = parser.parse_args()
    metrics = run_benchmark(args.output)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
