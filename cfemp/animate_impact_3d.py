"""把三维板碰撞计算结果做成动画。"""

from __future__ import annotations

import argparse
import tempfile
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np

from .plate_impact_3d import (
    PlateImpact3DConfig,
    TaichiCFEMPPlateImpact3D,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402


@dataclass
class AnimationFrame:
    time: float
    fem_slice: np.ndarray
    mpm_slice: np.ndarray
    fem_mean_vx: float
    mpm_mean_vx: float
    gap: float
    contact: bool


def _sample(frames: int) -> tuple[PlateImpact3DConfig, list[AnimationFrame]]:
    config = PlateImpact3DConfig()
    solver = TaichiCFEMPPlateImpact3D(config)
    targets = np.linspace(0.0, config.end_time, frames)
    sampled: list[AnimationFrame] = []

    def append_frame(contact: bool) -> None:
        fem = solver.fem_x.to_numpy().reshape(
            solver.fem_nx + 1,
            solver.fem_ny + 1,
            solver.fem_nz + 1,
            3,
        )
        mpm = solver.mpm_x.to_numpy().reshape(
            solver.mpm_nx,
            solver.mpm_ny,
            solver.mpm_nz,
            3,
        )
        sampled.append(
            AnimationFrame(
                time=solver.time,
                fem_slice=fem[:, :, solver.fem_nz // 2, :].copy(),
                mpm_slice=mpm[:, :, solver.mpm_nz // 2, :].copy(),
                fem_mean_vx=float(
                    np.mean(solver.fem_v.to_numpy()[:, 0])
                ),
                mpm_mean_vx=float(
                    np.mean(solver.mpm_v.to_numpy()[:, 0])
                ),
                gap=solver._gap(),
                contact=contact,
            )
        )

    append_frame(False)
    target = 1
    contact = False
    while solver.time < config.end_time - 0.5 * config.dt:
        step = solver.step()
        contact = step.active
        while target < len(targets) and (
            solver.time >= targets[target] - 0.5 * config.dt
        ):
            append_frame(contact)
            target += 1
    return config, sampled


def _phase(frame: AnimationFrame, config: PlateImpact3DConfig) -> str:
    one_way = config.length / config.wave_speed
    if frame.time < 0.5 * config.dt:
        return "initial contact: +100 m/s  |  -100 m/s"
    if frame.contact and frame.time < one_way:
        return "compression waves move toward both free ends"
    if frame.contact:
        return "release waves return from the free ends"
    return "contact released: the plates separate"


def _draw_deformation(
    ax: plt.Axes,
    frame: AnimationFrame,
    reference: AnimationFrame,
    *,
    amplification: float,
) -> None:
    fem = reference.fem_slice + amplification * (
        frame.fem_slice - reference.fem_slice
    )
    mpm = reference.mpm_slice + amplification * (
        frame.mpm_slice - reference.mpm_slice
    )

    for j in range(fem.shape[1]):
        ax.plot(
            fem[:, j, 0] * 1.0e3,
            fem[:, j, 1] * 1.0e3,
            color="#183A52",
            linewidth=0.55,
            alpha=0.78,
        )
    for i in range(0, fem.shape[0], 3):
        ax.plot(
            fem[i, :, 0] * 1.0e3,
            fem[i, :, 1] * 1.0e3,
            color="#183A52",
            linewidth=0.5,
            alpha=0.68,
        )

    ax.scatter(
        mpm[:, :, 0].ravel() * 1.0e3,
        mpm[:, :, 1].ravel() * 1.0e3,
        s=2.6,
        color="#B14343",
        alpha=0.66,
        linewidths=0,
    )
    ax.scatter(
        fem[-1, :, 0] * 1.0e3,
        fem[-1, :, 1] * 1.0e3,
        s=12,
        marker="s",
        color="#183A52",
        zorder=4,
        label="FEM contact nodes",
    )
    ax.scatter(
        mpm[0, :, 0] * 1.0e3,
        mpm[0, :, 1] * 1.0e3,
        s=8,
        color="#B14343",
        zorder=4,
        label="MPM boundary particles",
    )
    ax.plot(
        reference.fem_slice[[0, -1], 0, 0] * 1.0e3,
        (-0.18, -0.18),
        color="#6B7280",
        linestyle=":",
        linewidth=0.8,
    )
    ax.plot(
        reference.mpm_slice[[0, -1], 0, 0] * 1.0e3,
        (-0.18, -0.18),
        color="#6B7280",
        linestyle=":",
        linewidth=0.8,
    )
    ax.axvline(0.0, color="#222222", linewidth=0.55, alpha=0.5)
    limit = 27.0 if amplification > 1.0 else 22.5
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-0.32, 3.32)
    ax.set_yticks((0.0, 1.5, 3.0))
    ax.set_ylabel("y (mm)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.12)


def _render_deformation_frame(
    frame: AnimationFrame,
    reference: AnimationFrame,
    config: PlateImpact3DConfig,
    target: Path,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 4.5), dpi=88)
    _draw_deformation(axes[0], frame, reference, amplification=1.0)
    _draw_deformation(axes[1], frame, reference, amplification=4.0)
    axes[0].set_title("true displacement", loc="left", fontsize=10)
    axes[1].set_title("displacement x4", loc="left", fontsize=10)
    axes[1].set_xlabel("x (mm)")
    axes[0].text(
        1.0,
        1.05,
        f"t = {frame.time * 1.0e6:5.2f} us   "
        f"gap = {frame.gap * 1.0e6:+6.2f} um",
        transform=axes[0].transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
    )
    fem_length = np.ptp(frame.fem_slice[:, :, 0]) * 1.0e3
    mpm_length = (
        np.ptp(frame.mpm_slice[:, :, 0]) + config.particle_spacing
    ) * 1.0e3
    axes[1].text(
        1.0,
        1.05,
        f"FEM: L={fem_length:5.3f} mm, "
        f"mean vx={frame.fem_mean_vx:+5.1f} m/s   |   "
        f"MPM: L={mpm_length:5.3f} mm, "
        f"mean vx={frame.mpm_mean_vx:+5.1f} m/s",
        transform=axes[1].transAxes,
        ha="right",
        va="bottom",
        fontsize=8.5,
    )
    axes[1].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.27),
        ncol=2,
        frameon=False,
    )
    fig.suptitle(
        f"Taichi 3D CFEMP deformation  |  {_phase(frame, config)}",
        y=0.99,
    )
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.15, top=0.88, hspace=0.46)
    fig.savefig(target, facecolor="white")
    plt.close(fig)


def _encode(
    pngs: list[Path],
    gif_path: Path,
    fps: int,
) -> None:
    images = [
        Image.open(path).convert(
            "P",
            palette=Image.ADAPTIVE,
            colors=32,
        )
        for path in pngs
    ]
    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=int(round(1000 / fps)),
        loop=0,
        disposal=2,
        optimize=True,
    )
    for image in images:
        image.close()


def render_animation(
    output_dir: str | Path,
    *,
    frames: int = 25,
    fps: int = 5,
) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config, sampled = _sample(frames)
    deformation_gif = output / "impact_deformation.gif"

    with tempfile.TemporaryDirectory(prefix="cfemp-impact-") as temp_name:
        deformation_temp = Path(temp_name) / "deformation"
        deformation_temp.mkdir()
        deformation_pngs: list[Path] = []
        for index, frame in enumerate(sampled):
            deformation_target = (
                deformation_temp / f"frame-{index:04d}.png"
            )
            _render_deformation_frame(
                frame,
                sampled[0],
                config,
                deformation_target,
            )
            deformation_pngs.append(deformation_target)

        _encode(deformation_pngs, deformation_gif, fps)
    return deformation_gif


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="results/symmetric_plate_impact_3d",
    )
    parser.add_argument("--frames", type=int, default=25)
    parser.add_argument("--fps", type=int, default=5)
    args = parser.parse_args()
    print(render_animation(args.output, frames=args.frames, fps=args.fps))


if __name__ == "__main__":
    main()
