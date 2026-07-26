"""二维线性背景网格的形函数。"""

from __future__ import annotations

import numpy as np


def hat_weights_and_grads(
    xp: np.ndarray,
    origin: np.ndarray,
    dx: float,
    nx: int,
    ny: int,
) -> tuple[list[tuple[int, int]], np.ndarray, np.ndarray]:
    rx = (float(xp[0]) - float(origin[0])) / dx
    ry = (float(xp[1]) - float(origin[1])) / dx
    if not np.isfinite(rx) or not np.isfinite(ry):
        raise FloatingPointError("粒子坐标出现 NaN/Inf")

    i = int(np.floor(rx))
    j = int(np.floor(ry))
    i = int(np.clip(i, 0, nx - 2))
    j = int(np.clip(j, 0, ny - 2))
    sx = float(np.clip(rx - i, 0.0, 1.0))
    sy = float(np.clip(ry - j, 0.0, 1.0))

    nodes = [(i, j), (i + 1, j), (i, j + 1), (i + 1, j + 1)]
    weights = np.array(
        [
            (1.0 - sx) * (1.0 - sy),
            sx * (1.0 - sy),
            (1.0 - sx) * sy,
            sx * sy,
        ],
        dtype=float,
    )
    grads = np.array(
        [
            [-(1.0 - sy), -(1.0 - sx)],
            [1.0 - sy, -sx],
            [-sy, 1.0 - sx],
            [sy, sx],
        ],
        dtype=float,
    ) / dx
    return nodes, weights, grads


def node_id(i: int, j: int, nx: int) -> int:
    return j * nx + i
