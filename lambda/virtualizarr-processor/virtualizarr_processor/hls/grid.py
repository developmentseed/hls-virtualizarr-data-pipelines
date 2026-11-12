"""MGRS tile grids for HLS granules and the coordinates of their resolution levels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TileGrid:
    """The full-resolution grid of one MGRS tile.

    ``transform`` is the nine-element row-major affine, as STAC ``proj:transform``.
    """

    mgrs_id: str
    epsg: int
    shape: tuple[int, int]
    transform: tuple[float, ...]


# Values recorded in docs/superpowers/spikes/2026-11-11-time-stack-findings.md
# (newest T13SDA granule's STAC ``proj:*``).
TILE_GRIDS: dict[str, TileGrid] = {
    "T13SDA": TileGrid(
        "T13SDA",
        32613,
        (3660, 3660),
        (30.0, 0.0, 399960.0, 0.0, -30.0, 4100040.0, 0.0, 0.0, 1.0),
    ),
}


def tile_grid(tile: str) -> TileGrid:
    """Return the grid for an MGRS tile.

    Raises ``ValueError`` if the tile has no embedded grid.
    """
    try:
        return TILE_GRIDS[tile]
    except KeyError:
        raise ValueError(f"no grid for {tile}") from None


def level_coords(grid: TileGrid, ny: int, nx: int) -> tuple[np.ndarray, np.ndarray]:
    """Pixel-centre ``y`` and ``x`` for a ``ny`` x ``nx`` level covering the whole tile.

    Resolution along each axis is the tile's extent divided by the level's size,
    so a level whose size does not divide the tile evenly still spans it exactly.
    """
    rows, cols = grid.shape
    x0, y0 = grid.transform[2], grid.transform[5]
    res_x = cols * grid.transform[0] / nx  # positive: east
    res_y = rows * grid.transform[4] / ny  # negative: north-up
    x = x0 + (np.arange(nx, dtype=np.float64) + 0.5) * res_x
    y = y0 + (np.arange(ny, dtype=np.float64) + 0.5) * res_y
    return y, x
