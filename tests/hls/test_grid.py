import pytest
from virtualizarr_processor.hls.grid import TileGrid, level_coords, tile_grid

# Measured from the newest T13SDA granule in the spike; see
# docs/superpowers/spikes/2026-11-11-time-stack-findings.md (the "Grid" section).
SPIKE_EPSG = 32613
SPIKE_X0 = 399960.0
SPIKE_Y0 = 4100040.0

# Test-local example grid; deliberately not T13SDA (its top edge is 4300020).
GRID = TileGrid(
    "T13SDA",
    32613,
    (3660, 3660),
    (30.0, 0.0, 399960.0, 0.0, -30.0, 4300020.0, 0.0, 0.0, 1.0),
)


def test_t13sda_grid_is_embedded():
    g = tile_grid("T13SDA")
    assert g.mgrs_id == "T13SDA" and g.shape == (3660, 3660)
    assert g.transform[0] == 30.0 and g.transform[4] == -30.0
    assert (
        g.epsg == SPIKE_EPSG
        and g.transform[2] == SPIKE_X0
        and g.transform[5] == SPIKE_Y0
    )


def test_unknown_tile_has_no_grid():
    with pytest.raises(ValueError, match="T99XXX"):
        tile_grid("T99XXX")


def test_level0_coords_are_pixel_centres():
    y, x = level_coords(GRID, 3660, 3660)
    assert x[0] == 399975.0 and x[-1] == 399960.0 + 3660 * 30 - 15
    assert y[0] == 4300005.0 and y[-1] == 4300020.0 - 3660 * 30 + 15
    assert len(x) == len(y) == 3660


def test_overview_coords_cover_cell_with_rounded_size():
    y, x = level_coords(GRID, 458, 458)
    res = 3660 * 30 / 458
    assert x[0] == pytest.approx(399960.0 + res / 2)
    assert x[-1] + res / 2 == pytest.approx(399960.0 + 3660 * 30)
