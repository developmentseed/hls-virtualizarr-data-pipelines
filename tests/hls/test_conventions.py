from virtualizarr_processor.hls.conventions import band_group_attrs
from virtualizarr_processor.hls.grid import TileGrid

# Test-local example grid; same literal as in test_grid.py.
GRID = TileGrid(
    "T13SDA",
    32613,
    (3660, 3660),
    (30.0, 0.0, 399960.0, 0.0, -30.0, 4300020.0, 0.0, 0.0, 1.0),
)


def test_band_group_attrs():
    attrs = band_group_attrs(GRID, levels=3, band="B04")
    assert attrs["proj:code"] == "EPSG:32613"
    assert attrs["spatial:shape"] == [3660, 3660]
    assert attrs["spatial:bbox"] == [
        399960.0,
        4300020.0 - 109800.0,
        399960.0 + 109800.0,
        4300020.0,
    ]
    layout = attrs["multiscales"]["layout"]
    assert [e["asset"] for e in layout] == [
        "multiscales/0/B04",
        "multiscales/1/B04",
        "multiscales/2/B04",
    ]
    assert layout[1]["derived_from"] == "multiscales/0/B04"
