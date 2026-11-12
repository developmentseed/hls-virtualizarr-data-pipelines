from datetime import UTC, datetime

import numpy as np
import pytest
import xarray as xr
import zarr
from fakes import FAKE_GRID, local_locator
from virtualizarr_processor.hls.assets import asset_key
from virtualizarr_processor.hls.cube import (
    declare_cube,
    granule_levels,
    level_group,
    level_groups,
    open_granule,
    store_times,
    write_granule,
)
from virtualizarr_processor.hls.granule import parse_granule_id
from virtualizarr_processor.hls.pyramid import GridMismatch

TIMES = [
    datetime(2013, 6, 1, 17, tzinfo=UTC),
    datetime(2025, 6, 1, 17, tzinfo=UTC),
    datetime(2025, 6, 17, 17, tzinfo=UTC),
]
# granule IDs whose parsed times are exactly TIMES
IDS = [
    "HLS.L30.T13SDA.2013152T170000.v2.0",
    "HLS.L30.T13SDA.2025152T170000.v2.0",
    "HLS.L30.T13SDA.2025168T170000.v2.0",
]
REF = "HLS.L30.T13SDA.2025100T170000.v2.0"
BANDS = ("B04", "Fmask")


def _open(fake_assets, gid, bands=BANDS):
    locator = local_locator(fake_assets.root)
    return open_granule(
        parse_granule_id(gid), bands, fake_assets.opener, None, locator.read_url
    )


def _write(s, fake_assets, gid, mode):
    locator = local_locator(fake_assets.root)
    write_granule(
        s.store,
        "T13SDA",
        parse_granule_id(gid),
        _open(fake_assets, gid),
        mode=mode,
        reference_url_for=locator.reference_url,
    )


def _level0(s):
    return xr.open_zarr(s.store, group="T13SDA/B04/multiscales/0", consolidated=False)


@pytest.fixture
def reference_pyramids(fake_assets):
    for band in BANDS:
        fake_assets.write(REF, band, 9)
    return _open(fake_assets, REF)


def test_ids_match_times():
    assert [parse_granule_id(g).time for g in IDS] == TIMES


def test_level_group():
    assert level_group("T13SDA", "B04", 2) == "T13SDA/B04/multiscales/2"


def test_declare_cube_shapes_coords_attrs(hls_repo, reference_pyramids):
    s = hls_repo.writable_session("main")
    declare_cube(s.store, FAKE_GRID, TIMES, reference_pyramids)
    ds0 = xr.open_zarr(s.store, group="T13SDA/B04/multiscales/0", consolidated=False)
    ds1 = xr.open_zarr(s.store, group="T13SDA/B04/multiscales/1", consolidated=False)
    assert ds0["B04"].dims == ("time", "y", "x") and ds0["B04"].shape == (3, 8, 8)
    assert ds1["B04"].shape == (3, 4, 4)
    assert list(ds0.time.values) == [
        np.datetime64(t.replace(tzinfo=None)) for t in TIMES
    ]
    assert ds0["B04"].encoding["chunks"] == (1, 4, 4)
    assert zarr.open_group(s.store, path="T13SDA/B04").attrs["proj:code"] == (
        "EPSG:32613"
    )


def test_declare_cube_details(hls_repo, reference_pyramids):
    s = hls_repo.writable_session("main")
    declare_cube(s.store, FAKE_GRID, TIMES, reference_pyramids)
    assert level_groups(s.store, "T13SDA") == [
        "T13SDA/B04/multiscales/0",
        "T13SDA/B04/multiscales/1",
        "T13SDA/Fmask/multiscales/0",
        "T13SDA/Fmask/multiscales/1",
    ]
    time = zarr.open_array(s.store, path="T13SDA/B04/multiscales/0/time")
    assert time.dtype == np.int64
    assert time.attrs["calendar"] == "proleptic_gregorian"
    assert time.attrs["units"].startswith("seconds since 1970-01-01")
    ds1 = xr.open_zarr(s.store, group="T13SDA/Fmask/multiscales/1", consolidated=False)
    assert ds1.x.values[0] == 399960.0 + 30.0  # 8 px of 30 m over 4 cells -> 60 m
    assert ds1.y.values[0] == 4100040.0 - 30.0
    layout = zarr.open_group(s.store, path="T13SDA/Fmask").attrs["multiscales"]
    assert len(layout["layout"]) == 2
    assert layout["layout"][0]["asset"] == "multiscales/0/Fmask"


def test_store_times(hls_repo, reference_pyramids):
    s = hls_repo.writable_session("main")
    assert store_times(s.store, "T13SDA") is None
    declare_cube(s.store, FAKE_GRID, TIMES, reference_pyramids)
    assert list(store_times(s.store, "T13SDA")) == [
        np.datetime64(t.replace(tzinfo=None)) for t in TIMES
    ]


def test_granule_levels(fake_assets):
    fake_assets.write(IDS[0], "B04", 1)
    granule = parse_granule_id(IDS[0])
    pyramid = _open(fake_assets, IDS[0], ("B04",))["B04"]
    pyramid[0].attrs["LANDSAT_SCENE_ID"] = "x"
    pyramid[0]["band_data"].attrs["scale"] = 1
    levels = granule_levels(granule, "B04", pyramid)
    assert list(levels) == [0, 1]
    ds = levels[0]
    assert sorted(ds.variables) == ["B04", "time"]
    assert ds["B04"].dims == ("time", "y", "x")
    assert ds.attrs == {} and ds["B04"].attrs == {}
    paths = ds["B04"].data.manifest._paths
    want = f"vcc://lpdaac/{asset_key(granule, 'B04')}"
    assert {str(p) for p in np.asarray(paths).ravel()} == {want}


def test_region_writes_land_at_their_time(hls_repo, fake_assets, reference_pyramids):
    s = hls_repo.writable_session("main")
    declare_cube(s.store, FAKE_GRID, TIMES, reference_pyramids)
    for band in BANDS:
        fake_assets.write(IDS[2], band, 3)
        fake_assets.write(IDS[0], band, 1)
    _write(s, fake_assets, IDS[2], "region")
    _write(s, fake_assets, IDS[0], "region")
    ds0 = _level0(s)
    assert ds0["B04"].isel(time=0).values.tolist() == [[1] * 8] * 8
    assert ds0["B04"].isel(time=2).values.max() == 3
    fill = ds0["B04"].encoding.get(
        "_FillValue",
        zarr.open_array(s.store, path="T13SDA/B04/multiscales/0/B04").fill_value,
    )
    assert (ds0["B04"].isel(time=1).values == fill).all() or np.isnan(
        ds0["B04"].isel(time=1).values
    ).all()  # untouched step reads fill
    level1 = xr.open_zarr(
        s.store, group="T13SDA/Fmask/multiscales/1", consolidated=False
    )
    assert level1["Fmask"].isel(time=2).values.tolist() == [[3] * 4] * 4


def test_append_extends_every_group(hls_repo, fake_assets, reference_pyramids):
    s = hls_repo.writable_session("main")
    declare_cube(s.store, FAKE_GRID, TIMES[:2], reference_pyramids)
    for band in BANDS:
        fake_assets.write(IDS[2], band, 3)
    _write(s, fake_assets, IDS[2], "append")
    for g in level_groups(s.store, "T13SDA"):
        assert xr.open_zarr(s.store, group=g, consolidated=False).sizes["time"] == 3
    assert store_times(s.store, "T13SDA")[-1] == np.datetime64(
        TIMES[2].replace(tzinfo=None)
    )
    assert _level0(s)["B04"].isel(time=2).values.max() == 3


def test_write_granule_mismatch_writes_nothing(
    hls_repo, fake_assets, reference_pyramids
):
    s = hls_repo.writable_session("main")
    declare_cube(s.store, FAKE_GRID, TIMES, reference_pyramids)
    s.commit("declare")
    s = hls_repo.writable_session("main")
    fake_assets.write(IDS[2], "B04", 3)
    fake_assets.write(IDS[2], "Fmask", 3, chunks=(2, 2))
    granule = parse_granule_id(IDS[2])
    pyramids = _open(fake_assets, IDS[2])
    with pytest.raises(GridMismatch):
        write_granule(s.store, "T13SDA", granule, pyramids, mode="region")
    assert not s.has_uncommitted_changes  # after declare was committed


def test_write_granule_missing_band_or_level_is_mismatch(
    hls_repo, fake_assets, reference_pyramids
):
    s = hls_repo.writable_session("main")
    declare_cube(s.store, FAKE_GRID, TIMES, reference_pyramids)
    s.commit("declare")
    s = hls_repo.writable_session("main")
    fake_assets.write(IDS[2], "B04", 3)
    fake_assets.write(IDS[2], "Fmask", 3, levels=((8, 8),))
    granule = parse_granule_id(IDS[2])
    with pytest.raises(GridMismatch, match="level"):
        write_granule(
            s.store, "T13SDA", granule, _open(fake_assets, IDS[2]), mode="region"
        )
    with pytest.raises(GridMismatch, match="band"):
        write_granule(
            s.store,
            "T13SDA",
            granule,
            _open(fake_assets, IDS[2], ("B04",)),
            mode="region",
        )
    assert not s.has_uncommitted_changes


def test_open_granule_opens_every_band_or_raises(fake_assets):
    for band in BANDS:
        fake_assets.write(IDS[0], band, 1)
    fake_assets.missing(IDS[0], "Fmask")
    with pytest.raises(FileNotFoundError):
        _open(fake_assets, IDS[0])
    assert len(fake_assets.opened) == 2
    assert fake_assets.opened[0] == local_locator(fake_assets.root).read_url(
        asset_key(parse_granule_id(IDS[0]), "B04")
    )
