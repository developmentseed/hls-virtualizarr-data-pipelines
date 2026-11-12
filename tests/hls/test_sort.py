import numpy as np
import pytest
import xarray as xr
import zarr
from fakes import FAKE_GRID, local_locator
from virtualizarr_processor.hls.cube import (
    declare_cube,
    granule_levels,
    level_groups,
    open_granule,
    write_granule,
)
from virtualizarr_processor.hls.granule import parse_granule_id
from virtualizarr_processor.hls.sort import sort_time

IDS = [
    "HLS.L30.T13SDA.2013152T170000.v2.0",
    "HLS.L30.T13SDA.2025152T170000.v2.0",
    "HLS.L30.T13SDA.2025168T170000.v2.0",
]
REF = "HLS.L30.T13SDA.2025100T170000.v2.0"
BANDS = ("B04", "Fmask")
T = "T13SDA"


def _open(fake_assets, gid, bands=BANDS):
    locator = local_locator(fake_assets.root)
    return open_granule(
        parse_granule_id(gid), bands, fake_assets.opener, None, locator.read_url
    )


def _write(s, fake_assets, gid, mode):
    locator = local_locator(fake_assets.root)
    write_granule(
        s.store,
        T,
        parse_granule_id(gid),
        _open(fake_assets, gid),
        mode=mode,
        reference_url_for=locator.reference_url,
    )


def _declare(repo, fake_assets, ids, values):
    for band in BANDS:
        fake_assets.write(REF, band, 9)
    s = repo.writable_session("main")
    declare_cube(
        s.store,
        FAKE_GRID,
        [parse_granule_id(i).time for i in ids],
        _open(fake_assets, REF),
    )
    for gid, v in zip(ids, values, strict=True):
        for band in BANDS:
            fake_assets.write(gid, band, v)
        _write(s, fake_assets, gid, "region")
    return s


def test_sort_noop_when_ordered(hls_repo, fake_assets):
    s = _declare(hls_repo, fake_assets, IDS[1:], [2, 3])
    s.commit("cube")
    s = hls_repo.writable_session("main")
    assert sort_time(s, T) == 0
    assert not s.has_uncommitted_changes


def test_out_of_order_append_is_sorted_in_every_group(hls_repo, fake_assets):
    s = _declare(hls_repo, fake_assets, IDS[1:], [2, 3])
    for band in BANDS:
        fake_assets.write(IDS[0], band, 1)
    _write(s, fake_assets, IDS[0], "append")
    assert sort_time(s, T) == 3
    for g in level_groups(s.store, T):
        ds = xr.open_zarr(s.store, group=g, consolidated=False)
        assert (np.diff(ds.time.values) > np.timedelta64(0)).all()
        assert zarr.open_array(s.store, path=f"{g}/time").dtype == np.int64
        band = g.split("/")[1]
        assert [int(ds[band].isel(time=i).max()) for i in range(3)] == [1, 2, 3]
    assert sort_time(s, T) == 0


def test_sort_refuses_duplicate_times(hls_repo, fake_assets):
    s = _declare(hls_repo, fake_assets, IDS[1:], [2, 3])
    for g in level_groups(s.store, T):
        zarr.open_array(s.store, path=f"{g}/time", mode="r+")[1] = zarr.open_array(
            s.store, path=f"{g}/time"
        )[0]
    with pytest.raises(ValueError, match="duplicate"):
        sort_time(s, T)


def test_sort_refuses_disagreeing_groups(hls_repo, fake_assets):
    s = _declare(hls_repo, fake_assets, IDS[1:], [2, 3])
    granule = parse_granule_id(IDS[0])
    for band in BANDS:
        fake_assets.write(IDS[0], band, 1)
    locator = local_locator(fake_assets.root)
    levels = granule_levels(
        granule,
        "B04",
        _open(fake_assets, IDS[0])["B04"],
        reference_url_for=locator.reference_url,
    )
    for level, ds in levels.items():
        ds.vz.to_icechunk(
            s.store,
            group=f"{T}/B04/multiscales/{level}",
            append_dim="time",
            validate_containers=False,
        )
    with pytest.raises(ValueError, match="disagree"):
        sort_time(s, T)
