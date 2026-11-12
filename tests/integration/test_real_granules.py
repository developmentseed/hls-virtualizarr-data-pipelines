"""Opt-in end-to-end test against real LP DAAC granules (T13SDA, L30).

Needs the network, an Earthdata login in ``~/.netrc`` and ``HLS_NETWORK_TESTS=1``;
reads over https, so no AWS credentials are required. Run with::

    HLS_NETWORK_TESTS=1 uv run --with rasterio pytest tests/integration -v -m network

The pixel comparison against a direct COG read needs rasterio, which is not a
project dependency (hence ``--with rasterio``).

Granule IDs come from docs/superpowers/spikes/2026-11-11-time-stack-findings.md
section 1: the three the spike used for backfill (1x2013, 2x2025) and, for the
forward append, another T13SDA row from the same 2025-06 archive listing. That
granule's time is not on the backfilled axis, so it exercises append + re-sort.
"""

from __future__ import annotations

import os
import pathlib

import numpy as np
import pytest
import xarray as xr
from virtualizarr_processor.backfill import create_fork, merge_and_commit, promote
from virtualizarr_processor.hls.credentials import earthdata_token
from virtualizarr_processor.processor import Processor

pytestmark = [
    pytest.mark.network,
    pytest.mark.skipif(
        os.environ.get("HLS_NETWORK_TESTS") != "1",
        reason="set HLS_NETWORK_TESTS=1 to run tests against LP DAAC",
    ),
]

TILE = "T13SDA"
BACKFILL_IDS = [
    "HLS.L30.T13SDA.2013157T174007.v2.0",
    "HLS.L30.T13SDA.2025157T174342.v2.0",
    "HLS.L30.T13SDA.2025181T174401.v2.0",
]
FORWARD_ID = "HLS.L30.T13SDA.2025173T174356.v2.0"
ALL_IDS_SORTED = sorted(BACKFILL_IDS + [FORWARD_ID], key=lambda g: g.split(".")[3])

# 3x3 windows (row, col) where B04 is valid in the three backfilled granules
# (findings doc section 5).
WINDOWS = {0: (1520, 752), 1: (760, 376)}
HTTPS_PREFIX = "https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/"


def _cube(session, band: str, level: int) -> xr.Dataset:
    return xr.open_zarr(
        session.store,
        group=f"{TILE}/{band}/multiscales/{level}",
        consolidated=False,
        zarr_format=3,
        mask_and_scale=False,
    )


def _direct(gid: str, band: str, level: int, window) -> np.ndarray:
    rasterio = pytest.importorskip("rasterio")
    from rasterio.windows import Window

    cookies = os.path.join(os.path.expanduser("~"), ".cache", "hls-gdal-cookies")
    # A GDAL bearer header fails against the signed-S3 redirect (spike
    # findings); the netrc + cookie-jar flow works.
    env = {
        "GDAL_HTTP_NETRC": "YES",
        "GDAL_HTTP_COOKIEFILE": cookies,
        "GDAL_HTTP_COOKIEJAR": cookies,
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    }
    r0, c0 = window
    kw = {} if level == 0 else {"overview_level": level - 1}
    url = f"{HTTPS_PREFIX}HLSL30.020/{gid}/{gid}.{band}.tif"
    with rasterio.Env(**env), rasterio.open(url, **kw) as src:
        return src.read(1, window=Window(c0, r0, 3, 3))


def _token_in_tree(root: pathlib.Path, token: str) -> list[pathlib.Path]:
    needle = token.encode()
    return [p for p in root.rglob("*") if p.is_file() and needle in p.read_bytes()]


def test_real_granules_backfill_then_forward(tmp_path, monkeypatch) -> None:
    pytest.importorskip("rasterio")
    repo_dir = tmp_path / "repo"
    monkeypatch.setenv("ICECHUNK_LOCAL_PATH", str(repo_dir))
    processor = Processor(tile=TILE, product="L30", access="https")
    repo = processor.initialize_repo()

    # R12: the https container config must not persist the bearer token.
    leaked = _token_in_tree(repo_dir, earthdata_token())
    assert not leaked, f"bearer token found in repo storage: {leaked}"

    # main needs a commit to branch from (the deployed init does this too)
    import zarr

    session = repo.writable_session("main")
    zarr.open_group(session.store, mode="a")
    session.commit("init main")

    processor.initialize_backfill_store(repo, BACKFILL_IDS)
    fork_bytes = create_fork(repo)
    import pickle

    fork = pickle.loads(fork_bytes).fork()
    for gid in BACKFILL_IDS:
        assert processor.process_backfill_file(gid, fork), processor.skipped
    merge_and_commit(repo, [pickle.dumps(fork)], message="backfill real granules")
    promote(repo)

    session = processor.initialize_session(repo)
    assert processor.process_file(FORWARD_ID, session)
    processor.commit_processed_files(session)

    read = repo.readonly_session("main")
    ds = _cube(read, "B04", 0)
    times = ds["time"].values
    assert len(times) == 4
    assert np.all(np.diff(times) > np.timedelta64(0)), times

    for band in ("B04", "Fmask"):
        for level, (r0, c0) in WINDOWS.items():
            arr = _cube(read, band, level)[band]
            for i, gid in enumerate(ALL_IDS_SORTED):
                got = np.asarray(arr[i, r0 : r0 + 3, c0 : c0 + 3])
                want = _direct(gid, band, level, (r0, c0))
                np.testing.assert_array_equal(
                    got, want, err_msg=f"{gid} {band} L{level}"
                )

    assert "Fmask" in _cube(read, "Fmask", 1)
