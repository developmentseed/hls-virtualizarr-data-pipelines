"""The HLS tile-cube processor: backfill, forward processing and their failures."""

import pickle
from datetime import UTC, datetime, timedelta
from unittest.mock import ANY

import icechunk
import numpy as np
import pytest
import xarray as xr
import zarr
from fakes import FAKE_GRID, local_locator, write_all_bands
from virtualizarr_processor import processor as processor_module
from virtualizarr_processor.backfill import merge_and_commit, promote
from virtualizarr_processor.hls.cube import level_group, level_groups
from virtualizarr_processor.hls.encoding import BAND_ENCODINGS
from virtualizarr_processor.hls.granule import BANDS, parse_granule_id
from virtualizarr_processor.processor import (
    LpDaacLocator,
    Processor,
    SkipRecord,
)
from virtualizarr_processor.typing import VirtualizarrProcessor

GID_2013 = "HLS.L30.T13SDA.2013152T170000.v2.0"
GID_2025A = "HLS.L30.T13SDA.2025152T170000.v2.0"
GID_2025B = "HLS.L30.T13SDA.2025168T170000.v2.0"
GID_NEW = "HLS.L30.T13SDA.2025200T170000.v2.0"
TILE = "T13SDA"


def open_level(repo, branch, band, level):
    return xr.open_zarr(
        repo.readonly_session(branch).store,
        group=level_group(TILE, band, level),
        consolidated=False,
        mask_and_scale=False,
    )


def time_len(repo, branch):
    return open_level(repo, branch, "B04", 0).sizes["time"]


def chunks_at_time(store, index):
    """(values, fill) at one time step, for every (band, level) array."""
    out = []
    for g in level_groups(store, TILE):
        band = g.split("/")[1]
        array = zarr.open_array(store, path=f"{g}/{band}")
        out.append((np.asarray(array[index]), array.fill_value))
    return out


def fork_has_no_chunks_at_time(fork, index):
    # unwritten chunks read as the array's fill value; the fakes write values >= 1
    return all((a == fill).all() for a, fill in chunks_at_time(fork.store, index))


#: Stands for "this band's documented fill" in ``assert_monotonic_and_values``.
FILL = "fill"


def assert_monotonic_and_values(repo, branch, values):
    store = repo.readonly_session(branch).store
    for g in level_groups(store, TILE):
        band = g.split("/")[1]
        ds = xr.open_zarr(store, group=g, consolidated=False, mask_and_scale=False)
        assert (np.diff(ds.time.values) > np.timedelta64(0)).all(), g
        got = [int(ds[band].isel(time=i).max()) for i in range(ds.sizes["time"])]
        fill = BAND_ENCODINGS["L30"][band].fill_value
        assert got == [fill if v == FILL else v for v in values], g


@pytest.fixture
def processor(fake_assets, hls_repo):
    return Processor(
        tile=TILE,
        product="L30",
        grid=FAKE_GRID,
        opener=fake_assets.opener,
        locator=local_locator(fake_assets.root),
        repo_opener=lambda: hls_repo,
    )


def backfill(processor, repo, fake_assets, ids_values):
    """Run a whole in-process backfill and promote it to main."""
    for gid, value in ids_values:
        write_all_bands(fake_assets, gid, value)
    ids = [gid for gid, _ in ids_values]
    processor.initialize_backfill_store(repo, ids)
    fork = repo.writable_session("backfill").fork()
    assert all(processor.process_backfill_file(g, fork) for g in ids)
    merge_and_commit(repo, [pickle.dumps(fork)], message="p0")
    promote(repo)


@pytest.fixture
def backfilled(processor, hls_repo, fake_assets):
    backfill(processor, hls_repo, fake_assets, [(GID_2025A, 2), (GID_2025B, 3)])
    return hls_repo


# --- construction -----------------------------------------------------------


def test_follows_protocol(processor):
    assert isinstance(processor, VirtualizarrProcessor)


def test_defaults_from_environment(monkeypatch):
    monkeypatch.setenv("MGRS_TILE", TILE)
    monkeypatch.delenv("HLS_PRODUCT", raising=False)
    monkeypatch.delenv("LPDAAC_ACCESS", raising=False)
    p = Processor()
    assert (p.tile, p.product, p.access) == (TILE, "L30", "s3")
    assert p.grid.mgrs_id == TILE and p.grid.shape == (3660, 3660)
    assert p.skipped == []


def test_missing_tile_is_an_error(monkeypatch):
    monkeypatch.delenv("MGRS_TILE", raising=False)
    with pytest.raises(ValueError, match="MGRS_TILE"):
        Processor()


def test_unknown_tile_product_or_access_is_an_error():
    with pytest.raises(ValueError):
        Processor(tile="T99XXX")
    with pytest.raises(ValueError):
        Processor(tile=TILE, product="X30")
    with pytest.raises(ValueError):
        Processor(tile=TILE, access="ftp")


def test_grid_for_another_tile_is_an_error():
    with pytest.raises(ValueError):
        Processor(tile="T13SDB", grid=FAKE_GRID)


def test_lpdaac_locator():
    key = f"HLSL30.020/{GID_2025A}/{GID_2025A}.B04.tif"
    s3 = LpDaacLocator("s3")
    https = LpDaacLocator("https")
    assert s3.read_url(key) == f"s3://lp-prod-protected/{key}"
    assert https.read_url(key) == (
        f"https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/{key}"
    )
    assert s3.reference_url(key) == https.reference_url(key) == f"vcc://lpdaac/{key}"


def test_skip_record_to_dict():
    record = SkipRecord(GID_2025A, "open", "boom")
    assert record.to_dict() == {
        "granule_id": GID_2025A,
        "stage": "open",
        "error": "boom",
    }


def test_default_repo_opener_local_path(tmp_path, monkeypatch):
    monkeypatch.delenv("ICECHUNK_BUCKET", raising=False)
    monkeypatch.delenv("EARTHDATA_SECRET_ARN", raising=False)
    monkeypatch.setenv("ICECHUNK_LOCAL_PATH", str(tmp_path / "repo"))
    p = Processor(tile=TILE)
    repo = p.open_backfill_repo()
    assert "main" in repo.list_branches() and repo.lookup_branch("main")
    assert p.initialize_repo().lookup_branch("main") == repo.lookup_branch("main")


def test_default_repo_opener_needs_a_location(monkeypatch):
    monkeypatch.delenv("ICECHUNK_BUCKET", raising=False)
    monkeypatch.delenv("ICECHUNK_LOCAL_PATH", raising=False)
    with pytest.raises(ValueError, match="ICECHUNK_BUCKET"):
        Processor(tile=TILE).open_backfill_repo()


def test_garbage_collect(processor, backfilled):
    expiry = datetime.now(UTC) - timedelta(days=2)
    assert isinstance(processor.garbage_collect(expiry), icechunk.GCSummary)


# --- backfill ---------------------------------------------------------------


def test_backfill_end_to_end_in_process(processor, fake_assets, hls_repo):
    ids = [GID_2013, GID_2025A, GID_2025B]
    for v, gid in enumerate(ids, 1):
        write_all_bands(fake_assets, gid, v)
    processor.initialize_backfill_store(hls_repo, ids)
    fork = hls_repo.writable_session("backfill").fork()
    assert all(processor.process_backfill_file(g, fork) for g in reversed(ids))
    merge_and_commit(hls_repo, [pickle.dumps(fork)], message="p0")
    ds = open_level(hls_repo, "backfill", "B04", 0)
    assert [int(ds.B04.isel(time=i).max()) for i in range(3)] == [1, 2, 3]
    assert_monotonic_and_values(hls_repo, "backfill", [1, 2, 3])
    assert processor.skipped == []


def test_initialize_backfill_store_declares_sorted_times_from_newest(
    processor, fake_assets, hls_repo
):
    ids = [GID_2025B, GID_2013, GID_2025A]
    write_all_bands(fake_assets, GID_2025B, 1)  # only the newest needs assets
    snapshot = processor.initialize_backfill_store(hls_repo, ids)
    assert snapshot == hls_repo.lookup_branch("backfill")
    assert not hls_repo.writable_session("backfill").has_uncommitted_changes
    assert all(GID_2025B in url for url in fake_assets.opened)
    store = hls_repo.readonly_session("backfill").store
    groups = level_groups(store, TILE)
    assert len(groups) == len(BANDS["L30"]) * 2
    want = np.array(
        [parse_granule_id(g).time.replace(tzinfo=None) for g in sorted(ids)],
        dtype="datetime64[ns]",
    )
    for g in groups:
        ds = xr.open_zarr(store, group=g, consolidated=False, mask_and_scale=False)
        np.testing.assert_array_equal(ds.time.values, want)


def test_initialize_backfill_store_rejects_foreign_ids(processor, hls_repo):
    with pytest.raises(ValueError):
        processor.initialize_backfill_store(
            hls_repo, [GID_2025A, "HLS.L30.T13SDB.2025001T173012.v2.0"]
        )
    with pytest.raises(ValueError):
        processor.initialize_backfill_store(
            hls_repo, [GID_2025A, "HLS.S30.T13SDA.2025001T173012.v2.0"]
        )
    assert "backfill" not in hls_repo.list_branches()


def test_initialize_backfill_store_rejects_bad_or_duplicate_or_empty(
    processor, hls_repo
):
    with pytest.raises(ValueError):
        processor.initialize_backfill_store(hls_repo, [GID_2025A, "not-an-id"])
    with pytest.raises(ValueError, match="duplicate"):
        processor.initialize_backfill_store(hls_repo, [GID_2025A, GID_2025A])
    with pytest.raises(ValueError):
        processor.initialize_backfill_store(hls_repo, [])
    assert "backfill" not in hls_repo.list_branches()


@pytest.fixture
def initialized(processor, fake_assets, hls_repo):
    ids = [GID_2013, GID_2025A, GID_2025B]
    for v, gid in enumerate(ids, 1):
        write_all_bands(fake_assets, gid, v)
    processor.initialize_backfill_store(hls_repo, ids)
    return hls_repo.writable_session("backfill").fork()


def test_backfill_skip_records_and_returns_false(processor, fake_assets, initialized):
    fake_assets.missing(GID_2025A, "Fmask")
    assert processor.process_backfill_file(GID_2025A, initialized) is False
    assert processor.skipped == [SkipRecord(GID_2025A, "open", ANY)]


def test_backfill_skip_writes_nothing_for_granule(processor, fake_assets, initialized):
    fork = initialized
    assert processor.process_backfill_file(GID_2013, fork)
    fake_assets.missing(GID_2025A, "Fmask")  # sorts after B04 and most bands
    assert processor.process_backfill_file(GID_2025A, fork) is False
    assert processor.process_backfill_file(GID_2025B, fork)
    assert fork_has_no_chunks_at_time(fork, 1)
    assert not fork_has_no_chunks_at_time(fork, 0)
    assert not fork_has_no_chunks_at_time(fork, 2)


def test_backfill_grid_mismatch_is_a_check_skip(processor, fake_assets, initialized):
    fake_assets.write(GID_2025A, "B05", 2, chunks=(2, 2))
    assert processor.process_backfill_file(GID_2025A, initialized) is False
    assert processor.skipped == [SkipRecord(GID_2025A, "check", ANY)]
    assert "chunks" in processor.skipped[0].error
    assert fork_has_no_chunks_at_time(initialized, 1)


def test_backfill_granule_not_in_inventory_is_a_check_skip(
    processor, fake_assets, initialized
):
    write_all_bands(fake_assets, GID_NEW, 4)
    assert processor.process_backfill_file(GID_NEW, initialized) is False
    assert processor.skipped == [SkipRecord(GID_NEW, "check", ANY)]


def test_backfill_unparseable_and_foreign_keys_are_skipped(processor, initialized):
    assert processor.process_backfill_file("nonsense", initialized) is False
    other = "HLS.L30.T13SDB.2025152T170000.v2.0"
    assert processor.process_backfill_file(other, initialized) is False
    assert processor.skipped == [
        SkipRecord("nonsense", "parse", ANY),
        SkipRecord(other, "check", ANY),
    ]


def test_backfill_write_failure_is_a_write_skip(processor, initialized, monkeypatch):
    def boom(*args, **kwargs):
        raise OSError("disk on fire")

    monkeypatch.setattr(processor_module, "write_granule", boom)
    assert processor.process_backfill_file(GID_2025A, initialized) is False
    assert processor.skipped == [SkipRecord(GID_2025A, "write", "disk on fire")]


def test_transient_open_errors_are_retried(processor, fake_assets, initialized):
    real = fake_assets.opener
    calls = {"n": 0}

    def flaky(url, registry):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("throttled")
        return real(url, registry)

    processor.opener = flaky
    processor.retry_delay = 0
    assert processor.process_backfill_file(GID_2025A, initialized)
    assert processor.skipped == []


def test_not_found_is_not_retried(processor, fake_assets, initialized):
    processor.retry_delay = 0
    fake_assets.missing(GID_2025A, "B01")
    assert processor.process_backfill_file(GID_2025A, initialized) is False
    assert sum(f"{GID_2025A}.B01" in u for u in fake_assets.opened) == 1


# --- forward ----------------------------------------------------------------


def test_process_file_appends_and_sorts_out_of_order(
    processor, fake_assets, backfilled
):
    write_all_bands(fake_assets, GID_2013, 1)
    session = processor.initialize_session(backfilled)
    assert processor.process_file(f"HLSL30.020/{GID_2013}/{GID_2013}.B04.tif", session)
    processor.commit_processed_files(session)
    assert_monotonic_and_values(backfilled, "main", [1, 2, 3])


def test_process_file_in_order_append(processor, fake_assets, backfilled):
    write_all_bands(fake_assets, GID_NEW, 4)
    session = processor.initialize_session(backfilled)
    assert processor.process_file(GID_NEW, session)
    snapshot = processor.commit_processed_files(session)
    assert snapshot == backfilled.lookup_branch("main")
    assert_monotonic_and_values(backfilled, "main", [2, 3, 4])


def test_process_file_redelivery_is_region_write(processor, fake_assets, backfilled):
    write_all_bands(fake_assets, GID_NEW, 4)
    for _ in range(2):
        session = processor.initialize_session(backfilled)
        assert processor.process_file(GID_NEW, session)
        processor.commit_processed_files(session)
    assert time_len(backfilled, "main") == 3
    assert_monotonic_and_values(backfilled, "main", [2, 3, 4])


def test_process_file_fills_a_skipped_backfill_slot(processor, fake_assets, hls_repo):
    """A granule skipped at backfill keeps a fill-valued step; forward fills it."""
    for gid, v in [(GID_2013, 1), (GID_2025A, 2), (GID_2025B, 3)]:
        write_all_bands(fake_assets, gid, v)
    processor.initialize_backfill_store(hls_repo, [GID_2013, GID_2025A, GID_2025B])
    fork = hls_repo.writable_session("backfill").fork()
    fake_assets.missing(GID_2025A, "B04")
    assert processor.process_backfill_file(GID_2013, fork)
    assert not processor.process_backfill_file(GID_2025A, fork)
    assert processor.process_backfill_file(GID_2025B, fork)
    merge_and_commit(hls_repo, [pickle.dumps(fork)], message="p0")
    promote(hls_repo)
    assert_monotonic_and_values(hls_repo, "main", [1, FILL, 3])

    write_all_bands(fake_assets, GID_2025A, 2)
    session = processor.initialize_session(hls_repo)
    assert processor.process_file(GID_2025A, session)
    processor.commit_processed_files(session)
    assert time_len(hls_repo, "main") == 3
    assert_monotonic_and_values(hls_repo, "main", [1, 2, 3])


def test_duplicate_messages_in_one_batch_write_one_step(
    processor, fake_assets, backfilled
):
    write_all_bands(fake_assets, GID_NEW, 4)
    session = processor.initialize_session(backfilled)
    for band in ("B04", "Fmask"):
        assert processor.process_file(
            f"HLSL30.020/{GID_NEW}/{GID_NEW}.{band}.tif", session
        )
    processor.commit_processed_files(session)
    assert time_len(backfilled, "main") == 3


def test_process_file_rejects_other_tile_or_product(processor, backfilled):
    session = processor.initialize_session(backfilled)
    assert not processor.process_file("HLS.S30.T13SDA.2025001T173012.v2.0", session)
    assert not processor.process_file("HLS.L30.T13SDB.2025001T173012.v2.0", session)
    assert not processor.process_file("not-a-granule", session)
    assert not session.has_uncommitted_changes


def test_process_file_without_backfilled_store_fails(
    processor, fake_assets, hls_repo, caplog
):
    write_all_bands(fake_assets, GID_2025A, 2)
    fresh_main_session = processor.initialize_session(hls_repo)
    # the forward handler's powertools setup may reroute this logger; listen on it
    processor_module.logger.addHandler(caplog.handler)
    try:
        assert processor.process_file(GID_2025A, fresh_main_session) is False
    finally:
        processor_module.logger.removeHandler(caplog.handler)
    assert "store not backfilled" in caplog.text
    assert not fresh_main_session.has_uncommitted_changes
    assert fake_assets.opened == []


def test_process_file_open_or_check_failure_changes_nothing(
    processor, fake_assets, backfilled
):
    write_all_bands(fake_assets, GID_NEW, 4)
    fake_assets.missing(GID_NEW, "VAA")
    write_all_bands(fake_assets, GID_2013, 1)
    fake_assets.write(GID_2013, "B05", 1, levels=((8, 8),))
    session = processor.initialize_session(backfilled)
    processor.retry_delay = 0
    assert processor.process_file(GID_NEW, session) is False
    assert processor.process_file(GID_2013, session) is False
    assert not session.has_uncommitted_changes


def test_write_failure_poisons_the_session_commit(
    processor, fake_assets, backfilled, monkeypatch
):
    """A write that fails part-way may leave the session diverged: refuse to commit
    it so the whole batch is redelivered."""
    write_all_bands(fake_assets, GID_NEW, 4)
    session = processor.initialize_session(backfilled)

    def boom(*args, **kwargs):
        raise OSError("half written")

    monkeypatch.setattr(processor_module, "write_granule", boom)
    assert processor.process_file(GID_NEW, session) is False
    with pytest.raises(RuntimeError, match="half written|discard"):
        processor.commit_processed_files(session)
    # a fresh session for the redelivery is not poisoned
    monkeypatch.undo()
    session = processor.initialize_session(backfilled)
    assert processor.process_file(GID_NEW, session)
    processor.commit_processed_files(session)
    assert time_len(backfilled, "main") == 3


def test_registry_failure_fails_the_worker_rather_than_skipping(
    fake_assets, hls_repo, initialized
):
    def no_credentials():
        raise PermissionError("earthdata login failed")

    p = Processor(
        tile=TILE,
        grid=FAKE_GRID,
        opener=fake_assets.opener,
        locator=local_locator(fake_assets.root),
        repo_opener=lambda: hls_repo,
        registry=no_credentials,
    )
    with pytest.raises(PermissionError):
        p.process_backfill_file(GID_2025A, initialized)
    assert p.skipped == []


# --- documented encodings ---------------------------------------------------


def test_untagged_newest_granule_does_not_skip_the_archive(
    processor, fake_assets, hls_repo
):
    """The 2026 regression: the newest granule's angle COGs lost their nodata
    tag (fill reads 0). The cube still declares the documented fill, so the
    tagged older granules and the untagged newest one all write."""
    write_all_bands(fake_assets, GID_2013, 1)
    write_all_bands(fake_assets, GID_2025A, 2)
    write_all_bands(fake_assets, GID_2025B, 3)
    for band in ("SAA", "SZA", "VAA", "VZA"):
        fake_assets.write(GID_2025B, band, 3, fill_value=0)
    ids = [GID_2013, GID_2025A, GID_2025B]
    processor.initialize_backfill_store(hls_repo, ids)
    fork = hls_repo.writable_session("backfill").fork()
    assert [processor.process_backfill_file(g, fork) for g in ids] == [True] * 3
    assert processor.skipped == []

    store = fork.store
    for band, encoding in BAND_ENCODINGS["L30"].items():
        for level in (0, 1):
            array = zarr.open_array(
                store, path=f"{level_group(TILE, band, level)}/{band}"
            )
            assert array.fill_value == encoding.fill_value, (band, level)
            assert dict(array.attrs) == encoding.attributes(), (band, level)


def test_cube_decodes_with_documented_scale_and_fill(processor, fake_assets, hls_repo):
    write_all_bands(fake_assets, GID_2025A, 2)
    write_all_bands(fake_assets, GID_2025B, 3)
    fake_assets.missing(GID_2025A, "SAA")  # step 0 stays fill-valued
    processor.initialize_backfill_store(hls_repo, [GID_2025A, GID_2025B])
    fork = hls_repo.writable_session("backfill").fork()
    assert not processor.process_backfill_file(GID_2025A, fork)
    assert processor.process_backfill_file(GID_2025B, fork)
    merge_and_commit(hls_repo, [pickle.dumps(fork)], message="p0")

    saa = xr.open_zarr(  # decoded, as a reader would open it
        hls_repo.readonly_session("backfill").store,
        group=level_group(TILE, "SAA", 0),
        consolidated=False,
    )
    assert np.isnan(saa.SAA.isel(time=0)).all()
    np.testing.assert_allclose(saa.SAA.isel(time=1), 0.03)


def test_attributes_survive_forward_append_and_sort(processor, fake_assets, backfilled):
    write_all_bands(fake_assets, GID_2013, 1)  # out of order: append, then sort
    session = processor.initialize_session(backfilled)
    assert processor.process_file(GID_2013, session)
    processor.commit_processed_files(session)
    store = backfilled.readonly_session("main").store
    for band in ("B04", "Fmask", "SAA"):
        array = zarr.open_array(store, path=f"{level_group(TILE, band, 0)}/{band}")
        assert dict(array.attrs) == BAND_ENCODINGS["L30"][band].attributes(), band
