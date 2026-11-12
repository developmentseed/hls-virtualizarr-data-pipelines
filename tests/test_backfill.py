"""The template's fork/merge/promote helpers driving the HLS processor."""

import pickle
from collections.abc import Callable

import xarray as xr
from fakes import FakeAssets, write_all_bands
from virtualizarr_processor import backfill
from virtualizarr_processor.processor import Processor

IDS = [
    f"HLS.L30.T13SDA.{day}T170000.v2.0"
    for day in ("2013152", "2014152", "2015152", "2025152", "2025168", "2025184")
]


def _worker(processor: Processor, shared_fork_bytes: bytes, keys: list[str]) -> bytes:
    child = pickle.loads(shared_fork_bytes).fork()
    for key in keys:
        assert processor.process_backfill_file(key, child)
    return pickle.dumps(child)


def _b04_values(processor: Processor, branch: str) -> list[int]:
    store = processor.open_backfill_repo().readonly_session(branch).store
    ds = xr.open_zarr(store, group="T13SDA/B04/multiscales/0", consolidated=False)
    return [int(ds.B04.isel(time=i).max()) for i in range(ds.sizes["time"])]


def test_full_backfill_round_trip(
    fake_assets: FakeAssets, fake_processor_factory: Callable[[], Processor]
) -> None:
    for value, gid in enumerate(IDS, 1):
        write_all_bands(fake_assets, gid, value)
    processor = fake_processor_factory()
    repo = processor.open_backfill_repo()
    base = processor.initialize_backfill_store(repo, IDS)
    assert base == repo.lookup_branch("backfill")

    shared = backfill.create_fork(repo)
    # two workers in separate Processor instances, given out-of-order keys
    child_a = _worker(fake_processor_factory(), shared, IDS[:3][::-1])
    child_b = _worker(fake_processor_factory(), shared, IDS[3:])

    tip = backfill.merge_and_commit(repo, [child_a, child_b], message="backfill")
    assert tip == repo.lookup_branch("backfill")
    assert _b04_values(processor, "backfill") == [1, 2, 3, 4, 5, 6]

    backfill.promote(repo)
    assert repo.lookup_branch("main") == tip
    assert _b04_values(processor, "main") == [1, 2, 3, 4, 5, 6]


def test_a_skipped_granule_leaves_a_fill_valued_step(
    fake_assets: FakeAssets, fake_processor_factory: Callable[[], Processor]
) -> None:
    for value, gid in enumerate(IDS, 1):
        write_all_bands(fake_assets, gid, value)
    fake_assets.missing(IDS[2], "SZA")
    processor = fake_processor_factory()
    repo = processor.open_backfill_repo()
    processor.initialize_backfill_store(repo, IDS)

    child = pickle.loads(backfill.create_fork(repo)).fork()
    results = [processor.process_backfill_file(k, child) for k in IDS]
    backfill.merge_and_commit(repo, [pickle.dumps(child)], message="backfill")

    assert results == [True, True, False, True, True, True]
    assert [r.granule_id for r in processor.skipped] == [IDS[2]]
    assert _b04_values(processor, "backfill") == [1, 2, 0, 4, 5, 6]
