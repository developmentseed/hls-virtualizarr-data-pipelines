"""Forward handler: a batch is one commit, with two distinct failure modes.

The contract these pin down is the ordering. A per-file failure is reported to
SQS only *after* the commit has succeeded, so the records left off the failure
list really are stored. A commit failure is raised instead, because then nothing
is stored and the whole batch has to come back.

These run the real HLS processor against the test fakes: a local repo holding a
two-granule T13SDA cube, and forward granules addressed by asset key.
"""

import json
import pickle
import sys
from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import xarray as xr
from fakes import FakeAssets, write_all_bands
from virtualizarr_processor import processor as processor_module
from virtualizarr_processor.backfill import create_fork, merge_and_commit, promote
from virtualizarr_processor.processor import Processor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lambda"))

import process_messages.handler as handler_module  # noqa: E402
from process_messages.handler import handler  # noqa: E402


def make_sqs_event(keys: list[str], bucket: str = "test-bucket") -> dict:
    """Build a minimal SQS event with S3 notification bodies."""
    records = []
    for i, key in enumerate(keys):
        body = {
            "Records": [
                {
                    "s3": {
                        "bucket": {"name": bucket},
                        "object": {"key": key},
                    }
                }
            ]
        }
        records.append(
            {
                "messageId": f"msg-{i:03d}",
                "receiptHandle": f"receipt-{i}",
                "body": json.dumps(body),
                "attributes": {
                    "ApproximateReceiveCount": "1",
                    "SentTimestamp": "1717600000000",
                    "ApproximateFirstReceiveTimestamp": "1717600000000",
                },
                "messageAttributes": {},
                "md5OfBody": "abc",
                "eventSource": "aws:sqs",
                "eventSourceARN": "arn:aws:sqs:us-east-1:123456789:test-queue",
                "awsRegion": "us-east-1",
            }
        )
    return {"Records": records}


BACKFILLED = [
    "HLS.L30.T13SDA.2025152T170000.v2.0",
    "HLS.L30.T13SDA.2025168T170000.v2.0",
]
NEW_A = "HLS.L30.T13SDA.2025184T170000.v2.0"
NEW_B = "HLS.L30.T13SDA.2025200T170000.v2.0"
OLD = "HLS.L30.T13SDA.2013152T170000.v2.0"


def key(granule_id: str, band: str = "B04") -> str:
    return f"HLSL30.020/{granule_id}/{granule_id}.{band}.tif"


def main_times(factory: Callable[[], Processor]) -> list[str]:
    store = factory().initialize_repo().readonly_session("main").store
    ds = xr.open_zarr(store, group="T13SDA/B04/multiscales/0", consolidated=False)
    return [str(t)[:10] for t in ds.time.values]


def main_commits(factory: Callable[[], Processor]) -> int:
    return len(list(factory().initialize_repo().ancestry(branch="main")))


@pytest.fixture
def forward(
    fake_assets: FakeAssets,
    fake_processor_factory: Callable[[], Processor],
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[], Processor]:
    """A backfilled cube on main, the handler's Processor wired to it, and assets
    for the forward granules."""
    for value, gid in enumerate([*BACKFILLED, NEW_A, NEW_B, OLD], 1):
        write_all_bands(fake_assets, gid, value)
    processor = fake_processor_factory()
    repo = processor.open_backfill_repo()
    processor.initialize_backfill_store(repo, BACKFILLED)
    child = pickle.loads(create_fork(repo)).fork()
    for gid in BACKFILLED:
        assert processor.process_backfill_file(gid, child)
    merge_and_commit(repo, [pickle.dumps(child)], message="backfill")
    promote(repo)
    monkeypatch.setattr(handler_module, "Processor", fake_processor_factory)
    return fake_processor_factory


def test_a_batch_that_writes_and_commits_reports_no_failures(
    forward: Callable[[], Processor],
) -> None:
    before = main_commits(forward)

    response = handler(make_sqs_event([key(NEW_A), key(NEW_B)]), MagicMock())

    assert response == {"batchItemFailures": []}
    # one session shared across the batch, one commit at the end
    assert main_commits(forward) == before + 1
    assert main_times(forward) == [
        "2025-06-01",
        "2025-06-17",
        "2025-07-03",
        "2025-07-19",
    ]


def test_an_out_of_order_granule_is_sorted_before_commit(
    forward: Callable[[], Processor],
) -> None:
    response = handler(make_sqs_event([key(OLD)]), MagicMock())

    assert response == {"batchItemFailures": []}
    assert main_times(forward) == ["2013-06-01", "2025-06-01", "2025-06-17"]


def test_one_failed_file_returns_only_its_own_message(
    forward: Callable[[], Processor], fake_assets: FakeAssets
) -> None:
    """The rest of the batch still commits, so those files really are stored and
    only the failed message needs redelivering."""
    fake_assets.missing(NEW_B, "Fmask")

    response = handler(make_sqs_event([key(NEW_A), key(NEW_B), key(OLD)]), MagicMock())

    assert response == {"batchItemFailures": [{"itemIdentifier": "msg-001"}]}
    assert main_times(forward) == [
        "2013-06-01",
        "2025-06-01",
        "2025-06-17",
        "2025-07-03",
    ]


def test_a_file_that_raises_is_held_rather_than_aborting_the_batch(
    forward: Callable[[], Processor], monkeypatch: pytest.MonkeyPatch
) -> None:
    def factory() -> Processor:
        p = forward()
        real = p.process_file

        def process_file(file_key: str, session: object) -> bool:
            if file_key == "explode":
                raise Exception("boom")
            return real(file_key, session)  # type: ignore[arg-type]

        p.process_file = process_file  # type: ignore[method-assign]
        return p

    monkeypatch.setattr(handler_module, "Processor", factory)

    response = handler(make_sqs_event(["explode", key(NEW_A)]), MagicMock())

    assert response == {"batchItemFailures": [{"itemIdentifier": "msg-000"}]}
    assert len(main_times(forward)) == 3


def test_a_failed_commit_fails_the_whole_invocation(
    forward: Callable[[], Processor], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Files written into a session that never commits have not been stored.
    Reporting failures here would delete their messages; raising redelivers
    every one of them. A write that fails part-way makes the commit refuse."""
    real = processor_module.write_granule

    def half_written(store, tile, granule, *args, **kwargs):  # type: ignore[no-untyped-def]
        if granule.id == NEW_B:
            raise OSError("half written")
        return real(store, tile, granule, *args, **kwargs)

    monkeypatch.setattr(processor_module, "write_granule", half_written)
    before = main_commits(forward)

    with pytest.raises(RuntimeError, match="refusing to commit"):
        handler(make_sqs_event([key(NEW_A), key(NEW_B)]), MagicMock())

    assert main_commits(forward) == before
    assert len(main_times(forward)) == 2


def test_a_batch_where_nothing_could_be_written_fails_the_invocation(
    forward: Callable[[], Processor],
) -> None:
    """There is no commit to hang a partial failure report off, and Icechunk
    refuses an empty commit, so the batch goes back whole."""
    other_tile = "HLS.L30.T13SDB.2025184T170000.v2.0"

    with pytest.raises(RuntimeError, match="no file in this batch could be written"):
        handler(make_sqs_event([key(other_tile), "not-a-granule"]), MagicMock())


def test_a_store_that_was_never_backfilled_rejects_the_batch(
    fake_assets: FakeAssets,
    fake_processor_factory: Callable[[], Processor],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_all_bands(fake_assets, NEW_A, 1)
    monkeypatch.setattr(handler_module, "Processor", fake_processor_factory)

    with pytest.raises(RuntimeError, match="no file in this batch could be written"):
        handler(make_sqs_event([key(NEW_A)]), MagicMock())


def test_a_store_that_cannot_be_opened_is_logged_before_it_raises(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A misconfigured deployment fails here -- no bucket, no permission on it --
    and it happens before any per-file logging. Unguarded it leaves nothing but
    the runtime's plain-text traceback, which is invisible to a level filter."""
    monkeypatch.delenv("ICECHUNK_BUCKET", raising=False)
    monkeypatch.delenv("ICECHUNK_LOCAL_PATH", raising=False)
    monkeypatch.setattr(handler_module, "Processor", lambda: Processor(tile="T13SDA"))

    with caplog.at_level("ERROR"):
        with pytest.raises(ValueError, match="ICECHUNK_BUCKET"):
            handler(make_sqs_event([key(NEW_A)]), MagicMock())

    assert "Could not open the store" in caplog.text


def test_an_sns_wrapped_message_is_unwrapped(forward: Callable[[], Processor]) -> None:
    inner = {
        "Records": [
            {"s3": {"bucket": {"name": "test-bucket"}, "object": {"key": key(NEW_A)}}}
        ]
    }
    event = make_sqs_event(["placeholder"])
    event["Records"][0]["body"] = json.dumps({"Message": json.dumps(inner)})

    assert handler(event, MagicMock()) == {"batchItemFailures": []}
    assert main_times(forward)[-1] == "2025-07-03"
