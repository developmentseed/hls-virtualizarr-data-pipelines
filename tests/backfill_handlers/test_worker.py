from unittest.mock import MagicMock

import pytest
from backfill_handlers import fork, fork_store, init, inventory, worker
from fakes import Backfill
from virtualizarr_processor.processor import SkipRecord


def test_worker_writes_child_fork(
    s3_bucket: str,
    hls_backfill: Backfill,
    lambda_context: MagicMock,
) -> None:
    init.handler({"inventory_uri": hls_backfill.inventory_uri}, lambda_context)
    fork_result = fork.handler(
        {
            "partition_id": "0",
            "manifest_uri": f"s3://{s3_bucket}/run/partitions/0.json",
            "run_prefix": f"s3://{s3_bucket}/run/",
        },
        lambda_context,
    )

    event = {
        "fork_in_uri": fork_result["fork_in_uri"],
        "forks_out_prefix": fork_result["forks_out_prefix"],
        "skipped_out_prefix": fork_result["skipped_out_prefix"],
        "file_keys": hls_backfill.granule_ids[:3],
    }
    result = worker.handler(event, lambda_context)

    assert result["child_fork_uri"].startswith(fork_result["forks_out_prefix"])
    assert len(fork_store.load_fork(result["child_fork_uri"])) > 0
    assert len(fork_store.list_forks(fork_result["forks_out_prefix"])) == 1


class _FakeProcessor:
    """Processor stand-in; ``outcomes`` maps key -> skip stage (absent = ok)."""

    outcomes: dict[str, str | None] = {}

    def __init__(self) -> None:
        self.skipped: list[SkipRecord] = []

    def process_backfill_file(self, key: str, fork: object) -> bool:
        stage = self.outcomes.get(key)
        if stage is None:
            return True
        self.skipped.append(SkipRecord(key, stage, "boom"))  # type: ignore[arg-type]
        return False


def _setup_fork(s3_bucket: str, hls_backfill: Backfill, ctx: MagicMock) -> dict:
    init.handler({"inventory_uri": hls_backfill.inventory_uri}, ctx)
    fr = fork.handler(
        {
            "partition_id": "0",
            "manifest_uri": f"s3://{s3_bucket}/run/partitions/0.json",
            "run_prefix": f"s3://{s3_bucket}/run/",
        },
        ctx,
    )
    return {
        "fork_in_uri": fr["fork_in_uri"],
        "forks_out_prefix": fr["forks_out_prefix"],
        "skipped_out_prefix": fr["skipped_out_prefix"],
    }


def test_worker_records_skips_and_does_not_raise(
    s3_bucket: str,
    hls_backfill: Backfill,
    lambda_context: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _setup_fork(s3_bucket, hls_backfill, lambda_context)
    bad = hls_backfill.granule_ids[1]
    _FakeProcessor.outcomes = {bad: "open"}
    monkeypatch.setattr(worker, "Processor", _FakeProcessor)

    result = worker.handler(
        {**base, "file_keys": hls_backfill.granule_ids[:3]}, lambda_context
    )

    keys = fork_store.list_forks(base["skipped_out_prefix"])
    assert len(keys) == 1 and keys[0].endswith(".json")
    stem = result["child_fork_uri"].rsplit("/", 1)[1].removesuffix(".pkl")
    assert keys[0].endswith(f"{stem}.json")
    assert inventory.read_json(keys[0]) == [
        {"granule_id": bad, "stage": "open", "error": "boom"}
    ]
    assert len(fork_store.list_forks(base["forks_out_prefix"])) == 1


def test_worker_writes_no_skip_file_when_nothing_skipped(
    s3_bucket: str,
    hls_backfill: Backfill,
    lambda_context: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _setup_fork(s3_bucket, hls_backfill, lambda_context)
    _FakeProcessor.outcomes = {}
    monkeypatch.setattr(worker, "Processor", _FakeProcessor)
    worker.handler({**base, "file_keys": hls_backfill.granule_ids[:2]}, lambda_context)
    assert fork_store.list_forks(base["skipped_out_prefix"]) == []


def test_worker_raises_on_write_stage_skip_and_saves_nothing(
    s3_bucket: str,
    hls_backfill: Backfill,
    lambda_context: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _setup_fork(s3_bucket, hls_backfill, lambda_context)
    _FakeProcessor.outcomes = {hls_backfill.granule_ids[1]: "write"}
    monkeypatch.setattr(worker, "Processor", _FakeProcessor)
    with pytest.raises(RuntimeError):
        worker.handler(
            {**base, "file_keys": hls_backfill.granule_ids[:3]}, lambda_context
        )
    assert fork_store.list_forks(base["forks_out_prefix"]) == []


def test_worker_propagates_processor_exceptions(
    s3_bucket: str,
    hls_backfill: Backfill,
    lambda_context: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _setup_fork(s3_bucket, hls_backfill, lambda_context)

    class Exploding(_FakeProcessor):
        def process_backfill_file(self, key: str, fork: object) -> bool:
            raise ValueError("kaboom")

    monkeypatch.setattr(worker, "Processor", Exploding)
    with pytest.raises(ValueError):
        worker.handler(
            {**base, "file_keys": hls_backfill.granule_ids[:1]}, lambda_context
        )
    assert fork_store.list_forks(base["forks_out_prefix"]) == []
