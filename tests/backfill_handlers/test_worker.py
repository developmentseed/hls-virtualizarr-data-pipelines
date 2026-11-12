from unittest.mock import MagicMock

from backfill_handlers import fork, fork_store, init, worker
from fakes import Backfill


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
        "file_keys": hls_backfill.granule_ids[:3],
    }
    result = worker.handler(event, lambda_context)

    assert result["child_fork_uri"].startswith(fork_result["forks_out_prefix"])
    assert len(fork_store.load_fork(result["child_fork_uri"])) > 0
    assert len(fork_store.list_forks(fork_result["forks_out_prefix"])) == 1
