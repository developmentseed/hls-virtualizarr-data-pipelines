from unittest.mock import MagicMock

from backfill_handlers import fork, init, reduce, worker
from fakes import Backfill


def test_reduce_commits_all_worker_forks(
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
    base = {
        "fork_in_uri": fork_result["fork_in_uri"],
        "forks_out_prefix": fork_result["forks_out_prefix"],
    }
    worker.handler({**base, "file_keys": hls_backfill.granule_ids[:3]}, lambda_context)
    worker.handler({**base, "file_keys": hls_backfill.granule_ids[3:]}, lambda_context)

    result = reduce.handler(
        {"partition_id": "0", "forks_out_prefix": fork_result["forks_out_prefix"]},
        lambda_context,
    )

    assert isinstance(result["tip"], str) and result["tip"]
    assert result["partition_id"] == "0"
