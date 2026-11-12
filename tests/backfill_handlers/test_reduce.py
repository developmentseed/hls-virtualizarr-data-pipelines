from unittest.mock import MagicMock

from backfill_handlers import fork, init, inventory, reduce, worker
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
        "skipped_out_prefix": fork_result["skipped_out_prefix"],
    }
    worker.handler({**base, "file_keys": hls_backfill.granule_ids[:3]}, lambda_context)
    worker.handler({**base, "file_keys": hls_backfill.granule_ids[3:]}, lambda_context)

    result = reduce.handler(
        {
            "partition_id": "0",
            "run_prefix": f"s3://{s3_bucket}/run/",
            "forks_out_prefix": fork_result["forks_out_prefix"],
            "skipped_out_prefix": fork_result["skipped_out_prefix"],
        },
        lambda_context,
    )

    assert isinstance(result["tip"], str) and result["tip"]
    assert result["partition_id"] == "0"
    assert result["skipped"] == 0
    assert inventory.read_json(f"s3://{s3_bucket}/run/partitions/0/skipped.json") == []


def test_reduce_writes_partition_skipped_json(
    s3_bucket: str,
    hls_backfill: Backfill,
    lambda_context: MagicMock,
) -> None:
    run_prefix = f"s3://{s3_bucket}/run/"
    init.handler({"inventory_uri": hls_backfill.inventory_uri}, lambda_context)
    fr = fork.handler(
        {
            "partition_id": "0",
            "manifest_uri": f"{run_prefix}partitions/0.json",
            "run_prefix": run_prefix,
        },
        lambda_context,
    )
    base = {
        "fork_in_uri": fr["fork_in_uri"],
        "forks_out_prefix": fr["forks_out_prefix"],
        "skipped_out_prefix": fr["skipped_out_prefix"],
    }
    stems = [
        worker.handler({**base, "file_keys": [gid]}, lambda_context)["child_fork_uri"]
        .rsplit("/", 1)[-1]
        .removesuffix(".pkl")
        for gid in hls_backfill.granule_ids[:2]
    ]
    a = {"granule_id": "B", "stage": "open", "error": "x"}
    b = {"granule_id": "A", "stage": "parse", "error": "y"}
    orphan = {"granule_id": "C", "stage": "open", "error": "z"}
    inventory.write_json(f"{fr['skipped_out_prefix']}{stems[0]}.json", [a, b])
    inventory.write_json(f"{fr['skipped_out_prefix']}{stems[1]}.json", [a])
    # a worker that crashed between its skip put and its fork put
    inventory.write_json(f"{fr['skipped_out_prefix']}orphan.json", [orphan])

    result = reduce.handler(
        {
            "partition_id": "0",
            "run_prefix": run_prefix,
            "forks_out_prefix": fr["forks_out_prefix"],
            "skipped_out_prefix": fr["skipped_out_prefix"],
        },
        lambda_context,
    )

    assert result["skipped"] == 2
    assert inventory.read_json(f"{run_prefix}partitions/0/skipped.json") == [b, a]
