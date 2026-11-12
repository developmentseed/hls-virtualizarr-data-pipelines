from unittest.mock import MagicMock

import numpy as np
import xarray as xr
from backfill_handlers import fork, init, inventory, partition, promote, reduce, worker
from fakes import Backfill
from virtualizarr_processor.hls.cube import level_groups


def test_full_backfill_chain(
    s3_bucket: str,
    hls_backfill: Backfill,
    lambda_context: MagicMock,
) -> None:
    run_prefix = f"s3://{s3_bucket}/run/"

    # partition -> init
    parts = partition.handler(
        {
            "inventory_uri": hls_backfill.inventory_uri,
            "run_prefix": run_prefix,
            "partition_size": 3,
        },
        lambda_context,
    )["partitions"]
    init.handler({"inventory_uri": hls_backfill.inventory_uri}, lambda_context)

    # serial over partitions: fork -> workers (one per granule) -> reduce
    for part in parts:
        fork_result = fork.handler(
            {
                "partition_id": part["partition_id"],
                "manifest_uri": part["manifest_uri"],
                "run_prefix": run_prefix,
            },
            lambda_context,
        )
        for file_key in inventory.read_manifest(part["manifest_uri"]):
            worker.handler(
                {
                    "fork_in_uri": fork_result["fork_in_uri"],
                    "forks_out_prefix": fork_result["forks_out_prefix"],
                    "skipped_out_prefix": fork_result["skipped_out_prefix"],
                    "file_keys": [file_key],
                },
                lambda_context,
            )
        reduce.handler(
            {
                "partition_id": part["partition_id"],
                "run_prefix": run_prefix,
                "forks_out_prefix": fork_result["forks_out_prefix"],
                "skipped_out_prefix": fork_result["skipped_out_prefix"],
            },
            lambda_context,
        )

    # promote and verify every granule on main, in every (band, level) group
    promote.handler({}, lambda_context)
    store = hls_backfill.processor().open_backfill_repo().readonly_session("main").store
    groups = level_groups(store, "T13SDA")
    assert len(groups) == 15 * 2
    for g in groups:
        band = g.split("/")[1]
        ds = xr.open_zarr(store, group=g, consolidated=False)
        assert [int(ds[band].isel(time=i).max()) for i in range(6)] == [
            1,
            2,
            3,
            4,
            5,
            6,
        ]
        assert (np.diff(ds.time.values) > np.timedelta64(0)).all()
