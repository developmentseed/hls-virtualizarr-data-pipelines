from unittest.mock import MagicMock

import numpy as np
import xarray as xr
from fakes import Backfill


def test_init_declares_the_cube_from_the_inventory(
    hls_backfill: Backfill, lambda_context: MagicMock
) -> None:
    from backfill_handlers import init

    result = init.handler({"inventory_uri": hls_backfill.inventory_uri}, lambda_context)

    repo = hls_backfill.processor().open_backfill_repo()
    assert result["base_snapshot"] == repo.lookup_branch("backfill")
    ds = xr.open_zarr(
        repo.readonly_session("backfill").store,
        group="T13SDA/B04/multiscales/0",
        consolidated=False,
    )
    assert ds.sizes["time"] == len(hls_backfill.granule_ids)
    assert (np.diff(ds.time.values) > np.timedelta64(0)).all()
