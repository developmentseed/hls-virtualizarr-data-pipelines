import json
from collections.abc import Callable, Iterator
from unittest.mock import MagicMock

import boto3
import pytest
from fakes import Backfill, FakeAssets, write_all_bands
from moto import mock_aws
from virtualizarr_processor.processor import Processor

BUCKET = "test-backfill-bucket"

#: Six T13SDA granules in time order; granule i is written with value i + 1.
GRANULE_IDS = [
    f"HLS.L30.T13SDA.{day}T170000.v2.0"
    for day in ("2013152", "2014152", "2015152", "2025152", "2025168", "2025184")
]


@pytest.fixture()
def s3_bucket() -> Iterator[str]:
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=BUCKET)
        yield BUCKET


@pytest.fixture()
def lambda_context() -> MagicMock:
    """A stand-in Lambda context.

    powertools' @logger.inject_lambda_context reads context.function_name etc.
    at invocation, so handlers cannot be called with None; a MagicMock supplies
    any attribute (matching the existing tests/test_handler.py convention).
    """
    return MagicMock()


@pytest.fixture()
def hls_backfill(
    s3_bucket: str,
    fake_assets: FakeAssets,
    fake_processor_factory: Callable[[], Processor],
    monkeypatch: pytest.MonkeyPatch,
) -> Backfill:
    """Fake assets for ``GRANULE_IDS``, their inventory in S3, and every backfill
    handler's ``Processor`` replaced by one wired to the fakes and a local repo."""
    from backfill_handlers import fork, init, promote, reduce, worker

    for module in (fork, init, promote, reduce, worker):
        monkeypatch.setattr(module, "Processor", fake_processor_factory)
    for value, gid in enumerate(GRANULE_IDS, 1):
        write_all_bands(fake_assets, gid, value)
    boto3.client("s3", region_name="us-east-1").put_object(
        Bucket=s3_bucket, Key="inventory.json", Body=json.dumps(GRANULE_IDS).encode()
    )
    return Backfill(
        f"s3://{s3_bucket}/inventory.json", list(GRANULE_IDS), fake_processor_factory
    )
