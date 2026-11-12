import importlib.util
import json
import sys
from datetime import date
from pathlib import Path

import boto3
import pyarrow as pa
import pytest
from moto import mock_aws
from virtualizarr_processor.hls.assets import asset_key
from virtualizarr_processor.hls.granule import granule_from_key, parse_granule_id

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lambda"))

from process_messages.handler import process_notification  # noqa: E402

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


build_inventory = _load("build_inventory")
enqueue_granules = _load("enqueue_granules")

IDS = [f"HLS.L30.T13SDA.2025{d:03d}T173012.v2.0" for d in range(1, 26)]


@pytest.fixture
def sqs_queue():
    with mock_aws():
        client = boto3.client("sqs", region_name="us-west-2")
        url = client.create_queue(QueueName="q")["QueueUrl"]
        yield client, url


def _drain(client, url):
    bodies = []
    while True:
        msgs = client.receive_message(
            QueueUrl=url, MaxNumberOfMessages=10, WaitTimeSeconds=0
        ).get("Messages", [])
        if not msgs:
            return bodies
        bodies += [m["Body"] for m in msgs]
        for m in msgs:
            client.delete_message(QueueUrl=url, ReceiptHandle=m["ReceiptHandle"])


def test_enqueue_message_shape_round_trips_through_handler(sqs_queue):
    client, url = sqs_queue
    assert enqueue_granules.main(["--queue-url", url, *IDS], sqs_client=client) == 0
    bodies = _drain(client, url)
    assert len(bodies) == 25

    seen = []

    class StubProcessor:
        def process_file(self, file_key, session):
            seen.append(file_key)
            return True

    for body in bodies:
        msg = json.loads(body)
        assert msg["Records"][0]["s3"]["bucket"]["name"] == "lp-prod-protected"
        process_notification(msg, session=None, processor=StubProcessor())
    assert sorted(granule_from_key(k).id for k in seen) == IDS
    assert all(k.endswith(".Fmask.tif") for k in seen)
    assert seen[0] == asset_key(parse_granule_id(seen[0].split("/")[1]), "Fmask")


def test_enqueue_since_and_from_file(sqs_queue, tmp_path):
    client, url = sqs_queue
    inv = tmp_path / "inventory.json"
    inv.write_text(json.dumps({"ids": IDS}))
    enqueue_granules.main(
        ["--queue-url", url, "--from", str(inv), "--since", "2025-01-21"],
        sqs_client=client,
    )
    assert len(_drain(client, url)) == 5  # days 21..25


def test_build_inventory_writes_and_uploads(tmp_path):
    table = pa.table({"id": IDS[:3] + ["HLS.S30.T13SDA.2025001T173012.v2.0"]})
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-west-2")
        s3.create_bucket(
            Bucket="inv-bucket",
            CreateBucketConfiguration={"LocationConstraint": "us-west-2"},
        )
        rc = build_inventory.main(
            [
                "--tile",
                "T13SDA",
                "--product",
                "L30",
                "--start",
                "2025-01",
                "--end",
                "2025-01",
                "--out",
                str(tmp_path),
                "--upload",
                "s3://inv-bucket/pre/",
            ],
            reader=lambda uri: table,
            today=date(2025, 3, 1),
            s3_client=s3,
        )
        assert rc == 0
        body = s3.get_object(Bucket="inv-bucket", Key="pre/inventory.json")[
            "Body"
        ].read()
    assert json.loads(body)["ids"] == IDS[:3]
    assert json.loads((tmp_path / "inventory.json").read_text())["ids"] == IDS[:3]
