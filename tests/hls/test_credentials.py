import json
import os

import boto3
import pytest
from moto import mock_aws
from virtualizarr_processor.hls.credentials import (
    bootstrap_earthdata_env,
    virtual_chunk_container,
)

SECRET_NAME = "earthdata-test"


@pytest.fixture
def aws_region(monkeypatch):
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-west-2")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    # setenv-then-delenv makes monkeypatch record the original (absent) state,
    # so vars the code under test sets are removed again at teardown.
    for name in ("EARTHDATA_USERNAME", "EARTHDATA_PASSWORD", "EARTHDATA_SECRET_ARN"):
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)


@pytest.fixture
def secret_arn(aws_region):
    with mock_aws():
        client = boto3.client("secretsmanager", region_name="us-west-2")
        arn = client.create_secret(
            Name=SECRET_NAME,
            SecretString=json.dumps({"username": "u", "password": "p"}),
        )["ARN"]
        yield arn


def test_bootstrap_sets_env_from_secret(monkeypatch, secret_arn):
    monkeypatch.setenv("EARTHDATA_SECRET_ARN", secret_arn)
    bootstrap_earthdata_env()
    assert os.environ["EARTHDATA_USERNAME"] == "u"
    assert os.environ["EARTHDATA_PASSWORD"] == "p"


def test_bootstrap_noop_without_arn(monkeypatch, aws_region):
    def no_boto3(*args, **kwargs):
        raise AssertionError("boto3 must not be called without a secret ARN")

    monkeypatch.setattr(boto3, "client", no_boto3)
    bootstrap_earthdata_env()  # no boto3 call, no error
    assert "EARTHDATA_USERNAME" not in os.environ


def test_container_is_named_lpdaac():
    assert virtual_chunk_container("s3").name == "lpdaac"
