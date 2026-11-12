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


FAKE_TOKEN = "fake-edl-token-0123456789abcdef"


def test_https_bearer_token_never_reaches_storage(tmp_path, monkeypatch):
    """R13: the https container is persisted without headers; the token is
    only in the in-memory config of the opened repo."""
    import icechunk
    import zarr
    from virtualizarr_processor import processor as processor_module

    repo_dir = tmp_path / "repo"
    monkeypatch.setattr(processor_module, "earthdata_token", lambda: FAKE_TOKEN)
    for name in ("ICECHUNK_BUCKET", "EARTHDATA_SECRET_ARN"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ICECHUNK_LOCAL_PATH", str(repo_dir))
    opener = processor_module.default_repo_opener("https")

    created = opener()  # create path
    assert FAKE_TOKEN in repr(created.config.virtual_chunk_containers)
    session = created.writable_session("main")
    zarr.open_group(session.store, mode="a")
    session.commit("init main")
    reopened = opener()  # existing-repo path
    assert FAKE_TOKEN in repr(reopened.config.virtual_chunk_containers)
    session = reopened.writable_session("main")
    zarr.open_group(session.store, mode="a").attrs["x"] = 1
    session.commit("second commit")

    assert any(repo_dir.rglob("*")), "repo dir is empty; the check would be vacuous"
    stored = icechunk.Repository.open(
        icechunk.local_filesystem_storage(str(repo_dir))
    ).config.virtual_chunk_containers
    assert "lpdaac" in repr(stored)
    assert FAKE_TOKEN not in repr(stored)
    assert "Bearer" not in repr(stored)
    needle = FAKE_TOKEN.encode()
    leaked = [
        p for p in repo_dir.rglob("*") if p.is_file() and needle in p.read_bytes()
    ]
    assert not leaked


def test_persisted_https_container_has_no_headers():
    container = virtual_chunk_container("https", persisted=True)
    assert "Bearer" not in repr(container)
    with pytest.raises(ValueError, match="bearer token"):
        virtual_chunk_container("https")
