import os
from typing import Any, Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings

print("STAGE from env:", os.getenv("STAGE"))


def include_trailing_slash(value: Any) -> Any:
    """Make sure the value includes a trailing slash if str"""
    if isinstance(value, str):
        return value.rstrip("/") + "/"
    return value


class StackSettings(BaseSettings):
    PROJECT_NAME: str = "virtualizarr-data-pipelines"
    STACK_NAME: str = "virtualizarr-data-pipelines"
    STAGE: Literal["dev", "prod"]
    # Optional: when blank, app.py falls back to CDK_DEFAULT_ACCOUNT (the account
    # of the active AWS credentials) so synth/deploy still resolve an environment.
    ACCOUNT_ID: str | None = None
    ACCOUNT_REGION: str = "us-east-1"
    ICECHUNK_BUCKET_NAME: str = "icechunk-outuput"
    ICECHUNK_BUCKET: str | None = None
    # Key prefix for this dataset's repo. Icechunk >=2.1.0 refuses to CREATE a
    # repo at an empty prefix (bucket root), so this must be non-empty to bootstrap
    # a new store. Passed into the Lambda env as ICECHUNK_PREFIX.
    ICECHUNK_PREFIX: str | None = None
    # Region of the Icechunk bucket, passed into the Lambda env as
    # ICECHUNK_REGION. Only needed when the bucket is not in the region the
    # stack is deployed into: left unset, no ICECHUNK_REGION reaches the
    # handlers and icechunk resolves the region from the running Lambda.
    ICECHUNK_REGION: str | None = None
    DATA_BUCKET_NAME: str | None = None
    PROJECT: str = "virtualizarr-data-pipelines"
    SNS_TOPIC: str | None = None
    MAX_CONCURRENCY: int = 50
    SQS_BATCH_SIZE: int = 10

    # ARN of the Secrets Manager secret holding Earthdata {username, password}.
    # Optional: required only for reading protected LP DAAC granules. When unset,
    # no secret resource or IAM grant is created. Granted (and exposed as an env
    # var) only to the functions that read COGs: backfill init and worker, and the
    # forward process_messages Lambda.
    EARTHDATA_SECRET_ARN: str | None = None

    # The one HLS tile this deployment builds a cube for, and its product.
    # Passed to every processor Lambda as MGRS_TILE / HLS_PRODUCT.
    MGRS_TILE: str = "T13SDA"
    HLS_PRODUCT: Literal["L30"] = "L30"
    # How granule COGs are read. Deployed Lambdas are s3-only (same-region,
    # us-west-2). The processor's https mode is local/laptop use only: its
    # bearer token must never sit in Lambda env, repo config or fork pickles
    # in shared storage.
    LPDAAC_ACCESS: Literal["s3"] = "s3"
    # Icechunk chunk-ref cache size (ICECHUNK_NUM_CHUNK_REFS). The spike measured
    # 8,010 chunk refs per granule: 15 bands x (225 + 225 + 64 + 16 + 4) = 534
    # chunks per band. One backfill partition of BACKFILL_PARTITION_SIZE = 500
    # granules is 500 x 8,010 = 4,005,000 refs; 5,000,000 leaves headroom.
    # Scale this with BACKFILL_PARTITION_SIZE (about 8,010 x partition size).
    ICECHUNK_NUM_CHUNK_REFS: int = 5_000_000

    # Freguency in days to run garbage collection.
    GARBAGE_COLLECTION_FREQUENCY: int | None = None

    VPC_ID: str | None = None
    # AWS Batch cluster reference to SSM parameter describing the AMI _or_ the AMI ID
    # If using SSM to resolve the AMI ID, prefix with `resolve:ssm`.
    # MCP_AMI_ID: str = "resolve:ssm:/mcp/amis/aml2023-ecs"
    AMI_ID: str = (
        "resolve:ssm:/aws/service/ecs/optimized-ami/amazon-linux-2/recommended/image_id"
    )

    # Cluster scaling max
    BATCH_MAX_VCPU: int = 10

    # Backfill (partitioned fork/merge) pipeline
    BACKFILL_ENABLED: bool = False
    # Bucket for backfill artifacts: the inventory, the partition manifests and
    # the pickled forks. Separate from the Icechunk bucket because
    # StepFunctions' S3JsonItemReader takes no region and assumes the stack's
    # own, so it cannot read manifests out of a bucket in another region. A
    # bucket this stack creates is always in-region; adopt an existing one with
    # BACKFILL_BUCKET only if you know it is local to the deployment.
    # S3 bucket names are globally unique, so BACKFILL_BUCKET_NAME has to be
    # changed from its default for a fresh deploy to succeed.
    BACKFILL_BUCKET_NAME: str = "backfill-artifacts"
    BACKFILL_BUCKET: str | None = None
    BACKFILL_PARTITION_SIZE: int = 500
    BACKFILL_MAX_ITEMS_PER_BATCH: int = 10
    BACKFILL_MAX_CONCURRENCY: int = 50

    # Forward SQS consumer. `None` resolves in the validator below:
    #   backfill enabled  -> default disabled (bootstrap via backfill, enable later)
    #   backfill disabled -> default enabled  (normal forward-only deployment)
    FORWARD_QUEUE_ENABLED: bool | None = None

    @model_validator(mode="after")
    def _resolve_forward_queue_enabled(self) -> "StackSettings":
        if self.FORWARD_QUEUE_ENABLED is None:
            self.FORWARD_QUEUE_ENABLED = not self.BACKFILL_ENABLED
        return self
