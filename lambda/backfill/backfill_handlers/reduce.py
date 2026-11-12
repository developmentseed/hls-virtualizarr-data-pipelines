"""Handler: merge all child forks for a partition into one commit."""

from typing import Any

from aws_lambda_powertools import Logger, Tracer
from aws_lambda_powertools.utilities.typing import LambdaContext
from virtualizarr_processor import backfill
from virtualizarr_processor.processor import Processor

from backfill_handlers import fork_store, inventory

logger = Logger()
tracer = Tracer()


def _stem(uri: str) -> str:
    return uri.rsplit("/", 1)[-1].rsplit(".", 1)[0]


@logger.inject_lambda_context()
@tracer.capture_lambda_handler
def handler(event: dict[str, Any], context: LambdaContext) -> dict[str, Any]:
    partition_id = event["partition_id"]
    processor = Processor()
    repo = processor.open_backfill_repo()

    child_uris = fork_store.list_forks(event["forks_out_prefix"])
    children = [fork_store.load_fork(uri) for uri in child_uris]
    tip = backfill.merge_and_commit(
        repo, children, message=f"Backfill partition {partition_id}"
    )

    # Only skip files whose fork was saved: a worker that crashed between its
    # skip put and its fork put is retried and writes a new pair under a new stem.
    fork_stems = {_stem(uri) for uri in child_uris if uri.endswith(".pkl")}
    by_granule: dict[str, dict[str, Any]] = {}
    for uri in fork_store.list_forks(event["skipped_out_prefix"]):
        if not uri.endswith(".json") or _stem(uri) not in fork_stems:
            continue
        for record in inventory.read_json(uri):
            by_granule.setdefault(record["granule_id"], record)
    skipped = [by_granule[g] for g in sorted(by_granule)]
    inventory.write_json(
        f"{event['run_prefix']}partitions/{partition_id}/skipped.json", skipped
    )

    logger.info(
        "Committed partition",
        extra={"partition_id": partition_id, "tip": tip, "skipped": len(skipped)},
    )
    return {"partition_id": partition_id, "tip": tip, "skipped": len(skipped)}
