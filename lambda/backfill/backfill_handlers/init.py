"""Handler: declare the full-shape backfill store from the inventory and commit."""

from typing import Any

from aws_lambda_powertools import Logger, Tracer
from aws_lambda_powertools.utilities.typing import LambdaContext
from virtualizarr_processor.processor import Processor

from backfill_handlers import inventory

logger = Logger()
tracer = Tracer()


@logger.inject_lambda_context()
@tracer.capture_lambda_handler
def handler(event: dict[str, Any], context: LambdaContext) -> dict[str, Any]:
    granule_ids = inventory.read_inventory(event["inventory_uri"])
    processor = Processor()
    repo = processor.open_backfill_repo()
    base_snapshot = processor.initialize_backfill_store(repo, granule_ids)
    logger.info(
        "Initialized backfill store",
        extra={"base_snapshot": base_snapshot, "granules": len(granule_ids)},
    )
    return {"base_snapshot": base_snapshot}
