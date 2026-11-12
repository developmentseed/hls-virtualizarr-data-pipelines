"""Handler: write one file-batch's virtual refs into a child fork on S3."""

import pickle
import uuid
from typing import Any

from aws_lambda_powertools import Logger, Tracer
from aws_lambda_powertools.utilities.typing import LambdaContext
from virtualizarr_processor.processor import Processor

from backfill_handlers import fork_store, inventory

logger = Logger()
tracer = Tracer()


@logger.inject_lambda_context()
@tracer.capture_lambda_handler
def handler(event: dict[str, Any], context: LambdaContext) -> dict[str, Any]:
    processor = Processor()
    shared = pickle.loads(fork_store.load_fork(event["fork_in_uri"]))
    child = shared.fork()
    for file_key in event["file_keys"]:
        if processor.process_backfill_file(file_key, child):
            continue
        # A "write" stage failure may have left the child fork partially
        # written; raise (without saving) so Step Functions retries the batch
        # with a fresh child fork. Other stages wrote nothing and are recorded.
        if processor.skipped and processor.skipped[-1].stage == "write":
            logger.error("Write failed", extra={"file_key": file_key})
            raise RuntimeError(f"process_backfill_file write failed for {file_key}")
        logger.warning("Skipped file", extra={"file_key": file_key})

    stem = uuid.uuid4().hex
    if processor.skipped:
        inventory.write_json(
            f"{event['skipped_out_prefix']}{stem}.json",
            [r.to_dict() for r in processor.skipped],
        )
    child_fork_uri = f"{event['forks_out_prefix']}{stem}.pkl"
    fork_store.save_fork(child_fork_uri, pickle.dumps(child))
    logger.info("Wrote child fork", extra={"child_fork_uri": child_fork_uri})
    return {"child_fork_uri": child_fork_uri}
