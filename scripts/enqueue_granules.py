"""Send one S3-notification-shaped SQS message per granule (Fmask asset).

    uv run python scripts/enqueue_granules.py --queue-url URL \
        [IDS... | --from inventory.json] [--since YYYY-MM-DD]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from virtualizarr_processor.hls.granule import parse_granule_id
from virtualizarr_processor.hls.inventory import fmask_message


def main(argv: list[str] | None = None, *, sqs_client=None) -> int:  # noqa: ANN001
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--queue-url", required=True)
    p.add_argument("ids", nargs="*")
    p.add_argument("--from", dest="from_file", type=Path)
    p.add_argument("--since", help="YYYY-MM-DD; skip granules acquired before")
    args = p.parse_args(argv)

    ids = list(args.ids)
    if args.from_file:
        ids += json.loads(args.from_file.read_text())["ids"]
    if not ids:
        p.error("give granule IDs or --from inventory.json")
    granules = [parse_granule_id(i) for i in ids]
    if args.since:
        since = datetime.strptime(args.since, "%Y-%m-%d").replace(tzinfo=UTC)
        granules = [g for g in granules if g.time >= since]

    if sqs_client is None:
        import boto3

        sqs_client = boto3.client("sqs")
    for i in range(0, len(granules), 10):
        batch = granules[i : i + 10]
        resp = sqs_client.send_message_batch(
            QueueUrl=args.queue_url,
            Entries=[
                {"Id": str(n), "MessageBody": json.dumps(fmask_message(g))}
                for n, g in enumerate(batch)
            ],
        )
        if resp.get("Failed"):
            raise RuntimeError(f"SQS rejected messages: {resp['Failed']}")
    print(f"enqueued {len(granules)} granules")
    return 0


if __name__ == "__main__":
    sys.exit(main())
