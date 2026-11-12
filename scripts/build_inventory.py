"""Build DIR/inventory.json: every granule ID for one tile/product in a month range.

    uv run python scripts/build_inventory.py --tile T13SDA --product L30 \
        --start 2013-04 --end 2025-11 --out build/ [--upload s3://bucket/prefix/]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from datetime import date
from pathlib import Path

import pyarrow as pa
from virtualizarr_processor.hls.granule import parse_granule_id
from virtualizarr_processor.hls.inventory import build_inventory, read_month_ids


def _month(text: str) -> date:
    year, month = text.split("-")
    return date(int(year), int(month), 1)


def main(
    argv: list[str] | None = None,
    *,
    reader: Callable[[str], pa.Table] = read_month_ids,
    today: date | None = None,
    s3_client=None,  # noqa: ANN001
) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tile", required=True)
    p.add_argument("--product", required=True)
    p.add_argument("--start", required=True, type=_month, help="YYYY-MM")
    p.add_argument("--end", required=True, type=_month, help="YYYY-MM")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--upload", help="s3://bucket/prefix/ to put inventory.json under")
    args = p.parse_args(argv)

    ids = build_inventory(
        args.tile,
        args.product,
        args.start,
        args.end,
        reader=reader,
        today=today,
        warn=lambda m: print(f"warning: {m}", file=sys.stderr),
    )
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "inventory.json"
    path.write_text(json.dumps(ids, indent=2))
    print(f"{args.tile} {args.product}")
    if ids:
        first, last = parse_granule_id(ids[0]).time, parse_granule_id(ids[-1]).time
        print(f"{len(ids)} granules, {first.isoformat()} to {last.isoformat()}")
    else:
        print("0 granules")

    if args.upload:
        import boto3

        bucket, _, prefix = args.upload.removeprefix("s3://").partition("/")
        key = f"{prefix.rstrip('/')}/inventory.json".lstrip("/")
        (s3_client or boto3.client("s3")).put_object(
            Bucket=bucket, Key=key, Body=path.read_bytes()
        )
        print(f"uploaded s3://{bucket}/{key}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
