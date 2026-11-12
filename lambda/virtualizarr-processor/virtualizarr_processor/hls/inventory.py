"""Build a tile's granule inventory from the HLS stac-geoparquet archive.

The archive holds one parquet file per product-month. Only the ``id`` column is
read: the 2013 files lack the ``proj:*`` columns and carry a wrong ``proj:epsg``,
so nothing but the granule ID can be trusted across the whole record.
"""

from __future__ import annotations

import shutil
import tempfile
import time
import urllib.error
import urllib.request
from collections import defaultdict
from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from virtualizarr_processor.hls.assets import asset_key
from virtualizarr_processor.hls.granule import Granule, parse_granule_id

ARCHIVE_BUCKET = "nasa-maap-data-store"
ARCHIVE_REGION = "us-west-2"
ARCHIVE_PREFIX = "file-staging/nasa-map/hls-stac-geoparquet-archive/v2"
ARCHIVE_COLLECTIONS = {"L30": "HLSL30_2.0"}
LPDAAC_BUCKET = "lp-prod-protected"


class MonthMissing(Exception):
    """A month's archive file does not exist (HTTP 404)."""


def _months(start: date, end: date) -> list[tuple[int, int]]:
    out = []
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        out.append((y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def month_uris(
    product: str, start: date, end: date, *, https: bool = True
) -> list[str]:
    """One archive parquet URI per month from ``start`` to ``end`` inclusive."""
    try:
        collection = ARCHIVE_COLLECTIONS[product]
    except KeyError:
        raise ValueError(f"unknown product: {product!r}") from None
    base = (
        f"https://{ARCHIVE_BUCKET}.s3.{ARCHIVE_REGION}.amazonaws.com/"
        if https
        else f"s3://{ARCHIVE_BUCKET}/"
    )
    return [
        f"{base}{ARCHIVE_PREFIX}/{collection}/year={y}/month={m}/"
        f"{collection}-{y}-{m}.parquet"
        for y, m in _months(start, end)
    ]


def select_tile(table: pa.Table, tile: str, product: str) -> list[str]:
    """IDs for ``tile``/``product``, deduplicated and sorted by acquisition time.

    IDs that do not parse as HLS v2.0 granule IDs are dropped.
    """
    return select_tile_counted(table, tile, product)[0]


def select_tile_counted(
    table: pa.Table, tile: str, product: str
) -> tuple[list[str], int]:
    """Like ``select_tile`` but also returns how many IDs were unparseable."""
    granules: dict[str, Granule] = {}
    unparsed = 0
    for gid in table["id"].to_pylist():
        try:
            g = parse_granule_id(gid)
        except (ValueError, TypeError):
            unparsed += 1
            continue
        if g.tile == tile and g.product == product:
            granules[g.id] = g
    ordered = sorted(granules.values(), key=lambda g: (g.time, g.id))
    return [g.id for g in ordered], unparsed


def check_unique_times(ids: list[str]) -> None:
    """Raise ``ValueError`` naming every group of IDs sharing an acquisition time."""
    groups: dict[datetime, list[str]] = defaultdict(list)
    for gid in ids:
        groups[parse_granule_id(gid).time].append(gid)
    collisions = {t: g for t, g in groups.items() if len(g) > 1}
    if collisions:
        lines = [
            f"{t.strftime('%Y%jT%H%M%S')}: {', '.join(g)}"
            for t, g in sorted(collisions.items())
        ]
        raise ValueError("granules share an acquisition time:\n" + "\n".join(lines))


class ArchiveForbidden(Exception):
    """The archive refused access to a month file (HTTP 403)."""


FETCH_ATTEMPTS = 3
FETCH_TIMEOUT = 60


def _download(uri: str, dest: Path) -> None:
    """Stream ``uri`` to ``dest``, retrying 5xx/connection/timeout errors."""
    for attempt in range(1, FETCH_ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(uri, timeout=FETCH_TIMEOUT) as resp:
                with open(dest, "wb") as out:
                    shutil.copyfileobj(resp, out)
            return
        except urllib.error.HTTPError as err:
            if err.code == 404:
                raise MonthMissing(uri) from err
            if err.code == 403:
                raise ArchiveForbidden(
                    f"403 forbidden fetching {uri}; check access/auth"
                ) from err
            if err.code < 500 or attempt == FETCH_ATTEMPTS:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == FETCH_ATTEMPTS:
                raise
        time.sleep(2**attempt)


def read_month_ids(uri: str) -> pa.Table:
    """Fetch one month's parquet over HTTPS (anonymous); read only ``id``."""
    with tempfile.TemporaryDirectory() as tmp:
        local = Path(tmp) / "month.parquet"
        _download(uri, local)
        return pq.read_table(local, columns=["id"])


def build_inventory(
    tile: str,
    product: str,
    start: date,
    end: date,
    *,
    reader: Callable[[str], pa.Table] = read_month_ids,
    today: date | None = None,
    warn: Callable[[str], None] = lambda msg: None,
) -> list[str]:
    """Collect the tile's sorted, unique-time granule IDs over ``start``..``end``.

    A missing month file is a warning only for the ``end`` month when that is
    the current UTC month (the archive lags); anywhere else it is fatal.
    """
    today = today or datetime.now(UTC).date()
    found: dict[str, None] = {}
    unparsed = 0
    for (y, m), uri in zip(
        _months(start, end), month_uris(product, start, end), strict=True
    ):
        try:
            table = reader(uri)
        except MonthMissing:
            if (y, m) == (end.year, end.month) == (today.year, today.month):
                warn(f"{y}-{m}: month file not yet published, skipped ({uri})")
                continue
            raise
        ids, bad = select_tile_counted(table, tile, product)
        unparsed += bad
        found.update(dict.fromkeys(ids))
    if unparsed:
        warn(f"dropped {unparsed} unparseable ids")
    ids = sorted(found, key=lambda i: (parse_granule_id(i).time, i))
    check_unique_times(ids)
    return ids


def fmask_message(granule: Granule) -> dict:
    """An S3-notification-shaped message naming the granule's Fmask asset."""
    return {
        "Records": [
            {
                "s3": {
                    "bucket": {"name": LPDAAC_BUCKET},
                    "object": {"key": asset_key(granule, "Fmask")},
                }
            }
        ]
    }
