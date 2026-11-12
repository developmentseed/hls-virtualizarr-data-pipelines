"""Parse HLS granule identifiers and the keys/URLs that name them."""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from urllib.parse import unquote

_GRANULE_ID = re.compile(
    r"HLS\.(L30|S30)\.(T\d{2}[A-Z]{3})\.(\d{7})T(\d{6})\.v2\.0"
)

BANDS: dict[str, tuple[str, ...]] = {
    "L30": (
        "B01",
        "B02",
        "B03",
        "B04",
        "B05",
        "B06",
        "B07",
        "B09",
        "B10",
        "B11",
        "Fmask",
        "SZA",
        "SAA",
        "VZA",
        "VAA",
    ),
}


@dataclass(frozen=True)
class Granule:
    id: str
    product: str
    tile: str
    time: datetime


def parse_granule_id(granule_id: str) -> Granule:
    """Parse a bare HLS granule ID such as ``HLS.L30.T13SDA.2025001T173012.v2.0``."""
    match = _GRANULE_ID.fullmatch(granule_id)
    if match is None:
        raise ValueError(f"not an HLS v2.0 granule id: {granule_id!r}")
    product, tile, day_part, time_part = match.groups()
    year, doy = int(day_part[:4]), int(day_part[4:])
    days_in_year = 366 if calendar.isleap(year) else 365
    if not 1 <= doy <= days_in_year:
        raise ValueError(f"day of year {doy} out of range for {year}: {granule_id!r}")
    hour, minute, second = int(time_part[:2]), int(time_part[2:4]), int(time_part[4:])
    time = datetime(year, 1, 1, hour, minute, second, tzinfo=UTC) + timedelta(
        days=doy - 1
    )
    return Granule(id=granule_id, product=product, tile=tile, time=time)


def granule_from_key(key: str) -> Granule:
    """Parse a granule from a bare ID, an asset key, or an s3/https asset URL.

    The asset's band suffix (``.B04.tif``) is stripped; URL-encoded keys are
    decoded first.
    """
    name = unquote(key).rsplit("/", 1)[-1]
    if name.endswith(".tif"):
        name, _, _band = name.removesuffix(".tif").rpartition(".")
    return parse_granule_id(name)
