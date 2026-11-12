"""Asset keys and the URL forms an HLS asset takes.

``asset_key`` names an object inside the LP DAAC protected bucket. ``read_url``
gives the URL a reader fetches it from; ``reference_url`` gives the endpoint-free
``vcc://`` form recorded in a manifest.
"""

from __future__ import annotations

from typing import Literal

from virtualizarr_processor.hls.granule import Granule

LPDAAC_S3_PREFIX = "s3://lp-prod-protected/"
LPDAAC_HTTPS_ASSET_PREFIX = (
    "https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/"
)
CONTAINER_NAME = "lpdaac"


def asset_key(granule: Granule, band: str) -> str:
    """The bucket key of one band's COG, e.g. ``HLSL30.020/{id}/{id}.B04.tif``."""
    return f"HLSL30.020/{granule.id}/{granule.id}.{band}.tif"


def read_url(key: str, access: Literal["s3", "https"]) -> str:
    """The URL to read ``key`` from, via in-region S3 or public HTTPS."""
    if access == "s3":
        return f"{LPDAAC_S3_PREFIX}{key}"
    if access == "https":
        return f"{LPDAAC_HTTPS_ASSET_PREFIX}{key}"
    raise ValueError(f"unknown access mode: {access!r}")


def reference_url(key: str) -> str:
    """The endpoint-free ``vcc://`` reference recorded in a manifest."""
    return f"vcc://{CONTAINER_NAME}/{key}"
