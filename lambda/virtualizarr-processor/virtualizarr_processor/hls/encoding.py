"""The documented encoding of every HLS band, embedded rather than inferred.

The cube used to copy each array's fill value from the newest granule. LP DAAC
COGs processed from about 2026-05-02 on dropped the nodata and scale tags from
the four angle bands (their pixels still use 40000 for no-data, stored x100),
so that granule declared fill 0 and every older granule then failed the grid
check. The values below are the HLS v2.0 documented ones, confirmed against
T13SDA COGs processed before that change (2026114) and after it (2026120,
2026145): reflectance bands int16, fill -9999, scale 0.0001; thermal bands
(B10, B11) int16, fill -9999, scale 0.01; Fmask uint8, fill 255, unscaled bit
flags; angle bands uint16, fill 40000, scale 0.01.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import xarray as xr
from zarr.core.metadata import ArrayV3Metadata

from virtualizarr_processor.hls.pyramid import Pyramid


@dataclass(frozen=True)
class BandEncoding:
    """How one band's integers are stored and how to decode them."""

    dtype: str
    fill_value: int
    #: ``None`` for bands that are not scaled (Fmask's bit flags).
    scale_factor: float | None = None
    add_offset: float = 0.0

    def attributes(self) -> dict[str, Any]:
        """CF attributes for the cube's array. xarray masks zarr v3 fill values
        only through ``_FillValue``; the zarr ``fill_value`` alone is not enough."""
        attrs: dict[str, Any] = {"_FillValue": self.fill_value}
        if self.scale_factor is not None:
            attrs["scale_factor"] = self.scale_factor
            attrs["add_offset"] = self.add_offset
        return attrs


_REFLECTANCE = BandEncoding("int16", -9999, 0.0001)
_THERMAL = BandEncoding("int16", -9999, 0.01)
_FMASK = BandEncoding("uint8", 255)
_ANGLE = BandEncoding("uint16", 40000, 0.01)

BAND_ENCODINGS: dict[str, dict[str, BandEncoding]] = {
    "L30": {
        **{
            band: _REFLECTANCE
            for band in ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B09")
        },
        "B10": _THERMAL,
        "B11": _THERMAL,
        "Fmask": _FMASK,
        **{band: _ANGLE for band in ("SAA", "SZA", "VAA", "VZA")},
    },
}

#: The fill value VirtualTIFF reports when a COG carries no nodata tag.
UNTAGGED_FILL = 0


def _with_fill(level: xr.Dataset, fill_value: int) -> xr.Dataset:
    (name,) = list(level.data_vars)
    variable = level[name].variable
    array = variable.data
    metadata = array.metadata.to_dict()
    metadata["fill_value"] = fill_value
    restamped = type(array)(
        chunkmanifest=array.manifest,
        metadata=ArrayV3Metadata.from_dict(metadata),
    )
    return level.assign(
        {name: xr.Variable(variable.dims, restamped, attrs=variable.attrs)}
    )


def restore_untagged_fill(
    encodings: Mapping[str, BandEncoding], pyramids: dict[str, Pyramid]
) -> dict[str, Pyramid]:
    """Give back the documented fill to bands whose COG lost its nodata tag.

    Only a level that has the documented dtype and reports ``UNTAGGED_FILL``
    (while the documented fill is something else) is changed: the fill value is
    array metadata, so the chunk bytes it references are untouched. Any other
    disagreement is left for the grid check to reject.
    """
    out: dict[str, Pyramid] = {}
    for band, pyramid in pyramids.items():
        encoding = encodings.get(band)
        if encoding is None or encoding.fill_value == UNTAGGED_FILL:
            out[band] = pyramid
            continue
        out[band] = {
            level: _with_fill(ds, encoding.fill_value)
            if _untagged(ds, encoding)
            else ds
            for level, ds in pyramid.items()
        }
    return out


def _untagged(level: xr.Dataset, encoding: BandEncoding) -> bool:
    (name,) = list(level.data_vars)
    metadata = level[name].data.metadata
    return bool(
        metadata.data_type.to_native_dtype() == np.dtype(encoding.dtype)
        and np.asarray(metadata.fill_value) == UNTAGGED_FILL
    )
