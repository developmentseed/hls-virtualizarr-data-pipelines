"""Opening an HLS COG's resolution pyramid and checking it against a declared array."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import xarray as xr
import zarr

Pyramid = dict[int, xr.Dataset]
PyramidOpener = Callable[[str, Any], Pyramid]


class GridMismatch(Exception):
    """A granule level does not match the array declared for it."""


def open_pyramid(url: str, registry: Any) -> Pyramid:
    """Open every resolution level of one COG in a single read.

    ``ifd=None`` parses the whole IFD chain in one pass. ``ifd_layout`` must be
    ``"nested"`` so each IFD becomes a child group of an
    ``open_virtual_datatree`` result. Returns one single-variable ``(y, x)``
    dataset per level, level 0 first.
    """
    from virtual_tiff import VirtualTIFF
    from virtualizarr import open_virtual_datatree

    tree = open_virtual_datatree(
        url=url,
        registry=registry,
        parser=VirtualTIFF(ifd=None, ifd_layout="nested"),
    )
    levels = {int(key): node.to_dataset() for key, node in tree.children.items()}
    return dict(sorted(levels.items()))


def _codecs(codecs: Any) -> list[Any]:
    return [c.to_dict() if hasattr(c, "to_dict") else c for c in codecs]


def check_level(declared: zarr.Array, level: xr.Dataset) -> None:
    """Raise ``GridMismatch`` naming the field if ``level`` differs from ``declared``.

    ``declared`` has dims ``(time, y, x)``; the leading ``time`` axis is ignored.
    CRS is deliberately not compared: the grid comes only from ``TILE_GRIDS``.
    """
    (name,) = list(level.data_vars)
    got = level[name].data.metadata
    want = declared.metadata

    if tuple(want.shape[1:]) != tuple(got.shape):
        raise GridMismatch(f"shape: declared {want.shape[1:]}, granule {got.shape}")
    if tuple(declared.chunks[1:]) != tuple(got.chunks):
        raise GridMismatch(
            f"chunks: declared {declared.chunks[1:]}, granule {got.chunks}"
        )
    if want.data_type != got.data_type:
        raise GridMismatch(f"dtype: declared {want.data_type}, granule {got.data_type}")
    if not np.array_equal(
        np.asarray(want.fill_value), np.asarray(got.fill_value), equal_nan=True
    ):
        raise GridMismatch(
            f"fill value: declared {want.fill_value}, granule {got.fill_value}"
        )
    if _codecs(want.codecs) != _codecs(got.codecs):
        raise GridMismatch(
            f"codecs: declared {_codecs(want.codecs)}, granule {_codecs(got.codecs)}"
        )
