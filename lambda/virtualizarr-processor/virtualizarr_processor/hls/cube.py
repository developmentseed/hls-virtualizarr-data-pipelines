"""The per-tile cube: declaring its arrays and writing granules into them.

Layout: ``/{tile}/{band}/multiscales/{level}`` holds the data array ``{band}``
with dims ``(time, y, x)`` plus the ``time``, ``y`` and ``x`` coordinates.
Every level group of a tile shares one time axis. The recipe follows
docs/superpowers/spikes/2026-11-11-time-stack-findings.md section 3.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from datetime import UTC, datetime
from typing import Any, Literal

import numpy as np
import xarray as xr
import zarr
from virtualizarr.manifests import ChunkManifest, ManifestArray
from zarr.core.metadata import ArrayV3Metadata

from virtualizarr_processor.hls.assets import asset_key, reference_url
from virtualizarr_processor.hls.conventions import (
    MULTISCALES_GROUP,
    band_group_attrs,
)
from virtualizarr_processor.hls.granule import Granule
from virtualizarr_processor.hls.grid import TileGrid, level_coords
from virtualizarr_processor.hls.pyramid import (
    GridMismatch,
    Pyramid,
    PyramidOpener,
    check_level,
)

#: Set only at declaration: region and append writes reuse the stored encoding,
#: and xarray rejects encoding on an existing variable in append mode.
TIME_ENCODING: dict[str, str] = {
    "units": "seconds since 1970-01-01T00:00:00",
    "calendar": "proleptic_gregorian",
    "dtype": "int64",
}


def level_group(tile: str, band: str, level: int) -> str:
    """The group holding one band's resolution level."""
    return f"{tile}/{band}/{MULTISCALES_GROUP}/{level}"


def _datetime64(times: Iterable[datetime]) -> np.ndarray:
    """Naive-UTC ``datetime64[ns]`` from (possibly aware) datetimes."""
    naive = [
        t.astimezone(UTC).replace(tzinfo=None) if t.tzinfo is not None else t
        for t in times
    ]
    return np.array(naive, dtype="datetime64[ns]")


def _child_groups(group: zarr.Group) -> list[str]:
    return sorted(name for name, _ in group.groups())


def level_groups(store: Any, tile: str) -> list[str]:
    """Every existing level group under ``/{tile}``, sorted."""
    try:
        root = zarr.open_group(store, path=tile, mode="r", zarr_format=3)
    except (FileNotFoundError, zarr.errors.GroupNotFoundError):
        return []
    out = []
    for band in _child_groups(root):
        band_group = root[band]
        if MULTISCALES_GROUP not in band_group:
            continue
        for level in _child_groups(band_group[MULTISCALES_GROUP]):
            out.append(f"{tile}/{band}/{MULTISCALES_GROUP}/{level}")
    return sorted(out)


def _declared_levels(store: Any, tile: str) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for group in level_groups(store, tile):
        _, band, _, level = group.split("/")
        out.setdefault(band, []).append(int(level))
    return {band: sorted(levels) for band, levels in out.items()}


def declare_cube(
    store: Any,
    grid: TileGrid,
    times: Sequence[datetime],
    reference: dict[str, Pyramid],
) -> None:
    """Declare every (band, level) of ``reference`` as an empty ``(time, y, x)`` array.

    Array metadata (dtype, fill, codecs, chunk shape per level) is copied from
    the reference granule's levels, with a leading ``time`` axis chunked by 1
    and no chunks written. ``y``/``x`` come from ``grid`` only (never the
    granule's CRS). Each band group gets its convention attributes.
    """
    tile = grid.mgrs_id
    time_values = _datetime64(times)
    for band, pyramid in reference.items():
        for level, level_ds in sorted(pyramid.items()):
            (name,) = list(level_ds.data_vars)
            src: ManifestArray = level_ds[name].data
            ny, nx = src.metadata.shape
            cy, cx = src.metadata.chunks
            spec = src.metadata.to_dict()
            spec["shape"] = [len(time_values), ny, nx]
            spec["chunk_grid"] = {
                "name": "regular",
                "configuration": {"chunk_shape": [1, cy, cx]},
            }
            spec["dimension_names"] = ["time", "y", "x"]
            spec["attributes"] = {}
            gy, gx = src.manifest.shape_chunk_grid
            empty = ManifestArray(
                chunkmanifest=ChunkManifest({}, shape=(len(time_values), gy, gx)),
                metadata=ArrayV3Metadata.from_dict(spec),
            )
            y, x = level_coords(grid, ny, nx)
            cube = xr.Dataset(
                {band: xr.Variable(("time", "y", "x"), empty)},
                coords={"time": ("time", time_values), "y": ("y", y), "x": ("x", x)},
            )
            cube["time"].encoding.update(TIME_ENCODING)
            cube.vz.to_icechunk(store, group=level_group(tile, band, level))
        zarr.open_group(store, path=f"{tile}/{band}", mode="a").attrs.update(
            band_group_attrs(grid, levels=len(pyramid), band=band)
        )


def _without_dimension_names(array: ManifestArray) -> ManifestArray:
    """Drop ``dimension_names`` so prepending ``time`` cannot leave them stale.

    virtualizarr's ``expand_dims`` keeps the 2-d names on 3-d metadata and
    fails validation; the written names come from the xarray dims anyway.
    """
    if array.metadata.dimension_names is None:
        return array
    spec = array.metadata.to_dict()
    spec.pop("dimension_names", None)
    return ManifestArray(
        chunkmanifest=array.manifest, metadata=ArrayV3Metadata.from_dict(spec)
    )


def granule_levels(
    granule: Granule,
    band: str,
    pyramid: Pyramid,
    *,
    reference_url_for: Callable[[str], str] = reference_url,
) -> dict[int, xr.Dataset]:
    """One granule band's levels as ``(time=1, y, x)`` datasets ready to write.

    The variable is renamed to ``band``, every other variable dropped, chunk
    references rewritten to ``reference_url_for(asset_key(granule, band))``,
    ``time`` prepended from the granule ID, and attrs cleared (to_icechunk
    stamps dataset attrs onto the group on every write). No time encoding is
    set: the declared one is reused.
    """
    url = reference_url_for(asset_key(granule, band))
    time = _datetime64([granule.time])
    out: dict[int, xr.Dataset] = {}
    for level, level_ds in sorted(pyramid.items()):
        (name,) = list(level_ds.data_vars)
        ds = level_ds.rename({name: band})
        ds = ds.drop_vars([v for v in ds.variables if v != band])
        ds[band] = ds[band].copy(data=_without_dimension_names(ds[band].data))
        ds = ds.vz.rename_paths(url)
        ds = ds.expand_dims(time=time)
        ds.attrs = {}
        ds[band].attrs = {}
        out[level] = ds
    return out


def open_granule(
    granule: Granule,
    bands: Iterable[str],
    opener: PyramidOpener,
    registry: Any,
    read_url_for: Callable[[str], str],
) -> dict[str, Pyramid]:
    """Open every band's pyramid; any failure raises before anything is written."""
    return {
        band: opener(read_url_for(asset_key(granule, band)), registry) for band in bands
    }


def write_granule(
    store: Any,
    tile: str,
    granule: Granule,
    pyramids: dict[str, Pyramid],
    *,
    mode: Literal["region", "append"],
    reference_url_for: Callable[[str], str] = reference_url,
) -> None:
    """Write one granule into every declared (band, level) of ``tile``.

    Every declared array is checked against the granule first -- the granule
    must supply exactly the declared bands and levels, each matching per
    ``check_level`` -- so a mismatch raises ``GridMismatch`` with nothing
    written. ``mode="region"`` fills an already-declared time step;
    ``"append"`` extends the time axis. ``reference_url_for`` maps an asset
    key to the URL recorded in the manifest.
    """
    declared = _declared_levels(store, tile)
    if set(pyramids) != set(declared):
        raise GridMismatch(
            f"bands: declared {sorted(declared)}, granule {sorted(pyramids)}"
        )
    for band, levels in declared.items():
        if sorted(pyramids[band]) != levels:
            raise GridMismatch(
                f"{band} levels: declared {levels}, granule {sorted(pyramids[band])}"
            )
        for level in levels:
            array = zarr.open_array(
                store, path=f"{level_group(tile, band, level)}/{band}", mode="r"
            )
            try:
                check_level(array, pyramids[band][level])
            except GridMismatch as err:
                raise GridMismatch(f"{band} level {level}: {err}") from err

    kwargs: dict[str, Any] = (
        {"region": "auto"} if mode == "region" else {"append_dim": "time"}
    )
    for band in sorted(declared):
        levels = granule_levels(
            granule, band, pyramids[band], reference_url_for=reference_url_for
        )
        for level, ds in levels.items():
            ds.vz.to_icechunk(
                store,
                group=level_group(tile, band, level),
                validate_containers=False,
                **kwargs,
            )


def store_times(store: Any, tile: str) -> np.ndarray | None:
    """The decoded ``time`` axis of ``tile``, or ``None`` if nothing is declared."""
    groups = level_groups(store, tile)
    if not groups:
        return None
    ds = xr.open_zarr(store, group=groups[0], consolidated=False, zarr_format=3)
    return ds["time"].values
