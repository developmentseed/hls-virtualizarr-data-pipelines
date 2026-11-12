"""Test fakes: local COG stand-ins, a pyramid opener over them, and a locator."""

from __future__ import annotations

import ast
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import icechunk
import numpy as np
import xarray as xr
import zarr
from virtualizarr.manifests import ChunkManifest, ManifestArray
from virtualizarr_processor.hls.assets import asset_key
from virtualizarr_processor.hls.granule import BANDS, parse_granule_id
from virtualizarr_processor.hls.grid import TileGrid
from virtualizarr_processor.hls.pyramid import Pyramid
from zarr.codecs import BytesCodec, ZstdCodec
from zarr.core.dtype import parse_data_type
from zarr.core.metadata import ArrayV3Metadata

GID = "HLS.L30.T13SDA.2025152T173012.v2.0"
GID2 = "HLS.L30.T13SDA.2025168T173012.v2.0"

FAKE_GRID = TileGrid(
    "T13SDA",
    32613,
    (8, 8),
    (30.0, 0.0, 399960.0, 0.0, -30.0, 4100040.0, 0.0, 0.0, 1.0),
)

_KEY = re.compile(r"HLSL30\.020/(?P<id>[^/]+)/(?P=id)\.(?P<band>[^./]+)\.tif$")


def url_for(granule_id: str, band: str) -> str:
    """The asset key of a fake COG; the opener accepts this or any locator URL."""
    return asset_key(parse_granule_id(granule_id), band)


@dataclass(frozen=True)
class LocalLocator:
    """Read URL and reference URL are the same ``file://`` URL."""

    root: Path

    def read_url(self, key: str) -> str:
        return f"file://{self.root}/{key}"

    def reference_url(self, key: str) -> str:
        return f"file://{self.root}/{key}"


def local_locator(root: Path) -> LocalLocator:
    return LocalLocator(Path(root))


class FakeAssets:
    """Raw-chunk files on disk laid out like the LP DAAC keys, plus an opener."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self._missing: set[tuple[str, str]] = set()
        self.opened: list[str] = []

    def path(self, granule_id: str, band: str) -> Path:
        return self.root / url_for(granule_id, band)

    def write(
        self,
        granule_id: str,
        band: str,
        value: int,
        *,
        levels: tuple[tuple[int, int], ...] = ((8, 8), (4, 4)),
        chunks: tuple[int, int] = (4, 4),
        dtype: str = "uint16",
        fill_value: int = 0,
        compress: bool = False,
    ) -> None:
        """Write one asset: every level's chunks, each filled with ``value``.

        Chunks are stored back to back, level 0 first, row-major within a level.
        """
        cy, cx = chunks
        chunk = np.full((cy, cx), value, dtype=dtype).tobytes()
        n_chunks = sum(-(-ny // cy) * -(-nx // cx) for ny, nx in levels)
        path = self.path(granule_id, band)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(chunk * n_chunks)
        # sidecar so the opener can rebuild the layout without parsing a TIFF
        path.with_suffix(".layout").write_text(
            repr(
                {
                    "levels": levels,
                    "chunks": chunks,
                    "dtype": dtype,
                    "fill_value": fill_value,
                    "compress": compress,
                }
            )
        )
        self._missing.discard((granule_id, band))

    def missing(self, granule_id: str, band: str) -> None:
        self._missing.add((granule_id, band))

    def opener(self, url: str, registry: Any) -> Pyramid:
        m = _KEY.search(url)
        if m is None:
            raise ValueError(f"not a fake asset URL: {url}")
        gid, band = m["id"], m["band"]
        self.opened.append(url)
        if (gid, band) in self._missing:
            raise FileNotFoundError(url)
        path = self.path(gid, band)
        layout = ast.literal_eval(path.with_suffix(".layout").read_text())
        return _pyramid(f"file://{path}", **layout)


def _pyramid(path: str, levels, chunks, dtype, fill_value=0, compress=False) -> Pyramid:
    cy, cx = chunks
    data_type = parse_data_type(np.dtype(dtype), zarr_format=3)
    chunk_bytes = cy * cx * np.dtype(dtype).itemsize
    offset = 0
    out: Pyramid = {}
    for level, (ny, nx) in enumerate(levels):
        gy, gx = -(-ny // cy), -(-nx // cx)
        entries = {}
        for i in range(gy):
            for j in range(gx):
                entries[f"{i}.{j}"] = {
                    "path": path,
                    "offset": offset,
                    "length": chunk_bytes,
                }
                offset += chunk_bytes
        array = ManifestArray(
            chunkmanifest=ChunkManifest(entries, shape=(gy, gx)),
            metadata=ArrayV3Metadata(
                shape=(ny, nx),
                data_type=data_type,
                chunk_grid={
                    "name": "regular",
                    "configuration": {"chunk_shape": chunks},
                },
                chunk_key_encoding={"name": "default"},
                fill_value=data_type.to_json_scalar(fill_value, zarr_format=3),
                codecs=[BytesCodec(), ZstdCodec()] if compress else [BytesCodec()],
                attributes={},
                dimension_names=("y", "x"),
                storage_transformers=None,
            ),
        )
        # wrap as virtualizarr does: to_virtual_variable moves dimension_names
        # (and attributes) off the ManifestArray and onto the xarray Variable
        out[level] = xr.Dataset({"band_data": array.to_virtual_variable()})
    return out


def open_local_repo(repo_path: Path, assets_root: Path) -> icechunk.Repository:
    """Open (or create) a filesystem repo that authorizes ``file://{assets_root}/``."""
    Path(assets_root).mkdir(parents=True, exist_ok=True)
    prefix = f"file://{assets_root}/"
    config = icechunk.RepositoryConfig.default()
    config.set_virtual_chunk_container(
        icechunk.VirtualChunkContainer(
            prefix, icechunk.local_filesystem_store(str(assets_root))
        )
    )
    return icechunk.Repository.open_or_create(
        storage=icechunk.local_filesystem_storage(str(repo_path)),
        config=config,
        authorize_virtual_chunk_access={
            prefix: icechunk.credentials.LocalFileSystemAccess
        },
    )


def write_all_bands(
    fake_assets: FakeAssets, granule_id: str, value: int, **layout: Any
) -> None:
    """Write every L30 band of one granule, each filled with ``value``."""
    for band in BANDS["L30"]:
        fake_assets.write(granule_id, band, value, **layout)


@dataclass
class Backfill:
    """What the backfill-handler tests' ``hls_backfill`` fixture sets up."""

    inventory_uri: str
    granule_ids: list[str]
    processor: Callable[[], Any]


def declare_like(repo: icechunk.Repository, level_ds: xr.Dataset, n_time: int = 3):
    """A (n_time, ny, nx) zarr array, chunked (1, cy, cx), matching ``level_ds``."""
    (name,) = list(level_ds.data_vars)
    md = level_ds[name].data.metadata
    ny, nx = md.shape
    cy, cx = md.chunks
    session = repo.writable_session("main")
    return zarr.create_array(
        session.store,
        name="declared",
        shape=(n_time, ny, nx),
        chunks=(1, cy, cx),
        dtype=md.data_type.to_native_dtype(),
        fill_value=md.fill_value,
        serializer=md.codecs[0],
        compressors=[c for c in md.codecs[1:]] or None,
        filters=None,
        dimension_names=("time", "y", "x"),
        zarr_format=3,
    )
