"""Spike: prove virtual time-stacking of HLS L30 COGs (throwaway).

Run (rasterio is spike-only, for the independent direct-COG reads):

    uv run --with rasterio python spike/time_stack.py

Without rasterio (`uv run python spike/time_stack.py`) the script still runs
steps 1-5 but skips the direct-COG comparison and exits 2 ("UNVERIFIED").
Results are written to $SPIKE_CACHE/results.json (default ~/.cache/hls-spike).

Needs an Earthdata login in ~/.netrc. No AWS credentials are used: the
stac-geoparquet archive is read anonymously over https and the COGs over https
with an Earthdata bearer token.

Steps (see task-2 brief):
  1. inventory: T13SDA IDs from 2013-6 and 2025-6 archive months
  2. open every L30 band pyramid for 3 granules; compare 2013 vs 2025 encodings
  3. declare a 3-step (time, y, x) cube for B04 and Fmask, empty manifests
  4. region-write the 3 granules in reverse time order; read back; compare
  5. 2-step declare, region-write 2025s, append 2013, reindex sort, verify
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import icechunk
import numpy as np
import obstore
import pyarrow.compute as pc
import pyarrow.parquet as pq
import virtual_tiff  # noqa: F401  registers HorizontalDeltaCodec with zarr
import xarray as xr
import zarr
from earthaccess_auth import login
from obspec_utils.registry import ObjectStoreRegistry
from virtual_tiff import VirtualTIFF
from virtualizarr import open_virtual_datatree
from virtualizarr.manifests import ChunkManifest, ManifestArray
from zarr.core.metadata import ArrayV3Metadata

TILE = "T13SDA"
ARCHIVE = (
    "https://nasa-maap-data-store.s3.us-west-2.amazonaws.com/file-staging/nasa-map/"
    "hls-stac-geoparquet-archive/v2/HLSL30_2.0/year={y}/month={m}/HLSL30_2.0-{y}-{m}.parquet"
)
LPDAAC_HTTPS_ASSET_PREFIX = (
    "https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/"
)
CONTAINER = "lpdaac"
ALL_BANDS = "B01 B02 B03 B04 B05 B06 B07 B09 B10 B11 Fmask SZA SAA VZA VAA".split()
CUBE_BANDS = ("B04", "Fmask")
TIME_ENCODING = {
    "units": "seconds since 1970-01-01T00:00:00",
    "calendar": "proleptic_gregorian",
    "dtype": "int64",
}
# 3x3 pixel windows (row0, col0) per level for the read-back comparison.
# Default only; with rasterio available, `pick_windows` replaces it with a
# window that holds valid (non-fill) B04 data in *every* granule, so a missing
# chunk reference (which reads as fill) cannot masquerade as a match.
WINDOWS = {0: (1830, 1830), 1: (915, 915)}
FILL = {"B04": -9999, "Fmask": 255}

CACHE = Path(os.environ.get("SPIKE_CACHE", Path.home() / ".cache" / "hls-spike"))
RESULTS: dict[str, Any] = {}


# --------------------------------------------------------------------------- utils
def granule_time(gid: str) -> np.datetime64:
    stamp = gid.split(".")[3]
    dt = datetime.strptime(stamp, "%Y%jT%H%M%S").replace(tzinfo=timezone.utc)
    return np.datetime64(dt.replace(tzinfo=None), "ns")


def asset_key(gid: str, band: str) -> str:
    return f"HLSL30.020/{gid}/{gid}.{band}.tif"


def https_url(gid: str, band: str) -> str:
    return LPDAAC_HTTPS_ASSET_PREFIX + asset_key(gid, band)


def vcc_url(gid: str, band: str) -> str:
    return f"vcc://{CONTAINER}/{asset_key(gid, band)}"


def level_group(band: str, level: int) -> str:
    return f"{TILE}/{band}/multiscales/{level}"


def array_path(band: str, level: int) -> str:
    return f"/{level_group(band, level)}/{band}"


# --------------------------------------------------------------------------- step 1
def step1_inventory() -> dict[str, Any]:
    CACHE.mkdir(parents=True, exist_ok=True)
    out: dict[str, Any] = {}
    for year in (2013, 2025):
        url = ARCHIVE.format(y=year, m=6)
        local = CACHE / url.rsplit("/", 1)[1]
        if not local.exists():
            print(f"downloading {url}")
            urllib.request.urlretrieve(url, local)
        names = pq.read_schema(local).names
        cols = [
            c for c in ("id", "proj:epsg", "proj:shape", "proj:transform") if c in names
        ]
        table = pq.read_table(local, columns=cols)
        rows = (
            table.filter(pc.match_substring(table["id"], f".{TILE}."))
            .sort_by("id")
            .to_pylist()
        )
        out[year] = {"columns": cols, "rows": rows}
        print(f"[step1] {year}-6: columns={cols} {TILE} rows={len(rows)}")
        for r in rows:
            print("   ", r)
    rows13, rows25 = out[2013]["rows"], out[2025]["rows"]
    newest = rows25[-1]
    granules = [rows13[0]["id"], rows25[0]["id"], newest["id"]]
    RESULTS["step1"] = {
        "granules": granules,
        "all_2013": [r["id"] for r in rows13],
        "all_2025": [r["id"] for r in rows25],
        "epsg_2013": sorted({r.get("proj:epsg") for r in rows13}),
        "epsg_2025": sorted({r.get("proj:epsg") for r in rows25}),
        "columns_2013": out[2013]["columns"],
        "newest_2025": newest,
    }
    return RESULTS["step1"]


# --------------------------------------------------------------------------- step 2
def make_registry(token: str) -> ObjectStoreRegistry:
    store = obstore.store.from_url(
        LPDAAC_HTTPS_ASSET_PREFIX.rstrip("/"),
        client_options={"default_headers": {"Authorization": f"Bearer {token}"}},
    )
    return ObjectStoreRegistry({LPDAAC_HTTPS_ASSET_PREFIX: store})


def open_pyramid(gid: str, band: str, registry: Any) -> dict[int, xr.Dataset]:
    tree = open_virtual_datatree(
        url=https_url(gid, band),
        registry=registry,
        parser=VirtualTIFF(ifd=None, ifd_layout="nested"),
    )
    return {int(k): node.to_dataset() for k, node in tree.children.items()}


def describe_level(ds: xr.Dataset) -> dict[str, Any]:
    (name,) = list(ds.data_vars)
    ma: ManifestArray = ds[name].data
    md = ma.metadata.to_dict()
    return {
        "shape": list(md["shape"]),
        "chunks": list(md["chunk_grid"]["configuration"]["chunk_shape"]),
        "chunk_grid": list(ma.manifest.shape_chunk_grid),
        "n_chunks": int(np.prod(ma.manifest.shape_chunk_grid)),
        "dtype": str(md["data_type"]),
        "fill_value": md["fill_value"].item()
        if hasattr(md["fill_value"], "item")
        else md["fill_value"],
        "codecs": json.loads(json.dumps(list(md["codecs"]), default=str)),
    }


def step2_open(granules: list[str], registry: Any) -> dict[str, dict[str, dict]]:
    pyramids: dict[str, dict[str, dict[int, xr.Dataset]]] = {}
    table: dict[str, dict[str, dict[int, dict]]] = {}
    for gid in granules:
        pyramids[gid] = {}
        table[gid] = {}
        for band in ALL_BANDS:
            levels = open_pyramid(gid, band, registry)
            pyramids[gid][band] = levels
            table[gid][band] = {lv: describe_level(ds) for lv, ds in levels.items()}
        print(f"[step2] opened {gid}")
    # CRS sanity from geokeys of level 0 (B04)
    crs = {
        gid: {
            k: pyramids[gid]["B04"][0].attrs.get(k)
            for k in (
                "citation",
                "projection",
                "projected_type",
                "model_tiepoint",
                "model_pixel_scale",
            )
        }
        for gid in granules
    }
    ref = table[granules[-1]]
    diffs = []
    for gid in granules[:-1]:
        for band in ALL_BANDS:
            a, b = table[gid][band], ref[band]
            if sorted(a) != sorted(b):
                diffs.append((gid, band, "levels", sorted(a), sorted(b)))
                continue
            for lv in a:
                for field in ("shape", "chunks", "dtype", "fill_value", "codecs"):
                    if a[lv][field] != b[lv][field]:
                        diffs.append((gid, band, lv, field, a[lv][field], b[lv][field]))
    RESULTS["step2"] = {
        "table": {b: {str(lv): d for lv, d in ref[b].items()} for b in ALL_BANDS},
        "diffs": [list(map(str, d)) for d in diffs],
        "verdict": "SAME" if not diffs else "DIFFERENT",
        "crs_geokeys": crs,
        "chunks_per_granule": sum(
            d["n_chunks"] for b in ALL_BANDS for d in ref[b].values()
        ),
    }
    print(f"[step2] verdict {RESULTS['step2']['verdict']} diffs={diffs}")
    for band in ALL_BANDS:
        for lv, d in ref[band].items():
            print(f"   {band} L{lv} {d}")
    return pyramids


# --------------------------------------------------------------------------- repo
def open_repo(path: Path, token: str) -> icechunk.Repository:
    config = icechunk.RepositoryConfig.default()
    config.set_virtual_chunk_container(
        icechunk.VirtualChunkContainer(
            url_prefix=LPDAAC_HTTPS_ASSET_PREFIX,
            store=icechunk.http_store(headers={"Authorization": f"Bearer {token}"}),
            name=CONTAINER,
        )
    )
    return icechunk.Repository.open_or_create(
        icechunk.local_filesystem_storage(str(path)),
        config=config,
        authorize_virtual_chunk_access=icechunk.containers_credentials(
            {LPDAAC_HTTPS_ASSET_PREFIX: icechunk.Credentials.HttpAccess()}
        ),
    )


def level_coords(
    shape: tuple[int, int], transform: list[float]
) -> dict[str, np.ndarray]:
    """Pixel-centre x/y for a level, from the level-0 affine and tile extent."""
    ny, nx = shape
    a, _, c, _, e, f = transform[:6]
    width, height = 3660 * a, 3660 * e  # level-0 extent in metres
    rx, ry = width / nx, height / ny
    return {
        "x": c + (np.arange(nx) + 0.5) * rx,
        "y": f + (np.arange(ny) + 0.5) * ry,
    }


def declare(
    store: Any,
    reference: dict[str, dict[int, xr.Dataset]],
    times: np.ndarray,
    transform: list[float],
) -> None:
    """Init-style declaration: full-shape (time, y, x) arrays, empty manifests."""
    for band in CUBE_BANDS:
        for level, ds in sorted(reference[band].items()):
            (name,) = list(ds.data_vars)
            src: ManifestArray = ds[name].data
            ny, nx = src.metadata.shape
            cy, cx = src.metadata.chunks
            spec = src.metadata.to_dict()
            spec["shape"] = [len(times), ny, nx]
            spec["chunk_grid"] = {
                "name": "regular",
                "configuration": {"chunk_shape": [1, cy, cx]},
            }
            spec["dimension_names"] = ["time", "y", "x"]
            spec["attributes"] = {}
            gy, gx = src.manifest.shape_chunk_grid
            empty = ManifestArray(
                chunkmanifest=ChunkManifest({}, shape=(len(times), gy, gx)),
                metadata=ArrayV3Metadata.from_dict(spec),
            )
            coords = level_coords((ny, nx), transform)
            cube = xr.Dataset(
                {band: xr.Variable(("time", "y", "x"), empty)},
                coords={
                    "time": ("time", times),
                    "y": ("y", coords["y"]),
                    "x": ("x", coords["x"]),
                },
            )
            cube["time"].encoding.update(TIME_ENCODING)
            cube.vz.to_icechunk(store, group=level_group(band, level))


def granule_level_ds(ds: xr.Dataset, band: str, gid: str) -> xr.Dataset:
    """A granule level as a (time=1, y, x) dataset ready for region/append."""
    (name,) = list(ds.data_vars)
    out = ds.rename({name: band})
    out = out.drop_vars([v for v in out.variables if v != band])
    out = out.vz.rename_paths(vcc_url(gid, band))
    out = out.expand_dims(time=[granule_time(gid)])
    out.attrs = {}  # group attrs would be overwritten by each write otherwise
    out[band].attrs = {}
    return out


def write_granule(
    store: Any, pyr: dict[str, dict[int, xr.Dataset]], gid: str, **kwargs: Any
) -> None:
    for band in CUBE_BANDS:
        for level, ds in sorted(pyr[band].items()):
            granule_level_ds(ds, band, gid).vz.to_icechunk(
                store,
                group=level_group(band, level),
                validate_containers=False,
                **kwargs,
            )


def read_windows(session: icechunk.Session) -> dict[tuple[str, int], xr.DataArray]:
    out = {}
    for band in CUBE_BANDS:
        for level, (r0, c0) in WINDOWS.items():
            ds = xr.open_zarr(
                session.store,
                group=level_group(band, level),
                consolidated=False,
                zarr_format=3,
                mask_and_scale=False,
            )
            out[(band, level)] = ds[band][:, r0 : r0 + 3, c0 : c0 + 3].load()
    return out


# --------------------------------------------------------------------------- direct
def gdal_env() -> dict[str, str]:
    # GDAL with a bearer header (GDAL_HTTP_HEADERS or GDAL_HTTP_AUTH=BEARER)
    # fails: "not recognized as being in a supported file format" (the header
    # is presumably forwarded to the signed S3 redirect). The classic netrc +
    # cookie-jar flow works.
    cookies = str(CACHE / "gdal-cookies")
    return {
        "GDAL_HTTP_NETRC": "YES",
        "GDAL_HTTP_COOKIEFILE": cookies,
        "GDAL_HTTP_COOKIEJAR": cookies,
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    }


def pick_windows(granules: list[str]) -> None:
    """Choose 3x3 windows where B04 is valid in every granule (from level 4)."""
    try:
        import rasterio
    except ImportError:
        return
    valid = None
    with rasterio.Env(**gdal_env()):
        for gid in granules:
            with rasterio.open(https_url(gid, "B04"), overview_level=3) as src:
                arr = src.read(1)
            v = arr != FILL["B04"]
            valid = v if valid is None else valid & v
    # require a 5x5 valid neighbourhood at level 4 (~16x coarser than level 0)
    from numpy.lib.stride_tricks import sliding_window_view

    ok = sliding_window_view(valid, (5, 5)).all(axis=(2, 3))
    rows, cols = np.nonzero(ok)
    if rows.size == 0:
        raise RuntimeError("no pixel valid in all granules")
    centre = np.array(ok.shape) / 2
    i = int(np.argmin((rows - centre[0]) ** 2 + (cols - centre[1]) ** 2))
    r4, c4 = int(rows[i]) + 2, int(cols[i]) + 2  # centre of the 5x5 block
    WINDOWS[0] = (r4 * 16, c4 * 16)
    WINDOWS[1] = (r4 * 8, c4 * 8)
    RESULTS["windows"] = {
        "level4_valid_pixels_all_granules": int(valid.sum()),
        "level4_centre": [r4, c4],
        "windows": {str(k): list(v) for k, v in WINDOWS.items()},
    }
    print(f"[windows] {RESULTS['windows']}")


def chunk_counts(session: icechunk.Session, ntimes: int) -> dict[str, list[int]]:
    """Number of stored chunk references per time index, per (band, level)."""
    import asyncio

    async def count(prefix: str) -> list[int]:
        per_t = [0] * ntimes
        async for key in session.store.list_prefix(prefix):
            rest = key[len(prefix) :].lstrip("/").split("/")
            if rest and rest[0] == "c":
                per_t[int(rest[1])] += 1
        return per_t

    out = {}
    for band in CUBE_BANDS:
        for level in _levels(session, band):
            out[f"{band}/{level}"] = asyncio.run(count(array_path(band, level)[1:]))
    return out


def direct_reads(granules: list[str], token: str) -> dict | None:
    try:
        import rasterio
        from rasterio.windows import Window
    except ImportError:
        print("[direct] rasterio not available; run with `uv run --with rasterio`")
        return None
    out = {}
    with rasterio.Env(**gdal_env()):
        for gid in granules:
            for band in CUBE_BANDS:
                for level, (r0, c0) in WINDOWS.items():
                    kw = {} if level == 0 else {"overview_level": level - 1}
                    with rasterio.open(https_url(gid, band), **kw) as src:
                        out[(gid, band, level)] = src.read(
                            1, window=Window(c0, r0, 3, 3)
                        )
                        if level == 0:
                            out[("crs", gid)] = str(src.crs)
                            out[("transform", gid)] = list(src.transform)
                            out[("overviews", gid, band)] = src.overviews(1)
    return out


def compare(read: dict, direct: dict | None, order: list[str]) -> tuple[bool, list]:
    """Check cube[time=i] == direct read of granule order[i] for every window."""
    evidence = []
    ok = True
    for (band, level), arr in read.items():
        for i, gid in enumerate(order):
            got = np.asarray(arr[i])
            entry = {
                "band": band,
                "level": level,
                "t": i,
                "gid": gid,
                "cube": got.tolist(),
            }
            if direct is not None:
                exp = direct[(gid, band, level)]
                match = bool(np.array_equal(got, exp))
                ok &= match
                # A missing reference reads as fill; demand B04 real data.
                if band == "B04":
                    entry["has_data"] = bool((got != FILL[band]).all())
                    ok &= entry["has_data"]
                entry["direct"] = exp.tolist()
                entry["match"] = match
            evidence.append(entry)
    return ok, evidence


# --------------------------------------------------------------------------- main
def main() -> int:
    s1 = step1_inventory()
    granules = s1["granules"]
    newest = s1["newest_2025"]
    transform = list(newest["proj:transform"])
    times_sorted = sorted(granules, key=granule_time)
    assert times_sorted == granules, granules

    token = login(strategy="netrc").token["access_token"]
    registry = make_registry(token)
    pyramids = step2_open(granules, registry)
    reference = pyramids[granules[-1]]

    pick_windows(granules)
    direct = direct_reads(granules, token)
    if direct is not None:
        RESULTS["direct_crs"] = {g: direct[("crs", g)] for g in granules}
        RESULTS["direct_transform"] = {g: direct[("transform", g)] for g in granules}

    workdir = Path(tempfile.mkdtemp(prefix="hls-spike-"))
    print(f"[repo] {workdir}")

    # ----- step 3: declare 3 steps
    s3: dict[str, Any] = {}
    try:
        repo = open_repo(workdir / "cube3", token)
        session = repo.writable_session("main")
        times = np.array([granule_time(g) for g in granules])
        declare(session.store, reference, times, transform)
        s3["snapshot"] = session.commit("declare 3-step cube")
        ro = repo.readonly_session("main")
        shapes = {}
        for band in CUBE_BANDS:
            for level in reference[band]:
                arr = zarr.open_array(ro.store, path=array_path(band, level), mode="r")
                shapes[f"{band}/{level}"] = {
                    "shape": list(arr.shape),
                    "chunks": list(arr.chunks),
                    "dims": list(arr.metadata.dimension_names),
                }
        ds0 = xr.open_zarr(
            ro.store, group=level_group("B04", 0), consolidated=False, zarr_format=3
        )
        tarr = zarr.open_array(
            ro.store, path=f"/{level_group('B04', 0)}/time", mode="r"
        )
        s3["arrays"] = shapes
        s3["time_decoded"] = [str(t) for t in ds0["time"].values]
        s3["time_raw"] = {
            "dtype": str(tarr.dtype),
            "values": tarr[:].tolist(),
            "attrs": dict(tarr.attrs),
        }
        s3["pass"] = (
            ds0["time"].dtype.kind == "M"
            and tarr.dtype == np.int64
            and all(v["shape"][0] == 3 and v["chunks"][0] == 1 for v in shapes.values())
        )
    except Exception as exc:  # noqa: BLE001
        s3["pass"] = False
        s3["error"] = repr(exc)
        raise
    finally:
        RESULTS["step3"] = s3
    print(f"[step3] pass={s3['pass']}")

    # ----- step 4: region-write in reverse order
    s4: dict[str, Any] = {}
    try:
        session = repo.writable_session("main")
        for gid in reversed(granules):
            write_granule(session.store, pyramids[gid], gid, region="auto")
        s4["snapshot"] = session.commit("region-write 3 granules (reverse order)")
        ro = repo.readonly_session("main")
        read = read_windows(ro)
        ok, evidence = compare(read, direct, granules)
        ds0 = xr.open_zarr(
            ro.store, group=level_group("B04", 0), consolidated=False, zarr_format=3
        )
        s4["time"] = [str(t) for t in ds0["time"].values]
        s4["evidence"] = evidence
        s4["chunk_refs_per_time"] = chunk_counts(ro, len(granules))
        expected_refs = {
            f"{b}/{lv}": [
                len(pyramids[g][b][lv][str(lv)].data.manifest.dict()) for g in granules
            ]
            for b in CUBE_BANDS
            for lv in reference[b]
        }
        s4["expected_refs_per_time"] = expected_refs
        s4["refs_ok"] = s4["chunk_refs_per_time"] == expected_refs
        # all_virtual_chunk_locations() returns *resolved* URLs, so to prove
        # the stored refs are relative vcc:// we reopen the repo with the
        # container pointed at the s3 prefix: relative refs follow it,
        # absolute https refs would not.
        alt = icechunk.RepositoryConfig.default()
        alt.set_virtual_chunk_container(
            icechunk.VirtualChunkContainer(
                url_prefix="s3://lp-prod-protected/",
                store=icechunk.s3_store(region="us-west-2"),
                name=CONTAINER,
            )
        )
        locs = list(
            icechunk.Repository.open(
                icechunk.local_filesystem_storage(str(workdir / "cube3")),
                config=alt,
            )
            .readonly_session("main")
            .all_virtual_chunk_locations()
        )
        s4["virtual_locations"] = {
            "count": len(locs),
            "sample_resolved_via_s3_container": sorted(locs)[:1],
            "all_vcc": all(x.startswith("s3://lp-prod-protected/") for x in locs),
        }
        s4["refs_ok"] &= s4["virtual_locations"]["all_vcc"]
        s4["group_attrs"] = dict(
            zarr.open_group(ro.store, path=level_group("B04", 0), mode="r").attrs
        )
        s4["verified_against_direct"] = direct is not None
        s4["pass"] = (
            ok
            and s4["refs_ok"]
            and s4["time"] == [str(granule_time(g)) for g in granules]
        )
    except Exception as exc:  # noqa: BLE001
        s4["pass"] = False
        s4["error"] = repr(exc)
        raise
    finally:
        RESULTS["step4"] = s4
    print(f"[step4] pass={s4['pass']} (direct={direct is not None})")

    # ----- step 5: 2-step declare, region 2025s, append 2013, reindex sort
    s5: dict[str, Any] = {}
    try:
        old, new = granules[0], granules[1:]
        repo5 = open_repo(workdir / "cube2", token)
        session = repo5.writable_session("main")
        declare(
            session.store,
            reference,
            np.array([granule_time(g) for g in new]),
            transform,
        )
        session.commit("declare 2-step cube")
        session = repo5.writable_session("main")
        for gid in new:
            write_granule(session.store, pyramids[gid], gid, region="auto")
        write_granule(session.store, pyramids[old], old, append_dim="time")
        before = xr.open_zarr(
            session.store,
            group=level_group("B04", 0),
            consolidated=False,
            zarr_format=3,
        )
        s5["time_before_sort"] = [str(t) for t in before["time"].values]
        s5["moved"] = sort_time(session)
        s5["snapshot"] = session.commit("append 2013 + sort")
        ro = repo5.readonly_session("main")
        times_by_group = {}
        for band in CUBE_BANDS:
            for level in reference[band]:
                ds = xr.open_zarr(
                    ro.store,
                    group=level_group(band, level),
                    consolidated=False,
                    zarr_format=3,
                )
                times_by_group[f"{band}/{level}"] = [str(t) for t in ds["time"].values]
        expected = [str(granule_time(g)) for g in granules]
        s5["times_by_group_all_sorted"] = all(
            v == expected for v in times_by_group.values()
        )
        s5["time_after_sort"] = times_by_group["B04/0"]
        read = read_windows(ro)
        ok, evidence = compare(read, direct, granules)
        # Also check against the step-4 cube read (independent of rasterio).
        ref4 = read_windows(repo.readonly_session("main"))
        same_as_step4 = all(
            np.array_equal(read[k].values, ref4[k].values) for k in read
        )
        s5["same_as_step4_cube"] = same_as_step4
        s5["chunk_refs_per_time"] = chunk_counts(ro, len(granules))
        s5["refs_ok"] = s5["chunk_refs_per_time"] == s4["expected_refs_per_time"]
        s5["evidence"] = evidence
        s5["verified_against_direct"] = direct is not None
        s5["pass"] = (
            ok and s5["refs_ok"] and same_as_step4 and s5["times_by_group_all_sorted"]
        )
    except Exception as exc:  # noqa: BLE001
        s5["pass"] = False
        s5["error"] = repr(exc)
        raise
    finally:
        RESULTS["step5"] = s5
    print(f"[step5] pass={s5['pass']}")

    out = CACHE / "results.json"
    out.write_text(json.dumps(RESULTS, indent=1, default=str))
    print(f"[done] results -> {out}")
    shutil.rmtree(workdir, ignore_errors=True)
    gate = RESULTS["step2"]["verdict"] == "SAME" and all(
        RESULTS[s]["pass"] for s in ("step3", "step4", "step5")
    )
    if gate and direct is None:
        print(
            "[gate] UNVERIFIED: steps ran, but pixel values were not checked "
            "against direct COG reads (run with `uv run --with rasterio`)"
        )
        return 2
    print(f"[gate] {'PASS' if gate else 'FAIL'}")
    return 0 if gate else 1


def sort_time(session: icechunk.Session) -> int:
    """Sort `time` in every level group with reindex_array (naqfc approach).

    Every group must agree on the time axis; the permutation is computed once
    and applied to every data array, then each group's raw time coordinate is
    permuted directly (bypassing xarray's encoding inference).
    """
    groups = [(b, lv) for b in CUBE_BANDS for lv in _levels(session, b)]
    axes = []
    for band, level in groups:
        t = zarr.open_array(
            session.store, path=f"/{level_group(band, level)}/time", mode="r"
        )
        axes.append(np.asarray(t[:]))
    if any(not np.array_equal(a, axes[0]) for a in axes):
        raise ValueError("level groups disagree on time")
    times = axes[0]
    if np.unique(times).size != times.size:
        raise ValueError("duplicate times")
    order = np.argsort(times, kind="stable")
    if np.array_equal(order, np.arange(len(order))):
        return 0
    old_to_new = np.empty_like(order)
    old_to_new[order] = np.arange(len(order))

    def forward(pos: Any) -> list[int]:
        p = list(pos)
        p[0] = int(old_to_new[p[0]])
        return p

    def backward(pos: Any) -> list[int]:
        p = list(pos)
        p[0] = int(order[p[0]])
        return p

    for band, level in groups:
        session.reindex_array(array_path(band, level), forward, backward)
        t = zarr.open_array(
            session.store, path=f"/{level_group(band, level)}/time", mode="r+"
        )
        t[:] = np.asarray(t[:])[order]
    return int((order != np.arange(len(order))).sum())


def _levels(session: icechunk.Session, band: str) -> list[int]:
    g = zarr.open_group(session.store, path=f"{TILE}/{band}/multiscales", mode="r")
    return sorted(int(k) for k in g.group_keys())


if __name__ == "__main__":
    sys.exit(main())
