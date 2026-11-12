# Spike findings: virtual time-stacking of HLS L30 COGs (T13SDA)

Date: 2026-11-11 · Spike code: `spike/time_stack.py` (throwaway; removed from
the tree, recoverable with `git show c598a71:spike/time_stack.py`).
Run: `uv run --with rasterio python spike/time_stack.py` (~2 min from a laptop
over https; writes `~/.cache/hls-spike/results.json`). Without rasterio it runs
steps 1–5 but cannot do the direct-COG comparison and exits 2 (`UNVERIFIED`).

Versions: virtualizarr 2.7.1, virtual-tiff 0.5.0, icechunk 2.1.1, xarray
2025.12.0, zarr 3.x; rasterio 1.5.2 / GDAL 3.12.2 (spike-only, for direct reads).

## Verdict

**GATE: PASS.** 2013 and 2025 encodings are identical in every compared field,
and steps 3, 4 and 5 all pass with pixel values checked against independent
rasterio reads of the same COGs. Stored references are relative
`vcc://lpdaac/...` and read back through an https container named `lpdaac`; no
absolute-URL fallback was needed.

Two things the spec/plan should take into account (details below):

1. **2013 archive rows carry the wrong `proj:epsg` (32713, UTM 13 *South*)**,
   and the 2013 parquet has **no `proj:shape`/`proj:transform` columns at all**
   (not just null values). The 2013 COG itself is UTM 13 *North*, same grid.
2. **Overview levels are chunked 128×128, level 0 is 256×256.** Chunk shape is
   per level, not one COG tile size.

## 1. Granules (step 1)

Archive months read anonymously over https:
`.../HLSL30_2.0/year={Y}/month=6/HLSL30_2.0-{Y}-6.parquet`.

| month | rows | T13SDA rows | columns present (of id/proj:epsg/proj:shape/proj:transform) |
|---|---|---|---|
| 2013-6 | 72,006 | 5 | `id`, `proj:epsg` only |
| 2025-6 | 184,523 | 9 | all four |

**Used (1×2013, 2×2025):**

- `HLS.L30.T13SDA.2013157T174007.v2.0` (2013-06-06T17:40:07Z)
- `HLS.L30.T13SDA.2025157T174342.v2.0` (2025-06-06T17:43:42Z)
- `HLS.L30.T13SDA.2025181T174401.v2.0` (2025-06-30T17:44:01Z) — newest row read

Other T13SDA IDs in those months:

- 2013-6: `2013164T174615`, `2013166T173417`, `2013173T174000`, `2013180T174614`
- 2025-6: `2025158T173736`, `2025159T173146`, `2025165T174354`, `2025166T173740`,
  `2025167T173157`, `2025173T174356`, `2025174T173748`
  (all `HLS.L30.T13SDA.{…}.v2.0`)

Note: 2025 has near-daily granules (157, 158, 159; 165, 166, 167; …): L30 now
includes Landsat 9 and adjacent-path overlaps, so expect ~3× the 2013 density.
The 2025-6 granules cover only part of the tile (the centre pixel used first was
fill in both 2025 granules); 2013157 reports `spatial_coverage: 100`.

**Grid (newest 2025 row, `HLS.L30.T13SDA.2025181T174401.v2.0`) — for
`TILE_GRIDS["T13SDA"]`:**

```python
epsg = 32613
shape = [3660, 3660]
transform = [30.0, 0.0, 399960.0, 0.0, -30.0, 4100040.0, 0.0, 0.0, 1.0]
```

All 9 T13SDA rows of 2025-6 have exactly these values.

**CRS discrepancy in old records:** all five 2013 T13SDA rows have
`proj:epsg = 32713` (UTM 13S). The 2013 COG says otherwise: GeoKeys
`citation="UTM Zone 13, Northern Hemisphere"`, `ProjectionGeoKey=16013`
(UTM 13N), `ProjectedCSTypeGeoKey=32767` (user-defined), tiepoint
`(399960, 4100040)`, pixel scale 30 — i.e. the same grid as 2025, expressed
without an EPSG code. rasterio reports the 2013 CRS as a WKT
"UTM Zone 13, Northern Hemisphere" and the 2025 ones as `EPSG:32613`; geotransform
is identical for all three: `[30, 0, 399960, 0, -30, 4100040]`.
Consequences:

- The inventory must **not** take the grid (or EPSG) from 2013 rows, and must
  not reject 2013 rows for `proj:epsg != 32613`; take the grid from the newest
  rows (as planned) and treat old-row `proj:epsg` as unreliable.
- A per-granule grid check that compares COG GeoKeys to `EPSG:32613` literally
  would reject every 2013 granule. Check shape + chunking + dtype/codecs (and
  if CRS is checked, accept UTM-13N in either encoding, or compare the
  tiepoint/pixel scale).

## 2. Encodings per (band, level) (step 2)

Opened with `open_virtual_datatree(url, registry=…, parser=VirtualTIFF(ifd=None,
ifd_layout="nested"))` over https; registry = obstore http store on
`https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected` (no trailing
slash) with `Authorization: Bearer <earthaccess_auth.login(strategy="netrc")
token>`, keyed by the prefix *with* trailing slash.

**Verdict 2013 vs 2025: SAME** — level count, shape, chunk shape, dtype,
fill value and codecs identical for all 15 bands × 5 levels across all three
granules (0 diffs). Level count = 5 everywhere (GDAL overviews 2, 4, 8, 16).

Per level (identical for every band):

| level | shape | chunk shape | chunk grid | chunks |
|---|---|---|---|---|
| 0 | 3660×3660 | 256×256 | 15×15 | 225 |
| 1 | 1830×1830 | 128×128 | 15×15 | 225 |
| 2 | 915×915 | 128×128 | 8×8 | 64 |
| 3 | 458×458 | 128×128 | 4×4 | 16 |
| 4 | 229×229 | 128×128 | 2×2 | 4 |

Per band (identical at every level):

| bands | dtype | fill_value | codecs |
|---|---|---|---|
| B01 B02 B03 B04 B05 B06 B07 B09 B10 B11 | int16 | -9999 | `HorizontalDeltaCodec`, `bytes{endian: little}`, `numcodecs.zlib{}` |
| Fmask | uint8 | 255 | `HorizontalDeltaCodec`, `bytes` (no config), `numcodecs.zlib{}` |
| SZA SAA VZA VAA | uint16 | 40000 | `HorizontalDeltaCodec`, `bytes{endian: little}`, `numcodecs.zlib{}` |

Exact codec list as virtual-tiff emits it (int16 example):
`[{"name": "HorizontalDeltaCodec"}, {"name": "bytes", "configuration": {"endian": "little"}}, {"name": "numcodecs.zlib", "configuration": {}}]`.
`HorizontalDeltaCodec` is registered by `import virtual_tiff`; any reader of the
cube (xarray/zarr) must import `virtual_tiff` first.

**Chunk-ref sizing (#6):** per band per granule 225+225+64+16+4 = **534** chunk
refs; per granule (15 bands) **8,010**. Every COG had a full manifest (no sparse
tiles), confirmed by counting stored refs per time index after the writes.

Overview pixel sizes are extent/n, not exactly 30·2^L: L1 60 m, L2 120 m,
L3 109800/458 = 239.738 m, L4 109800/229 = 479.476 m. The spike's `x`/`y` per
level are pixel centres `x0 + (i + 0.5)·(3660·30/nx)`.

Header open cost: ~100 s for 45 COGs (3 granules × 15 bands) from a laptop over
https, i.e. ~2 s per COG — relevant for backfill batch sizing (in-region s3 will
be faster).

## 3. Declare / region-write / append recipe (#4)

Layout used: group `T13SDA/{band}/multiscales/{level}` holds data array `{band}`
plus `time`, `y`, `x`. Bands in the cube spike: `B04`, `Fmask`.

### Declaration (empty, full shape, time prepended)

```python
from virtualizarr.manifests import ChunkManifest, ManifestArray
from zarr.core.metadata import ArrayV3Metadata

src: ManifestArray = level_ds[str(level)].data       # reference granule's level
ny, nx = src.metadata.shape
cy, cx = src.metadata.chunks
spec = src.metadata.to_dict()
spec["shape"] = [len(times), ny, nx]
spec["chunk_grid"] = {"name": "regular", "configuration": {"chunk_shape": [1, cy, cx]}}
spec["dimension_names"] = ["time", "y", "x"]
spec["attributes"] = {}
gy, gx = src.manifest.shape_chunk_grid
empty = ManifestArray(
    chunkmanifest=ChunkManifest({}, shape=(len(times), gy, gx)),
    metadata=ArrayV3Metadata.from_dict(spec),
)
cube = xr.Dataset(
    {band: xr.Variable(("time", "y", "x"), empty)},
    coords={"time": ("time", times), "y": ("y", y), "x": ("x", x)},
)  # times: datetime64[ns] array
cube["time"].encoding.update({
    "units": "seconds since 1970-01-01T00:00:00",
    "calendar": "proleptic_gregorian",
    "dtype": "int64",
})
cube.vz.to_icechunk(store, group=f"T13SDA/{band}/multiscales/{level}")
```

Result: arrays `(3, ny, nx)`, chunks `(1, cy, cx)`, `dimension_names
["time","y","x"]`; `time` stored as int64 `[1370540407, 1749231822,
1751305441]` with attrs `units="seconds since 1970-01-01"` (xarray normalises
the `T00:00:00` away — compare decoded values, not the units string),
`calendar="proleptic_gregorian"`; xarray decodes it as `datetime64[ns]`.
Parent groups (`T13SDA`, `T13SDA/B04`, `…/multiscales`) are created implicitly.

### Per-granule level dataset (region and append)

```python
(name,) = list(level_ds.data_vars)            # variable is named str(level)
ds = level_ds.rename({name: band})
ds = ds.drop_vars([v for v in ds.variables if v != band])   # no-op today: no coords
ds = ds.vz.rename_paths(f"vcc://lpdaac/HLSL30.020/{gid}/{gid}.{band}.tif")
ds = ds.expand_dims(time=[granule_time(gid)])   # datetime64[ns]; NO time encoding
ds.attrs = {}
ds[band].attrs = {}
```

Region write (time step already declared):

```python
ds.vz.to_icechunk(store, group=f"T13SDA/{band}/multiscales/{level}",
                  region="auto", validate_containers=False)
```

Append (time step not on the axis):

```python
ds.vz.to_icechunk(store, group=f"T13SDA/{band}/multiscales/{level}",
                  append_dim="time", validate_containers=False)
```

Gotchas:

- **Clear the level dataset's attrs.** Level 0's dataset attrs carry the whole
  granule's TIFF metadata (`LANDSAT_SCENE_ID`, `SENSING_TIME`, GeoKeys, …), and
  virtualizarr's `to_icechunk` calls `update_attributes(group, vds.attrs)` on
  *every* write (region and append included), so without clearing them each
  write would stamp one granule's metadata onto the level group. After clearing,
  the level group's attrs are just `{"coordinates": "time y x"}`.
- **Do not set `time` encoding on region/append datasets**: the declared
  encoding is reused by xarray (xarray rejects encoding for an existing variable
  in append mode). Set it only at declaration.
- `validate_containers=False` is required because the refs are `vcc://`.
- The time coordinate is the only coordinate on the write datasets; `x`/`y` are
  written once at declaration. (Including `x`/`y` in a region write was not
  tested; they have no `time` dim so `region="auto"` would reject them.)
- `granule_time` from the ID drops sub-seconds (STAC `datetime` for 2013157 is
  17:40:07.866; the ID gives 17:40:07). Fine as long as the ID is the sole source.
- Region/append do `check_compatible_arrays` (dtype, codecs, chunk shape, ndim,
  shape except along time), so a mis-gridded granule raises — but the declared
  metadata must come from the same parser (virtual-tiff) to compare equal.

### Repository / virtual chunk container (laptop, https)

```python
P = "https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/"
config = icechunk.RepositoryConfig.default()
config.set_virtual_chunk_container(icechunk.VirtualChunkContainer(
    url_prefix=P,
    store=icechunk.http_store(headers={"Authorization": f"Bearer {token}"}),
    name="lpdaac",
))
repo = icechunk.Repository.open_or_create(
    icechunk.local_filesystem_storage(path), config=config,
    authorize_virtual_chunk_access=icechunk.containers_credentials(
        {P: icechunk.Credentials.HttpAccess()}),
)
```

- Passing `None` as the https credential (as icechest does) is **deprecated in
  icechunk 2.1** (warning); use `icechunk.Credentials.HttpAccess()`.
- Relative `vcc://lpdaac/...` refs work end-to-end: read back over https with
  xarray. `Session.all_virtual_chunk_locations()` returns **resolved** URLs, so
  to prove refs are relative the spike reopened the repo with the container
  pointed at `s3://lp-prod-protected/` — all 3,204 locations then resolved to
  `s3://lp-prod-protected/HLSL30.020/...` (absolute refs would not have moved).

## 4. Reindex recipe across level groups (#5)

Run in the same session as the append(s), before commit:

```python
groups = [(b, lv) for b in bands for lv in levels(b)]  # every T13SDA/{b}/multiscales/{lv}
axes = [zarr.open_array(store, path=f"/{g}/time", mode="r")[:] for g in group_paths]
assert all equal; assert unique
order = np.argsort(times, kind="stable")
if identity: return 0
old_to_new = np.empty_like(order); old_to_new[order] = np.arange(len(order))
forward  = lambda pos: [int(old_to_new[pos[0]]), *list(pos)[1:]]
backward = lambda pos: [int(order[pos[0]]), *list(pos)[1:]]
for g in group_paths:
    session.reindex_array(f"/{g}/{band}", forward, backward)   # leading "/" path
    t = zarr.open_array(session.store, path=f"/{g}/time", mode="r+")
    t[:] = np.asarray(t[:])[order]          # raw int64, not via xarray
```

Works because the time chunk size is 1 (one granule per chunk row). The time
array is permuted raw (int64 seconds) so xarray does not re-infer encoding.

## 5. Results (#7)

Pixel windows: chosen automatically where B04 is valid in **all three**
granules (5×5 valid block at level 4, nearest the tile centre): level 0 rows
1520:1523, cols 752:755; level 1 rows 760:763, cols 376:379. B04 windows are
additionally required to contain no fill, and the number of stored chunk refs
per time index is checked against each granule's manifest, so a missing
reference (which reads as fill) cannot pass as a match.

### Step 3 — declare 3-step cube: **PASS**

All 10 arrays (B04, Fmask × 5 levels) `(3, ny, nx)`, chunks `(1, cy, cx)`,
dims `time,y,x`; `time` int64 seconds, decodes to
`2013-06-06T17:40:07, 2025-06-06T17:43:42, 2025-06-30T17:44:01`.

### Step 4 — region-write in reverse time order, read back: **PASS**

Wrote 2025181, 2025157, 2013157 (reverse) with `region="auto"`, committed,
re-opened read-only. `time` unchanged and ordered. Stored refs per time index =
225/225/64/16/4 for each (band, level), matching each granule's manifest. Every
3×3 window equals the rasterio read of the same COG/overview:

| band | L | t | granule | cube = rasterio (row-major 3×3) |
|---|---|---|---|---|
| B04 | 0 | 0 | 2013157 | 1625 1597 1563 / 1489 1513 1499 / 1524 1501 1485 |
| B04 | 0 | 1 | 2025157 | 1544 1303 563 / 1529 1440 974 / 1483 1465 1395 |
| B04 | 0 | 2 | 2025181 | 1517 1526 1431 / 1399 1126 783 / 1285 620 202 |
| B04 | 1 | 0 | 2013157 | 1625 1563 1619 / 1524 1485 1618 / 1413 1703 1578 |
| B04 | 1 | 1 | 2025157 | 1544 563 69 / 1483 1395 1095 / 1362 1631 1673 |
| B04 | 1 | 2 | 2025181 | 1517 1431 1325 / 1285 202 469 / 732 215 244 |
| Fmask | 0 | 0 | 2013157 | 128 128 128 / 128 128 128 / 128 128 128 |
| Fmask | 0 | 1 | 2025157 | 72 72 200 / 72 72 136 / 76 72 136 |
| Fmask | 0 | 2 | 2025181 | 128 128 128 / 128 128 128 / 128 128 160 |
| Fmask | 1 | 0 | 2013157 | 128 128 64 / 128 128 128 / 128 128 128 |
| Fmask | 1 | 1 | 2025157 | 72 200 200 / 76 136 200 / 76 72 72 |
| Fmask | 1 | 2 | 2025181 | 128 128 128 / 128 160 128 / 192 224 160 |

(Raw values read with `mask_and_scale=False`. Level-1 windows are rasterio
`overview_level=0`.)

### Step 5 — out-of-order append + reindex: **PASS**

Fresh repo, declared 2 steps (2025157, 2025181), committed; new session:
region-wrote both 2025 granules, appended 2013157 with `append_dim="time"` →
time `[2025-06-06, 2025-06-30, 2013-06-06]`; `sort_time` moved 3 rows across
all 10 level groups; committed. After commit every group's `time` is
`[2013-06-06T17:40:07, 2025-06-06T17:43:42, 2025-06-30T17:44:01]`, chunk refs
per time index again 225/225/64/16/4, and all 12 windows are identical to the
step-4 cube and to rasterio (same table as above).

## 6. Spec assumptions checked

- "`proj:shape`/`proj:transform` are null on old (2013) records" — **corrected**:
  the columns are absent from the 2013-6 file entirely; readers must select
  columns by presence, not just handle nulls. Also `proj:epsg` is wrong (32713)
  on 2013 rows.
- "Level count/levels same across archive" — **confirmed** (5 levels, same
  shapes, 2013 vs 2025).
- "Chunks `(1, cog_tile_y, cog_tile_x)`" — **refine**: tile size differs by
  level (256 at L0, 128 at L1–L4); take it from each level's metadata.
- "References stay relative (`vcc://lpdaac/...`)" — **confirmed workable** for
  region and append writes and read-back over https.
- Fmask/angle bands have the same pyramid depth as spectral bands for L30
  (icechest's caution about differing depth does not bite here, but keep per-band
  level discovery).

## 7. Other notes for later tasks

- Direct GDAL/rasterio reads of LP DAAC over https **fail with a bearer header**
  (`GDAL_HTTP_HEADERS` or `GDAL_HTTP_AUTH=BEARER` → "not recognized as being in
  a supported file format"); `GDAL_HTTP_NETRC=YES` + a cookie jar works. Only
  relevant for test tooling; obstore and icechunk's http store work with the
  bearer header.
- The spike did not exercise the band-group conventions attrs, multiple tiles,
  s3 access, or Fork/merge sessions.
