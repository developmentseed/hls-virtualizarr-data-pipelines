# HLS L30 single-MGRS-tile datacube — design

Date: 2026-11-11
Status: draft, awaiting approval

## Intent

Build a virtual (VirtualiZarr + Icechunk) **datacube for one MGRS tile** of
Harmonized Landsat Sentinel-2, where every granule of that tile is
**concatenated along a `time` dimension** — instead of one group per granule
as `~/projects/icechest-hls` does.

HLS is distributed per MGRS tile, and tiles sit in different UTM CRSs, so a
single cube across tiles is not meaningful. One tile has one CRS and one fixed
pixel grid, which is what makes a time-stacked cube possible.

The cube is built with the `virtualizarr-data-pipelines` template: a one-time
**backfill** loads the archive history, and **forward processing** appends new
granules as they are published.

## Goals

- A single Icechunk store for **HLS L30 v2.0, tile `T13SDA`** (Colorado),
  covering **the entire archive history** (April 2013 onward).
- Layout: **`/{mgrs_id}/{band}/multiscales/{level}/{band}`**, e.g.
  `/T13SDA/B04/multiscales/0/B04`, each array shaped `(time, y, x)`.
- Keep the zarr-conventions **multiscales / proj / spatial** attributes from
  icechest-hls, now describing the tile rather than a granule.
- Explicit **`x` and `y` coordinate arrays covering the whole MGRS cell**, so
  the cube is spatially self-describing.
- A **forward-processing path** that appends newly published granules, built
  and tested now even though no notification feed exists yet.

## Non-goals

- **S30.** It comes later as a *separate* cube (different band set and
  observation times). Nothing here should make that hard, but it is not built
  now.
- More than one tile. The `/{mgrs_id}` prefix leaves room for siblings later;
  multi-tile orchestration is not built.
- Wiring an SNS subscription. HLS has no notification topic we can subscribe
  to yet (see Forward processing).
- Deploying and running against the real archive. Like icechest-hls, the code,
  tests and `cdk synth` should be complete; running `cdk deploy` and dispatching
  the backfill is a deliberate operator step.
- Materialised data, reprojection, cloud masking or compositing. This is a
  virtual cube of references to the published COGs.

## Anti-goals (would be a failure even if it "works")

- Granules from a different grid silently stacked into the cube (wrong tile,
  wrong shape, wrong chunking, wrong CRS). Every granule must be verified
  against the declared grid.
- A non-monotonic or duplicated `time` axis.
- Absolute endpoint URLs baked into manifests. References stay relative
  (`vcc://lpdaac/...`) as in icechest, so the store reads from a laptop over
  https and from us-west-2 over s3.

## Decisions made with the user

1. **Product order:** L30 first (better coverage), S30 later as its own cube.
2. **Tile and range:** `T13SDA`, full archive history.
3. **Layout:** `/{mgrs_id}/{band}/multiscales/{level}/{band}`.
4. **Spatial coordinates:** create `x`/`y` for the entire MGRS cell.
5. **Inventory unit is the granule.** The inventory is a list of granule IDs;
   COG asset URLs are **inferred from the ID**, not read from STAC asset hrefs.
6. **Inventory source:** the HLS stac-geoparquet archive, filtered to one tile.
7. **Build forward/update processing**, knowing there is no SNS feed yet.
8. **Repo:** a new repo at `~/projects/hls-virtualizarr`, created from the
   `virtualizarr-data-pipelines` template (local copy at
   `~/projects/virtualizarr-data-pipelines`).

## Background facts

From icechest-hls / icechest specs (measured 2026-09):

- Archive: `s3://nasa-maap-data-store/file-staging/nasa-map/hls-stac-geoparquet-archive/v2/HLSL30_2.0/year={Y}/month={M}/HLSL30_2.0-{Y}-{M}.parquet`
  — one file per month, April 2013 → present, months not zero-padded, readable
  anonymously over HTTPS. Each month is ~100–185k rows, one row group.
- **`proj:shape` / `proj:transform` are null on old (2013) records** and
  populated on recent ones. This design does not depend on per-granule proj
  fields (see Tile grid).
- Asset URL pattern:
  `https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/HLSL30.020/{id}/{id}.{band}.tif`,
  also `s3://lp-prod-protected/HLSL30.020/{id}/{id}.{band}.tif` in us-west-2
  with Earthdata-minted temporary credentials.
- Granule ID: `HLS.L30.T13SDA.{YYYYDDD}T{HHMMSS}.v2.0` (day-of-year, UTC).
- L30 bands (icechest `DEFAULT_BANDS`): `B01`–`B07`, `B09`–`B11`, `Fmask`,
  `SZA`, `SAA`, `VZA`, `VAA`.
- COGs carry an overview pyramid; icechest opens all levels in one pass with
  `virtual_tiff.VirtualTIFF(ifd=None, ifd_layout="nested")` via
  `open_virtual_datatree`, one child per IFD.
- Template backfill inventory is a JSON array of string keys;
  `process_backfill_file(file_key, fork)` receives one key. A granule ID fits
  this without a protocol change.
- Template `Init` is currently invoked with an empty payload, and
  `initialize_backfill_store(repo)` takes no inventory.

Estimated scale (estimate, not measured): T13SDA gets on the order of a few
hundred L30 granules a year → a few thousand granules total.

## Design

### Store layout

```
/                                   (root group)
  T13SDA/                           (tile group)
    B04/                            (band group: conventions attrs live here)
      multiscales/
        0/   B04 (time, y, x), time, y, x
        1/   B04 (time, y, x), time, y, x
        ...
    Fmask/ ...
```

- One array per (band, level), named after the band, shaped
  `(n_time, ny_level, nx_level)`, **chunked `(1, cog_tile_y, cog_tile_x)`**
  — one time step per chunk, spatial chunks equal to the COG's internal tiles
  so each virtual chunk is exactly one COG tile.
- dtype, fill value and codecs equal to what the COG's virtual arrays carry
  (taken from a reference granule at Init; see below).
- Each level group holds its own `time`, `y`, `x` coordinate arrays so that
  `xr.open_zarr(store, group=".../multiscales/0")` (and each other level) opens
  as a self-contained dataset and so `region="auto"` can align writes within
  that group. Coordinate inheritance via DataTree is not relied on.
  *(Builder may revisit if they confirm a shared coordinate works with
  `region="auto"`.)*
- Band group attributes: the icechest `granule_attrs` content (multiscales
  layout, `proj:code`, `spatial:shape/dimensions/transform/bbox`), computed
  **once from the tile grid**. `spatial:dimensions` stays `["y", "x"]`.

### Tile grid

All granules of a tile share a CRS and pixel grid (MGRS cell, 3660×3660 at
30 m). The grid is captured **once**, as code:

- **Embedded in the processor (decided with the user):** a constant
  `TILE_GRIDS: dict[str, TileGrid]` in the processor package holds each
  supported tile's `epsg`, `shape` and `transform`. No external grid file and
  nothing extra in the Lambda environment. The processor refuses a tile with
  no entry.
- The T13SDA values are taken once from the newest T13SDA granule's STAC
  `proj:epsg`, `proj:shape`, `proj:transform` (old records have these null),
  recorded in the spike findings, and committed as the constant. Adding a tile
  later (or S30 for the same tile, which shares the MGRS grid) means adding an
  entry.
- `x`/`y` for level 0 are pixel-centre coordinates from the transform. For
  overview level *k*, use that level's actual pixel count and the cell's fixed
  extent (resolution = extent / n), because overview sizes round (3660 → 1830
  → 915 → 458 …) and are not always an exact factor of two.

### Granule → virtual dataset (shared by backfill and forward)

For one granule ID:

1. Parse `time` from the ID (`YYYYDDDTHHMMSS` → UTC `datetime64[ns]`).
2. Build each band's URL from the ID (read form: s3 or https per access
   mode); record references as `vcc://lpdaac/HLSL30.020/{id}/{id}.{band}.tif`.
3. Open the band's full pyramid (icechest's `_open_pyramid` approach).
4. **Verify the granule against the declared grid**: per band, level count,
   each level's shape, chunk shape, dtype and codecs must equal what the store
   declares. Mismatch → error (see failure handling).
5. Per level: rename the IFD variable to the band, `expand_dims` a length-1
   `time` dimension with the parsed time coordinate, drop anything without a
   `time` dimension, and write to `/{mgrs_id}/{band}/multiscales/{level}` with
   `region="auto"` (backfill / existing time) or `append_dim="time"`
   (forward, new time). `validate_containers=False` as in icechest.

Code is **vendored from icechest** (`assets`, `conventions`, `virtualize`
pyramid opening, `credentials`) into the processor package, adapted for
URL-from-ID and the time dimension. icechest itself is not a dependency: it is
a local path dependency today and cannot be resolved in the Lambda Docker
build.

### Inventory

A script (`scripts/build_inventory.py`), run by an operator, not a Lambda:

- Reads every monthly HLSL30_2.0 parquet file from April 2013 to the present
  month, projecting only `id`, filtering
  rows whose ID contains `.T13SDA.`.
- Writes:
  - `inventory.json` — JSON array of granule IDs, **sorted by time**.
- **Fails loudly on duplicate timestamps** for the tile (two granules with the
  same parsed time). Two Landsat paths and Landsat 8/9 can both cover a tile on
  one day, but at different times, so duplicates are not expected; if they
  occur a human decides.
- Uploads it to the backfill artifacts bucket (template requirement: same
  region as the stack).

### Backfill

Template flow unchanged (Init → Partition → [Fork → Workers → Reduce]* →
Promote), with these changes:

- **Init needs the inventory.** The state machine passes `inventory_uri` to
  Init as well as Partition; the init handler forwards it and the processor's
  `initialize_backfill_store` reads `inventory.json` and uses the embedded
  tile grid to declare
  every array at full length `n_time`, write the `time`/`y`/`x` coordinates and
  band-group conventions, then commit. This is a small, deliberate change to
  the template's protocol and state machine.
- **Array metadata comes from a reference granule.** Init opens the newest
  granule's pyramids and copies dtype, fill value, codecs and chunk shape per
  (band, level) into the declared arrays. These become the contract every
  other granule is verified against.
- **Worker:** `process_backfill_file(granule_id, fork)` runs the shared path
  with `region="auto"`.

### Forward processing

- `process_file(key, session)` derives the granule ID and runs the shared path,
  using the template's `write_plan` logic: time already on the axis → region
  rewrite (idempotent re-delivery); time not on the axis → **append, wherever
  it falls chronologically**; no store → error (this cube must be backfilled
  first).
- **Out-of-order granules (decided with the user): follow the
  `naqfc-virtualizarr-data-pipelines` PR #6 pattern** ("Use a direct append and
  reindex approach for out of order data"; `sort_reference_time` in that repo's
  `virtualizarr_processor/processor.py`). A late granule older than the cube's
  newest step is appended at the end, then `commit_processed_files` sorts the
  `time` axis back into order **inside the same session, before committing**,
  so no snapshot is ever non-monotonic. The sort:
  - computes the permutation from the `time` coordinate, short-circuits when
    already ordered, and raises on duplicate times;
  - checks every array to be moved has time chunk size 1 before moving
    anything;
  - moves virtual chunk references with `session.reindex_array(path, forward,
    backward)` (no data read), and permutes the raw encoded `time` values
    directly with zarr (not via xarray, to avoid re-inferring time encoding).
  
  Adaptation for this cube: there is no single root dataset. The sort runs per
  `(band, level)` group (`/{mgrs_id}/{band}/multiscales/{level}`), each with its
  own `time` coordinate; all groups share the same permutation and must end up
  with identical `time` arrays (assert this before committing).
- This also covers granules published between the inventory snapshot and
  Promote: they are enqueued after Promote and sorted into place.
- **Message format:** keep the template's S3-notification shape. The `key` may
  be either a bare granule ID or any asset key of the granule
  (`HLSL30.020/{id}/{id}.B04.tif`); the processor normalises it to the ID. This
  lets operators enqueue manually today, and a future per-object S3/SNS feed
  would work unchanged (redundant per-band notifications become harmless
  region rewrites).
- A `scripts/enqueue_granules.py` helper sends granule IDs (e.g. from re-running
  the inventory for recent months) to the forward queue.
- `SNS_TOPIC` stays unset.

### Failure handling

- **Backfill (decided with the user): skip and record per partition.** A
  granule that fails the grid check or still cannot be read after retries with
  backoff is skipped, not fatal. Each worker records its skips (granule ID,
  stage, error) alongside its fork; Reduce combines them into a
  **`skipped.json` per partition** under the run prefix
  (e.g. `{run_prefix}partitions/{partition_id}/skipped.json`, written even when
  empty). The worker batch only fails, as in the template, on errors that are not
  about a single granule (fork load/save, credentials).
  - Consequence (my call, flag if wrong): a skipped granule's time step was
    declared at Init, so it stays in the cube as a **fill-valued step** (no
    chunk references). It is not silent: it is listed in `skipped.json`, and a
    skipped granule can later be re-sent through forward processing, which
    region-writes it into its existing slot.
- **Forward:** a failed granule returns its message to the queue (template
  behaviour) and ends in the DLQ if it keeps failing.

### Configuration

- Tile (`MGRS_TILE=T13SDA`), product (`HLS_PRODUCT=L30`) and bands are settings,
  so the S30 cube later is a configuration + band-list change, not a fork.
- Region `us-west-2` (LP DAAC same-region access policy).
- Earthdata credentials as in icechest (`earthaccess-auth`).

### Testing

- Unit tests with injected pyramids (icechest's `open_pyramid` seam) and an
  on-disk Icechunk repo: Init declares correct shapes/coords/attrs; backfill
  region writes land at the right time index; forward append extends every
  (band, level) array and its `time`; region rewrite on re-delivery; grid
  mismatch rejected; duplicate-time detection in the inventory; a skipped
  granule appears in the partition's `skipped.json` and leaves a fill-valued
  step; an out-of-order forward append is sorted into place across every
  (band, level) group before commit, and an in-order batch is not touched.
- The template's existing backfill/fork-merge tests keep passing.
- One opt-in integration test (requires Earthdata + AWS credentials) that
  virtualizes 2–3 real T13SDA granules into a local store and reads a pixel
  time series back with xarray.

## Main technical risk

**Stacking virtual COG chunks into a pre-declared `(time, y, x)` array.** The
declared array's codecs/chunking must match the per-granule virtual arrays
exactly, and `region="auto"` must work with a `ManifestArray` that has been
`expand_dims`-ed to a length-1 time axis inside a nested group. The first
implementation task should be a **spike** proving this on 2 real granules
(including one from 2013 and one recent, to confirm the pyramid/codecs are
identical across the archive).

## Left to the builder

- Exact module/file split inside the processor package.
- Partition and batch sizes (template defaults are likely fine at this scale).
- How `inventory.json` is named/located in the backfill bucket.
- Logging, retries/backoff on COG header reads.
- Whether `time` is stored as `datetime64[ns]` or int64 + CF units, provided
  xarray decodes it as datetime.

## Spike exit criteria

The spike runs locally over https using the developer's Earthdata login
(`~/.netrc`); no AWS account is needed for it. It must show:

1. Two or more real T13SDA granules, **one from 2013 and one recent**, have
   identical per-(band, level) dtype, codecs, chunk shape, level count and
   level shapes.
2. Init-style full-shape declaration + `region="auto"` writes of those granules
   land at the right time indices and read back as correct pixel values with
   xarray.
3. An out-of-order append + per-group `reindex_array` sort produces a monotonic
   axis with correct values.

If (1) fails, **stop and bring the finding back to the user** before planning
further; the user expects it not to be an issue.

## Builder-check decisions (my calls, for the user to confirm while reading)

- Skipped backfill granules keep a fill-valued time slot rather than being
  removed from the axis (see Failure handling).
- Each level group carries its own `time`/`y`/`x` coordinates; the forward sort
  runs per group and asserts all groups agree.
- The s3 read path and the Lambda's Earthdata secret are verified only at deploy
  time, by an operator; the plan uses the https path for the spike and tests.
