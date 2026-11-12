# HLS L30 Single-MGRS-Tile Datacube Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A virtualizarr-data-pipelines deployment that builds and maintains one Icechunk datacube of every HLS L30 granule for MGRS tile `T13SDA`, stacked along `time`, at `/{mgrs_id}/{band}/multiscales/{level}/{band}`.

**Architecture:** A new repo copied from the `virtualizarr-data-pipelines` template. A dataset-specific processor package (`virtualizarr_processor/hls/`) turns one granule ID into per-(band, level) virtual datasets with a length-1 `time` axis. Backfill declares the full cube at Init from an inventory of granule IDs and a tile grid embedded in the processor code, and workers write `region="auto"` into forks, skipping bad granules into per-partition `skipped.json`. Forward processing appends granules wherever they fall in time and sorts the time axis (via `Session.reindex_array`) before every commit, following naqfc-virtualizarr-data-pipelines PR #6.

**Tech Stack:** Python 3.12, icechunk ≥2.1, virtualizarr ≥2.7, virtual-tiff ≥0.5, zarr ≥3.2.1, xarray, earthaccess-auth ≥0.3.1, obstore/obspec-utils, pyarrow, AWS CDK (Lambda, Step Functions, SQS), pytest + moto, uv.

**Spec:** `docs/superpowers/specs/2026-11-11-hls-l30-mgrs-datacube-design.md`

## Global Constraints

- Repo root: `~/projects/hls-virtualizarr`, copied from `~/projects/virtualizarr-data-pipelines` at commit `545cb11`.
- Product **HLS L30 v2.0**, tile **`T13SDA`**, full archive history (April 2013 → present).
- Store layout: `/{mgrs_id}/{band}/multiscales/{level}/{band}`; each level group also holds `time`, `y`, `x` coordinate arrays.
- Data arrays: dims `("time", "y", "x")`, chunks `(1, cog_tile_y, cog_tile_x)`; dtype/fill/codecs equal to the COG's virtual arrays.
- L30 bands: `B01 B02 B03 B04 B05 B06 B07 B09 B10 B11 Fmask SZA SAA VZA VAA`.
- Granule ID form: `HLS.L30.{tile}.{YYYYDDD}T{HHMMSS}.v2.0`; time = UTC from `%Y%jT%H%M%S`.
- Asset key: `HLSL30.020/{id}/{id}.{band}.tif`. Read URLs: `s3://lp-prod-protected/{key}` (us-west-2) or `https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/{key}`. **Stored references: `vcc://lpdaac/{key}` only**; container name `lpdaac`.
- Archive: `s3://nasa-maap-data-store/file-staging/nasa-map/hls-stac-geoparquet-archive/v2/HLSL30_2.0/year={Y}/month={M}/HLSL30_2.0-{Y}-{M}.parquet` (month not zero-padded); anonymous HTTPS form `https://nasa-maap-data-store.s3.us-west-2.amazonaws.com/<same key>`.
- `time` encoding fixed: int64, `units="seconds since 1970-01-01T00:00:00"`, `calendar="proleptic_gregorian"`; xarray must decode it as datetime64.
- Deploy region `us-west-2`.
- No dependency on `icechest`; the needed pieces are vendored into `virtualizarr_processor/hls/`.
- No `SNS_TOPIC`. Nothing is deployed; `cdk synth` and tests are the bar.
- Commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **A forward message for the wrong tile or product** (e.g. `HLS.S30.T13SDA…` or `HLS.L30.T13SDB…`) must be rejected, not written into the cube. Test: Task 9 `test_process_file_rejects_other_tile_or_product`.
2. **The same granule delivered several times in one forward batch**, as a per-object S3 feed would do (one notification per band), must produce one time step, not several. Test: Task 9 `test_duplicate_messages_in_one_batch_write_one_step`.
3. **Message key forms:** a bare ID, an asset key, a full `s3://`/`https://` URL, or a URL-encoded key must all resolve to the same granule. Test: Task 3 `test_parse_granule_key_forms`.
4. **Forward processing against a store that was never backfilled** must fail clearly (message returns to the queue), not create a half-shaped cube. Test: Task 9 `test_process_file_without_backfilled_store_fails`.
5. **A granule missing one band's COG** (404) during backfill is skipped and recorded, with no partial bands written into the fork. Test: Task 9 `test_backfill_skip_writes_nothing_for_granule`.

---

## File Structure

All new processor code goes in `lambda/virtualizarr-processor/virtualizarr_processor/hls/`:

| File | Responsibility |
|---|---|
| `granule.py` | Parse granule IDs and message keys; product/tile/time; asset keys |
| `assets.py` | Read-URL and `vcc://` reference forms; band lists per product |
| `grid.py` | `TileGrid` (JSON round-trip) and per-level `x`/`y` coordinates |
| `conventions.py` | multiscales/proj/spatial attributes for a band group (vendored from icechest) |
| `pyramid.py` | Open a COG's pyramid as `{level: xr.Dataset}`; compare against declared arrays |
| `cube.py` | Declare the full cube; turn a granule into level datasets; write them |
| `sort.py` | Sort the `time` axis across every level group inside a session |
| `credentials.py` | Earthdata secret bootstrap, LP DAAC container/registry (vendored from icechest) |
| `inventory.py` | Archive month URIs, tile selection, duplicate-time check, grid pick |

`virtualizarr_processor/processor.py` is replaced by the HLS `Processor`. Handlers are changed in `lambda/backfill/backfill_handlers/{init,fork,worker,reduce}.py`. CDK: `cdk/settings.py`, `cdk/stack.py`, `cdk/stack_constructs/backfill_pipeline.py`. Scripts: `scripts/build_inventory.py`, `scripts/enqueue_granules.py`. Shared test fakes: `tests/hls/fakes.py`.

---

### Task 1: Scaffold the repo from the template

**Files:**
- Create: `~/projects/hls-virtualizarr/*` (copy of template)
- Modify: `pyproject.toml` (`name = "hls-virtualizarr"`), `README.md` (one-paragraph header naming the cube and linking the spec; template README kept below it)

- [ ] **Step 1: Copy the template, excluding VCS and caches**

```bash
cd ~/projects
rsync -a --exclude .git --exclude .venv --exclude '__pycache__' --exclude '.*_cache' \
  --exclude .DS_Store --exclude cdk.out --exclude .claude --exclude 'docs/superpowers' \
  virtualizarr-data-pipelines/ hls-virtualizarr/
```
`hls-virtualizarr/docs/superpowers/specs/` and `plans/` already exist and are preserved.

- [ ] **Step 2: Rename the project and add the README header**
- [ ] **Step 3: Install and run the template's test suite**

Run: `cd ~/projects/hls-virtualizarr && uv sync --all-groups && uv run pytest -q`
Expected: all template tests PASS.

- [ ] **Step 4: Initialise git and commit**

```bash
git init -b main && git add -A
git commit -m "Import virtualizarr-data-pipelines template at 545cb11 with HLS cube design"
```

---

### Task 2: Spike — prove virtual time-stacking on real granules (gate)

Throwaway code; the findings document is the deliverable.

**Files:**
- Create: `spike/time_stack.py`, `docs/superpowers/spikes/2026-11-11-time-stack-findings.md`
- Modify: `lambda/virtualizarr-processor/pyproject.toml` (add `virtual-tiff>=0.5`, `earthaccess-auth>=0.3.1`, `obstore`, `obspec-utils`, `xarray`, `pyarrow`)

- [ ] **Step 1: Find real granule IDs.** Read `HLSL30_2.0-2013-6.parquet` and `HLSL30_2.0-2025-6.parquet` over anonymous HTTPS (columns `id`, `proj:epsg`, `proj:shape`, `proj:transform`), filter `.T13SDA.`, and record 1 ID from 2013 and 2 from 2025, plus the newest 2025 row's exact `proj:epsg`, `proj:shape`, `proj:transform` — these become the `TILE_GRIDS["T13SDA"]` constant in Task 4.
- [ ] **Step 2: Open every L30 band's pyramid for those 3 granules over https.** Use `open_virtual_datatree(url, registry, parser=VirtualTIFF(ifd=None, ifd_layout="nested"))` with an obstore registry using a bearer token from `earthaccess_auth.login(strategy="netrc")`. Tabulate per (band, level): shape, chunk shape, dtype, fill value, codecs. **Compare 2013 vs 2025.**
- [ ] **Step 3: Declare a 3-step cube for `B04` and `Fmask` in a local Icechunk repo.** For each level, write a Dataset whose data variable is a `ManifestArray` with an **empty** `ChunkManifest` of shape `(3, *chunk_grid)` and the reference level's metadata with `time` prepended (dims `("time","y","x")`), plus `time` (encoding per Global Constraints) and `x`/`y` coordinates, via `vz.to_icechunk(store, group=…)`. Commit.
- [ ] **Step 4: Region-write the 3 granules** (each level dataset: rename variable to band, `expand_dims(time=[t])`, drop non-time variables, `to_icechunk(region="auto", validate_containers=False)`) in **reverse time order**, commit, and read a 3×3 pixel time series at level 0 and level 1 back with xarray over https. Compare with `rasterio`/`tifffile` reads of the same pixels.
- [ ] **Step 5: Out-of-order append.** Re-declare with 2 steps, region-write the 2025 granules, then append the 2013 granule with `append_dim="time"`, and run a `reindex_array`-based permutation on every level group (naqfc PR #6 approach). Verify `time` is monotonic and pixel values follow their times.
- [ ] **Step 6: Write findings.** Record the tables, COG tile size, level count, level shapes, and pass/fail for steps 3–5. Confirm or correct any assumption in the spec.
- [ ] **Step 7: GATE.** If 2013 and 2025 encodings differ in any compared field, or steps 3–5 fail, **stop and report to the user** before Task 3. Otherwise commit:

```bash
git add spike docs/superpowers/spikes lambda/virtualizarr-processor/pyproject.toml uv.lock
git commit -m "spike: prove virtual time-stacking of HLS L30 COGs"
```

---

### Task 3: Granule IDs and asset URLs

**Files:**
- Create: `virtualizarr_processor/hls/__init__.py`, `hls/granule.py`, `hls/assets.py`
- Test: `tests/hls/test_granule.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) class Granule: id: str; product: str; tile: str; time: datetime` (UTC-aware)
  - `parse_granule_id(granule_id: str) -> Granule` — raises `ValueError` on anything not matching `^HLS\.(L30|S30)\.(T\d{2}[A-Z]{3})\.(\d{7}T\d{6})\.v2\.0$`
  - `granule_from_key(key: str) -> Granule` — accepts bare ID, asset key, `s3://…`, `https://…`, URL-encoded
  - `BANDS: dict[str, tuple[str, ...]]` — `{"L30": (<the 15 L30 bands>)}` (S30 not defined yet)
  - `asset_key(granule: Granule, band: str) -> str` → `HLSL30.020/{id}/{id}.{band}.tif`
  - `read_url(key: str, access: Literal["s3","https"]) -> str`; `reference_url(key: str) -> str` → `vcc://lpdaac/{key}`
  - Constants `CONTAINER_NAME = "lpdaac"`, `LPDAAC_S3_PREFIX`, `LPDAAC_HTTPS_ASSET_PREFIX` (values as in icechest `demo/assets.py`)

- [ ] **Step 1: Write the failing tests**

```python
GID = "HLS.L30.T13SDA.2025001T173012.v2.0"

def test_parse_granule_id():
    g = parse_granule_id(GID)
    assert (g.product, g.tile) == ("L30", "T13SDA")
    assert g.time == datetime(2025, 1, 1, 17, 30, 12, tzinfo=UTC)

def test_parse_granule_id_day_of_year():
    with pytest.raises(ValueError):
        parse_granule_id("HLS.L30.T13SDA.2013366T000000.v2.0")
    assert parse_granule_id("HLS.L30.T13SDA.2016366T000000.v2.0").time.date() == date(2016, 12, 31)

@pytest.mark.parametrize("key", [
    GID,
    f"HLSL30.020/{GID}/{GID}.B04.tif",
    f"s3://lp-prod-protected/HLSL30.020/{GID}/{GID}.Fmask.tif",
    f"https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/HLSL30.020/{GID}/{GID}.B04.tif",
    f"HLSL30.020/{GID}/{GID}.B04.tif".replace("/", "%2F"),
])
def test_parse_granule_key_forms(key):
    assert granule_from_key(key).id == GID

@pytest.mark.parametrize("bad", ["", "HLS.L30.T13SDA.v2.0", "HLS.L30.T13SDA.2025001T173012.v1.4", "foo/bar.tif"])
def test_parse_granule_rejects(bad):
    with pytest.raises(ValueError):
        granule_from_key(bad)

def test_asset_forms():
    key = asset_key(parse_granule_id(GID), "B04")
    assert key == f"HLSL30.020/{GID}/{GID}.B04.tif"
    assert read_url(key, "s3") == f"s3://lp-prod-protected/{key}"
    assert read_url(key, "https") == f"https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/{key}"
    assert reference_url(key) == f"vcc://lpdaac/{key}"

def test_l30_bands():
    assert BANDS["L30"] == ("B01","B02","B03","B04","B05","B06","B07","B09","B10","B11","Fmask","SZA","SAA","VZA","VAA")
```

- [ ] **Step 2: Run to verify failure** — `uv run pytest tests/hls/test_granule.py -v` → FAIL (import error)
- [ ] **Step 3: Implement** in `hls/granule.py` and `hls/assets.py`. `granule_from_key`: `urllib.parse.unquote`, take the last path segment, strip a trailing `.{band}.tif`, then `parse_granule_id`.
- [ ] **Step 4: Run to verify pass**
- [ ] **Step 5: Commit** — `feat(hls): granule id parsing and asset url forms`

---

### Task 4: Tile grid and convention attributes

**Files:**
- Create: `hls/grid.py`, `hls/conventions.py`
- Test: `tests/hls/test_grid.py`, `tests/hls/test_conventions.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) class TileGrid: mgrs_id: str; epsg: int; shape: tuple[int, int]; transform: tuple[float, ...]` (9-element row-major affine, as STAC `proj:transform`)
  - `TILE_GRIDS: dict[str, TileGrid]` — `{"T13SDA": TileGrid("T13SDA", <epsg>, <shape>, <transform>)}` with the **exact values recorded in the Task 2 findings** (newest T13SDA granule's STAC `proj:*`)
  - `tile_grid(tile: str) -> TileGrid` — `KeyError`-free lookup that raises `ValueError(f"no grid for {tile}")`
  - `level_coords(grid: TileGrid, ny: int, nx: int) -> tuple[np.ndarray, np.ndarray]` — pixel-centre `y`, `x` for a level of `ny × nx` covering the full cell; resolution = extent / n per axis
  - `band_group_attrs(grid: TileGrid, *, levels: int, band: str) -> dict[str, Any]` — icechest `granule_attrs` content (multiscales layout with `multiscales/{level}/{band}` assets, `proj:code`, `spatial:*`)

- [ ] **Step 1: Write the failing tests**

```python
GRID = TileGrid("T13SDA", 32613, (3660, 3660), (30.0, 0.0, 399960.0, 0.0, -30.0, 4300020.0, 0.0, 0.0, 1.0))

def test_t13sda_grid_is_embedded():
    g = tile_grid("T13SDA")
    assert g.mgrs_id == "T13SDA" and g.shape == (3660, 3660)
    assert g.transform[0] == 30.0 and g.transform[4] == -30.0
    assert g.epsg == SPIKE_EPSG and g.transform[2] == SPIKE_X0 and g.transform[5] == SPIKE_Y0  # constants copied from findings

def test_unknown_tile_has_no_grid():
    with pytest.raises(ValueError, match="T99XXX"):
        tile_grid("T99XXX")

def test_level0_coords_are_pixel_centres():
    y, x = level_coords(GRID, 3660, 3660)
    assert x[0] == 399975.0 and x[-1] == 399960.0 + 3660 * 30 - 15
    assert y[0] == 4300005.0 and y[-1] == 4300020.0 - 3660 * 30 + 15
    assert len(x) == len(y) == 3660

def test_overview_coords_cover_cell_with_rounded_size():
    y, x = level_coords(GRID, 458, 458)
    res = 3660 * 30 / 458
    assert x[0] == pytest.approx(399960.0 + res / 2)
    assert x[-1] + res / 2 == pytest.approx(399960.0 + 3660 * 30)

def test_band_group_attrs():
    attrs = band_group_attrs(GRID, levels=3, band="B04")
    assert attrs["proj:code"] == "EPSG:32613"
    assert attrs["spatial:shape"] == [3660, 3660]
    assert attrs["spatial:bbox"] == [399960.0, 4300020.0 - 109800.0, 399960.0 + 109800.0, 4300020.0]
    layout = attrs["multiscales"]["layout"]
    assert [e["asset"] for e in layout] == ["multiscales/0/B04", "multiscales/1/B04", "multiscales/2/B04"]
    assert layout[1]["derived_from"] == "multiscales/0/B04"
```

- [ ] **Step 2: Run to verify failure**
- [ ] **Step 3: Implement.** `conventions.py` is a vendored copy of icechest `demo/conventions.py` with `granule_attrs` renamed `band_group_attrs` and taking a `TileGrid`.
- [ ] **Step 4: Run to verify pass**
- [ ] **Step 5: Commit** — `feat(hls): tile grid, level coordinates and convention attributes`

---

### Task 5: Pyramid opening, contract check, and test fakes

**Files:**
- Create: `hls/pyramid.py`, `tests/hls/fakes.py`, `tests/hls/conftest.py`
- Test: `tests/hls/test_pyramid.py`

**Interfaces:**
- Produces:
  - `Pyramid = dict[int, xr.Dataset]` — one single-variable `(y, x)` virtual dataset per level, level 0 first
  - `PyramidOpener = Callable[[str, Any], Pyramid]` — `(read_url, registry) -> Pyramid`
  - `open_pyramid(url: str, registry: Any) -> Pyramid` — real opener (virtual-tiff, `ifd=None`, `ifd_layout="nested"`)
  - `class GridMismatch(Exception)`
  - `check_level(declared: zarr.Array, level: xr.Dataset) -> None` — raises `GridMismatch` naming the field when shape (excluding `time`), chunks (excluding `time`), dtype, fill value or codecs differ
  - `tests/hls/fakes.py`:
    - `FAKE_GRID: TileGrid` — `shape=(8, 8)`, 30 m, EPSG:32613
    - `FakeAssets(root: Path)` with `.write(granule_id: str, band: str, value: int, *, levels=((8, 8), (4, 4)), chunks=(4, 4), dtype="uint16") -> None` (writes raw chunk files on disk) and `.opener: PyramidOpener` (returns `ManifestArray` datasets pointing at those files, `BytesCodec` only); `.missing(granule_id, band)` makes the opener raise `FileNotFoundError` for that asset
    - `local_locator(root) ` returning read URL = reference URL = `file://{root}/{key}` so references resolve in tests
  - `tests/hls/conftest.py`: fixture `hls_repo(tmp_path) -> icechunk.Repository` (local filesystem storage, virtual chunk container for `file://{tmp_path}/assets/`)

- [ ] **Step 1: Write the failing tests**

```python
def test_check_level_accepts_matching(hls_repo, fake_assets):
    fake_assets.write(GID, "B04", 1)
    level0 = fake_assets.opener(url_for(GID, "B04"), None)[0]
    declared = declare_like(hls_repo, level0)  # helper: zarr array (3, 8, 8), chunks (1, 4, 4), same codecs
    check_level(declared, level0)

@pytest.mark.parametrize("field,kwargs", [
    ("shape", {"levels": ((16, 16), (8, 8))}),
    ("chunks", {"chunks": (2, 2)}),
    ("dtype", {"dtype": "int16"}),
])
def test_check_level_rejects(hls_repo, fake_assets, field, kwargs):
    fake_assets.write(GID, "B04", 1)
    declared = declare_like(hls_repo, fake_assets.opener(url_for(GID, "B04"), None)[0])
    fake_assets.write(GID2, "B04", 1, **kwargs)
    with pytest.raises(GridMismatch, match=field):
        check_level(declared, fake_assets.opener(url_for(GID2, "B04"), None)[0])
```
`declare_like` lives in `tests/hls/fakes.py`.

- [ ] **Step 2: Run to verify failure**
- [ ] **Step 3: Implement** `pyramid.py` (`open_pyramid` adapted from icechest `_open_pyramid`, returning `{int(k): node.to_dataset()}`) and the fakes, which follow naqfc `tests/test_example.py::virtual_cycle_cube` for building `ManifestArray`s over local files.
- [ ] **Step 4: Run to verify pass**
- [ ] **Step 5: Commit** — `feat(hls): pyramid opener, contract check and test fakes`

---

### Task 6: Cube declaration and granule writes

**Files:**
- Create: `hls/cube.py`
- Test: `tests/hls/test_cube.py`

**Interfaces:**
- Consumes: `Granule`, `asset_key`, `reference_url` (Task 3); `TileGrid`, `level_coords`, `band_group_attrs` (Task 4); `Pyramid`, `PyramidOpener`, `check_level`, `GridMismatch` (Task 5)
- Produces:
  - `level_group(tile: str, band: str, level: int) -> str` → `"{tile}/{band}/multiscales/{level}"`
  - `level_groups(store: Any, tile: str) -> list[str]` — every existing level group under `/{tile}`
  - `declare_cube(store, grid: TileGrid, times: Sequence[datetime], reference: dict[str, Pyramid]) -> None` — for every band in `reference`, every level: data array `(len(times), ny, nx)` with no chunks, `time`/`y`/`x` coordinates, and `band_group_attrs` on `/{tile}/{band}`. Uses the empty-manifest `vz.to_icechunk` approach proven in Task 2.
  - `granule_levels(granule: Granule, band: str, pyramid: Pyramid) -> dict[int, xr.Dataset]` — variable renamed to `band`, `time` expanded with `granule.time`, references rewritten to `reference_url(asset_key(...))`, no variables without `time`
  - `open_granule(granule, bands, opener, registry, read_url_for: Callable[[str], str]) -> dict[str, Pyramid]` — opens all bands **before** anything is written
  - `write_granule(store, tile: str, granule: Granule, pyramids: dict[str, Pyramid], *, mode: Literal["region", "append"]) -> None` — runs `check_level` against every declared array first, then writes every (band, level); `validate_containers=False`
  - `store_times(store, tile: str) -> np.ndarray | None` — decoded `time` of `level_group(tile, first_band, 0)`, or `None` if absent

- [ ] **Step 1: Write the failing tests**

```python
TIMES = [datetime(2013, 6, 1, 17, tzinfo=UTC), datetime(2025, 6, 1, 17, tzinfo=UTC), datetime(2025, 6, 17, 17, tzinfo=UTC)]

def test_declare_cube_shapes_coords_attrs(hls_repo, reference_pyramids):
    s = hls_repo.writable_session("main")
    declare_cube(s.store, FAKE_GRID, TIMES, reference_pyramids)
    ds0 = xr.open_zarr(s.store, group="T13SDA/B04/multiscales/0", consolidated=False)
    ds1 = xr.open_zarr(s.store, group="T13SDA/B04/multiscales/1", consolidated=False)
    assert ds0["B04"].dims == ("time", "y", "x") and ds0["B04"].shape == (3, 8, 8)
    assert ds1["B04"].shape == (3, 4, 4)
    assert list(ds0.time.values) == [np.datetime64(t.replace(tzinfo=None)) for t in TIMES]
    assert ds0["B04"].encoding["chunks"] == (1, 4, 4)
    assert zarr.open_group(s.store, path="T13SDA/B04").attrs["proj:code"] == "EPSG:32613"

def test_region_writes_land_at_their_time(hls_repo, fake_assets, reference_pyramids):
    ...  # declare with TIMES; write granules for TIMES[2] (value 3) then TIMES[0] (value 1) with mode="region"
    assert ds0["B04"].isel(time=0).values.tolist() == [[1] * 8] * 8
    assert ds0["B04"].isel(time=2).values.max() == 3
    fill = ds0["B04"].encoding.get("_FillValue", zarr.open_array(s.store, path="T13SDA/B04/multiscales/0/B04").fill_value)
    assert (ds0["B04"].isel(time=1).values == fill).all() or np.isnan(ds0["B04"].isel(time=1).values).all()  # untouched step reads fill

def test_append_extends_every_group(hls_repo, fake_assets, reference_pyramids):
    ...  # declare with TIMES[:2]; append a granule at TIMES[2]
    for g in level_groups(s.store, "T13SDA"):
        assert xr.open_zarr(s.store, group=g, consolidated=False).sizes["time"] == 3

def test_write_granule_mismatch_writes_nothing(hls_repo, fake_assets, reference_pyramids):
    ...  # B04 matches, Fmask has wrong chunks
    with pytest.raises(GridMismatch):
        write_granule(s.store, "T13SDA", granule, pyramids, mode="region")
    assert not s.has_uncommitted_changes  # after declare was committed
```
The implementer fills the `...` setup with the fakes from Task 5; the assertions are fixed.

- [ ] **Step 2: Run to verify failure**
- [ ] **Step 3: Implement `hls/cube.py`**
- [ ] **Step 4: Run to verify pass**
- [ ] **Step 5: Commit** — `feat(hls): declare the tile cube and write granules into it`

---

### Task 7: Sort the time axis across level groups

**Files:**
- Create: `hls/sort.py`
- Test: `tests/hls/test_sort.py`

**Interfaces:**
- Consumes: `level_groups`, `write_granule` (Task 6)
- Produces: `sort_time(session: icechunk.Session, tile: str) -> int` — rows moved (0 if already ordered)

Algorithm (adapted from naqfc PR #6 `sort_reference_time`, applied per group):
1. For every group in `level_groups(session.store, tile)`, read raw `time` values; **raise `ValueError` if any group's raw `time` differs from the first group's** or has duplicates.
2. `order = argsort(times, kind="stable")`; return 0 if identity.
3. Before moving anything, check every `time`-dimensioned array has time chunk size 1, else raise.
4. For every group and its data array: `session.reindex_array(f"/{group}/{band}", forward, backward)` with `old_to_new[order] = arange(n)`.
5. Permute each group's raw `time` with zarr (`coord[:] = coord[:][order]`), not via xarray.

- [ ] **Step 1: Write the failing tests**

```python
def test_sort_noop_when_ordered(...):
    assert sort_time(session, "T13SDA") == 0
    assert not session.has_uncommitted_changes  # nothing dirtied (after a clean commit)

def test_out_of_order_append_is_sorted_in_every_group(...):
    # declare [2025-06-01, 2025-06-17] with values 2, 3; append 2013-06-01 with value 1
    assert sort_time(session, "T13SDA") == 3
    for g in level_groups(session.store, "T13SDA"):
        ds = xr.open_zarr(session.store, group=g, consolidated=False)
        assert (np.diff(ds.time.values) > np.timedelta64(0)).all()
        band = g.split("/")[1]
        assert [int(ds[band].isel(time=i).max()) for i in range(3)] == [1, 2, 3]

def test_sort_refuses_duplicate_times(...):
    with pytest.raises(ValueError, match="duplicate"):
        sort_time(session, "T13SDA")

def test_sort_refuses_disagreeing_groups(...):
    # append to B04 groups only
    with pytest.raises(ValueError, match="disagree"):
        sort_time(session, "T13SDA")
```

- [ ] **Step 2: Run to verify failure**
- [ ] **Step 3: Implement `hls/sort.py`**
- [ ] **Step 4: Run to verify pass**
- [ ] **Step 5: Commit** — `feat(hls): sort the time axis across level groups before commit`

---

### Task 8: Credentials

**Files:**
- Create: `hls/credentials.py`
- Test: `tests/hls/test_credentials.py`

**Interfaces:**
- Produces:
  - `bootstrap_earthdata_env() -> None` — if `EARTHDATA_SECRET_ARN` is set and `EARTHDATA_USERNAME` is not, read the secret (`{"username", "password"}`) with boto3 Secrets Manager and set `EARTHDATA_USERNAME`/`EARTHDATA_PASSWORD`; otherwise do nothing. Idempotent.
  - `virtual_chunk_container(access, *, token=None)`, `container_credentials(access)`, `object_store_registry(access, *, token=None)` — vendored from icechest `demo/credentials.py`, importing constants from `hls/assets.py`

- [ ] **Step 1: Write the failing tests** (moto Secrets Manager)

```python
def test_bootstrap_sets_env_from_secret(monkeypatch, secret_arn):  # secret {"username": "u", "password": "p"}
    monkeypatch.setenv("EARTHDATA_SECRET_ARN", secret_arn)
    monkeypatch.delenv("EARTHDATA_USERNAME", raising=False)
    bootstrap_earthdata_env()
    assert os.environ["EARTHDATA_USERNAME"] == "u" and os.environ["EARTHDATA_PASSWORD"] == "p"

def test_bootstrap_noop_without_arn(monkeypatch):
    monkeypatch.delenv("EARTHDATA_SECRET_ARN", raising=False)
    bootstrap_earthdata_env()  # no boto3 call, no error

def test_container_is_named_lpdaac():
    assert virtual_chunk_container("s3").name == "lpdaac"
```

- [ ] **Step 2: Run to verify failure**
- [ ] **Step 3: Implement**
- [ ] **Step 4: Run to verify pass**
- [ ] **Step 5: Commit** — `feat(hls): earthdata secret bootstrap and LP DAAC container`

---

### Task 9: The HLS Processor

**Files:**
- Modify (replace): `virtualizarr_processor/processor.py`
- Modify: `virtualizarr_processor/typing.py` (new `initialize_backfill_store` signature; `skipped` attribute)
- Delete/replace template tests that exercise the synthetic processor: `tests/test_example.py`, `tests/test_write_modes.py`, `tests/conftest.py` synthetic fixtures; keep `tests/backfill_mechanics/` if it does not import the synthetic processor, otherwise point it at the fakes
- Test: `tests/hls/test_processor.py`

**Interfaces:**
- Consumes: everything in Tasks 3–8
- Produces:
  - `@dataclass class SkipRecord: granule_id: str; stage: Literal["parse", "open", "check", "write"]; error: str` with `.to_dict()`
  - `class Processor`:
    - `__init__(self, *, tile: str | None = None, product: str | None = None, access: str | None = None, opener: PyramidOpener = open_pyramid, locator: Locator | None = None, repo_opener: Callable[[], Repository] | None = None)` — `None` defaults read env `MGRS_TILE`, `HLS_PRODUCT` (default `"L30"`), `LPDAAC_ACCESS` (default `"s3"`); default `repo_opener` builds S3 storage from `ICECHUNK_BUCKET`/`ICECHUNK_PREFIX`/`ICECHUNK_REGION` (or `ICECHUNK_LOCAL_PATH`) with the `lpdaac` container and a raised `CachingConfig(num_chunk_refs=…)`, calling `bootstrap_earthdata_env()` first
    - `skipped: list[SkipRecord]`
    - `initialize_repo() -> Repository`; `open_backfill_repo() -> Repository` — both `repo_opener()`; neither writes data
    - `initialize_backfill_store(repo, granule_ids: list[str]) -> str` — uses `tile_grid(self.tile)`; creates `backfill` from `main`, opens the **newest** granule's pyramids as reference, `declare_cube` with all parsed times (sorted), commits, returns snapshot id; raises if any ID is for another tile/product
    - `process_backfill_file(file_key: str, fork: ForkSession) -> bool` — `region` write; on any per-granule failure appends a `SkipRecord` and returns `False` with **nothing written for that granule**
    - `initialize_session(repo) -> Session` (`main`)
    - `process_file(file_key: str, session: Session) -> bool` — rejects other tile/product; `store_times is None` → `False` with a logged "store not backfilled" error; time on axis → region; else append
    - `commit_processed_files(session) -> str` — `sort_time(session, tile)` then commit
    - `garbage_collect(expiry_time)` — as template, via `repo_opener`
  - `Locator` protocol: `read_url(key: str) -> str`, `reference(key: str) -> str`; default `LpDaacLocator(access)` built on Task 3

- [ ] **Step 1: Write the failing tests** (using fakes; `processor` fixture wires `FakeAssets.opener`, `local_locator`, `hls_repo`)

```python
def test_backfill_end_to_end_in_process(processor, fake_assets, hls_repo):
    ids = [GID_2013, GID_2025A, GID_2025B]
    for v, gid in enumerate(ids, 1): write_all_bands(fake_assets, gid, v)
    processor.initialize_backfill_store(hls_repo, ids, FAKE_GRID)
    fork = hls_repo.writable_session("backfill").fork()
    assert all(processor.process_backfill_file(g, fork) for g in reversed(ids))
    merge_and_commit(hls_repo, [pickle.dumps(fork)], message="p0")
    ds = open_level(hls_repo, "backfill", "B04", 0)
    assert [int(ds.B04.isel(time=i).max()) for i in range(3)] == [1, 2, 3]

def test_backfill_skip_records_and_returns_false(...):
    fake_assets.missing(GID_2025A, "Fmask")
    assert processor.process_backfill_file(GID_2025A, fork) is False
    assert processor.skipped == [SkipRecord(GID_2025A, "open", ANY)]

def test_backfill_skip_writes_nothing_for_granule(...):
    # Review Focus 5: one band 404s -> no band of that granule is referenced in the fork
    ...
    assert fork_has_no_chunks_at_time(fork, index_of(GID_2025A))

def test_process_file_appends_and_sorts_out_of_order(...):
    # backfilled with 2025 granules; forward 2013 granule via asset-key message
    assert processor.process_file(f"HLSL30.020/{GID_2013}/{GID_2013}.B04.tif", session)
    processor.commit_processed_files(session)
    assert monotonic_and_values(hls_repo, "main", [1, 2, 3])

def test_process_file_redelivery_is_region_write(...):
    # same granule twice across two batches -> time length unchanged

def test_duplicate_messages_in_one_batch_write_one_step(...):
    # Review Focus 2: same granule's B04 and Fmask keys in one session -> one new step
    for band in ("B04", "Fmask"):
        assert processor.process_file(f"HLSL30.020/{GID_NEW}/{GID_NEW}.{band}.tif", session)
    processor.commit_processed_files(session)
    assert time_len(hls_repo, "main") == 3

def test_process_file_rejects_other_tile_or_product(...):
    # Review Focus 1
    assert not processor.process_file("HLS.S30.T13SDA.2025001T173012.v2.0", session)
    assert not processor.process_file("HLS.L30.T13SDB.2025001T173012.v2.0", session)
    assert not session.has_uncommitted_changes

def test_process_file_without_backfilled_store_fails(...):
    # Review Focus 4
    assert processor.process_file(GID_2025A, fresh_main_session) is False
    assert not fresh_main_session.has_uncommitted_changes

def test_initialize_backfill_store_rejects_foreign_ids(...):
    with pytest.raises(ValueError):
        processor.initialize_backfill_store(hls_repo, [GID_2025A, "HLS.L30.T13SDB.2025001T173012.v2.0"], FAKE_GRID)
```

- [ ] **Step 2: Run to verify failure**
- [ ] **Step 3: Implement `processor.py` and update `typing.py`**
- [ ] **Step 4: Run** `uv run pytest -q` — the whole suite must pass, including `tests/test_handler.py` (forward handler) adjusted only for the new processor fixtures
- [ ] **Step 5: Commit** — `feat: HLS tile-cube processor for backfill and forward processing`

---

### Task 10: Backfill handlers and state machine

**Files:**
- Modify: `lambda/backfill/backfill_handlers/init.py`, `fork.py`, `worker.py`, `reduce.py`, `inventory.py` (add `read_json(uri) -> Any`, `write_json(uri, obj) -> None`)
- Modify: `cdk/stack_constructs/backfill_pipeline.py`
- Test: `tests/backfill_handlers/test_{init,fork,worker,reduce,end_to_end}.py`, `tests/cdk/test_backfill_pipeline.py`

**Interfaces:**
- Consumes: `Processor.initialize_backfill_store(repo, granule_ids)`, `Processor.skipped`, `SkipRecord.to_dict()`
- Produces (event contracts):
  - Init event `{"inventory_uri"}`; reads the inventory and passes the IDs to the processor
  - Fork result adds `"skipped_out_prefix": f"{run_prefix}forks/{partition_id}/skipped/"`
  - Worker `BatchInput` adds `skipped_out_prefix`; worker writes `{skipped_out_prefix}{uuid}.json` (list of skip dicts) **before** saving its fork, only if non-empty; `False` from `process_backfill_file` no longer raises; exceptions still raise
  - Reduce event adds `run_prefix`, `skipped_out_prefix`; writes `{run_prefix}partitions/{partition_id}/skipped.json` (dedupe by `granule_id`, sorted; `[]` when none) and returns `"skipped": <count>`

- [ ] **Step 1: Write the failing tests**

```python
def test_init_reads_inventory(s3_bucket, lambda_context, monkeypatch, fake_processor): ...
    # inventory.json in moto; asserts processor got the IDs

def test_worker_records_skips_and_does_not_raise(...):
    # processor fake returns False for one key and appends a SkipRecord
    out = worker.handler({...,"BatchInput": {..., "skipped_out_prefix": P}}, ctx)
    assert list_keys(P) == [ANY_JSON] and json.loads(body)[0]["granule_id"] == BAD

def test_reduce_writes_partition_skipped_json(...):
    # two worker skip files, one duplicate granule
    assert read_json(f"{run_prefix}partitions/0/skipped.json") == [{"granule_id": BAD, ...}]

def test_reduce_writes_empty_skipped_json(...):
    assert read_json(f"{run_prefix}partitions/0/skipped.json") == []

def test_state_machine_passes_inventory_to_init(template):
    # InitTask Parameters.Payload has "inventory_uri.$": "$.inventory_uri"

def test_state_machine_passes_skip_prefix(template):
    # ItemBatcher BatchInput has "skipped_out_prefix.$"; ReduceTask payload has run_prefix and skipped_out_prefix
```

- [ ] **Step 2: Run to verify failure**
- [ ] **Step 3: Implement handlers and state machine changes**
- [ ] **Step 4: Run** `uv run pytest tests/backfill_handlers tests/cdk -q` → PASS
- [ ] **Step 5: Commit** — `feat(backfill): inventory-driven init and per-partition skipped.json`

---

### Task 11: Settings and CDK wiring

**Files:**
- Modify: `cdk/settings.py`, `cdk/stack.py`, `cdk/stack_constructs/backfill_pipeline.py`, `.env.sample`, `lambda/*/Dockerfile` only if the new processor deps need system packages
- Test: `tests/cdk/test_settings.py`, `tests/cdk/test_hls_wiring.py`

**Interfaces:**
- Produces settings: `MGRS_TILE: str = "T13SDA"`, `HLS_PRODUCT: Literal["L30"] = "L30"`, `EARTHDATA_SECRET_ARN: str | None = None`, `LPDAAC_ACCESS: Literal["s3","https"] = "s3"`, `ICECHUNK_NUM_CHUNK_REFS: int` (default sized for `BACKFILL_PARTITION_SIZE × 15 bands × levels × chunks/level`, value from the Task 2 findings, with the arithmetic in a comment)

- [ ] **Step 1: Write the failing tests**

```python
def test_processor_lambdas_get_hls_env(template):
    # init, worker, reduce, fork, promote, process_messages, garbage_collect functions
    # have MGRS_TILE=T13SDA, HLS_PRODUCT=L30, LPDAAC_ACCESS=s3
def test_earthdata_secret_granted_to_readers(template_with_secret):
    # init, worker, process_messages: env EARTHDATA_SECRET_ARN and secretsmanager:GetSecretValue on that ARN
    # fork/reduce/promote: no grant
def test_no_sns_subscription_by_default(template):
    # no AWS::SNS::Subscription resource
def test_settings_defaults():
    assert Settings(...).MGRS_TILE == "T13SDA" and Settings(...).HLS_PRODUCT == "L30"
```

- [ ] **Step 2: Run to verify failure**
- [ ] **Step 3: Implement**
- [ ] **Step 4: Run** `uv run pytest tests/cdk -q` and `STAGE=dev ACCOUNT_ID=000000000000 ACCOUNT_REGION=us-west-2 uv run cdk synth > /dev/null` → PASS / exit 0
- [ ] **Step 5: Commit** — `feat(cdk): HLS tile settings and Earthdata secret wiring`

---

### Task 12: Inventory and enqueue scripts

**Files:**
- Create: `hls/inventory.py`, `scripts/build_inventory.py`, `scripts/enqueue_granules.py`
- Test: `tests/hls/test_inventory.py`, `tests/test_scripts.py`

**Interfaces:**
- Consumes: `parse_granule_id`, `TileGrid`
- Produces:
  - `month_uris(product: str, start: date, end: date, *, https: bool = True) -> list[str]` — one per month, inclusive, non-padded months, archive path per Global Constraints
  - `select_tile(table: pa.Table, tile: str, product: str) -> list[str]` — IDs for that tile/product, sorted by parsed time, deduplicated by ID
  - `check_unique_times(ids: list[str]) -> None` — `ValueError` listing every colliding pair
  - `scripts/build_inventory.py --tile T13SDA --product L30 --start 2013-04 --end YYYY-MM --out DIR [--upload s3://bucket/prefix/]` — writes `DIR/inventory.json`; prints count and time range; a month file that 404s is reported and skipped only if it is the current month, otherwise fatal
  - `scripts/enqueue_granules.py --queue-url URL [IDS... | --from inventory.json] [--since YYYY-MM-DD]` — sends one S3-notification-shaped message per granule (`Records[0].s3.bucket.name = "lp-prod-protected"`, `object.key = asset_key(granule, "Fmask")`), batches of 10

- [ ] **Step 1: Write the failing tests**

```python
def test_month_uris_span_and_padding():
    uris = month_uris("L30", date(2013, 4, 1), date(2013, 6, 1))
    assert [u.rsplit("/", 1)[-1] for u in uris] == ["HLSL30_2.0-2013-4.parquet", "HLSL30_2.0-2013-5.parquet", "HLSL30_2.0-2013-6.parquet"]
    assert "/year=2013/month=4/" in uris[0]

def test_select_tile_filters_and_sorts():
    t = pa.table({"id": [L30_B, S30_A, L30_A, L30_OTHER_TILE, L30_A]})
    assert select_tile(t, "T13SDA", "L30") == [L30_A, L30_B]

def test_check_unique_times_reports_collisions():
    # select_tile dedupes identical IDs, so a collision here means two distinct IDs
    # that parse to the same instant; simulate with a caller that skipped dedupe.
    gid = "HLS.L30.T13SDA.2025001T173012.v2.0"
    with pytest.raises(ValueError, match="2025001T173012"):
        check_unique_times([gid, gid])

def test_enqueue_message_shape(sqs_queue):
    # message body parses through process_messages handler's key extraction to the granule
```

- [ ] **Step 2: Run to verify failure**
- [ ] **Step 3: Implement**
- [ ] **Step 4: Run to verify pass**
- [ ] **Step 5: Commit** — `feat: tile inventory builder and granule enqueue script`

---

### Task 13: Opt-in integration test and README

**Files:**
- Create: `tests/integration/test_real_granules.py` (marked `@pytest.mark.network`, skipped unless `HLS_NETWORK_TESTS=1`)
- Modify: `pyproject.toml` (register `network` marker; default `-m "not network"`), `README.md` (HLS section: build inventory → upload → deploy with `BACKFILL_ENABLED=true` → `start_backfill.sh` → inspect `partitions/*/skipped.json` → enable forward → `enqueue_granules.py`; note there is no SNS feed)

- [ ] **Step 1: Write the test.** Using `LPDAAC_ACCESS=https` and `~/.netrc`, the 3 granule IDs from the Task 2 findings, a local repo: `initialize_backfill_store` → `process_backfill_file` ×3 into one fork → merge/commit → forward-append a 4th recent granule → `commit_processed_files`. Assert: `time` monotonic, length 4; a level-0 `B04` 3×3 pixel window per step equals a direct COG read of the same window; `Fmask` level 1 opens.
- [ ] **Step 2: Run** `HLS_NETWORK_TESTS=1 uv run pytest tests/integration -v -m network` → PASS
- [ ] **Step 3: Run the full suite and lint** — `uv run pytest -q && uv run ruff check . && uv run mypy` → all pass
- [ ] **Step 4: Commit** — `test: real-granule integration test; docs: HLS operating guide`
