from datetime import UTC, date, datetime

import pytest
from virtualizarr_processor.hls.assets import asset_key, read_url, reference_url
from virtualizarr_processor.hls.granule import (
    BANDS,
    granule_from_key,
    parse_granule_id,
)

GID = "HLS.L30.T13SDA.2025001T173012.v2.0"


def test_parse_granule_id():
    g = parse_granule_id(GID)
    assert (g.product, g.tile) == ("L30", "T13SDA")
    assert g.time == datetime(2025, 1, 1, 17, 30, 12, tzinfo=UTC)


def test_parse_granule_id_day_of_year():
    with pytest.raises(ValueError):
        parse_granule_id("HLS.L30.T13SDA.2013366T000000.v2.0")
    assert parse_granule_id("HLS.L30.T13SDA.2016366T000000.v2.0").time.date() == date(
        2016, 12, 31
    )


@pytest.mark.parametrize(
    "key",
    [
        GID,
        f"HLSL30.020/{GID}/{GID}.B04.tif",
        f"s3://lp-prod-protected/HLSL30.020/{GID}/{GID}.Fmask.tif",
        f"https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/HLSL30.020/{GID}/{GID}.B04.tif",
        f"HLSL30.020/{GID}/{GID}.B04.tif".replace("/", "%2F"),
    ],
)
def test_parse_granule_key_forms(key):
    assert granule_from_key(key).id == GID


@pytest.mark.parametrize(
    "bad",
    ["", "HLS.L30.T13SDA.v2.0", "HLS.L30.T13SDA.2025001T173012.v1.4", "foo/bar.tif"],
)
def test_parse_granule_rejects(bad):
    with pytest.raises(ValueError):
        granule_from_key(bad)


def test_asset_forms():
    key = asset_key(parse_granule_id(GID), "B04")
    assert key == f"HLSL30.020/{GID}/{GID}.B04.tif"
    assert read_url(key, "s3") == f"s3://lp-prod-protected/{key}"
    assert (
        read_url(key, "https")
        == f"https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/{key}"
    )
    assert reference_url(key) == f"vcc://lpdaac/{key}"


def test_read_url_rejects_unknown_access():
    key = f"HLSL30.020/{GID}/{GID}.B04.tif"
    with pytest.raises(ValueError):
        read_url(key, "ftp")  # type: ignore[arg-type]


def test_l30_bands():
    assert BANDS["L30"] == (
        "B01", "B02", "B03", "B04", "B05", "B06", "B07", "B09", "B10", "B11",
        "Fmask", "SZA", "SAA", "VZA", "VAA",
    )
