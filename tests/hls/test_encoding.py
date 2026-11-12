"""Documented band encodings, and restoring a fill lost with a COG's nodata tag."""

import pytest
from fakes import url_for
from virtualizarr_processor.hls.encoding import (
    BAND_ENCODINGS,
    BandEncoding,
    restore_untagged_fill,
)
from virtualizarr_processor.hls.granule import BANDS

GID = "HLS.L30.T13SDA.2026120T174345.v2.0"
L30 = BAND_ENCODINGS["L30"]


def _fill(pyramid, level=0):
    return pyramid[level]["band_data"].data.metadata.fill_value


def test_every_l30_band_has_an_encoding():
    assert set(L30) == set(BANDS["L30"])


@pytest.mark.parametrize(
    "bands, expected",
    [
        (
            ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B09"),
            BandEncoding("int16", -9999, 0.0001),
        ),
        (("B10", "B11"), BandEncoding("int16", -9999, 0.01)),
        (("Fmask",), BandEncoding("uint8", 255)),
        (("SAA", "SZA", "VAA", "VZA"), BandEncoding("uint16", 40000, 0.01)),
    ],
)
def test_documented_values(bands, expected):
    assert all(L30[band] == expected for band in bands)


def test_attributes():
    assert L30["SAA"].attributes() == {
        "_FillValue": 40000,
        "scale_factor": 0.01,
        "add_offset": 0.0,
    }
    # Fmask is bit flags: masked, never scaled
    assert L30["Fmask"].attributes() == {"_FillValue": 255}


def test_untagged_angle_band_gets_documented_fill(fake_assets):
    fake_assets.write(GID, "SAA", 13500, fill_value=0)  # nodata tag missing
    pyramid = fake_assets.opener(url_for(GID, "SAA"), None)
    restored = restore_untagged_fill(L30, {"SAA": pyramid})["SAA"]
    assert [_fill(restored, lvl) for lvl in restored] == [40000, 40000]
    # metadata only: the chunk references are the same
    before = pyramid[0]["band_data"].data.manifest.dict()
    assert restored[0]["band_data"].data.manifest.dict() == before


def test_tagged_band_is_unchanged(fake_assets):
    fake_assets.write(GID, "SAA", 13500)
    pyramid = fake_assets.opener(url_for(GID, "SAA"), None)
    restored = restore_untagged_fill(L30, {"SAA": pyramid})["SAA"]
    assert all(restored[lvl] is pyramid[lvl] for lvl in pyramid)


@pytest.mark.parametrize(
    "layout",
    [
        {"fill_value": 7},  # a different fill: for the grid check to reject
        {"dtype": "int16", "fill_value": 0},  # wrong dtype: not ours to repair
    ],
)
def test_other_disagreements_are_left_alone(fake_assets, layout):
    fake_assets.write(GID, "SAA", 1, **layout)
    pyramid = fake_assets.opener(url_for(GID, "SAA"), None)
    restored = restore_untagged_fill(L30, {"SAA": pyramid})["SAA"]
    assert _fill(restored) == layout["fill_value"]


def test_band_without_encoding_is_unchanged(fake_assets):
    fake_assets.write(GID, "XYZ", 1)
    pyramid = fake_assets.opener(url_for(GID, "XYZ"), None)
    assert restore_untagged_fill(L30, {"XYZ": pyramid})["XYZ"] is pyramid
