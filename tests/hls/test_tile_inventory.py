import io
import urllib.error
from datetime import date

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from virtualizarr_processor.hls import inventory
from virtualizarr_processor.hls.inventory import (
    ArchiveForbidden,
    MonthMissing,
    build_inventory,
    check_unique_times,
    month_uris,
    select_tile,
)

L30_A = "HLS.L30.T13SDA.2025001T173012.v2.0"
L30_B = "HLS.L30.T13SDA.2025010T173012.v2.0"
S30_A = "HLS.S30.T13SDA.2025002T173012.v2.0"
L30_OTHER_TILE = "HLS.L30.T13SDB.2025003T173012.v2.0"


def test_month_uris_span_and_padding():
    uris = month_uris("L30", date(2013, 4, 1), date(2013, 6, 1))
    assert [u.rsplit("/", 1)[-1] for u in uris] == [
        "HLSL30_2.0-2013-4.parquet",
        "HLSL30_2.0-2013-5.parquet",
        "HLSL30_2.0-2013-6.parquet",
    ]
    assert "/year=2013/month=4/" in uris[0]
    assert uris[0].startswith("https://nasa-maap-data-store.s3.us-west-2.")


def test_month_uris_s3_year_rollover_and_unknown_product():
    uris = month_uris("L30", date(2013, 12, 1), date(2014, 1, 1), https=False)
    assert uris[0].startswith("s3://nasa-maap-data-store/file-staging/")
    assert len(uris) == 2 and "year=2014/month=1/" in uris[1]
    with pytest.raises(ValueError):
        month_uris("S30", date(2013, 1, 1), date(2013, 1, 1))


def test_select_tile_filters_and_sorts():
    t = pa.table({"id": [L30_B, S30_A, L30_A, L30_OTHER_TILE, L30_A, "junk"]})
    assert select_tile(t, "T13SDA", "L30") == [L30_A, L30_B]


def test_check_unique_times_reports_collisions():
    with pytest.raises(ValueError, match="2025001T173012"):
        check_unique_times([L30_A, L30_A])
    check_unique_times([L30_A, L30_B])


def _reader(missing=()):
    def read(uri):
        if any(m in uri for m in missing):
            raise MonthMissing(uri)
        return pa.table({"id": [L30_A, L30_B]})

    return read


def test_build_inventory_current_month_missing_is_warning():
    warnings = []
    ids = build_inventory(
        "T13SDA",
        "L30",
        date(2025, 1, 1),
        date(2025, 2, 1),
        reader=_reader(missing=["month=2/"]),
        today=date(2025, 2, 10),
        warn=warnings.append,
    )
    assert ids == [L30_A, L30_B]
    assert any("2025-2" in w for w in warnings)


def test_build_inventory_other_missing_is_fatal():
    with pytest.raises(MonthMissing):
        build_inventory(
            "T13SDA",
            "L30",
            date(2025, 1, 1),
            date(2025, 2, 1),
            reader=_reader(missing=["month=1/"]),
            today=date(2025, 2, 10),
        )
    with pytest.raises(MonthMissing):  # end month, but not the current month
        build_inventory(
            "T13SDA",
            "L30",
            date(2025, 1, 1),
            date(2025, 2, 1),
            reader=_reader(missing=["month=2/"]),
            today=date(2025, 6, 1),
        )


def _parquet_bytes():
    buf = io.BytesIO()
    pq.write_table(pa.table({"id": [L30_A]}), buf)
    return buf.getvalue()


def _http_error(code):
    return urllib.error.HTTPError("u", code, "x", {}, None)


def _patch_urlopen(monkeypatch, outcomes):
    calls = []

    def fake(uri, timeout):
        calls.append(timeout)
        o = outcomes.pop(0)
        if isinstance(o, Exception):
            raise o
        return io.BytesIO(o)

    monkeypatch.setattr(inventory.urllib.request, "urlopen", fake)
    monkeypatch.setattr(inventory.time, "sleep", lambda s: None)
    return calls


def test_read_month_ids_404_is_month_missing(monkeypatch):
    _patch_urlopen(monkeypatch, [_http_error(404)])
    with pytest.raises(MonthMissing):
        inventory.read_month_ids("https://x/y.parquet")


def test_read_month_ids_403_is_clear_error(monkeypatch):
    calls = _patch_urlopen(monkeypatch, [_http_error(403)])
    with pytest.raises(ArchiveForbidden, match="forbidden"):
        inventory.read_month_ids("https://x/y.parquet")
    assert len(calls) == 1


def test_read_month_ids_retries_5xx(monkeypatch):
    calls = _patch_urlopen(monkeypatch, [_http_error(500), _parquet_bytes()])
    assert inventory.read_month_ids("https://x/y.parquet")["id"].to_pylist() == [L30_A]
    assert calls == [60, 60]
