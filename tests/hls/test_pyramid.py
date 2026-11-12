import pytest
import zarr
from fakes import GID, GID2, declare_like, url_for
from virtualizarr_processor.hls.pyramid import GridMismatch, _codecs, check_level
from zarr.codecs import ZstdCodec


def test_opener_returns_levels_in_order(fake_assets):
    fake_assets.write(GID, "B04", 1)
    pyramid = fake_assets.opener(url_for(GID, "B04"), None)
    assert list(pyramid) == [0, 1]
    assert dict(pyramid[1].sizes) == {"y": 4, "x": 4}


def test_missing_raises(fake_assets):
    fake_assets.write(GID, "B04", 1)
    fake_assets.missing(GID, "B04")
    with pytest.raises(FileNotFoundError):
        fake_assets.opener(url_for(GID, "B04"), None)


def test_check_level_accepts_matching(hls_repo, fake_assets):
    fake_assets.write(GID, "B04", 1)
    level0 = fake_assets.opener(url_for(GID, "B04"), None)[0]
    declared = declare_like(hls_repo, level0)
    check_level(declared, level0)


@pytest.mark.parametrize(
    "field,kwargs",
    [
        ("shape", {"levels": ((16, 16), (8, 8))}),
        ("chunks", {"chunks": (2, 2)}),
        ("dtype", {"dtype": "int16"}),
    ],
)
def test_check_level_rejects(hls_repo, fake_assets, field, kwargs):
    fake_assets.write(GID, "B04", 1)
    declared = declare_like(hls_repo, fake_assets.opener(url_for(GID, "B04"), None)[0])
    fake_assets.write(GID2, "B04", 1, **kwargs)
    with pytest.raises(GridMismatch, match=field):
        check_level(declared, fake_assets.opener(url_for(GID2, "B04"), None)[0])


@pytest.mark.parametrize(
    "field,kwargs",
    [("fill", {"fill_value": 7}), ("codecs", {"compress": True})],
)
def test_check_level_rejects_fill_and_codecs(hls_repo, fake_assets, field, kwargs):
    fake_assets.write(GID, "B04", 1)
    declared = declare_like(hls_repo, fake_assets.opener(url_for(GID, "B04"), None)[0])
    fake_assets.write(GID2, "B04", 1, **kwargs)
    with pytest.raises(GridMismatch, match=field):
        check_level(declared, fake_assets.opener(url_for(GID2, "B04"), None)[0])


def test_check_level_accepts_reopened_declared(hls_repo, fake_assets):
    fake_assets.write(GID, "B04", 1, compress=True)
    level0 = fake_assets.opener(url_for(GID, "B04"), None)[0]
    session = hls_repo.writable_session("main")
    zarr.create_array(
        session.store,
        name="d2",
        shape=(3, 8, 8),
        chunks=(1, 4, 4),
        dtype="uint16",
        compressors=[ZstdCodec()],
        zarr_format=3,
    )
    session.commit("declare")
    reopened = zarr.open_array(
        hls_repo.readonly_session("main").store, path="d2", zarr_format=3
    )
    check_level(reopened, level0)


def test_codecs_tuple_equals_list():
    assert _codecs([{"name": "transpose", "configuration": {"order": (0, 1)}}]) == (
        _codecs([{"name": "transpose", "configuration": {"order": [0, 1]}}])
    )
