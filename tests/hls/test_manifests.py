"""Manifest splitting: band arrays split along time, finer at levels 0-1."""

from pathlib import Path

import icechunk
import pytest
import zarr
from virtualizarr_processor import processor as processor_module
from virtualizarr_processor.hls.granule import BANDS
from virtualizarr_processor.hls.manifests import (
    COARSE_TIME_SPLIT,
    FINE_TIME_SPLIT,
    manifest_splitting,
)

TILE = "T13SDA"
N_TIME = 8


def _manifest_count(repo_dir: Path) -> int:
    return len(list((repo_dir / "manifests").iterdir()))


def _write(repo_dir: Path, config: icechunk.RepositoryConfig, paths: list[str]) -> int:
    """Commit one (N_TIME, 2, 2) array per path; return manifests written."""
    repo = icechunk.Repository.create(
        icechunk.local_filesystem_storage(str(repo_dir)), config=config
    )
    session = repo.writable_session("main")
    root = zarr.group(session.store)
    for path in paths:
        dims = ("time",) if path.endswith("/time") else ("time", "y", "x")
        shape = (N_TIME,) if len(dims) == 1 else (N_TIME, 2, 2)
        chunks = (1,) if len(dims) == 1 else (1, 2, 2)
        array = root.create_array(
            path, shape=shape, chunks=chunks, dtype="i2", dimension_names=dims
        )
        array[:] = 1
    session.commit("write")
    return _manifest_count(repo_dir)


def _config(fine: int, coarse: int) -> icechunk.RepositoryConfig:
    config = icechunk.RepositoryConfig.default()
    config.manifest = icechunk.ManifestConfig(
        splitting=manifest_splitting("L30", fine=fine, coarse=coarse)
    )
    return config


def test_split_sizes_are_the_recommended_ones():
    assert (FINE_TIME_SPLIT, COARSE_TIME_SPLIT) == (128, 512)


@pytest.mark.parametrize("level, expected", [(0, 4), (1, 4), (2, 2), (3, 2), (4, 2)])
def test_band_split_size_depends_on_level(tmp_path, level, expected):
    # fine=2, coarse=4 over 8 steps: levels 0-1 -> 4 splits, levels 2-4 -> 2.
    path = f"{TILE}/B04/multiscales/{level}/B04"
    assert _write(tmp_path / "repo", _config(2, 4), [path]) == expected


@pytest.mark.parametrize("band", BANDS["L30"])
def test_every_band_is_split(tmp_path, band):
    path = f"{TILE}/{band}/multiscales/0/{band}"
    assert _write(tmp_path / "repo", _config(2, 4), [path]) == N_TIME // 2


def test_coordinate_arrays_are_not_split(tmp_path):
    path = f"{TILE}/B04/multiscales/0/time"
    assert _write(tmp_path / "repo", _config(2, 4), [path]) == 1


def test_band_name_must_match_whole_name(tmp_path):
    # "B04_extra" is not a band: no split.
    path = f"{TILE}/B04/multiscales/0/B04_extra"
    assert _write(tmp_path / "repo", _config(2, 4), [path]) == 1


def test_unknown_product_raises():
    with pytest.raises(KeyError):
        manifest_splitting("X99")


def test_default_repo_opener_persists_splitting(tmp_path, monkeypatch):
    monkeypatch.delenv("ICECHUNK_BUCKET", raising=False)
    monkeypatch.delenv("EARTHDATA_SECRET_ARN", raising=False)
    repo_dir = tmp_path / "repo"
    monkeypatch.setenv("ICECHUNK_LOCAL_PATH", str(repo_dir))
    processor_module.default_repo_opener("s3", product="L30")()
    stored = icechunk.Repository.fetch_config(
        icechunk.local_filesystem_storage(str(repo_dir))
    )
    assert stored is not None and stored.manifest is not None
    assert stored.manifest.splitting is not None
    # Split conditions have no value equality; their reprs spell out every rule.
    assert repr(stored.manifest.splitting.to_dict()) == repr(
        manifest_splitting("L30").to_dict()
    )
