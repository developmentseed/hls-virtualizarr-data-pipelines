import pathlib

import icechunk
import pytest
import zarr
from fakes import FakeAssets


@pytest.fixture
def fake_assets(tmp_path: pathlib.Path) -> FakeAssets:
    return FakeAssets(tmp_path / "assets")


@pytest.fixture
def hls_repo(tmp_path: pathlib.Path) -> icechunk.Repository:
    """A filesystem repo with a committed `main`; authorizes `file://{tmp}/assets/`."""
    (tmp_path / "assets").mkdir(exist_ok=True)
    prefix = f"file://{tmp_path / 'assets'}/"
    config = icechunk.RepositoryConfig.default()
    config.set_virtual_chunk_container(
        icechunk.VirtualChunkContainer(
            prefix, icechunk.local_filesystem_store(str(tmp_path / "assets"))
        )
    )
    repo = icechunk.Repository.open_or_create(
        storage=icechunk.local_filesystem_storage(str(tmp_path / "repo")),
        config=config,
        authorize_virtual_chunk_access={
            prefix: icechunk.credentials.LocalFileSystemAccess
        },
    )
    session = repo.writable_session("main")
    zarr.open_group(session.store, mode="a").create_group("placeholder")
    session.commit("init main")
    return repo
