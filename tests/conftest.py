"""Shared fixtures: on-disk fake HLS assets, a local repo, and a wired processor.

``tests/hls`` is put on ``sys.path`` so every test package can import ``fakes``.
"""

import pathlib
import sys

import icechunk
import pytest
import zarr

sys.path.insert(0, str(pathlib.Path(__file__).parent / "hls"))

from fakes import FAKE_GRID, FakeAssets, local_locator, open_local_repo  # noqa: E402
from virtualizarr_processor.processor import Processor  # noqa: E402


@pytest.fixture
def fake_assets(tmp_path: pathlib.Path) -> FakeAssets:
    return FakeAssets(tmp_path / "assets")


@pytest.fixture
def hls_repo(tmp_path: pathlib.Path) -> icechunk.Repository:
    """A filesystem repo with a committed `main`; authorizes `file://{tmp}/assets/`."""
    repo = open_local_repo(tmp_path / "repo", tmp_path / "assets")
    session = repo.writable_session("main")
    zarr.open_group(session.store, mode="a").create_group("placeholder")
    session.commit("init main")
    return repo


@pytest.fixture
def fake_processor_factory(tmp_path: pathlib.Path, fake_assets: FakeAssets):  # type: ignore[no-untyped-def]
    """A zero-argument ``Processor`` factory wired to the fakes and a local repo
    at ``{tmp}/repo``, for patching over the handlers' ``Processor``."""

    def factory() -> Processor:
        return Processor(
            tile="T13SDA",
            product="L30",
            grid=FAKE_GRID,
            opener=fake_assets.opener,
            locator=local_locator(fake_assets.root),
            repo_opener=lambda: open_local_repo(tmp_path / "repo", fake_assets.root),
        )

    return factory
