"""The HLS tile-cube processor: one MGRS tile, one HLS product, one Icechunk repo.

Backfill declares the whole cube up front from an inventory of granule IDs and
region-writes each granule into its declared time step. Forward processing
region-writes a granule whose time is already on the axis (a redelivery, or a
slot a backfill skipped) and appends any other, then sorts the time axis back
into order before committing. See
docs/superpowers/specs/2026-11-11-hls-l30-mgrs-datacube-design.md.
"""

from __future__ import annotations

import logging
import os
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any, Literal, Protocol, cast

import icechunk
import numpy as np
from icechunk import ForkSession, Repository, Session

from virtualizarr_processor.hls import assets
from virtualizarr_processor.hls.credentials import (
    ACCESS_MODES,
    bootstrap_earthdata_env,
    container_credentials,
    earthdata_token,
    object_store_registry,
    virtual_chunk_container,
)
from virtualizarr_processor.hls.cube import (
    check_granule,
    declare_cube,
    open_granule,
    store_times,
    write_granule,
)
from virtualizarr_processor.hls.granule import BANDS, Granule, granule_from_key
from virtualizarr_processor.hls.grid import TileGrid, tile_grid
from virtualizarr_processor.hls.manifests import manifest_splitting
from virtualizarr_processor.hls.pyramid import Pyramid, PyramidOpener, open_pyramid
from virtualizarr_processor.hls.sort import sort_time

logger = logging.getLogger(__name__)

#: icechunk's ``CachingConfig.num_chunk_refs`` raised above its default, after
#: icechest-hls's measurement (a 14x commit slowdown with the default cache).
#: A T13SDA L30 granule carries 8,010 chunk refs (15 bands x 534 chunks over
#: all levels), and the whole archive is ~5,000 granules: 5,000 x 8,010 is
#: ~40M refs, the full cube that Reduce, forward commits and the time sort
#: touch. Matches the CDK default; override with ``ICECHUNK_NUM_CHUNK_REFS``.
DEFAULT_NUM_CHUNK_REFS = 40_000_000

#: Attempts and initial backoff (seconds, doubled per retry) for opening one
#: band's COG headers. A missing object (``FileNotFoundError``) is not retried.
OPEN_ATTEMPTS = 3
RETRY_DELAY = 1.0

#: Attribute set on a session whose granule write failed part-way. Sessions are
#: unhashable, so the mark lives on the session itself.
_FAILED_WRITES = "_hls_failed_writes"

Stage = Literal["parse", "open", "check", "write"]


@dataclass
class SkipRecord:
    """A granule a backfill worker skipped, and why."""

    granule_id: str
    stage: Stage
    error: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


class Locator(Protocol):
    """Maps an asset key to the URL it is read from and the URL it is recorded as."""

    def read_url(self, key: str) -> str: ...

    def reference_url(self, key: str) -> str: ...


@dataclass(frozen=True)
class LpDaacLocator:
    """Reads LP DAAC over ``s3`` or ``https``; records endpoint-free ``vcc://`` refs."""

    access: str = "s3"

    def read_url(self, key: str) -> str:
        return assets.read_url(key, cast(Literal["s3", "https"], self.access))

    def reference_url(self, key: str) -> str:
        return assets.reference_url(key)


class _Rejected(Exception):
    """A granule this processor's cube does not take."""


def default_repo_opener(
    access: str, *, product: str = "L30"
) -> Callable[[], Repository]:
    """Open (or create) the cube repo from the environment.

    ``ICECHUNK_BUCKET`` (with optional ``ICECHUNK_PREFIX``/``ICECHUNK_REGION``)
    selects S3 storage, else ``ICECHUNK_LOCAL_PATH`` a filesystem repo. The
    ``lpdaac`` virtual chunk container resolves ``vcc://lpdaac/...`` refs.

    ``access="https"`` is a local/laptop mode (the deployed Lambdas are
    s3-only): the repo is created with a header-free https container, then
    reopened with the Earthdata bearer token in the in-memory config only, so
    the token never reaches storage. A pickled https-mode repo or backfill
    fork still carries the token -- keep such pickles local.
    """

    def opener() -> Repository:
        bootstrap_earthdata_env()
        bucket = os.environ.get("ICECHUNK_BUCKET")
        local = os.environ.get("ICECHUNK_LOCAL_PATH")
        if bucket:
            storage = icechunk.s3_storage(
                bucket=bucket,
                prefix=os.environ.get("ICECHUNK_PREFIX"),
                region=os.environ.get("ICECHUNK_REGION"),
                from_env=True,
            )
        elif local:
            storage = icechunk.local_filesystem_storage(local)
        else:
            raise ValueError("set ICECHUNK_BUCKET or ICECHUNK_LOCAL_PATH")
        num_chunk_refs = int(
            os.environ.get("ICECHUNK_NUM_CHUNK_REFS", DEFAULT_NUM_CHUNK_REFS)
        )

        def config(
            container: icechunk.VirtualChunkContainer,
        ) -> icechunk.RepositoryConfig:
            cfg = icechunk.RepositoryConfig.default()
            cfg.set_virtual_chunk_container(container)
            cfg.caching = icechunk.CachingConfig(num_chunk_refs=num_chunk_refs)
            cfg.manifest = icechunk.ManifestConfig(
                splitting=manifest_splitting(product)
            )
            return cfg

        credentials = container_credentials(access)
        # Create (or open) with the header-free container: icechunk persists the
        # config a repo is created with, so a token here would land in storage.
        repo = icechunk.Repository.open_or_create(
            storage=storage,
            config=config(virtual_chunk_container(access, persisted=True)),
            authorize_virtual_chunk_access=credentials,
        )
        if access != "https":
            return repo
        # The bearer token lives only in this in-memory config; opening does
        # not write the config back (nothing here calls ``save_config``).
        token = earthdata_token()
        return icechunk.Repository.open(
            storage=storage,
            config=config(virtual_chunk_container(access, token=token)),
            authorize_virtual_chunk_access=credentials,
        )

    return opener


class Processor:
    """The ``VirtualizarrProcessor`` for one HLS tile cube.

    ``None`` arguments default from the environment: ``MGRS_TILE`` (required),
    ``HLS_PRODUCT`` (``"L30"``), ``LPDAAC_ACCESS`` (``"s3"``). ``grid``
    defaults to the tile's embedded ``TILE_GRIDS`` entry. ``opener``,
    ``locator``, ``repo_opener`` and ``registry`` are seams for tests; the
    registry (how COG headers are read) is only built for the real opener.

    ``skipped`` accumulates the backfill granules this instance skipped.
    """

    def __init__(
        self,
        *,
        tile: str | None = None,
        product: str | None = None,
        access: str | None = None,
        grid: TileGrid | None = None,
        opener: PyramidOpener = open_pyramid,
        locator: Locator | None = None,
        repo_opener: Callable[[], Repository] | None = None,
        registry: Callable[[], Any] | None = None,
    ) -> None:
        tile = tile if tile is not None else os.environ.get("MGRS_TILE")
        if not tile:
            raise ValueError("no MGRS tile: pass tile= or set MGRS_TILE")
        product = product if product is not None else os.environ.get("HLS_PRODUCT")
        product = product or "L30"
        if product not in BANDS:
            raise ValueError(f"unsupported HLS product {product!r}")
        access = access if access is not None else os.environ.get("LPDAAC_ACCESS")
        access = access or "s3"
        if access not in ACCESS_MODES:
            raise ValueError(f"unknown access mode {access!r}; expected {ACCESS_MODES}")
        self.tile, self.product, self.access = tile, product, access
        self.grid = grid if grid is not None else tile_grid(self.tile)
        if self.grid.mgrs_id != self.tile:
            raise ValueError(f"grid is for {self.grid.mgrs_id}, not {self.tile}")
        self.opener = opener
        self.locator = locator if locator is not None else LpDaacLocator(self.access)
        self.repo_opener = (
            repo_opener
            if repo_opener is not None
            else default_repo_opener(self.access, product=self.product)
        )
        if registry is None:
            registry = self._default_registry if opener is open_pyramid else _none
        self._registry_factory = registry
        self._registry: Any = None
        self._registry_built = False
        self.skipped: list[SkipRecord] = []
        self.open_attempts = OPEN_ATTEMPTS
        self.retry_delay = RETRY_DELAY

    # --- shared -------------------------------------------------------------

    def _default_registry(self) -> Any:
        bootstrap_earthdata_env()
        return object_store_registry(self.access)

    def _registry_now(self) -> Any:
        if not self._registry_built:
            self._registry = self._registry_factory()
            self._registry_built = True
        return self._registry

    @property
    def bands(self) -> tuple[str, ...]:
        return BANDS[self.product]

    def _granule(self, key: str) -> Granule:
        """Parse ``key`` (``ValueError``); ``_Rejected`` if it is not this cube's."""
        return self._accept(granule_from_key(key))

    def _accept(self, granule: Granule) -> Granule:
        if granule.tile != self.tile or granule.product != self.product:
            raise _Rejected(
                f"{granule.id} is {granule.product}/{granule.tile}; "
                f"this cube is {self.product}/{self.tile}"
            )
        return granule

    def _open_one(self, url: str, registry: Any) -> Pyramid:
        delay = self.retry_delay
        for attempt in range(1, self.open_attempts + 1):
            try:
                return self.opener(url, registry)
            except FileNotFoundError:
                raise
            except Exception as err:
                if attempt == self.open_attempts:
                    raise
                logger.warning(
                    "open %s failed (attempt %d/%d): %s; retrying",
                    url,
                    attempt,
                    self.open_attempts,
                    err,
                )
                time.sleep(delay)
                delay *= 2
        raise AssertionError("unreachable")

    def _open(self, granule: Granule, registry: Any) -> dict[str, Pyramid]:
        """Every band's pyramid, each open retried; raises before any write."""
        return open_granule(
            granule, self.bands, self._open_one, registry, self.locator.read_url
        )

    # --- repositories -------------------------------------------------------

    def initialize_repo(self) -> Repository:
        """Open (or create) the cube repo. Writes no data: the cube must be
        backfilled before forward processing will write to it."""
        return self.repo_opener()

    def open_backfill_repo(self) -> Repository:
        return self.repo_opener()

    def garbage_collect(self, expiry_time: datetime) -> icechunk.GCSummary:
        repo = self.repo_opener()
        repo.expire_snapshots(older_than=expiry_time)
        return repo.garbage_collect(delete_object_older_than=expiry_time)

    # --- backfill -----------------------------------------------------------

    def initialize_backfill_store(
        self, repo: Repository, granule_ids: list[str]
    ) -> str:
        """Declare the full cube on a new ``backfill`` branch and commit.

        Every ID must parse and belong to this tile and product, with no two
        sharing a time (``ValueError`` otherwise, before any branch is made).
        The newest granule is opened as the reference for every array's
        metadata; the time axis is every granule's time, sorted.
        """
        if not granule_ids:
            raise ValueError("empty inventory")
        granules = []
        for gid in granule_ids:
            try:
                granules.append(self._granule(gid))
            except _Rejected as err:
                raise ValueError(str(err)) from None
        counts = Counter(g.time for g in granules)
        dupes = sorted(t.isoformat() for t, n in counts.items() if n > 1)
        if dupes:
            raise ValueError(f"duplicate granule times in inventory: {dupes}")
        times = sorted(counts)
        newest = max(granules, key=lambda g: g.time)
        reference = self._open(newest, self._registry_now())

        repo.create_branch("backfill", repo.lookup_branch("main"))
        session = repo.writable_session("backfill")
        declare_cube(session.store, self.grid, times, reference)
        snapshot = cast(
            str,
            session.commit(
                f"Declare {self.product} {self.tile} cube: {len(times)} time steps"
            ),
        )
        logger.info(
            "declared %s cube with %d time steps (reference %s)",
            self.tile,
            len(times),
            newest.id,
        )
        return snapshot

    def _skip(self, granule_id: str, stage: Stage, err: BaseException) -> bool:
        record = SkipRecord(granule_id, stage, str(err))
        self.skipped.append(record)
        logger.error("skipped %s at %s: %s", granule_id, stage, err)
        return False

    def process_backfill_file(self, file_key: str, fork: ForkSession) -> bool:
        """Region-write one granule into its declared step of ``fork``.

        Any per-granule failure appends a ``SkipRecord`` to ``skipped`` and
        returns ``False``. For the parse, open and check stages nothing has
        been written for the granule (every band is opened and checked before
        the first write). A ``"write"`` skip may have left the fork partly
        written: the worker must then discard the whole fork, not save it.
        """
        try:
            granule = granule_from_key(file_key)
        except ValueError as err:
            return self._skip(file_key, "parse", err)
        try:
            self._accept(granule)
        except _Rejected as err:
            return self._skip(granule.id, "check", err)
        # outside the per-granule handling: a registry (credentials) failure is
        # not about this granule, so it fails the worker rather than skipping
        registry = self._registry_now()
        try:
            pyramids = self._open(granule, registry)
        except Exception as err:
            return self._skip(granule.id, "open", err)
        try:
            check_granule(fork.store, self.tile, granule, pyramids, mode="region")
        except Exception as err:
            return self._skip(granule.id, "check", err)
        try:
            write_granule(
                fork.store,
                self.tile,
                granule,
                pyramids,
                mode="region",
                reference_url_for=self.locator.reference_url,
            )
        except Exception as err:
            logger.exception("write failed part-way for %s", granule.id)
            return self._skip(granule.id, "write", err)
        return True

    # --- forward ------------------------------------------------------------

    def initialize_session(self, repo: Repository) -> Session:
        return repo.writable_session("main")

    def process_file(self, file_key: str, session: Session) -> bool:
        """Write one granule into ``main``: region if its time is on the axis,
        else append. Never raises.

        Returns ``False`` for a key that is not this cube's granule, a store
        that was never backfilled, or a granule that cannot be opened or fails
        the grid check -- each before anything is written. A write that fails
        part-way also returns ``False`` and marks ``session`` so that
        ``commit_processed_files`` refuses it.
        """
        try:
            granule = self._granule(file_key)
        except (ValueError, _Rejected) as err:
            logger.error("rejected %s: %s", file_key, err)
            return False
        try:
            times = store_times(session.store, self.tile)
        except Exception:
            logger.exception("could not read the %s time axis", self.tile)
            return False
        if times is None:
            logger.error(
                "store not backfilled: %s has no %s cube; cannot write %s",
                self.product,
                self.tile,
                granule.id,
            )
            return False
        when = np.datetime64(granule.time.astimezone(UTC).replace(tzinfo=None), "ns")
        mode: Literal["region", "append"] = (
            "region" if bool(np.any(times == when)) else "append"
        )
        try:
            pyramids = self._open(granule, self._registry_now())
            check_granule(session.store, self.tile, granule, pyramids, mode=mode)
        except Exception:
            logger.exception("could not %s %s", mode, granule.id)
            return False
        try:
            write_granule(
                session.store,
                self.tile,
                granule,
                pyramids,
                mode=mode,
                reference_url_for=self.locator.reference_url,
            )
        except Exception as err:
            logger.exception("%s write failed part-way for %s", mode, granule.id)
            failed = getattr(session, _FAILED_WRITES, [])
            setattr(session, _FAILED_WRITES, [*failed, f"{granule.id}: {err}"])
            return False
        logger.info("%s write of %s", mode, granule.id)
        return True

    def commit_processed_files(self, session: Session) -> str:
        """Sort the time axis into order across every level group, then commit.

        Raises, committing nothing, if a write in this session failed part-way:
        the session may hold diverged groups and has to be discarded.
        """
        failed = getattr(session, _FAILED_WRITES, None)
        if failed:
            raise RuntimeError(
                f"refusing to commit: writes failed part-way ({failed}); "
                "discard the session"
            )
        moved = sort_time(session, self.tile)
        logger.info("sorted %s time axis: %d rows moved", self.tile, moved)
        return cast(str, session.commit(f"Forward update of {self.tile}"))


def _none() -> None:
    return None
