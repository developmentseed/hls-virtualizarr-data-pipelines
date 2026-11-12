"""Sort a tile's time axis across all its level groups.

Adapted from the naqfc ``sort_reference_time``; the recipe is in
docs/superpowers/spikes/2026-11-11-time-stack-findings.md section 4. Run it in
the same writable session as the appends, before commit.
"""

from __future__ import annotations

from typing import Any

import icechunk
import numpy as np
import zarr

from virtualizarr_processor.hls.cube import level_groups


def sort_time(session: icechunk.Session, tile: str) -> int:
    """Sort ``time`` ascending in every level group; return the rows moved.

    Every group must hold identical, duplicate-free raw ``time`` values
    (``ValueError`` otherwise), and every data array must chunk time by 1.
    Nothing is touched unless all checks pass. Data arrays are reindexed with
    ``Session.reindex_array``; the raw int64 ``time`` coordinate is permuted
    directly with zarr so xarray never re-infers its encoding.
    """
    store = session.store
    groups = level_groups(store, tile)
    if not groups:
        return 0
    axes = {
        g: np.asarray(zarr.open_array(store, path=f"/{g}/time", mode="r")[:])
        for g in groups
    }
    times = axes[groups[0]]
    bad = [g for g, a in axes.items() if not np.array_equal(a, times)]
    if bad:
        raise ValueError(f"{tile}: level groups disagree on time: {bad}")
    if np.unique(times).size != times.size:
        raise ValueError(f"{tile}: duplicate times on the time axis")
    order = np.argsort(times, kind="stable")
    if np.array_equal(order, np.arange(len(order))):
        return 0

    paths = {g: f"/{g}/{g.split('/')[1]}" for g in groups}
    for g, path in paths.items():
        chunks = zarr.open_array(store, path=path, mode="r").chunks
        if chunks[0] != 1:
            raise ValueError(f"{path}: time chunk size {chunks[0]} != 1; cannot sort")

    old_to_new = np.empty_like(order)
    old_to_new[order] = np.arange(len(order))

    def forward(pos: Any) -> list[int]:
        return [int(old_to_new[pos[0]]), *list(pos)[1:]]

    def backward(pos: Any) -> list[int]:
        return [int(order[pos[0]]), *list(pos)[1:]]

    for g, path in paths.items():
        session.reindex_array(path, forward, backward)
        coord = zarr.open_array(store, path=f"/{g}/time", mode="r+")
        coord[:] = np.asarray(coord[:])[order]
    return int((order != np.arange(len(order))).sum())
