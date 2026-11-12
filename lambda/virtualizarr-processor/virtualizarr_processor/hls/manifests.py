"""How the tile cube's Icechunk manifests are split.

Without splitting, every band array keeps all of its chunk refs in one
manifest: ~1.1M refs at level 0 for ~5,000 granules (225 chunks per step),
and a forward append rewrites all 75 of them (15 bands x 5 levels). Splitting
along ``time`` bounds each manifest and confines an append to the last split.

Sizes aim for ~30k refs per split: levels 0-1 carry 225 refs per time step,
so 128 steps is ~29k; levels 2-4 carry at most 64, so 512 steps is <=33k.
Only ``time`` is split (one chunk per step, so a split of N chunks is N
steps); a step's spatial refs are few enough to keep together. Coordinate
arrays (``time``, ``x``, ``y``) are tiny and left unsplit.

Icechunk applies the first rule that matches an array, so the level 0-1 rule
comes first. The config is saved with the repo when it is created and governs
manifests written after that, so it must be in place before the backfill.
"""

from __future__ import annotations

import re

import icechunk

from virtualizarr_processor.hls.granule import BANDS

#: Time steps per manifest split for band arrays at levels 0 and 1.
FINE_TIME_SPLIT = 128
#: Time steps per manifest split for band arrays at levels 2 to 4.
COARSE_TIME_SPLIT = 512


def manifest_splitting(
    product: str,
    *,
    fine: int = FINE_TIME_SPLIT,
    coarse: int = COARSE_TIME_SPLIT,
) -> icechunk.ManifestSplittingConfig:
    """Split ``product``'s band arrays along ``time``: ``fine`` steps per split
    at levels 0-1, ``coarse`` at every other level."""
    bands = icechunk.ManifestSplitCondition.name_matches(
        "^(" + "|".join(re.escape(band) for band in BANDS[product]) + ")$"
    )
    fine_levels = icechunk.ManifestSplitCondition.and_conditions(
        [
            bands,
            icechunk.ManifestSplitCondition.path_matches(r".*/multiscales/[01]/[^/]+$"),
        ]
    )
    time = icechunk.ManifestSplitDimCondition.DimensionName("^time$")
    return icechunk.ManifestSplittingConfig.from_dict(
        {fine_levels: {time: fine}, bands: {time: coarse}}
    )
