from unittest.mock import MagicMock

from backfill_handlers import init, promote
from fakes import Backfill


def test_promote_moves_main_to_backfill_tip(
    hls_backfill: Backfill,
    lambda_context: MagicMock,
) -> None:
    init.handler({"inventory_uri": hls_backfill.inventory_uri}, lambda_context)

    result = promote.handler({}, lambda_context)

    assert result["promoted"] is True
    repo = hls_backfill.processor().open_backfill_repo()
    assert repo.lookup_branch("main") == repo.lookup_branch("backfill")
