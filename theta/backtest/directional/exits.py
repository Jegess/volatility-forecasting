"""Intra-month exit rules for held positions.

Entries are monthly-only, but positions can be closed early if the signal
that put them in the portfolio degrades. Freed cash sits idle until the
next monthly rebalance (no intra-month re-entries).
"""
from __future__ import annotations

import polars as pl


def signal_degradation_exits(held: set[str], ranked_today: pl.DataFrame,
                             exit_pct_rank: float = 0.70) -> set[str]:
    """Names to close at next open because their VRP rank has degraded.

    Args:
        held: symbols currently in the portfolio.
        ranked_today: one day's output of rank_daily, columns include
            `symbol` and `vrp_pct_rank`.
        exit_pct_rank: close if `vrp_pct_rank` < this threshold. Default
            0.70 = exit once the name leaves the bottom 3 deciles (bottom
            decile has pct_rank closest to 1.0).

    Names absent from ranked_today (no signal that day — stale panel, etc.)
    are held, not forced to exit.
    """
    if not held:
        return set()

    degraded = ranked_today.filter(
        pl.col("symbol").is_in(list(held))
        & (pl.col("vrp_pct_rank") < exit_pct_rank)
    )
    return set(degraded["symbol"].to_list())
