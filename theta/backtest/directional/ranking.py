"""Cross-sectional VRP ranking for the directional backtest.

Different from theta.backtest.signal.build_candidates: we do NOT apply the
sell-premium filter chain (VRP>0, VRP>hist_median, VIX window, earnings
skip, FOMC skip). This strategy takes the full cross-section and ranks
every symbol with a valid (y_pred, atm_iv) pair each day.
"""
from __future__ import annotations

import polars as pl

from theta.backtest.signal import compute_vrp


def compute_vrp_frame(signals: pl.DataFrame) -> pl.DataFrame:
    """Add `vrp = atm_iv^2 - y_pred`."""
    return signals.with_columns(
        compute_vrp(pl.col("atm_iv"), pl.col("y_pred")).alias("vrp")
    )


def rank_daily(vrp_frame: pl.DataFrame) -> pl.DataFrame:
    """Add `vrp_rank` (1 = highest VRP) and `vrp_pct_rank` (0-1) per date.

    Ties broken by symbol alphabetically for determinism.
    """
    return (
        vrp_frame
        .sort(["date", "vrp", "symbol"], descending=[False, True, False])
        .with_columns(
            pl.col("vrp").rank("ordinal", descending=True).over("date").alias("vrp_rank"),
            pl.col("vrp").rank("ordinal", descending=True).over("date")
                .truediv(pl.len().over("date"))
                .alias("vrp_pct_rank"),
        )
    )


def select_decile(ranked: pl.DataFrame, decile: str = "bottom",
                  n_deciles: int = 10) -> pl.DataFrame:
    """Return (date, symbol, vrp, vrp_rank, vrp_pct_rank) pairs in the target decile.

    decile="bottom" -> highest pct_rank (lowest VRP)
    decile="top"    -> lowest pct_rank (highest VRP)
    decile=int      -> 1-indexed decile (1=top, n_deciles=bottom)
    """
    if decile == "bottom":
        lo, hi = 1.0 - 1.0 / n_deciles, 1.0
    elif decile == "top":
        lo, hi = 0.0, 1.0 / n_deciles
    elif isinstance(decile, int):
        if not 1 <= decile <= n_deciles:
            raise ValueError(f"decile int must be in [1, {n_deciles}]")
        lo = (decile - 1) / n_deciles
        hi = decile / n_deciles
    else:
        raise ValueError(f"decile must be 'bottom', 'top', or int; got {decile!r}")

    return ranked.filter(
        (pl.col("vrp_pct_rank") > lo) & (pl.col("vrp_pct_rank") <= hi)
    )
