"""Signal construction and symbol-level filters.

The VRP signal and the filters that gate a symbol from becoming a trade
candidate on a given day. Contract-level liquidity checks (OI, spread) live
in spreads.py because they depend on the chosen strike, not the symbol.

Pipeline:
    df = daily_signals()            # from data.py
    df = add_vrp(df)                # atm_iv**2 - y_pred
    df = add_historical_median_vrp(df)
    df = apply_filters(df)          # drop rows that don't qualify
    df = rank_candidates(df)        # adds vrp_rank (1 = best per day)
"""
from __future__ import annotations

from datetime import date

import polars as pl

# Filter bounds mirror BACKTEST_PLAN.md §Filters.
VIX_MIN = 14.0
VIX_MAX = 30.0
DEFAULT_TRADE_DTE = 25  # midpoint of the 21–30 entry window


# ----- VRP ---------------------------------------------------------------

def compute_vrp(atm_iv: pl.Expr | float, rv_forecast: pl.Expr | float) -> pl.Expr | float:
    """VRP = IV² - RV_forecast. Both inputs are in annualized-variance units."""
    return atm_iv ** 2 - rv_forecast


def add_vrp(df: pl.DataFrame) -> pl.DataFrame:
    """Add a `vrp` column using LightGBM's `y_pred` as the RV forecast."""
    return df.with_columns(compute_vrp(pl.col("atm_iv"), pl.col("y_pred")).alias("vrp"))


# ----- historical median VRP --------------------------------------------

def add_historical_median_vrp(df: pl.DataFrame) -> pl.DataFrame:
    """Add `hist_median_vrp`: per-symbol expanding median of VRP, strictly
    from dates *before* the current row. The first observation per symbol
    therefore has a null hist_median and will be filtered out.
    """
    return (
        df
        .sort(["symbol", "date"])
        .with_columns(
            pl.col("vrp")
            .shift(1)
            .cumulative_eval(pl.element().median())
            .over("symbol")
            .alias("hist_median_vrp")
        )
    )


def historical_median_vrp(signals: pl.DataFrame, symbol: str,
                          before_date: date) -> float | None:
    """Point query: median VRP for `symbol` across all dates strictly before
    `before_date`. Returns None if the symbol has no prior observations.
    """
    past = signals.filter(
        (pl.col("symbol") == symbol) & (pl.col("date") < before_date)
    )
    if past.height == 0:
        return None
    return float(past["vrp"].median())


# ----- symbol-level filters ---------------------------------------------

def apply_filters(df: pl.DataFrame,
                  trade_dte: int = DEFAULT_TRADE_DTE,
                  vix_min: float = VIX_MIN,
                  vix_max: float = VIX_MAX,
                  earnings_buffer_days: int | None = None) -> pl.DataFrame:
    """Keep only rows that pass every symbol-level filter from the plan.

    Args:
        trade_dte: target DTE at entry (used for the earnings buffer if
            `earnings_buffer_days` is not supplied).
        vix_min: lower bound on VIX at entry. Sinclair: low-VIX regimes
            yield tiny premium vs margin and require frequent hedging; the
            mid-VIX regime is the sweet spot for harvesting VRP.
        vix_max: upper bound on VIX at entry. Plan default is 30 (the
            Sinclair/Natenberg "tradeable vol" ceiling); tighter values
            like 22 skip the stress-vol regimes where breaches cluster.
        earnings_buffer_days: days_to_earnings must strictly exceed this.
            Defaults to `trade_dte + 5` (earnings fall after expiration
            with ≥5 buffer). Pass a larger value to keep a hard skip
            around quarterly announcements.
    """
    earnings_buffer = earnings_buffer_days if earnings_buffer_days is not None \
        else trade_dte + 5
    return df.filter(
        (pl.col("vrp") > 0)
        & pl.col("hist_median_vrp").is_not_null()
        & (pl.col("vrp") > pl.col("hist_median_vrp"))
        & (pl.col("vix") >= vix_min)
        & (pl.col("vix") <= vix_max)
        & (pl.col("days_to_earnings") > earnings_buffer)
        & (pl.col("is_fomc_week") == 0)
    )


# ----- daily ranking -----------------------------------------------------

def rank_candidates(df: pl.DataFrame) -> pl.DataFrame:
    """Add `vrp_rank` — 1 = highest VRP on that date. Ties broken by symbol
    alphabetically for determinism.
    """
    return (
        df
        .sort(["date", "vrp", "symbol"], descending=[False, True, False])
        .with_columns(
            pl.col("vrp").rank("ordinal", descending=True).over("date").alias("vrp_rank")
        )
    )


# ----- convenience orchestrator -----------------------------------------

def build_candidates(daily: pl.DataFrame,
                     trade_dte: int = DEFAULT_TRADE_DTE,
                     vix_min: float = VIX_MIN,
                     vix_max: float = VIX_MAX,
                     earnings_buffer_days: int | None = None) -> pl.DataFrame:
    """Full pipeline: VRP → hist median → filters → rank. Returns a frame
    ordered by (date, vrp_rank) with every row a tradeable candidate.
    """
    return (
        daily
        .pipe(add_vrp)
        .pipe(add_historical_median_vrp)
        .pipe(apply_filters, trade_dte=trade_dte,
              vix_min=vix_min, vix_max=vix_max,
              earnings_buffer_days=earnings_buffer_days)
        .pipe(rank_candidates)
    )
