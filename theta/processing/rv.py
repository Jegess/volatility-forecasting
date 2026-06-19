"""Realized volatility features from underlying prices.

Computes HAR/HARQ/SHAR RV decomposition plus the 21-day forward
RV target variable. Handles stock splits via hardcoded table +
safety filter (|log_return| > 0.5).

Literature: Corsi (2009) HAR, Bollerslev et al. HARQ,
Patton & Sheppard SHAR, Carr & Wu (split adjustment).

Usage:
    Called by compute_features.py orchestrator, not directly.
"""

from __future__ import annotations

import numpy as np
import polars as pl


# Known stock splits in our universe (unadjusted underlying prices)
SPLITS: dict[str, list[tuple[str, int]]] = {
    "GOOGL": [("2022-07-15", 20)],
    "AMZN": [("2022-06-06", 20)],
    "TSLA": [("2022-08-25", 3)],
}

# Warmup period: max rolling window needed before features are valid
WARMUP_DAYS = 252


def adjust_splits(df: pl.DataFrame, symbol: str) -> pl.DataFrame:
    """Divide pre-split prices by split ratio.

    Modifies underlying_price in-place so log returns are continuous
    across split dates.
    """
    if symbol not in SPLITS:
        return df

    for split_date, ratio in SPLITS[symbol]:
        df = df.with_columns(
            pl.when(pl.col("date") < pl.lit(split_date).str.to_date("%Y-%m-%d"))
            .then(pl.col("underlying_price") / ratio)
            .otherwise(pl.col("underlying_price"))
            .alias("underlying_price")
        )

    return df


def compute_log_returns(df: pl.DataFrame) -> pl.DataFrame:
    """Compute daily log returns from underlying_price.

    Flags suspect returns (|log_return| > 0.5) as null — these
    indicate unhandled splits or data errors.
    """
    df = df.with_columns(
        (pl.col("underlying_price") / pl.col("underlying_price").shift(1))
        .log()
        .alias("log_return")
    )

    # Safety filter: null out extreme returns (Jansen: >100% moves are suspect)
    df = df.with_columns(
        pl.when(pl.col("log_return").abs() > 0.5)
        .then(None)
        .otherwise(pl.col("log_return"))
        .alias("log_return")
    )

    return df


def compute_rv_features(df: pl.DataFrame) -> pl.DataFrame:
    """Compute all RV features from a DataFrame with log_return column.

    Expects df sorted by date with one row per trading day.

    Features computed:
        rv_d:   daily squared return (annualized: x252)
        rv_w:   mean of rv_d over past 5 days (annualized)
        rv_m:   mean of rv_d over past 22 days (annualized)
        rq:     realized quarticity over 22d (annualized)
        rs_pos: sum of squared positive returns over 22d (annualized)
        rs_neg: sum of squared negative returns over 22d (annualized)
        rv_21d_forward: sum of squared returns over next 21 days (annualized) — TARGET
    """
    # Daily squared return (not annualized yet — annualize after rolling)
    df = df.with_columns(
        (pl.col("log_return") ** 2).alias("_r2"),
        pl.when(pl.col("log_return") > 0)
        .then(pl.col("log_return") ** 2)
        .otherwise(0.0)
        .alias("_r2_pos"),
        pl.when(pl.col("log_return") < 0)
        .then(pl.col("log_return") ** 2)
        .otherwise(0.0)
        .alias("_r2_neg"),
        (pl.col("log_return") ** 4).alias("_r4"),
    )

    df = df.with_columns(
        # rv_d: daily squared return, annualized
        (pl.col("_r2") * 252).alias("rv_d"),
        # rv_w: mean of daily squared returns over 5 days, annualized
        (pl.col("_r2").rolling_mean(5) * 252).alias("rv_w"),
        # rv_m: mean of daily squared returns over 22 days, annualized
        (pl.col("_r2").rolling_mean(22) * 252).alias("rv_m"),
        # rq: realized quarticity = (n/3) * sum(r^4) over 22d
        # Annualize: multiply by 252 (quarticity scales linearly with time)
        ((22 / 3) * pl.col("_r4").rolling_sum(22) * 252).alias("rq"),
        # rs_pos: sum of squared positive returns over 22d, annualized
        (pl.col("_r2_pos").rolling_sum(22) * (252 / 22)).alias("rs_pos"),
        # rs_neg: sum of squared negative returns over 22d, annualized
        (pl.col("_r2_neg").rolling_sum(22) * (252 / 22)).alias("rs_neg"),
    )

    # rv_21d_forward: TARGET — sum of squared returns over next 21 trading days
    # Use shift(-21) on a rolling_sum(21) offset by 1 to get the *future* window
    df = df.with_columns(
        (pl.col("_r2").shift(-1).rolling_sum(21) * (252 / 21)).alias("rv_21d_forward")
    )

    # Drop intermediate columns
    df = df.drop("_r2", "_r2_pos", "_r2_neg", "_r4")

    return df


def compute_rv_for_symbol(
    underlying_path: str | pl.DataFrame,
    symbol: str,
    *,
    truncate_warmup: bool = True,
) -> pl.DataFrame:
    """Full RV pipeline for one symbol.

    Args:
        underlying_path: Path to underlying parquet or pre-loaded DataFrame.
        symbol: Symbol name (for split adjustment).
        truncate_warmup: If True (default), drop first 252 rows. Set False
            to return full series (used by technical features which need
            their own lookback into the warmup period).

    Returns:
        DataFrame with columns: symbol, date, underlying_price, log_return,
        rv_d, rv_w, rv_m, rq, rs_pos, rs_neg, rv_21d_forward.
    """
    if isinstance(underlying_path, pl.DataFrame):
        df = underlying_path
    else:
        df = pl.read_parquet(underlying_path)

    # Need date and underlying_price at minimum
    df = df.select("date", "underlying_price").unique("date").sort("date")

    # Add symbol column if not present
    df = df.with_columns(pl.lit(symbol).alias("symbol"))

    # Adjust for known splits
    df = adjust_splits(df, symbol)

    # Compute log returns
    df = compute_log_returns(df)

    # Compute RV features
    df = compute_rv_features(df)

    # Select final columns
    df = df.select(
        "symbol",
        "date",
        "underlying_price",
        "log_return",
        "rv_d",
        "rv_w",
        "rv_m",
        "rq",
        "rs_pos",
        "rs_neg",
        "rv_21d_forward",
    )

    # Drop warmup rows (first 252 days don't have valid 252d windows)
    if truncate_warmup and len(df) > WARMUP_DAYS:
        df = df.slice(WARMUP_DAYS)

    return df
