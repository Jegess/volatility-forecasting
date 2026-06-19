"""Technical / price-based features from underlying prices.

Computes 9 features: 5 momentum horizons, max daily return,
RSI(14), and 2 MA crossover ratios.

Literature: Gu/Kelly/Xiu (2020), Souropanis & Vivian (2023),
Jansen (ML for Algo Trading).

Usage:
    Called by compute_features.py orchestrator, not directly.
"""

from __future__ import annotations

import polars as pl


def compute_momentum(df: pl.DataFrame) -> pl.DataFrame:
    """Compute cumulative return momentum at 5 horizons.

    mom_Nd = P_t / P_{t-N} - 1  (N-day cumulative return).
    """
    for n, name in [(5, "mom_5d"), (22, "mom_22d"), (63, "mom_63d"),
                     (126, "mom_126d"), (252, "mom_252d")]:
        df = df.with_columns(
            (pl.col("underlying_price") / pl.col("underlying_price").shift(n) - 1)
            .alias(name)
        )

    # Null out any infinite values (from zero/missing lookback prices)
    mom_cols = ["mom_5d", "mom_22d", "mom_63d", "mom_126d", "mom_252d"]
    df = df.with_columns(
        pl.when(pl.col(c).is_infinite()).then(None).otherwise(pl.col(c)).alias(c)
        for c in mom_cols
    )

    return df


def compute_max_daily_ret(df: pl.DataFrame) -> pl.DataFrame:
    """Max absolute daily return over past 22 days.

    Gu/Kelly/Xiu: one of the top cross-sectional predictors.
    """
    df = df.with_columns(
        pl.col("log_return").abs().rolling_max(22).alias("max_daily_ret")
    )
    return df


def compute_rsi(df: pl.DataFrame, period: int = 14) -> pl.DataFrame:
    """Compute RSI(14) — Relative Strength Index.

    RSI = 100 - 100 / (1 + RS)
    RS = avg_gain / avg_loss over `period` days (exponential smoothing).

    We use simple rolling mean (Wilder's original used EMA, but SMA
    is standard in academic work and the difference is negligible
    for our purposes).
    """
    df = df.with_columns(
        pl.when(pl.col("log_return") > 0)
        .then(pl.col("log_return"))
        .otherwise(0.0)
        .alias("_gain"),
        pl.when(pl.col("log_return") < 0)
        .then(pl.col("log_return").abs())
        .otherwise(0.0)
        .alias("_loss"),
    )

    df = df.with_columns(
        pl.col("_gain").rolling_mean(period).alias("_avg_gain"),
        pl.col("_loss").rolling_mean(period).alias("_avg_loss"),
    )

    df = df.with_columns(
        (100.0 - 100.0 / (1.0 + pl.col("_avg_gain") / pl.col("_avg_loss")))
        .alias("rsi_14")
    )

    # Handle division by zero (all losses = 0 -> RSI = 100)
    df = df.with_columns(
        pl.when(pl.col("_avg_loss") == 0)
        .then(100.0)
        .when(pl.col("_avg_gain") == 0)
        .then(0.0)
        .otherwise(pl.col("rsi_14"))
        .alias("rsi_14")
    )

    df = df.drop("_gain", "_loss", "_avg_gain", "_avg_loss")
    return df


def compute_ma_crossovers(df: pl.DataFrame) -> pl.DataFrame:
    """Compute MA crossover ratios in ratio form.

    ma_cross_1_9:  MA(1) / MA(9) - 1
    ma_cross_2_12: MA(2) / MA(12) - 1

    Ratio form avoids scale dependence across stocks.
    """
    df = df.with_columns(
        (
            pl.col("underlying_price")
            / pl.col("underlying_price").rolling_mean(9)
            - 1
        ).alias("ma_cross_1_9"),
        (
            pl.col("underlying_price").rolling_mean(2)
            / pl.col("underlying_price").rolling_mean(12)
            - 1
        ).alias("ma_cross_2_12"),
    )
    return df


def compute_technical_features(df: pl.DataFrame) -> pl.DataFrame:
    """Compute all 9 technical features.

    Args:
        df: DataFrame with columns: date, underlying_price, log_return.
            Must be sorted by date, one row per trading day.

    Returns:
        Input DataFrame with 9 added feature columns.
    """
    df = compute_momentum(df)
    df = compute_max_daily_ret(df)
    df = compute_rsi(df)
    df = compute_ma_crossovers(df)
    return df
