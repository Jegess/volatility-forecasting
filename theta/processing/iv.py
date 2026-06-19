"""Implied volatility and delta calculation via Black-Scholes-Merton.

Uses py_vollib_vectorized for fast vectorized IV inversion on mid-quotes.
European BSM is standard in the literature (Bali et al., Carr & Wu) —
early exercise premium is negligible for short-dated OTM options.

Usage:
    Called by compute_iv.py orchestrator, not directly.
"""

from __future__ import annotations

import numpy as np
import polars as pl


def compute_iv_and_delta(df: pl.DataFrame, rates: pl.DataFrame) -> pl.DataFrame:
    """Calculate IV and delta for all rows in a cleaned options DataFrame.

    Args:
        df: Options data from options_clean/ with columns including
            mid_quote, underlying_price, strike, right, dte, date.
        rates: Risk-free rate DataFrame with columns date, rate.

    Returns:
        Input DataFrame with added columns: rate, t_years, iv, delta.
        Rows where IV solver fails get iv=null, delta=null.
    """
    import py_vollib_vectorized as pv

    # Merge risk-free rate
    df = df.join(rates, on="date", how="left")

    # Forward-fill any remaining rate gaps (shouldn't happen but safety)
    df = df.with_columns(pl.col("rate").forward_fill())

    # Time to expiration in years
    df = df.with_columns(
        (pl.col("dte").cast(pl.Float64) / 365.0).alias("t_years")
    )

    # Extract numpy arrays for py_vollib
    price = df["mid_quote"].to_numpy().copy()
    S = df["underlying_price"].to_numpy().copy()
    K = df["strike"].to_numpy().copy()
    t = df["t_years"].to_numpy().copy()
    r = df["rate"].to_numpy().copy()
    flag = df["right"].to_numpy()

    # py_vollib expects 'c'/'p' lowercase
    flag = np.where(flag == "CALL", "c", "p")

    # Clamp t to small positive value (DTE >= 14 from filters, but safety)
    t = np.maximum(t, 1e-4)

    # Mask invalid rows: price/S/K must be positive for BSM
    valid_mask = (price > 0) & (S > 0) & (K > 0) & np.isfinite(price) & np.isfinite(S)

    # Initialize output arrays
    iv = np.full(len(price), np.nan)
    delta = np.full(len(price), np.nan)

    if valid_mask.sum() > 0:
        # --- IV calculation (valid rows only) ---
        iv[valid_mask] = pv.vectorized_implied_volatility(
            price[valid_mask], S[valid_mask], K[valid_mask],
            t[valid_mask], r[valid_mask], flag[valid_mask],
            q=0.0,
            model="black_scholes_merton",
            on_error="ignore",
            return_as="numpy",
        )

        # --- Delta calculation (where IV is valid) ---
        iv_ok = valid_mask & np.isfinite(iv) & (iv > 0)
        if iv_ok.sum() > 0:
            delta[iv_ok] = pv.vectorized_delta(
                flag[iv_ok], S[iv_ok], K[iv_ok],
                t[iv_ok], r[iv_ok], iv[iv_ok],
                q=0.0,
                model="black_scholes_merton",
                return_as="numpy",
            )

    # Add columns back to polars DataFrame
    df = df.with_columns(
        pl.Series("iv", iv, dtype=pl.Float64),
        pl.Series("delta", delta, dtype=pl.Float64),
    )

    # Sanity bounds: IV in [0.01, 5.0]
    df = df.with_columns(
        pl.when((pl.col("iv") < 0.01) | (pl.col("iv") > 5.0))
        .then(None)
        .otherwise(pl.col("iv"))
        .alias("iv"),
    )

    # Null out delta where IV was nulled
    df = df.with_columns(
        pl.when(pl.col("iv").is_null())
        .then(None)
        .otherwise(pl.col("delta"))
        .alias("delta"),
    )

    return df


def filter_delta(df: pl.DataFrame) -> pl.DataFrame:
    """Apply delta-based moneyness filter per Driessen et al.

    Calls:  0.15 <= delta <= 0.50
    Puts:  -0.50 <= delta <= -0.05

    Rows with null delta are dropped.
    """
    return df.filter(
        pl.col("delta").is_not_null()
        & (
            # Calls
            (
                (pl.col("right") == "CALL")
                & (pl.col("delta") >= 0.15)
                & (pl.col("delta") <= 0.50)
            )
            |
            # Puts
            (
                (pl.col("right") == "PUT")
                & (pl.col("delta") >= -0.50)
                & (pl.col("delta") <= -0.05)
            )
        )
    )
