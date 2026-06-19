"""Download and cache risk-free rate from FRED.

Uses the 3-month Treasury Bill secondary market rate (DTB3),
which is the standard proxy for the risk-free rate in options
pricing literature (Bali et al., Carr & Wu).

The rate is returned as a decimal (e.g., 5.25% -> 0.0525) and
forward-filled to cover weekends/holidays.

Usage:
    python -m theta.processing.rates
"""

from __future__ import annotations

from pathlib import Path

import polars as pl


def download_risk_free_rate(
    api_key: str,
    start_date: str = "2020-12-01",
    end_date: str = "2026-03-31",
) -> pl.DataFrame:
    """Download 3-month T-Bill rate from FRED.

    Returns DataFrame with columns: date (Date), rate (Float64).
    Rate is in decimal form (e.g., 0.0425 for 4.25%).
    Missing days (weekends/holidays) are forward-filled.
    """
    from fredapi import Fred

    fred = Fred(api_key=api_key)
    series = fred.get_series("DTB3", observation_start=start_date, observation_end=end_date)

    # Convert pandas Series to polars
    df = pl.DataFrame({
        "date": series.index,
        "rate": series.values,
    }).with_columns(
        pl.col("date").cast(pl.Date),
        pl.col("rate").cast(pl.Float64) / 100.0,  # percent to decimal
    )

    # Drop nulls (FRED uses '.' for missing), then forward-fill across all
    # calendar dates so every trading day has a rate
    date_range = pl.DataFrame({
        "date": pl.date_range(
            pl.Series([df["date"].min()]).item(),
            pl.Series([df["date"].max()]).item(),
            "1d",
            eager=True,
        )
    })
    df = date_range.join(df, on="date", how="left").with_columns(
        pl.col("rate").forward_fill()
    )

    return df


def load_risk_free_rate(
    cache_path: Path,
    api_key: str | None = None,
    force_refresh: bool = False,
) -> pl.DataFrame:
    """Load risk-free rate, downloading from FRED if not cached.

    Args:
        cache_path: Path to parquet cache file.
        api_key: FRED API key. Required if cache doesn't exist.
        force_refresh: Re-download even if cache exists.

    Returns:
        DataFrame with columns: date (Date), rate (Float64).
    """
    if cache_path.exists() and not force_refresh:
        return pl.read_parquet(cache_path)

    if api_key is None:
        raise ValueError("FRED API key required to download rates (no cache found)")

    df = download_risk_free_rate(api_key)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(cache_path)
    print(f"  Risk-free rate cached: {len(df)} days, {cache_path}")

    return df


if __name__ == "__main__":
    import os

    from dotenv import load_dotenv

    load_dotenv()
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise RuntimeError("Set FRED_API_KEY in .env")

    cache = Path("data/processed/rates/risk_free_rate.parquet")
    df = load_risk_free_rate(cache, api_key=api_key, force_refresh=True)
    print(f"\nDate range: {df['date'].min()} to {df['date'].max()}")
    print(f"Rate range: {df['rate'].min():.4f} to {df['rate'].max():.4f}")
    print(f"Nulls: {df['rate'].null_count()}")
