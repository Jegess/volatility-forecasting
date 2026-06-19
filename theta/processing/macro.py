"""Download and cache macro features from FRED and Yahoo Finance.

8 market-wide features that capture the macro volatility environment.
All features are constant across symbols on a given date — they get
joined to the panel by date in panel.py.

FRED series (5):
    - VIXCLS: VIX (market fear index)
    - DGS10 - DTB3: term spread (recession signal)
    - BAMLH0A0HYM2: high-yield OAS credit spread (risk appetite)
    - USEPUINDXD: economic policy uncertainty
    - DTB3 first difference: T-bill rate change

Philadelphia Fed (1):
    - ADS Index: Aruoba-Diebold-Scotti business conditions index

Yahoo Finance (2):
    - ^VVIX: volatility of VIX (mean-reversion signal)
    - ^HSI: Hang Seng overnight volatility spillover

Literature: Gu/Kelly/Xiu, Christensen et al., Welch & Goyal,
Souropanis & Vivian, Zhang et al., Sinclair.

Usage:
    python -m theta.processing.macro
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl


# ── FRED downloads ──────────────────────────────────────────────────────


def _download_fred_series(
    api_key: str,
    series_id: str,
    start_date: str = "2020-12-01",
    end_date: str = "2026-03-31",
    col_name: str | None = None,
) -> pl.DataFrame:
    """Download a single FRED series and return as polars DataFrame.

    Returns columns: date (Date), {col_name} (Float64).
    """
    from fredapi import Fred

    fred = Fred(api_key=api_key)
    series = fred.get_series(
        series_id,
        observation_start=start_date,
        observation_end=end_date,
    )

    name = col_name or series_id.lower()
    df = pl.DataFrame({
        "date": series.index,
        name: series.values,
    }).with_columns(
        pl.col("date").cast(pl.Date),
        # FRED returns NaN for holidays — convert to Polars null
        # so forward_fill works correctly
        pl.col(name).cast(pl.Float64).fill_nan(None),
    )

    return df


def _download_ads_index() -> pl.DataFrame:
    """Download ADS business conditions index from Philadelphia Fed.

    The ADS index is not on FRED — it's published as an Excel file by
    the Philly Fed. Date format in the file uses colons (e.g., 2024:03:15).
    """
    import tempfile
    import urllib.request

    url = "https://www.philadelphiafed.org/-/media/FRBP/Assets/Surveys-And-Data/ads/ADS_Index_Most_Current_Vintage.xlsx"

    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
        urllib.request.urlretrieve(url, f.name)
        df = pl.read_excel(f.name)

    # Date column uses colon separators (e.g., "2024:03:15")
    df = df.select([
        pl.col("Date").str.replace_all(":", "-").str.to_date("%Y-%m-%d").alias("date"),
        pl.col("ADS_Index").cast(pl.Float64).alias("ads_index"),
    ]).filter(
        pl.col("date") >= pl.lit("2020-12-01").str.to_date("%Y-%m-%d")
    )

    return df


def download_fred_macro(api_key: str) -> pl.DataFrame:
    """Download all FRED + Philly Fed macro features.

    Returns a single DataFrame indexed by date with columns:
    vix, term_spread, credit_spread, epu, tbill_change, ads_index.
    """
    print("  Downloading FRED series...")

    # 1. VIX
    vix = _download_fred_series(api_key, "VIXCLS", col_name="vix")
    print(f"    VIXCLS: {len(vix)} obs")

    # 2. Term spread = 10Y yield - 3M T-bill
    dgs10 = _download_fred_series(api_key, "DGS10", col_name="dgs10")
    dtb3 = _download_fred_series(api_key, "DTB3", col_name="dtb3")
    print(f"    DGS10: {len(dgs10)} obs, DTB3: {len(dtb3)} obs")

    # 3. Credit spread (high-yield option-adjusted spread)
    credit = _download_fred_series(
        api_key, "BAMLH0A0HYM2", col_name="credit_spread"
    )
    print(f"    BAMLH0A0HYM2: {len(credit)} obs")

    # 4. Economic Policy Uncertainty (daily)
    epu = _download_fred_series(api_key, "USEPUINDXD", col_name="epu")
    print(f"    USEPUINDXD: {len(epu)} obs")

    # 5. ADS business conditions index (from Philly Fed)
    print("  Downloading ADS index from Philadelphia Fed...")
    ads = _download_ads_index()
    print(f"    ADS Index: {len(ads)} obs")

    # Build a complete calendar date range, then join everything
    all_dates = pl.concat([
        vix.select("date"),
        dgs10.select("date"),
        dtb3.select("date"),
        credit.select("date"),
        epu.select("date"),
        ads.select("date"),
    ]).unique().sort("date")

    macro = (
        all_dates
        .join(vix, on="date", how="left")
        .join(dgs10, on="date", how="left")
        .join(dtb3, on="date", how="left")
        .join(credit, on="date", how="left")
        .join(epu, on="date", how="left")
        .join(ads, on="date", how="left")
    )

    # Compute derived features
    macro = macro.with_columns([
        # Term spread: 10Y - 3M (both in percent, keep in percent)
        (pl.col("dgs10") - pl.col("dtb3")).alias("term_spread"),
        # T-bill daily change (first difference, in percent)
        pl.col("dtb3").diff().alias("tbill_change"),
    ])

    # Drop intermediate columns, keep only final features
    macro = macro.select([
        "date", "vix", "term_spread", "credit_spread",
        "epu", "tbill_change", "ads_index",
    ])

    # Ensure any remaining NaN → null, then forward-fill
    macro = macro.with_columns(
        pl.all().exclude("date").fill_nan(None).forward_fill()
    )

    return macro


# ── Yahoo Finance downloads ─────────────────────────────────────────────


def download_yahoo_macro() -> pl.DataFrame:
    """Download VVIX and HSI from Yahoo Finance.

    Returns DataFrame with columns: date, vvix, hsi_overnight_vol.

    VVIX: volatility of VIX — high values predict IV mean reversion.
    HSI overnight vol: squared log return of Hang Seng index — captures
    Asian session volatility that spills into US open.
    """
    import yfinance as yf

    print("  Downloading Yahoo Finance series...")

    # VVIX — volatility of VIX
    vvix_raw = yf.download(
        "^VVIX",
        start="2020-12-01",
        end="2026-03-31",
        progress=False,
    )
    # yfinance returns MultiIndex columns: flatten them
    if isinstance(vvix_raw.columns, __import__("pandas").MultiIndex):
        vvix_raw.columns = [c[0] for c in vvix_raw.columns]
    vvix = pl.from_pandas(vvix_raw.reset_index()).select([
        pl.col("Date").cast(pl.Date).alias("date"),
        pl.col("Close").cast(pl.Float64).alias("vvix"),
    ])
    print(f"    ^VVIX: {len(vvix)} obs")

    # HSI — Hang Seng Index
    hsi_raw = yf.download(
        "^HSI",
        start="2020-12-01",
        end="2026-03-31",
        progress=False,
    )
    if isinstance(hsi_raw.columns, __import__("pandas").MultiIndex):
        hsi_raw.columns = [c[0] for c in hsi_raw.columns]
    hsi = pl.from_pandas(hsi_raw.reset_index()).select([
        pl.col("Date").cast(pl.Date).alias("date"),
        pl.col("Close").cast(pl.Float64).alias("hsi_close"),
    ])
    # Overnight vol = squared log return
    hsi = hsi.with_columns(
        (pl.col("hsi_close").log() - pl.col("hsi_close").shift(1).log())
        .pow(2)
        .alias("hsi_overnight_vol")
    ).select(["date", "hsi_overnight_vol"])
    print(f"    ^HSI: {len(hsi)} obs")

    # Join on date
    yahoo = vvix.join(hsi, on="date", how="full", coalesce=True).sort("date")

    # Forward-fill gaps (different holiday calendars)
    yahoo = yahoo.with_columns(
        pl.all().exclude("date").forward_fill()
    )

    return yahoo


# ── Combined ─────────────────────────────────────────────────────────────


def build_macro_features(api_key: str) -> pl.DataFrame:
    """Download and combine all 8 macro features.

    Returns DataFrame with columns:
        date, vix, term_spread, credit_spread, epu, tbill_change,
        ads_index, vvix, hsi_overnight_vol
    """
    fred = download_fred_macro(api_key)
    yahoo = download_yahoo_macro()

    # Join FRED and Yahoo on date
    macro = fred.join(yahoo, on="date", how="full", coalesce=True).sort("date")

    # Final forward-fill for any remaining gaps at the boundary
    macro = macro.with_columns(
        pl.all().exclude("date").forward_fill()
    )

    print(f"\n  Combined macro: {len(macro)} days, {len(macro.columns)} columns")
    print(f"  Date range: {macro['date'].min()} to {macro['date'].max()}")
    print(f"  Null counts:")
    for col in macro.columns:
        if col != "date":
            nulls = macro[col].null_count()
            print(f"    {col}: {nulls}")

    return macro


def load_macro_features(
    cache_path: Path,
    api_key: str | None = None,
    force_refresh: bool = False,
) -> pl.DataFrame:
    """Load macro features, downloading if not cached.

    Args:
        cache_path: Path to parquet cache file.
        api_key: FRED API key. Required if cache doesn't exist.
        force_refresh: Re-download even if cache exists.

    Returns:
        DataFrame with 8 macro feature columns + date.
    """
    if cache_path.exists() and not force_refresh:
        df = pl.read_parquet(cache_path)
        print(f"  Macro loaded from cache: {len(df)} days")
        return df

    if api_key is None:
        raise ValueError("FRED API key required to download macro data (no cache found)")

    df = build_macro_features(api_key)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(cache_path)
    print(f"  Macro cached: {cache_path}")

    return df


if __name__ == "__main__":
    import os

    from dotenv import load_dotenv

    load_dotenv()
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise RuntimeError("Set FRED_API_KEY in .env")

    cache = Path("data/processed/macro/macro.parquet")
    df = load_macro_features(cache, api_key=api_key, force_refresh=True)
    print(f"\nFinal shape: {df.shape}")
    print(df.head(5).to_pandas().to_string())
