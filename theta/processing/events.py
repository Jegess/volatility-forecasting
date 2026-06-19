"""Event proximity features for volatility forecasting.

6 features capturing scheduled event effects on implied/realized volatility:
  - days_to_fomc, is_fomc_week      (market-wide, hardcoded calendar)
  - days_to_earnings, is_earnings_week (per-symbol, yfinance + cache)
  - days_to_cpi, is_cpi_week        (market-wide, hardcoded calendar)

Literature basis:
  - Bali et al.: option return spreads 3x larger during earnings weeks
  - Christensen et al.: binary EA dummy in ML variance models
  - Sinclair/Natenberg: IV ramps before earnings, collapses after
  - Lucca & Moench: VIX declines post-FOMC, excess returns pre-FOMC
  - CPI: VIX systematically declines after release

Design: both continuous (days_to_*) and binary (is_*_week) features.
  - Continuous captures pre-event IV ramp (Sinclair)
  - Binary matches literature standard (Bali et al.)
  - Comparing both is an originality contribution.

Usage:
    python -m theta.processing.events
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import polars as pl


# ── FOMC announcement dates (2nd day of 2-day meetings) ─────────────────
# Source: https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
# Only the announcement day matters for the "event" signal.

FOMC_DATES: list[dt.date] = [
    # 2021
    dt.date(2021, 1, 27),
    dt.date(2021, 3, 17),
    dt.date(2021, 4, 28),
    dt.date(2021, 6, 16),
    dt.date(2021, 7, 28),
    dt.date(2021, 9, 22),
    dt.date(2021, 11, 3),
    dt.date(2021, 12, 15),
    # 2022
    dt.date(2022, 1, 26),
    dt.date(2022, 3, 16),
    dt.date(2022, 5, 4),
    dt.date(2022, 6, 15),
    dt.date(2022, 7, 27),
    dt.date(2022, 9, 21),
    dt.date(2022, 11, 2),
    dt.date(2022, 12, 14),
    # 2023
    dt.date(2023, 2, 1),
    dt.date(2023, 3, 22),
    dt.date(2023, 5, 3),
    dt.date(2023, 6, 14),
    dt.date(2023, 7, 26),
    dt.date(2023, 9, 20),
    dt.date(2023, 11, 1),
    dt.date(2023, 12, 13),
    # 2024
    dt.date(2024, 1, 31),
    dt.date(2024, 3, 20),
    dt.date(2024, 5, 1),
    dt.date(2024, 6, 12),
    dt.date(2024, 7, 31),
    dt.date(2024, 9, 18),
    dt.date(2024, 11, 7),
    dt.date(2024, 12, 18),
    # 2025
    dt.date(2025, 1, 29),
    dt.date(2025, 3, 19),
    dt.date(2025, 5, 7),
    dt.date(2025, 6, 18),
    dt.date(2025, 7, 30),
    dt.date(2025, 9, 17),
    dt.date(2025, 10, 29),
    dt.date(2025, 12, 10),
    # 2026 (verified against federalreserve.gov 2026-03-20)
    dt.date(2026, 1, 28),
    dt.date(2026, 3, 18),
    dt.date(2026, 4, 29),
    dt.date(2026, 6, 17),
    dt.date(2026, 7, 29),
    dt.date(2026, 9, 16),
    dt.date(2026, 10, 28),
    dt.date(2026, 12, 9),
]


# ── CPI release dates ───────────────────────────────────────────────────
# Sources: FRED release calendar (release_id=10) for 2021-2025,
#          BLS schedule page for 2026.
# Verified 2026-03-20. Note: 2025 had government shutdown — Oct/Nov/Dec
# releases were delayed (Oct 24, Nov skipped, Dec 18).

CPI_DATES: list[dt.date] = [
    # 2021 (verified via FRED)
    dt.date(2021, 1, 13),
    dt.date(2021, 2, 10),
    dt.date(2021, 3, 10),
    dt.date(2021, 4, 13),
    dt.date(2021, 5, 12),
    dt.date(2021, 6, 10),
    dt.date(2021, 7, 13),
    dt.date(2021, 8, 11),
    dt.date(2021, 9, 14),
    dt.date(2021, 10, 13),
    dt.date(2021, 11, 10),
    dt.date(2021, 12, 10),
    # 2022 (verified via FRED)
    dt.date(2022, 1, 12),
    dt.date(2022, 2, 10),
    dt.date(2022, 3, 10),
    dt.date(2022, 4, 12),
    dt.date(2022, 5, 11),
    dt.date(2022, 6, 10),
    dt.date(2022, 7, 13),
    dt.date(2022, 8, 10),
    dt.date(2022, 9, 13),
    dt.date(2022, 10, 13),
    dt.date(2022, 11, 10),
    dt.date(2022, 12, 13),
    # 2023 (verified via FRED + BLS archive URL)
    dt.date(2023, 1, 12),
    dt.date(2023, 2, 14),
    dt.date(2023, 3, 14),
    dt.date(2023, 4, 12),
    dt.date(2023, 5, 10),
    dt.date(2023, 6, 13),
    dt.date(2023, 7, 12),
    dt.date(2023, 8, 10),
    dt.date(2023, 9, 13),
    dt.date(2023, 10, 12),
    dt.date(2023, 11, 14),
    dt.date(2023, 12, 12),
    # 2024 (verified via FRED)
    dt.date(2024, 1, 11),
    dt.date(2024, 2, 13),
    dt.date(2024, 3, 12),
    dt.date(2024, 4, 10),
    dt.date(2024, 5, 15),
    dt.date(2024, 6, 12),
    dt.date(2024, 7, 11),
    dt.date(2024, 8, 14),
    dt.date(2024, 9, 11),
    dt.date(2024, 10, 10),
    dt.date(2024, 11, 13),
    dt.date(2024, 12, 11),
    # 2025 (verified via FRED — shutdown delayed Oct/Nov/Dec)
    dt.date(2025, 1, 15),
    dt.date(2025, 2, 12),
    dt.date(2025, 3, 12),
    dt.date(2025, 4, 10),
    dt.date(2025, 5, 13),
    dt.date(2025, 6, 11),
    dt.date(2025, 7, 15),
    dt.date(2025, 8, 12),
    dt.date(2025, 9, 11),
    dt.date(2025, 10, 24),  # delayed by government shutdown
    # Nov 2025 CPI skipped (shutdown)
    dt.date(2025, 12, 18),  # delayed by government shutdown
    # 2026 (verified via BLS schedule page 2026-03-20)
    dt.date(2026, 1, 13),
    dt.date(2026, 2, 13),
    dt.date(2026, 3, 11),
    dt.date(2026, 4, 10),
    dt.date(2026, 5, 12),
    dt.date(2026, 6, 10),
    dt.date(2026, 7, 14),
    dt.date(2026, 8, 12),
    dt.date(2026, 9, 11),
    dt.date(2026, 10, 14),
    dt.date(2026, 11, 10),
    dt.date(2026, 12, 10),
]


# ETFs have no earnings
ETFS: set[str] = {"SPY", "QQQ", "IWM", "GLD", "TLT"}


# ── Earnings cache (EDGAR primary + yfinance fallback) ──────────────────

# SEC EDGAR requires a User-Agent header identifying the requester.
_SEC_HEADERS = {"User-Agent": "Theta Research research@example.com"}
_SEC_TICKER_URL = "https://www.sec.gov/files/company_tickers.json"
_SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"

# We need earnings from 2020-12-01 onward (252-day warmup before 2022 features).
_EARNINGS_START = "2020-01-01"


def _load_ticker_to_cik() -> dict[str, int]:
    """Fetch SEC's ticker -> CIK mapping (single JSON, cached in-memory)."""
    import json
    import urllib.request

    req = urllib.request.Request(_SEC_TICKER_URL, headers=_SEC_HEADERS)
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read())
    return {v["ticker"]: v["cik_str"] for v in data.values()}


def _fetch_edgar_earnings(cik: int) -> list[dt.date]:
    """Fetch 8-K Item 2.02 filing dates from EDGAR submissions API.

    Returns sorted list of earnings dates found in the 'recent' filings page.
    Does NOT crawl overflow files (would be too slow for large filers like JPM).
    """
    import json
    import urllib.request

    cik_padded = str(cik).zfill(10)
    url = _SEC_SUBMISSIONS_URL.format(cik=cik_padded)
    req = urllib.request.Request(url, headers=_SEC_HEADERS)
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read())

    recent = data["filings"]["recent"]
    dates: list[dt.date] = []
    for filing_date, form, items in zip(
        recent["filingDate"], recent["form"], recent["items"]
    ):
        if form == "8-K" and "2.02" in items and filing_date >= _EARNINGS_START:
            dates.append(dt.date.fromisoformat(filing_date))

    return sorted(set(dates))


def _fetch_yfinance_earnings(symbol: str) -> list[dt.date]:
    """Fetch earnings dates from yfinance as fallback."""
    import yfinance as yf

    ticker = yf.Ticker(symbol)
    try:
        eds = ticker.get_earnings_dates(limit=40)
        if eds is not None and len(eds) > 0:
            return sorted({d.date() for d in eds.index})
    except Exception:
        pass
    return []


def download_earnings_dates(
    symbol: str,
    cache_dir: Path,
    ticker_to_cik: dict[str, int] | None = None,
) -> pl.DataFrame:
    """Fetch earnings dates via EDGAR + yfinance fallback, cache to parquet.

    Strategy:
    1. EDGAR 8-K Item 2.02 (authoritative, covers through recent filings)
    2. yfinance (covers older dates EDGAR's 'recent' page may miss)
    3. Merge both, deduplicate (within 3-day window to handle filing lag)

    Returns DataFrame with single column: earnings_date (Date).
    Skips download if cache file exists.
    """
    cache_path = cache_dir / f"{symbol}.parquet"
    if cache_path.exists():
        return pl.read_parquet(cache_path)

    import time

    edgar_dates: list[dt.date] = []
    yf_dates: list[dt.date] = []

    # EDGAR
    cik = ticker_to_cik.get(symbol) if ticker_to_cik else None
    if cik:
        try:
            edgar_dates = _fetch_edgar_earnings(cik)
            time.sleep(0.12)  # SEC rate limit: 10 req/sec
        except Exception as e:
            print(f"    EDGAR failed for {symbol}: {e}")

    # yfinance (always fetch — fills gaps for large filers)
    try:
        yf_dates = _fetch_yfinance_earnings(symbol)
    except Exception as e:
        print(f"    yfinance failed for {symbol}: {e}")

    # Merge: keep all EDGAR dates, add yfinance dates that aren't within
    # 3 days of any EDGAR date (avoids duplicates from filing lag)
    merged = set(edgar_dates)
    for yd in yf_dates:
        if not any(abs((yd - ed).days) <= 3 for ed in edgar_dates):
            merged.add(yd)

    all_dates = sorted(merged)

    df = pl.DataFrame({"earnings_date": all_dates}).cast({"earnings_date": pl.Date})
    cache_dir.mkdir(parents=True, exist_ok=True)
    df.write_parquet(cache_path)
    return df


def download_all_earnings(
    symbols: list[str],
    cache_dir: Path,
) -> dict[str, pl.DataFrame]:
    """Download and cache earnings dates for all symbols.

    Skips ETFs (no earnings). Returns dict of symbol -> earnings DataFrame.
    """
    results: dict[str, pl.DataFrame] = {}
    to_fetch = [s for s in symbols if s not in ETFS]
    cached = sum(1 for s in to_fetch if (cache_dir / f"{s}.parquet").exists())

    if cached:
        print(f"  Earnings cache: {cached}/{len(to_fetch)} symbols already cached")

    # Load SEC ticker map once
    print("  Loading SEC ticker -> CIK map...")
    try:
        ticker_to_cik = _load_ticker_to_cik()
        print(f"  SEC map loaded: {len(ticker_to_cik)} tickers")
    except Exception as e:
        print(f"  WARNING: SEC ticker map failed ({e}), using yfinance only")
        ticker_to_cik = {}

    for i, symbol in enumerate(to_fetch, 1):
        is_cached = (cache_dir / f"{symbol}.parquet").exists()
        if not is_cached:
            print(f"  [{i}/{len(to_fetch)}] {symbol}...", end=" ")
            results[symbol] = download_earnings_dates(
                symbol, cache_dir, ticker_to_cik
            )
            n = len(results[symbol])
            print(f"{n} dates")
        else:
            results[symbol] = download_earnings_dates(
                symbol, cache_dir, ticker_to_cik
            )

    return results


# ── Core computation ────────────────────────────────────────────────────


def _days_to_next_event(
    dates: pl.Series,
    event_dates: list[dt.date],
) -> pl.Series:
    """For each date in `dates`, compute calendar days to next event.

    Returns 0 on event day itself. Returns null if no future event exists.
    """
    event_set = sorted(event_dates)

    def _find_next(d: dt.date) -> int | None:
        for ed in event_set:
            if ed >= d:
                return (ed - d).days
        return None

    return dates.map_elements(_find_next, return_dtype=pl.Int32)


def compute_event_features(
    dates: pl.Series,
    symbol: str | None = None,
    earnings_dates: list[dt.date] | None = None,
) -> pl.DataFrame:
    """Compute all 6 event features for a series of dates.

    Args:
        dates: pl.Series of Date type (trading dates for one symbol).
        symbol: symbol name (used to skip earnings for ETFs).
        earnings_dates: list of earnings announcement dates for this symbol.

    Returns:
        DataFrame with 6 columns: days_to_fomc, is_fomc_week,
        days_to_earnings, is_earnings_week, days_to_cpi, is_cpi_week.
    """
    # FOMC
    days_fomc = _days_to_next_event(dates, FOMC_DATES)
    is_fomc = (days_fomc <= 5).cast(pl.Int8)

    # CPI
    days_cpi = _days_to_next_event(dates, CPI_DATES)
    is_cpi = (days_cpi <= 5).cast(pl.Int8)

    # Earnings (per-symbol)
    is_etf = symbol is not None and symbol in ETFS
    if is_etf or earnings_dates is None or len(earnings_dates) == 0:
        days_earn = pl.Series("days_to_earnings", [None] * len(dates), dtype=pl.Int32)
        is_earn = pl.Series("is_earnings_week", [None] * len(dates), dtype=pl.Int8)
    else:
        days_earn = _days_to_next_event(dates, earnings_dates)
        is_earn = (days_earn <= 5).cast(pl.Int8)

    return pl.DataFrame({
        "days_to_fomc": days_fomc,
        "is_fomc_week": is_fomc,
        "days_to_earnings": days_earn,
        "is_earnings_week": is_earn,
        "days_to_cpi": days_cpi,
        "is_cpi_week": is_cpi,
    })


# ── CLI entry point ─────────────────────────────────────────────────────


def main() -> None:
    """Download earnings cache and compute event features for all symbols."""
    features_dir = Path("data/processed/features")
    earnings_dir = Path("data/processed/earnings")

    # Get symbol list from existing feature files
    symbols = sorted(p.stem for p in features_dir.glob("*.parquet"))
    print(f"Event features: {len(symbols)} symbols")
    print(f"FOMC dates: {len(FOMC_DATES)}, CPI dates: {len(CPI_DATES)}")

    # Step 1: download/cache all earnings dates
    print("\n-- Downloading earnings calendars --")
    earnings_cache = download_all_earnings(symbols, earnings_dir)

    # Step 2: compute event features per symbol and add to feature files
    print("\n-- Computing event features --")
    for i, symbol in enumerate(symbols, 1):
        path = features_dir / f"{symbol}.parquet"
        df = pl.read_parquet(path)

        # Get earnings dates for this symbol
        earn_dates: list[dt.date] = []
        if symbol in earnings_cache:
            earn_df = earnings_cache[symbol]
            if len(earn_df) > 0:
                earn_dates = earn_df["earnings_date"].to_list()

        # Compute
        events = compute_event_features(
            df["date"],
            symbol=symbol,
            earnings_dates=earn_dates,
        )

        # Drop existing event columns if re-running
        existing_event_cols = [
            c for c in events.columns if c in df.columns
        ]
        if existing_event_cols:
            df = df.drop(existing_event_cols)

        # Append event columns
        df = pl.concat([df, events], how="horizontal")
        df.write_parquet(path)

        if i % 25 == 0 or i == len(symbols):
            print(f"  [{i}/{len(symbols)}] done")

    # Sanity check
    print("\n-- Sanity check --")
    sample_symbols = ["AAPL", "SPY", "TSLA"]
    for sym in sample_symbols:
        path = features_dir / f"{sym}.parquet"
        if not path.exists():
            continue
        df = pl.read_parquet(path)
        print(f"\n  {sym}: {len(df)} rows")
        for col in ["days_to_fomc", "days_to_earnings", "days_to_cpi"]:
            if col in df.columns:
                s = df[col]
                non_null = s.drop_nulls()
                if len(non_null) > 0:
                    print(f"    {col}: min={non_null.min()}, max={non_null.max()}, "
                          f"nulls={s.null_count()}/{len(s)}")
                else:
                    print(f"    {col}: all null")
        # Show earnings count
        if sym in earnings_cache:
            print(f"    earnings dates cached: {len(earnings_cache.get(sym, []))}")


if __name__ == "__main__":
    main()
