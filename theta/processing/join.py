"""Join EOD, OI, and underlying data per symbol.

Produces a single DataFrame per symbol with all raw fields plus
derived columns (mid_quote, dte, moneyness, relative_spread).

No filtering is applied — the output is the unfiltered joined dataset.

Usage:
    python -m theta.processing.join
"""

from __future__ import annotations

from pathlib import Path

import polars as pl


def join_option_data(
    eod_path: Path,
    oi_path: Path,
    underlying_path: Path,
) -> pl.DataFrame:
    """Join trimmed EOD with OI and underlying for one symbol.

    Steps:
        1. Load trimmed EOD, deduplicate exact-copy rows
        2. Load OI, extract date from timestamp
        3. Load underlying (already has date column)
        4. Left-join EOD with OI on (date, expiration, strike, right)
        5. Left-join with underlying on (date)
        6. Compute derived columns

    Returns:
        Joined DataFrame with columns:
            symbol, date, expiration, strike, right, bid, ask, volume, close,
            open_interest, underlying_price, mid_quote, dte, moneyness,
            relative_spread
    """
    # --- EOD ---
    eod = pl.read_parquet(eod_path)

    # Deduplicate exact-copy rows (~19% are dupes from ThetaData)
    eod = eod.unique(subset=["date", "expiration", "strike", "right"])

    # Parse expiration to Date for DTE calculation
    eod = eod.with_columns(
        pl.col("expiration").str.to_date("%Y-%m-%d").alias("expiration_date")
    )

    # --- OI ---
    oi = pl.read_parquet(oi_path)
    oi = oi.with_columns(
        pl.col("timestamp").str.slice(0, 10).str.to_date("%Y-%m-%d").alias("date")
    )
    # OI can also have duplicates
    oi = oi.unique(subset=["date", "expiration", "strike", "right"])

    oi_join = oi.select("date", "expiration", "strike", "right", "open_interest")

    # --- Underlying ---
    underlying = pl.read_parquet(underlying_path)
    underlying_join = underlying.select("date", "underlying_price")

    # --- Joins ---
    df = eod.join(oi_join, on=["date", "expiration", "strike", "right"], how="left")
    df = df.join(underlying_join, on="date", how="left")

    # --- Derived columns ---
    df = df.with_columns(
        # Mid-quote: average of bid and ask
        ((pl.col("bid") + pl.col("ask")) / 2).alias("mid_quote"),
        # DTE: days to expiration
        (pl.col("expiration_date") - pl.col("date")).dt.total_days().alias("dte"),
        # Moneyness: K/S ratio
        (pl.col("strike") / pl.col("underlying_price")).alias("moneyness"),
        # Relative bid-ask spread
        ((pl.col("ask") - pl.col("bid")) / ((pl.col("bid") + pl.col("ask")) / 2)).alias(
            "relative_spread"
        ),
    )

    # Sort chronologically, then by contract
    df = df.sort("date", "expiration", "strike", "right")

    # Select final columns (drop intermediate expiration_date)
    return df.select(
        "symbol",
        "date",
        "expiration",
        "expiration_date",
        "strike",
        "right",
        "bid",
        "ask",
        "volume",
        "close",
        "open_interest",
        "underlying_price",
        "mid_quote",
        "dte",
        "moneyness",
        "relative_spread",
    )


def join_symbol(
    symbol: str,
    trimmed_dir: Path,
    oi_dir: Path,
    underlying_dir: Path,
    output_dir: Path,
) -> int:
    """Join data for a single symbol and save to parquet.

    Returns row count written.
    """
    eod_path = trimmed_dir / f"{symbol}.parquet"
    oi_path = oi_dir / f"{symbol}.parquet"
    underlying_path = underlying_dir / f"{symbol}.parquet"

    if not eod_path.exists():
        print(f"  {symbol}: SKIPPED (no trimmed EOD)")
        return 0
    if not oi_path.exists():
        print(f"  {symbol}: SKIPPED (no OI)")
        return 0
    if not underlying_path.exists():
        print(f"  {symbol}: SKIPPED (no underlying)")
        return 0

    df = join_option_data(eod_path, oi_path, underlying_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{symbol}.parquet"
    df.write_parquet(output_path)

    return len(df)


def join_all(
    symbols: list[str],
    trimmed_dir: Path,
    oi_dir: Path,
    underlying_dir: Path,
    output_dir: Path,
) -> dict[str, int]:
    """Join data for all symbols.

    Returns dict mapping symbol to row count.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, int] = {}

    for symbol in symbols:
        rows = join_symbol(symbol, trimmed_dir, oi_dir, underlying_dir, output_dir)
        if rows > 0:
            print(f"  {symbol}: {rows:,} rows")
            results[symbol] = rows

    return results


if __name__ == "__main__":
    from theta.config import load_config

    config = load_config()
    trimmed_dir = Path("data/processed/eod_trimmed")
    oi_dir = Path("data/raw/open_interest")
    underlying_dir = Path("data/raw/underlying")
    output_dir = Path("data/processed/joined")

    print(f"Joining {len(config.symbols.universe)} symbols (EOD + OI + underlying)")
    print(f"  Trimmed EOD: {trimmed_dir}")
    print(f"  OI:          {oi_dir}")
    print(f"  Underlying:  {underlying_dir}")
    print(f"  Output:      {output_dir}\n")

    results = join_all(
        config.symbols.universe, trimmed_dir, oi_dir, underlying_dir, output_dir
    )

    total = sum(results.values())
    print(f"\nDone: {len(results)} symbols, {total:,} total rows")
