"""Literature-prescribed option data filters.

Applies filters in the correct order per Bali et al. (2021),
Driessen et al. (2009), and Carr & Wu (2009):

    1. Data integrity  (bid=0, ask<=bid, null underlying)
    2. Liquidity        (mid < $0.125, spread > 50%, zero OI)
    3. Volume           (zero trailing 7-day volume)
    4. DTE              (< 14 days)
    5. Monthly only     (3rd Friday expirations, no weeklies)

Delta/moneyness filtering is deferred to Phase 3 (requires IV).

Usage:
    python -m theta.processing.filters
"""

from __future__ import annotations

from pathlib import Path

import polars as pl


# ---------------------------------------------------------------------------
# Individual filter functions
# ---------------------------------------------------------------------------


def filter_integrity(df: pl.DataFrame) -> pl.DataFrame:
    """Remove rows with broken quotes or missing underlying.

    Drops rows where:
        - bid <= 0 (no market maker willing to buy)
        - ask <= bid (inverted/corrupt quote)
        - underlying_price is null (can't compute moneyness/Greeks)
    """
    return df.filter(
        (pl.col("bid") > 0)
        & (pl.col("ask") > pl.col("bid"))
        & pl.col("underlying_price").is_not_null()
    )


def filter_liquidity(df: pl.DataFrame) -> pl.DataFrame:
    """Remove illiquid options.

    Drops rows where:
        - mid_quote < $0.125 (minimum viable option price, Bali et al.)
        - relative_spread > 0.50 (bid-ask spread > 50% of mid)
        - open_interest is null or 0 (nobody holds this contract)
    """
    return df.filter(
        (pl.col("mid_quote") >= 0.125)
        & (pl.col("relative_spread") <= 0.50)
        & pl.col("open_interest").is_not_null()
        & (pl.col("open_interest") > 0)
    )


def filter_volume_7d(df: pl.DataFrame) -> pl.DataFrame:
    """Remove options with zero volume over trailing 7 calendar days.

    Per Bali et al. (2021): option must have traded at least once
    in the previous 7 calendar days. Uses polars rolling_sum_by
    with a 7-day temporal window per contract.
    """
    df_sorted = df.sort("date")

    df_with_vol = df_sorted.with_columns(
        pl.col("volume")
        .rolling_sum_by("date", window_size="7d")
        .over("expiration", "strike", "right")
        .alias("_volume_7d")
    )

    return df_with_vol.filter(pl.col("_volume_7d") > 0).drop("_volume_7d")


def filter_dte(df: pl.DataFrame, min_dte: int = 14) -> pl.DataFrame:
    """Remove short-dated options.

    Per Bali et al. (2021) and Driessen et al. (2009): exclude
    options with fewer than 14 days to expiration. Near-expiry
    options have erratic pricing (gamma explosion, non-linear decay).
    """
    return df.filter(pl.col("dte") >= min_dte)


def _is_standard_monthly(expiration_date: pl.Expr) -> pl.Expr:
    """Check if expiration is a standard monthly (3rd Friday or holiday-shifted Thursday).

    Standard monthly options expire on the 3rd Friday of the month.
    If that Friday is a market holiday (e.g., Good Friday), the
    exchange shifts expiration to Thursday.

    3rd Friday:   weekday == 5 (Fri) AND day in [15, 21]
    Holiday shift: weekday == 4 (Thu) AND day in [14, 20]
    """
    day = expiration_date.dt.day()
    weekday = expiration_date.dt.weekday()  # 1=Mon ... 7=Sun

    is_third_friday = (weekday == 5) & (day >= 15) & (day <= 21)
    is_holiday_thursday = (weekday == 4) & (day >= 14) & (day <= 20)

    return is_third_friday | is_holiday_thursday


def filter_monthly(df: pl.DataFrame) -> pl.DataFrame:
    """Keep only standard monthly expirations (3rd Friday).

    Per Bali et al. (2021): exclude weekly and non-standard
    expirations. Weeklies have different liquidity dynamics and
    create unbalanced panels where recent years dominate.
    """
    return df.filter(_is_standard_monthly(pl.col("expiration_date")))


# ---------------------------------------------------------------------------
# Filter chain
# ---------------------------------------------------------------------------


def filter_all(
    df: pl.DataFrame,
    *,
    min_dte: int = 14,
    verbose: bool = True,
) -> pl.DataFrame:
    """Apply the full literature-prescribed filter chain.

    Returns filtered DataFrame and prints per-step statistics
    if verbose=True.
    """
    n_start = len(df)

    steps: list[tuple[str, pl.DataFrame]] = []

    # Step 1: Data integrity
    df = filter_integrity(df)
    steps.append(("integrity", df))

    # Step 2: Liquidity
    df = filter_liquidity(df)
    steps.append(("liquidity", df))

    # Step 3: Trailing 7-day volume
    df = filter_volume_7d(df)
    steps.append(("volume_7d", df))

    # Step 4: DTE
    df = filter_dte(df, min_dte=min_dte)
    steps.append(("dte", df))

    # Step 5: Monthly expirations
    df = filter_monthly(df)
    steps.append(("monthly", df))

    if verbose:
        prev = n_start
        for name, step_df in steps:
            n = len(step_df)
            removed = prev - n
            pct = removed / prev * 100 if prev > 0 else 0
            print(f"    {name:12s}: {prev:>10,} -> {n:>10,}  ({removed:>10,} removed, {pct:5.1f}%)")
            prev = n
        total_removed = n_start - len(df)
        pct_total = total_removed / n_start * 100 if n_start > 0 else 0
        print(f"    {'TOTAL':12s}: {n_start:>10,} -> {len(df):>10,}  ({total_removed:>10,} removed, {pct_total:5.1f}%)")

    return df


# ---------------------------------------------------------------------------
# Per-symbol processing
# ---------------------------------------------------------------------------


def filter_symbol(
    symbol: str,
    joined_dir: Path,
    output_dir: Path,
    *,
    min_dte: int = 14,
    verbose: bool = True,
) -> int:
    """Filter a single symbol's joined data and save to parquet.

    Returns row count written (0 if input file missing).
    """
    input_path = joined_dir / f"{symbol}.parquet"
    if not input_path.exists():
        if verbose:
            print(f"  {symbol}: SKIPPED (no joined file)")
        return 0

    df = pl.read_parquet(input_path)
    if verbose:
        print(f"  {symbol}: {len(df):,} rows")

    df = filter_all(df, min_dte=min_dte, verbose=verbose)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{symbol}.parquet"
    df.write_parquet(output_path)

    return len(df)


def filter_all_symbols(
    symbols: list[str],
    joined_dir: Path,
    output_dir: Path,
    *,
    min_dte: int = 14,
    verbose: bool = True,
) -> dict[str, int]:
    """Filter all symbols. Returns dict mapping symbol to row count."""
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, int] = {}

    for symbol in symbols:
        rows = filter_symbol(
            symbol, joined_dir, output_dir,
            min_dte=min_dte, verbose=verbose,
        )
        if rows > 0:
            results[symbol] = rows

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    from theta.config import load_config

    config = load_config()
    joined_dir = Path("data/processed/joined")
    output_dir = Path("data/processed/options_clean")

    print(f"Filtering {len(config.symbols.universe)} symbols")
    print(f"  Input:  {joined_dir}")
    print(f"  Output: {output_dir}\n")

    results = filter_all_symbols(
        config.symbols.universe, joined_dir, output_dir,
    )

    total = sum(results.values())
    print(f"\nDone: {len(results)} symbols, {total:,} total rows")
