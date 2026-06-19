"""Phase 3 orchestrator: compute IV + delta, apply delta filter.

Reads from data/processed/options_clean/, writes to data/processed/options_iv/.

Usage:
    python -m theta.processing.compute_iv
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import polars as pl
from dotenv import load_dotenv

from theta.processing.iv import compute_iv_and_delta, filter_delta
from theta.processing.rates import load_risk_free_rate


def process_symbol(
    symbol: str,
    clean_dir: Path,
    output_dir: Path,
    rates: pl.DataFrame,
) -> dict[str, int | float] | None:
    """Process IV for one symbol.

    Returns dict with row counts and timing, or None if skipped.
    """
    clean_path = clean_dir / f"{symbol}.parquet"
    if not clean_path.exists():
        return None

    t0 = time.time()

    df = pl.read_parquet(clean_path)
    rows_in = len(df)

    # Compute IV and delta
    df = compute_iv_and_delta(df, rates)

    iv_null = df["iv"].null_count()
    iv_null_pct = iv_null / rows_in * 100 if rows_in > 0 else 0

    # Apply delta filter
    df_filtered = filter_delta(df)
    rows_out = len(df_filtered)
    delta_removed = rows_in - iv_null - rows_out  # rows with valid IV but outside delta range

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{symbol}.parquet"
    df_filtered.write_parquet(output_path)

    elapsed = time.time() - t0

    return {
        "rows_in": rows_in,
        "rows_out": rows_out,
        "iv_null": iv_null,
        "iv_null_pct": iv_null_pct,
        "survival_pct": rows_out / rows_in * 100 if rows_in > 0 else 0,
        "elapsed": elapsed,
    }


def process_all(
    symbols: list[str],
    clean_dir: Path,
    output_dir: Path,
    rates: pl.DataFrame,
) -> dict[str, dict]:
    """Process IV for all symbols.

    Returns dict mapping symbol to stats.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}
    total_in = 0
    total_out = 0

    for i, symbol in enumerate(symbols, 1):
        stats = process_symbol(symbol, clean_dir, output_dir, rates)
        if stats is None:
            print(f"  [{i:3d}/{len(symbols)}] {symbol}: SKIPPED (no clean file)")
            continue

        results[symbol] = stats
        total_in += stats["rows_in"]
        total_out += stats["rows_out"]

        print(
            f"  [{i:3d}/{len(symbols)}] {symbol}: "
            f"{stats['rows_in']:>9,} -> {stats['rows_out']:>8,} rows "
            f"({stats['survival_pct']:5.1f}% survive, "
            f"IV null {stats['iv_null_pct']:.1f}%) "
            f"[{stats['elapsed']:.1f}s]"
        )

    return results


if __name__ == "__main__":
    load_dotenv()

    from theta.config import load_config

    config = load_config()
    clean_dir = Path("data/processed/options_clean")
    output_dir = Path("data/processed/options_iv")
    rates_cache = Path("data/processed/rates/risk_free_rate.parquet")

    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise RuntimeError("Set FRED_API_KEY in .env")

    print("Phase 3: IV Calculation + Delta Filter")
    print("=" * 60)

    # Step 1: Load/download risk-free rate
    print("\n1. Loading risk-free rate from FRED...")
    rates = load_risk_free_rate(rates_cache, api_key=api_key)
    print(f"   {len(rates)} days, {rates['date'].min()} to {rates['date'].max()}")
    print(f"   Rate range: {rates['rate'].min():.4f} to {rates['rate'].max():.4f}")

    # Step 2+3: Compute IV + delta filter for all symbols
    symbols = config.symbols.universe
    print(f"\n2. Computing IV + delta filter for {len(symbols)} symbols...")
    print(f"   Input:  {clean_dir}")
    print(f"   Output: {output_dir}\n")

    t_start = time.time()
    results = process_all(symbols, clean_dir, output_dir, rates)
    t_total = time.time() - t_start

    # Summary
    total_in = sum(r["rows_in"] for r in results.values())
    total_out = sum(r["rows_out"] for r in results.values())
    total_iv_null = sum(r["iv_null"] for r in results.values())

    print(f"\n{'=' * 60}")
    print(f"Done: {len(results)} symbols in {t_total:.0f}s ({t_total / 60:.1f}min)")
    print(f"  Rows in:      {total_in:>12,}")
    print(f"  IV null:      {total_iv_null:>12,} ({total_iv_null / total_in * 100:.1f}%)")
    print(f"  Rows out:     {total_out:>12,} ({total_out / total_in * 100:.1f}% survival)")
    print(f"  Removed:      {total_in - total_out:>12,} ({(total_in - total_out) / total_in * 100:.1f}%)")
