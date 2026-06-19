"""Phase 4 orchestrator: compute all features per symbol.

Reads underlying prices, options_iv (delta-filtered), and options_clean
(full chain) to produce features/{SYMBOL}.parquet per symbol at the
(symbol, date) level with 27 features + target.

BKM risk-neutral moments are computed from the full strike chain
(options_clean + IV) per Carr & Wu (2009). All other option features
use the delta-filtered options_iv data.

Usage:
    python -m theta.processing.compute_features
"""

from __future__ import annotations

import time
from pathlib import Path

import polars as pl

from theta.processing.rv import compute_rv_for_symbol
from theta.processing.option_features import compute_option_features
from theta.processing.technical import compute_technical_features
from theta.processing.iv import compute_iv_and_delta


def process_symbol(
    symbol: str,
    underlying_dir: Path,
    options_iv_dir: Path,
    options_clean_dir: Path,
    output_dir: Path,
    rates: pl.DataFrame,
) -> dict[str, int | float] | None:
    """Compute all Phase 4 features for one symbol.

    Returns dict with row count and timing, or None if skipped.
    """
    underlying_path = underlying_dir / f"{symbol}.parquet"
    options_path = options_iv_dir / f"{symbol}.parquet"
    clean_path = options_clean_dir / f"{symbol}.parquet"

    if not underlying_path.exists():
        return None
    if not options_path.exists():
        return None

    t0 = time.time()

    # --- 1. RV features (full series, no warmup truncation) ---
    rv_full = compute_rv_for_symbol(underlying_path, symbol, truncate_warmup=False)

    # --- 2. Technical features on FULL series (needs lookback into warmup) ---
    tech_df = compute_technical_features(
        rv_full.select("date", "underlying_price", "log_return")
    )
    tech_cols = [
        "date", "mom_5d", "mom_22d", "mom_63d", "mom_126d", "mom_252d",
        "max_daily_ret", "rsi_14", "ma_cross_1_9", "ma_cross_2_12",
    ]
    tech_df = tech_df.select(tech_cols)

    # --- 3. Truncate RV to post-warmup ---
    from theta.processing.rv import WARMUP_DAYS
    rv_df = rv_full.slice(WARMUP_DAYS) if len(rv_full) > WARMUP_DAYS else rv_full

    # --- 4. Option-implied features ---
    options_df = pl.read_parquet(options_path)

    # BKM: compute IV on full chain (options_clean) for wide strike coverage
    # Per Carr & Wu (2009): BKM needs all OTM strikes, not delta-filtered subset
    df_wide = None
    if clean_path.exists():
        df_clean = pl.read_parquet(clean_path)
        df_wide = compute_iv_and_delta(df_clean, rates)
        # Keep only rows with valid IV (drop solver failures)
        df_wide = df_wide.filter(pl.col("iv").is_not_null() & (pl.col("iv") > 0))

    opt_features = compute_option_features(options_df, rv_df, df_wide=df_wide)

    # --- 5. Join everything on date (rv_df is the base = post-warmup) ---
    features = rv_df.select(
        "symbol", "date",
        "rv_d", "rv_w", "rv_m", "rq", "rs_pos", "rs_neg", "rv_21d_forward",
    )
    features = features.join(tech_df, on="date", how="left")
    features = features.join(opt_features, on="date", how="left")

    # Sort and save
    features = features.sort("date")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{symbol}.parquet"
    features.write_parquet(output_path)

    elapsed = time.time() - t0

    # Feature null stats
    feature_cols = [c for c in features.columns if c not in ("symbol", "date")]
    null_pcts = {
        c: features[c].null_count() / len(features) * 100
        for c in feature_cols
    }
    max_null_col = max(null_pcts, key=null_pcts.get) if null_pcts else ""
    max_null_pct = null_pcts.get(max_null_col, 0)

    return {
        "rows": len(features),
        "n_features": len(feature_cols),
        "max_null_col": max_null_col,
        "max_null_pct": max_null_pct,
        "elapsed": elapsed,
    }


def process_all(
    symbols: list[str],
    underlying_dir: Path,
    options_iv_dir: Path,
    options_clean_dir: Path,
    output_dir: Path,
    rates: pl.DataFrame,
) -> dict[str, dict]:
    """Compute features for all symbols.

    Returns dict mapping symbol to stats.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}

    for i, symbol in enumerate(symbols, 1):
        stats = process_symbol(
            symbol, underlying_dir, options_iv_dir,
            options_clean_dir, output_dir, rates,
        )
        if stats is None:
            print(f"  [{i:3d}/{len(symbols)}] {symbol}: SKIPPED (missing data)")
            continue

        results[symbol] = stats
        print(
            f"  [{i:3d}/{len(symbols)}] {symbol}: "
            f"{stats['rows']:>6,} rows x {stats['n_features']} features "
            f"(worst null: {stats['max_null_col']} {stats['max_null_pct']:.1f}%) "
            f"[{stats['elapsed']:.1f}s]"
        )

    return results


if __name__ == "__main__":
    import os

    from dotenv import load_dotenv

    from theta.config import load_config
    from theta.processing.rates import load_risk_free_rate

    load_dotenv()
    config = load_config()

    underlying_dir = Path("data/raw/underlying")
    options_iv_dir = Path("data/processed/options_iv")
    options_clean_dir = Path("data/processed/options_clean")
    output_dir = Path("data/processed/features")
    rates_cache = Path("data/processed/rates/risk_free_rate.parquet")

    symbols = config.symbols.universe

    # Load risk-free rate (needed for IV on clean data for BKM)
    api_key = os.getenv("FRED_API_KEY")
    rates = load_risk_free_rate(rates_cache, api_key=api_key)

    print("Phase 4: Feature Engineering")
    print("=" * 60)
    print(f"  Underlying:     {underlying_dir}")
    print(f"  Options IV:     {options_iv_dir}")
    print(f"  Options clean:  {options_clean_dir} (full chain for BKM)")
    print(f"  Output:         {output_dir}")
    print(f"  Symbols:        {len(symbols)}\n")

    t_start = time.time()
    results = process_all(
        symbols, underlying_dir, options_iv_dir,
        options_clean_dir, output_dir, rates,
    )
    t_total = time.time() - t_start

    # Summary
    total_rows = sum(r["rows"] for r in results.values())
    print(f"\n{'=' * 60}")
    print(f"Done: {len(results)} symbols in {t_total:.0f}s ({t_total / 60:.1f}min)")
    print(f"  Total rows:    {total_rows:>10,}")
    print(f"  Rows/symbol:   ~{total_rows // max(len(results), 1):,}")

    # Worst null features across all symbols
    all_max_nulls = [(s, r["max_null_col"], r["max_null_pct"]) for s, r in results.items()]
    all_max_nulls.sort(key=lambda x: x[2], reverse=True)
    print(f"\n  Top 5 worst null features:")
    for sym, col, pct in all_max_nulls[:5]:
        print(f"    {sym:6s} {col:20s} {pct:.1f}%")
