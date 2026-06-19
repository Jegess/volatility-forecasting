"""Assemble pooled panel dataset from per-symbol features + macro.

Steps:
    1. Concat all features/{SYMBOL}.parquet into one DataFrame
    2. Left-join macro features on date
    3. Drop rows where rv_21d_forward is null (last 21 days per symbol)
    4. Fill remaining nulls with cross-sectional median per date (Gu/Kelly/Xiu)

Output: data/processed/panel.parquet

Literature: Gu/Kelly/Xiu (2020) — pooled panel with cross-sectional
median fill for missing features.

Usage:
    python -m theta.processing.panel
"""

from __future__ import annotations

from pathlib import Path

import polars as pl


FEATURES_DIR = Path("data/processed/features")
MACRO_PATH = Path("data/processed/macro/macro.parquet")
OUTPUT_PATH = Path("data/processed/panel.parquet")

# Columns that are identifiers, not features
ID_COLS = {"symbol", "date"}
TARGET_COL = "rv_21d_forward"


def load_features(features_dir: Path) -> pl.DataFrame:
    """Concat all per-symbol feature files into one DataFrame."""
    files = sorted(features_dir.glob("*.parquet"))
    dfs = [pl.read_parquet(f) for f in files]
    panel = pl.concat(dfs, how="diagonal")
    return panel


def join_macro(panel: pl.DataFrame, macro_path: Path) -> pl.DataFrame:
    """Left-join macro features onto panel by date."""
    macro = pl.read_parquet(macro_path)
    return panel.join(macro, on="date", how="left")


def drop_null_target(panel: pl.DataFrame) -> tuple[pl.DataFrame, int]:
    """Drop rows where target (rv_21d_forward) is null."""
    before = len(panel)
    panel = panel.filter(pl.col(TARGET_COL).is_not_null())
    dropped = before - len(panel)
    return panel, dropped


def fill_nulls_cross_sectional_median(panel: pl.DataFrame) -> pl.DataFrame:
    """Fill remaining nulls with cross-sectional median per date.

    For each date, compute the median of each feature across all symbols,
    then fill nulls with that median. This is the Gu/Kelly/Xiu approach
    for handling missing features in pooled panels.
    """
    feature_cols = [
        c for c in panel.columns
        if c not in ID_COLS and c != TARGET_COL
    ]

    # Compute per-date medians for all feature columns
    medians = panel.group_by("date").agg(
        [pl.col(c).median().alias(f"_med_{c}") for c in feature_cols]
    )

    # Join medians, fill nulls, drop median columns
    panel = panel.join(medians, on="date", how="left")

    for c in feature_cols:
        med_col = f"_med_{c}"
        panel = panel.with_columns(
            pl.when(pl.col(c).is_null())
            .then(pl.col(med_col))
            .otherwise(pl.col(c))
            .alias(c)
        )

    # Drop temporary median columns
    med_cols = [f"_med_{c}" for c in feature_cols]
    panel = panel.drop(med_cols)

    return panel


def main() -> None:
    """Build the panel dataset."""
    # Step 1: concat features
    print("Step 1: Loading per-symbol feature files...")
    panel = load_features(FEATURES_DIR)
    n_symbols = panel["symbol"].n_unique()
    print(f"  {n_symbols} symbols, {len(panel):,} rows, {len(panel.columns)} columns")

    # Step 2: join macro
    print("\nStep 2: Joining macro features...")
    cols_before = len(panel.columns)
    panel = join_macro(panel, MACRO_PATH)
    macro_added = len(panel.columns) - cols_before
    print(f"  +{macro_added} macro columns -> {len(panel.columns)} total")

    # Check macro join coverage
    macro_nulls = panel["vix"].null_count()
    if macro_nulls > 0:
        print(f"  WARNING: {macro_nulls} rows have no macro match (date not in macro)")

    # Step 3: drop null target
    print("\nStep 3: Dropping rows with null target...")
    panel, dropped = drop_null_target(panel)
    print(f"  Dropped {dropped:,} rows -> {len(panel):,} remaining")

    # Step 4: fill nulls
    print("\nStep 4: Filling nulls with cross-sectional median per date...")
    # Report null rates before fill
    feature_cols = [
        c for c in panel.columns
        if c not in ID_COLS and c != TARGET_COL
    ]
    nulls_before = {
        c: panel[c].null_count()
        for c in feature_cols
        if panel[c].null_count() > 0
    }
    if nulls_before:
        print("  Nulls before fill:")
        for c, n in sorted(nulls_before.items(), key=lambda x: -x[1]):
            print(f"    {c:<24} {n:>6} ({n/len(panel)*100:.1f}%)")
    else:
        print("  No nulls to fill")

    panel = fill_nulls_cross_sectional_median(panel)

    # Check for any remaining nulls — these are dates where ALL symbols
    # had null (e.g., market holidays with no option data). Drop them.
    remaining = {
        c: panel[c].null_count()
        for c in feature_cols
        if panel[c].null_count() > 0
    }
    if remaining:
        # Drop rows that still have nulls in core option features
        before = len(panel)
        panel = panel.filter(pl.col("atm_iv").is_not_null())
        dropped_nulls = before - len(panel)
        print(f"  Dropped {dropped_nulls} rows with unfillable nulls "
              f"(all-null dates, likely market holidays)")
    else:
        print("  All nulls filled")

    # Step 5: sort and save
    print("\nStep 5: Saving panel...")
    panel = panel.sort(["symbol", "date"])
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    panel.write_parquet(OUTPUT_PATH)
    size_mb = OUTPUT_PATH.stat().st_size / 1024 / 1024
    print(f"  Saved: {OUTPUT_PATH} ({size_mb:.1f} MB)")

    # Summary stats
    print(f"\n{'='*50}")
    print(f"Panel summary:")
    print(f"  Symbols: {panel['symbol'].n_unique()}")
    print(f"  Rows: {len(panel):,}")
    print(f"  Columns: {len(panel.columns)} ({len(feature_cols)} features + target + symbol + date)")
    print(f"  Date range: {panel['date'].min()} to {panel['date'].max()}")
    print(f"  Target (rv_21d_forward):")
    t = panel[TARGET_COL]
    print(f"    mean={t.mean():.4f}, median={t.median():.4f}, "
          f"std={t.std():.4f}, min={t.min():.4f}, max={t.max():.4f}")

    # Per-symbol row counts
    counts = panel.group_by("symbol").len().sort("len")
    print(f"  Rows per symbol: min={counts['len'].min()}, "
          f"max={counts['len'].max()}, "
          f"median={counts['len'].median():.0f}")
    print(f"  Fewest rows: {counts.head(5).to_dict()}")


if __name__ == "__main__":
    main()
