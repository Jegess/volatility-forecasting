"""Trim EOD parquet files from 20 columns to 9 (8 literature columns + date).

Reads raw EOD files and outputs trimmed versions with only the columns
needed for downstream processing.

Usage:
    python -m theta.processing.trim
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

# 8 columns per literature (Bali et al., Driessen et al., Carr & Wu)
# plus date extracted from created timestamp
KEEP_COLUMNS = [
    "symbol",
    "date",
    "expiration",
    "strike",
    "right",
    "bid",
    "ask",
    "volume",
    "close",
]


def trim_eod_file(input_path: Path, output_path: Path) -> int:
    """Trim a single EOD parquet file to 9 columns.

    Extracts date from the 'created' timestamp string (e.g. '2021-01-04T18:00:17.048').

    Returns row count written.
    """
    df = pl.read_parquet(input_path)

    df = df.with_columns(
        pl.col("created").str.slice(0, 10).str.to_date("%Y-%m-%d").alias("date")
    )

    df = df.select(KEEP_COLUMNS)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output_path)

    return len(df)


def trim_eod(
    input_dir: Path,
    output_dir: Path,
    symbols: list[str],
) -> dict[str, int]:
    """Trim all EOD files for the given symbols.

    Args:
        input_dir: Directory containing raw EOD parquets (e.g. data/raw/eod/).
        output_dir: Directory for trimmed output (e.g. data/processed/eod_trimmed/).
        symbols: List of symbols to process.

    Returns:
        Dict mapping symbol to row count written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, int] = {}

    for symbol in symbols:
        input_path = input_dir / f"{symbol}.parquet"
        if not input_path.exists():
            print(f"  {symbol}: SKIPPED (no raw file)")
            continue

        output_path = output_dir / f"{symbol}.parquet"
        rows = trim_eod_file(input_path, output_path)
        results[symbol] = rows
        print(f"  {symbol}: {rows:,} rows")

    return results


if __name__ == "__main__":
    from theta.config import load_config

    config = load_config()
    raw_eod = Path("data/raw/eod")
    trimmed = Path("data/processed/eod_trimmed")

    print(f"Trimming {len(config.symbols.universe)} EOD files: 20 -> {len(KEEP_COLUMNS)} columns")
    print(f"  Input:  {raw_eod}")
    print(f"  Output: {trimmed}\n")

    results = trim_eod(raw_eod, trimmed, config.symbols.universe)

    total = sum(results.values())
    print(f"\nDone: {len(results)} files, {total:,} total rows")
