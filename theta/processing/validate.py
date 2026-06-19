"""Validate raw parquet files for completeness and data integrity.

Usage:
    python -m theta.processing.validate
    python -m theta.processing.validate --dir data/raw --kind eod
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import polars as pl

EOD_COLUMNS = [
    "symbol", "expiration", "strike", "right", "bid", "ask",
    "volume", "close", "open", "high", "low", "count",
    "last_trade", "bid_size", "bid_exchange", "bid_condition",
    "ask_size", "ask_exchange", "ask_condition", "created",
]

OI_COLUMNS = [
    "symbol", "strike", "open_interest", "expiration", "right", "timestamp",
]

UNDERLYING_COLUMNS = ["symbol", "date", "underlying_price"]

# Symbols with later listing dates
LATE_START = {
    "META": "2021-07-08",
    "COIN": "2021-04-14",
    "RIVN": "2021-11-10",
    "ABNB": "2020-12-10",
}

MIN_ROWS = {"eod": 500_000, "oi": 100_000, "underlying": 500}
MIN_TRADING_DAYS = {"eod": 1000, "oi": 1000, "underlying": 500}


@dataclass
class ValidationResult:
    symbol: str
    kind: str
    row_count: int
    passed: bool
    issues: list[str] = field(default_factory=list)


def _extract_date_series(df: pl.DataFrame, col: str) -> pl.Series | None:
    """Extract a Date series from various column formats."""
    s = df[col]
    if s.dtype == pl.Date:
        return s
    if s.dtype == pl.Datetime:
        return s.cast(pl.Date)
    # String timestamps like "2021-01-04T18:00:17.048"
    if s.dtype == pl.Utf8:
        try:
            return s.str.slice(0, 10).str.to_date("%Y-%m-%d")
        except Exception:
            pass
    return None


def validate_eod(path: Path) -> ValidationResult:
    """Validate a single EOD parquet file."""
    return _validate_file(path, kind="eod", expected_cols=EOD_COLUMNS, date_col="created")


def validate_oi(path: Path) -> ValidationResult:
    """Validate a single open interest parquet file."""
    return _validate_file(path, kind="oi", expected_cols=OI_COLUMNS, date_col="timestamp")


def validate_underlying(path: Path) -> ValidationResult:
    """Validate a single underlying price parquet file."""
    return _validate_file(path, kind="underlying", expected_cols=UNDERLYING_COLUMNS, date_col="date")


def _validate_file(
    path: Path,
    *,
    kind: str,
    expected_cols: list[str],
    date_col: str,
) -> ValidationResult:
    """Core validation logic for any parquet file type."""
    symbol = path.stem
    issues: list[str] = []

    try:
        df = pl.read_parquet(path)
    except Exception as exc:
        return ValidationResult(symbol, kind, 0, False, [f"Cannot read: {exc}"])

    nrows = len(df)
    if nrows == 0:
        return ValidationResult(symbol, kind, 0, False, ["Empty file"])

    # Column check
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        issues.append(f"Missing columns: {missing}")

    # All-null columns
    for col in expected_cols:
        if col in df.columns and df[col].null_count() == nrows:
            issues.append(f"Column '{col}' entirely null")

    # Row count
    min_rows = MIN_ROWS.get(kind, 0)
    if nrows < min_rows:
        issues.append(f"Low rows: {nrows:,} (min {min_rows:,})")

    # Date coverage
    if date_col in df.columns:
        dates = _extract_date_series(df, date_col)
        if dates is not None:
            min_date = dates.min()
            max_date = dates.max()
            n_days = dates.n_unique()

            expected_start = LATE_START.get(symbol, "2021-01-01")
            if min_date is not None:
                expected_dt = date.fromisoformat(expected_start)
                if min_date > expected_dt + timedelta(days=30):
                    issues.append(f"Starts late: {min_date} (expected ~{expected_start})")

            if max_date is not None and str(max_date) < "2026-02-01":
                issues.append(f"Ends early: {max_date}")

            min_days = MIN_TRADING_DAYS.get(kind, 0)
            if n_days < min_days:
                issues.append(f"Only {n_days} trading days (min {min_days})")

    # Value checks
    if "strike" in df.columns:
        bad = (df["strike"] <= 0).sum()
        if bad > 0:
            issues.append(f"{bad:,} non-positive strikes")

    if "bid" in df.columns:
        bad = (df["bid"] < 0).sum()
        if bad > 0:
            issues.append(f"{bad:,} negative bids")

    if "underlying_price" in df.columns:
        bad = (df["underlying_price"] <= 0).sum()
        if bad > 0:
            issues.append(f"{bad:,} non-positive underlying prices")

    return ValidationResult(symbol, kind, nrows, len(issues) == 0, issues)


def validate_all(
    raw_dir: Path,
    symbols: list[str],
) -> dict[str, list[ValidationResult]]:
    """Validate all three data types for all symbols.

    Returns dict mapping kind -> list of ValidationResults.
    """
    results: dict[str, list[ValidationResult]] = {"eod": [], "oi": [], "underlying": []}

    for symbol in symbols:
        eod_path = raw_dir / "eod" / f"{symbol}.parquet"
        if eod_path.exists():
            results["eod"].append(validate_eod(eod_path))

        oi_path = raw_dir / "open_interest" / f"{symbol}.parquet"
        if oi_path.exists():
            results["oi"].append(validate_oi(oi_path))

        underlying_path = raw_dir / "underlying" / f"{symbol}.parquet"
        if underlying_path.exists():
            results["underlying"].append(validate_underlying(underlying_path))

    return results


def print_results(results: dict[str, list[ValidationResult]]) -> None:
    """Print validation summary."""
    for kind, vrs in results.items():
        ok = sum(1 for v in vrs if v.passed)
        fail = sum(1 for v in vrs if not v.passed)
        print(f"\n{kind.upper()}: {ok} OK, {fail} with issues")
        for v in vrs:
            if not v.passed:
                print(f"  {v.symbol}: {v.issues}")


if __name__ == "__main__":
    from theta.config import load_config

    config = load_config()
    raw_dir = Path("data/raw")

    print(f"Validating {len(config.symbols.universe)} symbols in {raw_dir}")
    results = validate_all(raw_dir, config.symbols.universe)
    print_results(results)
