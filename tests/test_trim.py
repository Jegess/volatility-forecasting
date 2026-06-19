"""Tests for theta.processing.trim."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from theta.processing.trim import KEEP_COLUMNS, trim_eod, trim_eod_file


@pytest.fixture
def eod_file(tmp_path: Path) -> Path:
    """Create a sample raw EOD parquet with 20 columns."""
    n = 100
    df = pl.DataFrame({
        "symbol": ["AAPL"] * n,
        "expiration": ["2025-03-21"] * n,
        "strike": [150.0] * n,
        "right": ["C"] * n,
        "bid": [5.0] * n,
        "ask": [5.5] * n,
        "volume": [100] * n,
        "close": [5.25] * n,
        "open": [5.0] * n,
        "high": [5.5] * n,
        "low": [4.9] * n,
        "count": [50] * n,
        "last_trade": ["2025-03-06T16:00:00.000"] * n,
        "bid_size": [10] * n,
        "bid_exchange": [0] * n,
        "bid_condition": [0] * n,
        "ask_size": [10] * n,
        "ask_exchange": [0] * n,
        "ask_condition": [0] * n,
        "created": [f"2021-01-{(i % 28) + 1:02d}T18:00:17.048" for i in range(n)],
    })
    path = tmp_path / "raw" / "eod" / "AAPL.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    return path


class TestTrimEodFile:
    def test_output_has_correct_columns(self, eod_file: Path, tmp_path: Path) -> None:
        out = tmp_path / "trimmed" / "AAPL.parquet"
        trim_eod_file(eod_file, out)

        df = pl.read_parquet(out)
        assert df.columns == KEEP_COLUMNS

    def test_row_count_preserved(self, eod_file: Path, tmp_path: Path) -> None:
        out = tmp_path / "trimmed" / "AAPL.parquet"
        rows = trim_eod_file(eod_file, out)

        assert rows == 100
        df = pl.read_parquet(out)
        assert len(df) == 100

    def test_date_extracted_correctly(self, eod_file: Path, tmp_path: Path) -> None:
        out = tmp_path / "trimmed" / "AAPL.parquet"
        trim_eod_file(eod_file, out)

        df = pl.read_parquet(out)
        assert df["date"].dtype == pl.Date
        # First row created was "2021-01-01T18:00:17.048"
        import datetime
        assert df["date"][0] == datetime.date(2021, 1, 1)

    def test_dropped_columns_absent(self, eod_file: Path, tmp_path: Path) -> None:
        out = tmp_path / "trimmed" / "AAPL.parquet"
        trim_eod_file(eod_file, out)

        df = pl.read_parquet(out)
        dropped = {"open", "high", "low", "count", "last_trade", "bid_size",
                    "bid_exchange", "bid_condition", "ask_size", "ask_exchange",
                    "ask_condition", "created"}
        assert dropped.isdisjoint(set(df.columns))


class TestTrimEod:
    def test_processes_all_symbols(self, eod_file: Path, tmp_path: Path) -> None:
        out_dir = tmp_path / "trimmed"
        results = trim_eod(eod_file.parent, out_dir, ["AAPL"])

        assert "AAPL" in results
        assert results["AAPL"] == 100
        assert (out_dir / "AAPL.parquet").exists()

    def test_skips_missing_symbol(self, eod_file: Path, tmp_path: Path) -> None:
        out_dir = tmp_path / "trimmed"
        results = trim_eod(eod_file.parent, out_dir, ["MISSING"])

        assert "MISSING" not in results
