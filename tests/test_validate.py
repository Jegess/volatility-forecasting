"""Tests for theta.processing.validate."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from theta.processing.validate import (
    ValidationResult,
    validate_eod,
    validate_oi,
    validate_underlying,
)


@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    return tmp_path


def _make_dates(rows: int, start: str = "2021-01-04", end: str = "2026-03-06") -> list[str]:
    """Generate evenly spaced date strings across the full range."""
    import datetime
    s = datetime.date.fromisoformat(start)
    e = datetime.date.fromisoformat(end)
    total_days = (e - s).days
    return [(s + datetime.timedelta(days=int(i * total_days / max(rows, 1)))).isoformat() + "T18:00:17.048" for i in range(rows)]


def _write_eod(path: Path, rows: int = 1000, **overrides: object) -> Path:
    """Create a minimal valid EOD parquet file."""
    timestamps = _make_dates(rows)
    data = {
        "symbol": ["AAPL"] * rows,
        "expiration": ["2025-03-21"] * rows,
        "strike": [150.0] * rows,
        "right": ["C"] * rows,
        "bid": [5.0] * rows,
        "ask": [5.5] * rows,
        "volume": [100] * rows,
        "close": [5.25] * rows,
        "open": [5.0] * rows,
        "high": [5.5] * rows,
        "low": [4.9] * rows,
        "count": [50] * rows,
        "last_trade": ["2025-03-06T16:00:00.000"] * rows,
        "bid_size": [10] * rows,
        "bid_exchange": [0] * rows,
        "bid_condition": [0] * rows,
        "ask_size": [10] * rows,
        "ask_exchange": [0] * rows,
        "ask_condition": [0] * rows,
        "created": timestamps,
    }
    data.update(overrides)
    df = pl.DataFrame(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    return path


def _write_oi(path: Path, rows: int = 1000, **overrides: object) -> Path:
    """Create a minimal valid OI parquet file."""
    timestamps = _make_dates(rows)
    data = {
        "symbol": ["AAPL"] * rows,
        "strike": [150.0] * rows,
        "open_interest": [500] * rows,
        "expiration": ["2025-03-21"] * rows,
        "right": ["C"] * rows,
        "timestamp": timestamps,
    }
    data.update(overrides)
    df = pl.DataFrame(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    return path


def _write_underlying(path: Path, rows: int = 1000) -> Path:
    """Create a minimal valid underlying parquet file."""
    import datetime

    start = datetime.date(2021, 1, 4)
    end = datetime.date(2026, 3, 6)
    total_days = (end - start).days
    dates = [start + datetime.timedelta(days=int(i * total_days / max(rows, 1))) for i in range(rows)]
    df = pl.DataFrame({
        "symbol": ["AAPL"] * rows,
        "date": dates,
        "underlying_price": [150.0 + i * 0.1 for i in range(rows)],
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    return path


class TestValidateEod:
    def test_valid_file(self, tmp_dir: Path) -> None:
        path = _write_eod(tmp_dir / "AAPL.parquet", rows=600_000)
        result = validate_eod(path)
        assert result.passed
        assert result.row_count == 600_000
        assert result.issues == []

    def test_low_row_count(self, tmp_dir: Path) -> None:
        path = _write_eod(tmp_dir / "AAPL.parquet", rows=100)
        result = validate_eod(path)
        assert not result.passed
        assert any("Low rows" in i for i in result.issues)

    def test_negative_bid(self, tmp_dir: Path) -> None:
        path = _write_eod(tmp_dir / "AAPL.parquet", rows=600_000, bid=[-1.0] * 600_000)
        result = validate_eod(path)
        assert not result.passed
        assert any("negative bids" in i for i in result.issues)

    def test_empty_file(self, tmp_dir: Path) -> None:
        path = _write_eod(tmp_dir / "AAPL.parquet", rows=0)
        result = validate_eod(path)
        assert not result.passed
        assert any("Empty" in i for i in result.issues)


class TestValidateOi:
    def test_valid_file(self, tmp_dir: Path) -> None:
        path = _write_oi(tmp_dir / "AAPL.parquet", rows=200_000)
        result = validate_oi(path)
        assert result.passed

    def test_low_row_count(self, tmp_dir: Path) -> None:
        path = _write_oi(tmp_dir / "AAPL.parquet", rows=50)
        result = validate_oi(path)
        assert not result.passed


class TestValidateUnderlying:
    def test_valid_file(self, tmp_dir: Path) -> None:
        path = _write_underlying(tmp_dir / "AAPL.parquet", rows=1000)
        result = validate_underlying(path)
        assert result.passed
        assert result.row_count == 1000

    def test_low_row_count(self, tmp_dir: Path) -> None:
        path = _write_underlying(tmp_dir / "AAPL.parquet", rows=10)
        result = validate_underlying(path)
        assert not result.passed
