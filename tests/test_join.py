"""Tests for theta.processing.join."""

from __future__ import annotations

import datetime
from pathlib import Path

import polars as pl
import pytest

from theta.processing.join import join_option_data, join_symbol


def _make_eod(path: Path, n_dates: int = 5, n_strikes: int = 3) -> Path:
    """Create a trimmed EOD parquet with realistic structure."""
    rows = []
    base = datetime.date(2021, 1, 4)
    for d in range(n_dates):
        dt = base + datetime.timedelta(days=d)
        for s in range(n_strikes):
            strike = 130.0 + s * 5
            for right in ["CALL", "PUT"]:
                rows.append({
                    "symbol": "TEST",
                    "date": dt,
                    "expiration": "2021-02-19",
                    "strike": strike,
                    "right": right,
                    "bid": 3.0 + s * 0.5,
                    "ask": 3.5 + s * 0.5,
                    "volume": 100 + s * 10,
                    "close": 3.25 + s * 0.5,
                })
    # Add a duplicate row
    rows.append(rows[0].copy())

    df = pl.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    return path


def _make_oi(path: Path, n_dates: int = 5, n_strikes: int = 3) -> Path:
    """Create an OI parquet matching the EOD structure."""
    rows = []
    base = datetime.date(2021, 1, 4)
    for d in range(n_dates):
        dt = base + datetime.timedelta(days=d)
        for s in range(n_strikes):
            strike = 130.0 + s * 5
            for right in ["CALL", "PUT"]:
                rows.append({
                    "symbol": "TEST",
                    "strike": strike,
                    "open_interest": 500 + s * 100,
                    "expiration": "2021-02-19",
                    "right": right,
                    "timestamp": f"{dt}T07:01:00.000",
                })
    df = pl.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    return path


def _make_underlying(path: Path, n_dates: int = 5) -> Path:
    """Create an underlying parquet."""
    base = datetime.date(2021, 1, 4)
    dates = [base + datetime.timedelta(days=i) for i in range(n_dates)]
    df = pl.DataFrame({
        "symbol": ["TEST"] * n_dates,
        "date": dates,
        "underlying_price": [135.0 + i * 0.5 for i in range(n_dates)],
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    return path


@pytest.fixture
def data_paths(tmp_path: Path) -> tuple[Path, Path, Path]:
    eod = _make_eod(tmp_path / "eod" / "TEST.parquet")
    oi = _make_oi(tmp_path / "oi" / "TEST.parquet")
    und = _make_underlying(tmp_path / "underlying" / "TEST.parquet")
    return eod, oi, und


class TestJoinOptionData:
    def test_output_columns(self, data_paths: tuple[Path, Path, Path]) -> None:
        eod, oi, und = data_paths
        df = join_option_data(eod, oi, und)
        expected = [
            "symbol", "date", "expiration", "expiration_date", "strike", "right",
            "bid", "ask", "volume", "close", "open_interest", "underlying_price",
            "mid_quote", "dte", "moneyness", "relative_spread",
        ]
        assert df.columns == expected

    def test_deduplicates_eod(self, data_paths: tuple[Path, Path, Path]) -> None:
        eod, oi, und = data_paths
        df = join_option_data(eod, oi, und)
        # 5 dates * 3 strikes * 2 rights = 30 unique rows (duplicate removed)
        assert len(df) == 30

    def test_mid_quote_calculated(self, data_paths: tuple[Path, Path, Path]) -> None:
        eod, oi, und = data_paths
        df = join_option_data(eod, oi, und)
        row = df.filter(
            (pl.col("strike") == 130.0) & (pl.col("right") == "CALL")
        ).head(1)
        bid = row["bid"][0]
        ask = row["ask"][0]
        assert row["mid_quote"][0] == pytest.approx((bid + ask) / 2)

    def test_dte_calculated(self, data_paths: tuple[Path, Path, Path]) -> None:
        eod, oi, und = data_paths
        df = join_option_data(eod, oi, und)
        # First date is 2021-01-04, expiration is 2021-02-19 = 46 days
        row = df.filter(pl.col("date") == datetime.date(2021, 1, 4)).head(1)
        assert row["dte"][0] == 46

    def test_moneyness_calculated(self, data_paths: tuple[Path, Path, Path]) -> None:
        eod, oi, und = data_paths
        df = join_option_data(eod, oi, und)
        row = df.filter(
            (pl.col("date") == datetime.date(2021, 1, 4)) & (pl.col("strike") == 135.0)
        ).head(1)
        # underlying_price on 2021-01-04 = 135.0, strike = 135.0
        assert row["moneyness"][0] == pytest.approx(1.0)

    def test_open_interest_joined(self, data_paths: tuple[Path, Path, Path]) -> None:
        eod, oi, und = data_paths
        df = join_option_data(eod, oi, und)
        assert df["open_interest"].null_count() == 0

    def test_underlying_price_joined(self, data_paths: tuple[Path, Path, Path]) -> None:
        eod, oi, und = data_paths
        df = join_option_data(eod, oi, und)
        assert df["underlying_price"].null_count() == 0

    def test_relative_spread_positive(self, data_paths: tuple[Path, Path, Path]) -> None:
        eod, oi, und = data_paths
        df = join_option_data(eod, oi, und)
        assert (df["relative_spread"] >= 0).all()


class TestJoinSymbol:
    def test_writes_output(self, data_paths: tuple[Path, Path, Path], tmp_path: Path) -> None:
        eod, oi, und = data_paths
        out_dir = tmp_path / "output"
        rows = join_symbol(
            "TEST", eod.parent, oi.parent, und.parent, out_dir
        )
        assert rows == 30
        assert (out_dir / "TEST.parquet").exists()

    def test_missing_file_returns_zero(self, tmp_path: Path) -> None:
        rows = join_symbol(
            "MISSING",
            tmp_path / "eod",
            tmp_path / "oi",
            tmp_path / "underlying",
            tmp_path / "output",
        )
        assert rows == 0
