"""Tests for theta.processing.filters."""

from __future__ import annotations

import datetime
from pathlib import Path

import polars as pl
import pytest

from theta.processing.filters import (
    filter_all,
    filter_dte,
    filter_integrity,
    filter_liquidity,
    filter_monthly,
    filter_symbol,
    filter_volume_7d,
)


def _base_df(**overrides) -> pl.DataFrame:
    """Create a single-row DataFrame with valid defaults, overridable per test."""
    defaults = {
        "symbol": "TEST",
        "date": datetime.date(2021, 1, 4),
        "expiration": "2021-01-15",
        "expiration_date": datetime.date(2021, 1, 15),  # 3rd Friday
        "strike": 130.0,
        "right": "CALL",
        "bid": 3.0,
        "ask": 3.5,
        "volume": 100,
        "close": 3.25,
        "open_interest": 500,
        "underlying_price": 135.0,
        "mid_quote": 3.25,
        "dte": 46,
        "moneyness": 130.0 / 135.0,
        "relative_spread": 0.5 / 3.25,
    }
    defaults.update(overrides)
    return pl.DataFrame([defaults])


def _multi_row_df(rows: list[dict]) -> pl.DataFrame:
    """Create a multi-row DataFrame from partial dicts, filling defaults."""
    defaults = {
        "symbol": "TEST",
        "date": datetime.date(2021, 1, 4),
        "expiration": "2021-01-15",
        "expiration_date": datetime.date(2021, 1, 15),
        "strike": 130.0,
        "right": "CALL",
        "bid": 3.0,
        "ask": 3.5,
        "volume": 100,
        "close": 3.25,
        "open_interest": 500,
        "underlying_price": 135.0,
        "mid_quote": 3.25,
        "dte": 46,
        "moneyness": 130.0 / 135.0,
        "relative_spread": 0.5 / 3.25,
    }
    full_rows = []
    for r in rows:
        row = defaults.copy()
        row.update(r)
        full_rows.append(row)
    return pl.DataFrame(full_rows)


# ---------------------------------------------------------------------------
# filter_integrity
# ---------------------------------------------------------------------------


class TestFilterIntegrity:
    def test_keeps_valid_row(self) -> None:
        df = _base_df()
        assert len(filter_integrity(df)) == 1

    def test_removes_zero_bid(self) -> None:
        df = _base_df(bid=0.0)
        assert len(filter_integrity(df)) == 0

    def test_removes_negative_bid(self) -> None:
        df = _base_df(bid=-1.0)
        assert len(filter_integrity(df)) == 0

    def test_removes_ask_equals_bid(self) -> None:
        df = _base_df(bid=3.0, ask=3.0)
        assert len(filter_integrity(df)) == 0

    def test_removes_ask_less_than_bid(self) -> None:
        df = _base_df(bid=3.5, ask=3.0)
        assert len(filter_integrity(df)) == 0

    def test_removes_null_underlying(self) -> None:
        df = _base_df(underlying_price=None)
        assert len(filter_integrity(df)) == 0


# ---------------------------------------------------------------------------
# filter_liquidity
# ---------------------------------------------------------------------------


class TestFilterLiquidity:
    def test_keeps_liquid_row(self) -> None:
        df = _base_df()
        assert len(filter_liquidity(df)) == 1

    def test_removes_low_mid_quote(self) -> None:
        df = _base_df(mid_quote=0.10)
        assert len(filter_liquidity(df)) == 0

    def test_keeps_exact_threshold_mid_quote(self) -> None:
        df = _base_df(mid_quote=0.125)
        assert len(filter_liquidity(df)) == 1

    def test_removes_wide_spread(self) -> None:
        df = _base_df(relative_spread=0.60)
        assert len(filter_liquidity(df)) == 0

    def test_keeps_exact_threshold_spread(self) -> None:
        df = _base_df(relative_spread=0.50)
        assert len(filter_liquidity(df)) == 1

    def test_removes_zero_oi(self) -> None:
        df = _base_df(open_interest=0)
        assert len(filter_liquidity(df)) == 0

    def test_removes_null_oi(self) -> None:
        df = _base_df(open_interest=None)
        assert len(filter_liquidity(df)) == 0


# ---------------------------------------------------------------------------
# filter_volume_7d
# ---------------------------------------------------------------------------


class TestFilterVolume7d:
    def test_keeps_row_with_volume(self) -> None:
        df = _base_df(volume=100)
        assert len(filter_volume_7d(df)) == 1

    def test_removes_zero_volume_contract(self) -> None:
        """A contract with zero volume on all days in the 7-day window is removed."""
        df = _multi_row_df([
            {"date": datetime.date(2021, 1, 4), "volume": 0},
            {"date": datetime.date(2021, 1, 5), "volume": 0},
            {"date": datetime.date(2021, 1, 6), "volume": 0},
        ])
        result = filter_volume_7d(df)
        assert len(result) == 0

    def test_keeps_contract_with_recent_volume(self) -> None:
        """A contract with zero today but volume 2 days ago survives."""
        df = _multi_row_df([
            {"date": datetime.date(2021, 1, 4), "volume": 50},
            {"date": datetime.date(2021, 1, 5), "volume": 0},
            {"date": datetime.date(2021, 1, 6), "volume": 0},
        ])
        result = filter_volume_7d(df)
        # All 3 rows survive because rolling 7d window includes the day with volume
        assert len(result) == 3

    def test_volume_outside_7d_window_does_not_count(self) -> None:
        """Volume from 8+ days ago doesn't help."""
        df = _multi_row_df([
            {"date": datetime.date(2021, 1, 4), "volume": 50},
            {"date": datetime.date(2021, 1, 12), "volume": 0},  # 8 days later
        ])
        result = filter_volume_7d(df)
        # Jan 4 row survives (it has volume), Jan 12 does not (8 days gap)
        assert len(result) == 1
        assert result["date"][0] == datetime.date(2021, 1, 4)


# ---------------------------------------------------------------------------
# filter_dte
# ---------------------------------------------------------------------------


class TestFilterDte:
    def test_keeps_above_threshold(self) -> None:
        df = _base_df(dte=30)
        assert len(filter_dte(df)) == 1

    def test_keeps_exact_threshold(self) -> None:
        df = _base_df(dte=14)
        assert len(filter_dte(df)) == 1

    def test_removes_below_threshold(self) -> None:
        df = _base_df(dte=13)
        assert len(filter_dte(df)) == 0

    def test_removes_zero_dte(self) -> None:
        df = _base_df(dte=0)
        assert len(filter_dte(df)) == 0

    def test_removes_negative_dte(self) -> None:
        df = _base_df(dte=-1)
        assert len(filter_dte(df)) == 0

    def test_custom_min_dte(self) -> None:
        df = _base_df(dte=20)
        assert len(filter_dte(df, min_dte=21)) == 0
        assert len(filter_dte(df, min_dte=20)) == 1


# ---------------------------------------------------------------------------
# filter_monthly
# ---------------------------------------------------------------------------


class TestFilterMonthly:
    def test_keeps_third_friday(self) -> None:
        # 2021-01-15 is a Friday and 15th -> 3rd Friday
        df = _base_df(expiration_date=datetime.date(2021, 1, 15))
        assert len(filter_monthly(df)) == 1

    def test_keeps_holiday_shifted_thursday(self) -> None:
        # Good Friday 2021: April 2, so monthly shifts to Thursday April 15
        # 2021-04-15 is Thursday, day 15 -> holiday-shifted 3rd Friday
        df = _base_df(expiration_date=datetime.date(2021, 4, 15))
        assert len(filter_monthly(df)) == 1

    def test_removes_weekly_first_friday(self) -> None:
        # 2021-01-08 is the 1st Friday of January
        df = _base_df(expiration_date=datetime.date(2021, 1, 8))
        assert len(filter_monthly(df)) == 0

    def test_removes_weekly_second_friday(self) -> None:
        # 2021-02-12 is the 2nd Friday of February (day 12)
        df = _base_df(expiration_date=datetime.date(2021, 2, 12))
        assert len(filter_monthly(df)) == 0

    def test_removes_weekly_fourth_friday(self) -> None:
        # 2021-01-22 is the 4th Friday of January (day 22)
        df = _base_df(expiration_date=datetime.date(2021, 1, 22))
        assert len(filter_monthly(df)) == 0

    def test_removes_monday_expiration(self) -> None:
        # Non-Friday/Thursday expirations are always filtered
        df = _base_df(expiration_date=datetime.date(2021, 1, 18))  # Monday
        assert len(filter_monthly(df)) == 0

    def test_third_friday_various_months(self) -> None:
        """Verify 3rd Friday detection across multiple months."""
        third_fridays_2021 = [
            datetime.date(2021, 1, 15),
            datetime.date(2021, 2, 19),
            datetime.date(2021, 3, 19),
            datetime.date(2021, 4, 16),
            datetime.date(2021, 5, 21),
            datetime.date(2021, 6, 18),
            datetime.date(2021, 7, 16),
            datetime.date(2021, 8, 20),
            datetime.date(2021, 9, 17),
            datetime.date(2021, 10, 15),
            datetime.date(2021, 11, 19),
            datetime.date(2021, 12, 17),
        ]
        for d in third_fridays_2021:
            df = _base_df(expiration_date=d)
            assert len(filter_monthly(df)) == 1, f"Failed for {d}"


# ---------------------------------------------------------------------------
# filter_all
# ---------------------------------------------------------------------------


class TestFilterAll:
    def test_chains_all_filters(self) -> None:
        """A fully valid row survives the entire chain."""
        df = _base_df()
        result = filter_all(df, verbose=False)
        assert len(result) == 1

    def test_removes_bad_row(self) -> None:
        """A row that fails integrity is removed."""
        df = _base_df(bid=0.0)
        result = filter_all(df, verbose=False)
        assert len(result) == 0

    def test_output_columns_unchanged(self) -> None:
        """Filter chain does not add or remove columns."""
        df = _base_df()
        result = filter_all(df, verbose=False)
        assert result.columns == df.columns


# ---------------------------------------------------------------------------
# filter_symbol
# ---------------------------------------------------------------------------


class TestFilterSymbol:
    def test_writes_output(self, tmp_path: Path) -> None:
        """Filtered output is written to disk."""
        # Create a valid joined parquet
        df = _base_df()
        joined_dir = tmp_path / "joined"
        joined_dir.mkdir()
        df.write_parquet(joined_dir / "TEST.parquet")

        output_dir = tmp_path / "output"
        rows = filter_symbol("TEST", joined_dir, output_dir, verbose=False)
        assert rows == 1
        assert (output_dir / "TEST.parquet").exists()

    def test_missing_file_returns_zero(self, tmp_path: Path) -> None:
        rows = filter_symbol(
            "MISSING", tmp_path / "joined", tmp_path / "output", verbose=False
        )
        assert rows == 0
