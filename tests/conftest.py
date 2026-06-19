"""Shared test fixtures for the theta pipeline test suite."""

import pytest
from pathlib import Path


@pytest.fixture
def sample_toml_content() -> str:
    """Return a minimal valid TOML string for testing."""
    return """\
[symbols]
universe = ["SPY", "QQQ", "AAPL"]

[api]
base_url = "http://127.0.0.1:25503/v3"
concurrency = 2

[dates]
start_date = "2020-01-01"
end_date = "2024-12-31"
"""


@pytest.fixture
def sample_config_path(tmp_path: Path, sample_toml_content: str) -> Path:
    """Write sample TOML to a temp file and return the path."""
    config_file = tmp_path / "test_pipeline.toml"
    config_file.write_text(sample_toml_content, encoding="utf-8")
    return config_file


@pytest.fixture
def default_config():
    """Return an AppConfig with all defaults."""
    from theta.config import AppConfig
    return AppConfig()


@pytest.fixture
def mock_ndjson_response() -> str:
    """Return a multi-line ndjson string with realistic ThetaData fields."""
    return (
        '{"ms_of_day":56700000,"bid":1.5,"ask":1.6,"bid_size":10,"ask_size":20}\n'
        '{"ms_of_day":56700001,"bid":2.0,"ask":2.1,"bid_size":15,"ask_size":25}\n'
        '{"ms_of_day":56700002,"bid":3.5,"ask":3.7,"bid_size":5,"ask_size":10}\n'
    )


@pytest.fixture
def empty_response() -> str:
    """Return an empty string simulating an empty API response."""
    return ""


# ---------------------------------------------------------------------------
# Endpoint-specific mock DataFrames
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_quote_df():
    """Return a realistic polars DataFrame matching at_time/quote response columns."""
    import polars as pl

    return pl.DataFrame({
        "symbol": ["AAPL", "AAPL"],
        "expiration": ["2024-01-19", "2024-01-19"],
        "strike": [190000, 195000],
        "right": ["C", "P"],
        "timestamp": [1705600000000, 1705600000000],
        "bid_size": [10, 15],
        "bid_exchange": ["CBOE", "CBOE"],
        "bid": [5.20, 2.10],
        "bid_condition": [0, 0],
        "ask_size": [20, 25],
        "ask_exchange": ["CBOE", "CBOE"],
        "ask": [5.40, 2.30],
        "ask_condition": [0, 0],
    })


@pytest.fixture
def mock_eod_df():
    """Return a realistic polars DataFrame matching history/eod response columns."""
    import polars as pl

    return pl.DataFrame({
        "symbol": ["AAPL", "AAPL"],
        "expiration": ["2024-01-19", "2024-01-19"],
        "strike": [190000, 195000],
        "right": ["C", "P"],
        "created": ["2024-01-02", "2024-01-02"],
        "last_trade": ["2024-01-02", "2024-01-02"],
        "open": [5.00, 2.00],
        "high": [5.50, 2.50],
        "low": [4.80, 1.90],
        "close": [5.20, 2.10],
        "volume": [1000, 500],
        "count": [50, 25],
        "bid_size": [10, 15],
        "bid_exchange": ["CBOE", "CBOE"],
        "bid": [5.20, 2.10],
        "bid_condition": [0, 0],
        "ask_size": [20, 25],
        "ask_exchange": ["CBOE", "CBOE"],
        "ask": [5.40, 2.30],
        "ask_condition": [0, 0],
    })


@pytest.fixture
def mock_oi_df():
    """Return a realistic polars DataFrame matching history/open_interest columns."""
    import polars as pl

    return pl.DataFrame({
        "symbol": ["AAPL", "AAPL"],
        "expiration": ["2024-01-19", "2024-01-19"],
        "strike": [190000, 195000],
        "right": ["C", "P"],
        "timestamp": [1704153600000, 1704153600000],
        "open_interest": [5000, 3000],
    })
