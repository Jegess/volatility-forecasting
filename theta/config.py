"""Configuration loading and validation for the Theta data pipeline.

Loads pipeline settings from a TOML file and validates them using
pydantic models. Provides sensible defaults for all fields.
"""

import tomllib
from pathlib import Path

from pydantic import BaseModel, Field


class SymbolsConfig(BaseModel):
    """Configuration for the symbol universe."""

    universe: list[str] = Field(
        default=[
            # Benchmarks / ETFs (5)
            "SPY", "QQQ", "IWM", "GLD", "TLT",
            # Mega-cap tech (7)
            "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA",
            # Financials (10)
            "JPM", "V", "MA", "GS", "MS", "C", "BAC", "WFC", "SCHW", "AXP",
            # Healthcare (16)
            "UNH", "JNJ", "ABBV", "MRK", "LLY", "PFE", "BMY", "TMO",
            "MRNA", "GILD", "AMGN", "ISRG", "REGN", "VRTX",
            "CI", "MCK",
            # Healthcare devices (3)
            "SYK", "BSX", "DXCM",
            # Consumer (16)
            "HD", "PG", "COST", "WMT", "KO", "PEP", "MCD", "NKE", "SBUX",
            "TGT", "CMG", "ABNB", "LULU", "MDLZ", "CL", "MNST",
            # Industrials (12)
            "CAT", "HON", "DE", "RTX", "LMT", "GE", "BA",
            "FDX", "UPS", "WM", "ETN", "ADP",
            # Tech / Software (19)
            "ACN", "ADBE", "CRM", "NFLX", "ORCL", "NOW",
            "CRWD", "PANW", "SHOP", "SNOW", "DDOG",
            "ZS", "FTNT", "WDAY", "TEAM",
            "DELL", "ZM", "PINS", "TWLO",
            # Semiconductors (13)
            "AVGO", "AMD", "INTC", "QCOM", "TXN",
            "AMAT", "LRCX", "MU", "KLAC", "MRVL", "ON", "NXPI", "FSLR",
            # Energy (9)
            "XOM", "CVX", "SLB", "OXY", "NEE", "COP", "EOG", "MPC", "PSX",
            # Telecom / Media (4)
            "T", "VZ", "CMCSA", "DIS",
            # Growth / Speculative (17)
            "PLTR", "COIN", "UBER", "SQ", "NET", "SNAP", "ROKU",
            "DASH", "SPOT", "HOOD", "SOFI", "AFRM", "RBLX", "TTD",
            "MSTR", "ENPH", "PATH",
            # Other (16)
            "PYPL", "LOW", "CSCO", "IBM",
            "MARA", "RIVN", "NIO", "BABA", "SMCI",
            "BX", "KKR", "ALLY",
            "GM", "F", "SE", "HPQ",
        ],
        description="List of underlying symbols to download",
    )


class ApiConfig(BaseModel):
    """Configuration for the ThetaData API connection."""

    base_url: str = "http://127.0.0.1:25503/v3"
    concurrency: int = Field(default=4, ge=1, le=8)
    timeout_default: float = Field(
        default=60.0, description="Default timeout in seconds"
    )
    timeout_large: float = Field(
        default=300.0, description="Timeout for large requests (SPY/QQQ)"
    )
    large_symbols: list[str] = Field(
        default=["SPY", "QQQ", "IWM"],
        description="Symbols that need chunked requests and longer timeouts",
    )


class DateRange(BaseModel):
    """Configuration for the date range to download."""

    start_date: str = "2020-01-01"
    end_date: str = "2026-03-09"


class AppConfig(BaseModel):
    """Top-level application configuration."""

    symbols: SymbolsConfig = SymbolsConfig()
    api: ApiConfig = ApiConfig()
    dates: DateRange = DateRange()


def load_config(path: Path = Path("pipeline.toml")) -> AppConfig:
    """Load and validate pipeline configuration from a TOML file.

    Args:
        path: Path to the TOML configuration file.
              Defaults to 'pipeline.toml' in the current directory.

    Returns:
        Validated AppConfig instance.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    with open(path, "rb") as f:
        data = tomllib.load(f)
    return AppConfig(**data)
