"""Data loaders for the backtest.

Single source of truth for paths and schemas. Every other module asks this one
for data and never touches disk directly. Centralizing this makes look-ahead
bugs harder to introduce — there's exactly one place that filters by date.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl

# ----- paths --------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2] / "data"
WF_PREDS = ROOT / "processed" / "evaluation" / "walk_forward" / "all_predictions.parquet"
OPTIONS_IV_DIR = ROOT / "processed" / "options_iv"
RAW_EOD_DIR = ROOT / "raw" / "eod"
FEATURES_DIR = ROOT / "processed" / "features"
MACRO = ROOT / "processed" / "macro" / "macro.parquet"

OUTPUT_DIR = ROOT / "processed" / "backtest"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ----- universe ----------------------------------------------------------

# Excluded from the backtest: basket ETFs have damped single-stock dynamics,
# overlap each other in holdings, and their put VRP is a known hedging-demand
# effect (Bollen & Whaley 2004) unrelated to this study's single-equity thesis.
# Literature (Bali et al., Cremers & Weinbaum, Gu/Kelly/Xiu) uses equities only.
EXCLUDED_ETFS = {"SPY", "QQQ", "IWM", "GLD", "TLT"}


def list_symbols(include_etfs: bool = False) -> list[str]:
    """Backtest universe. Default excludes the 5 basket ETFs (188 equities).

    Pass `include_etfs=True` to get the full 193-symbol panel — useful for
    ex-post robustness slicing, not for the primary backtest.
    """
    all_syms = sorted(p.stem for p in OPTIONS_IV_DIR.glob("*.parquet"))
    if include_etfs:
        return all_syms
    return [s for s in all_syms if s not in EXCLUDED_ETFS]


# ----- loaders -----------------------------------------------------------

def load_wf_predictions(model: str = "LightGBM") -> pl.DataFrame:
    """Walk-forward out-of-sample predictions for the given model."""
    df = pl.read_parquet(WF_PREDS).filter(pl.col("model") == model)
    return df.select("symbol", "date", "y_true", "y_pred", "window_id")


def load_features(symbol: str) -> pl.DataFrame:
    """Per-symbol feature panel (atm_iv, vrp, days_to_earnings, ...)."""
    return pl.read_parquet(FEATURES_DIR / f"{symbol}.parquet")


def load_options_iv(symbol: str) -> pl.DataFrame:
    """Filtered/IV-enriched option chain for a symbol."""
    return pl.read_parquet(OPTIONS_IV_DIR / f"{symbol}.parquet")


def load_raw_eod(symbol: str) -> pl.DataFrame:
    """Raw EOD chain (no filters). Use only as fallback when a tracked
    contract drops out of options_iv (delta moved outside the trade window).
    """
    return pl.read_parquet(RAW_EOD_DIR / f"{symbol}.parquet")


def load_macro() -> pl.DataFrame:
    """Daily macro panel (vix, vvix, term_spread, ...)."""
    return pl.read_parquet(MACRO)


# ----- assembled cross-section ------------------------------------------

def daily_signals(model: str = "LightGBM",
                  include_etfs: bool = False) -> pl.DataFrame:
    """Per-(symbol, date) frame with everything needed for the entry decision.

    Columns:
        symbol, date, y_pred (RV forecast), y_true (realized RV — only used
        for ex-post diagnostics, NEVER for trade decisions), atm_iv,
        days_to_earnings, is_fomc_week, vix.

    By default excludes the 5 basket ETFs — see `list_symbols()`. Pass
    `include_etfs=True` for the full 193-symbol panel.

    Joining all of this once is fine because every column is observable
    at end-of-day on its own date. The look-ahead guarantee comes from
    downstream code never peeking at rows with `date > current_date`.
    """
    symbols = list_symbols(include_etfs=include_etfs)

    preds = (
        load_wf_predictions(model)
        .filter(pl.col("symbol").is_in(symbols))
    )

    feat_frames = [
        load_features(sym).select(
            "symbol", "date", "atm_iv", "days_to_earnings", "is_fomc_week",
            "is_earnings_week",
        )
        for sym in symbols
    ]
    feats = pl.concat(feat_frames)

    macro = load_macro().select("date", "vix")

    return (
        preds
        .join(feats, on=["symbol", "date"], how="inner")
        .join(macro, on="date", how="inner")
        .sort(["date", "symbol"])
    )


# ----- contract lookup ---------------------------------------------------

def get_chain_on(symbol: str, on_date: date, right: str = "PUT") -> pl.DataFrame:
    """All option contracts for symbol/date/right — used at entry time."""
    df = load_options_iv(symbol)
    return df.filter((pl.col("date") == on_date) & (pl.col("right") == right))


def get_quote(symbol: str, on_date: date, strike: float,
              expiration_date: date, right: str = "PUT",
              fallback_to_raw: bool = True) -> dict | None:
    """Bid/ask/mid for a specific contract on a specific date.

    Returns None if the contract has no quote anywhere on that date
    (will happen if it expired or simply had no market).

    Tries options_iv first; if missing (because delta moved outside the
    -0.05 to -0.50 filter window) and `fallback_to_raw=True`, looks in
    the unfiltered raw EOD data.
    """
    iv = load_options_iv(symbol).filter(
        (pl.col("date") == on_date)
        & (pl.col("strike") == strike)
        & (pl.col("expiration_date") == expiration_date)
        & (pl.col("right") == right)
    )
    if iv.height > 0:
        row = iv.row(0, named=True)
        return {
            "bid": row["bid"], "ask": row["ask"], "mid_quote": row["mid_quote"],
            "underlying_price": row["underlying_price"], "delta": row["delta"],
            "dte": row["dte"], "source": "options_iv",
        }

    if not fallback_to_raw:
        return None

    # `created` is an ISO timestamp like "2021-01-04T18:00:17.048" — match on the
    # date prefix. `expiration` is also ISO "YYYY-MM-DD". Multiple snapshots
    # may exist on the same date (different ingest times); take the first.
    raw = load_raw_eod(symbol).filter(
        pl.col("created").str.starts_with(str(on_date))
        & (pl.col("strike") == strike)
        & (pl.col("expiration") == expiration_date.isoformat())
        & (pl.col("right") == right)
    )
    if raw.height == 0:
        return None
    row = raw.row(0, named=True)
    # Raw EOD has no underlying_price column, but any options_iv row on
    # this date carries the spot — pull it if available so the breach
    # check still works during the DTE < 14 tail.
    spot = get_underlying_price(symbol, on_date)
    return {
        "bid": row["bid"], "ask": row["ask"],
        "mid_quote": (row["bid"] + row["ask"]) / 2.0,
        "underlying_price": spot, "delta": None,
        "dte": (expiration_date - on_date).days,
        "source": "raw_eod",
    }


def get_underlying_price(symbol: str, on_date: date) -> float | None:
    """Closing underlying price on a given date. Pulled from options_iv
    (every row has it) — pick any contract for that date."""
    df = load_options_iv(symbol).filter(pl.col("date") == on_date)
    if df.height == 0:
        return None
    return df["underlying_price"][0]
