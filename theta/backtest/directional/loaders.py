"""Data loaders for the directional backtest.

Thin layer over theta.backtest.data. Adds underlying-close loading (the
put-spread backtest only needed point-in-time spot, this one needs
daily close series for P&L) and the signals frame used by ranking.
"""
from __future__ import annotations

import polars as pl

from theta.backtest import data as bt_data

RAW_UNDERLYING_DIR = bt_data.ROOT / "raw" / "underlying"


def load_underlying_closes(symbols: list[str]) -> pl.DataFrame:
    """Long-format (symbol, date, close) for every requested symbol.

    Source: data/raw/underlying/{symbol}.parquet with columns
    (symbol, date, underlying_price). Renamed to `close` for portfolio
    code that treats any price series uniformly.
    """
    frames = []
    for sym in symbols:
        path = RAW_UNDERLYING_DIR / f"{sym}.parquet"
        if not path.exists():
            continue
        frames.append(
            pl.read_parquet(path)
            .select("symbol", "date", pl.col("underlying_price").alias("close"))
        )
    return pl.concat(frames).sort(["symbol", "date"])


def load_spy_closes() -> pl.DataFrame:
    """SPY daily closes for the benchmark portfolio."""
    return load_underlying_closes(["SPY"])


def load_signals() -> pl.DataFrame:
    """Per-(symbol, date) frame with y_pred and atm_iv, 188-equity universe.

    Returns columns: symbol, date, y_pred, atm_iv, window_id.
    Uses LightGBM walk-forward OOS predictions. ETFs excluded.
    """
    symbols = bt_data.list_symbols(include_etfs=False)

    preds = (
        bt_data.load_wf_predictions("LightGBM")
        .filter(pl.col("symbol").is_in(symbols))
        .select("symbol", "date", "y_pred", "window_id")
    )

    feat_frames = [
        bt_data.load_features(sym).select("symbol", "date", "atm_iv")
        for sym in symbols
    ]
    feats = pl.concat(feat_frames)

    return (
        preds
        .join(feats, on=["symbol", "date"], how="inner")
        .sort(["date", "symbol"])
    )
