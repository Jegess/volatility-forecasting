"""Predict 21-day-forward realized volatility for any stock.

Two models:
  * loghar    - needs only a daily close series (any ticker, free).
  * lightgbm  - lean 9-feature model: prices + one ATM implied-vol number
                (+ optionally the next earnings date).

The script assembles the features it needs from data you supply or that it
pulls from Yahoo Finance. Getting the data is your job; this just runs the
trained model on it.

Examples
--------
    python -m prediction.predict --symbol AAPL --model loghar
    python -m prediction.predict --symbol AAPL --model loghar --prices aapl.csv
    python -m prediction.predict --symbol AAPL --model lightgbm --atm-iv 0.28 \
        --next-earnings 2026-04-30

Notes
-----
  * A daily price history of at least ~22 trading days is required (the
    longest feature window is the 22-day rv_m). Use a few months to be safe.
  * --atm-iv is a single annualized ATM implied-vol as a decimal (0.28 = 28%).
  * days_to_fomc comes from a built-in FOMC calendar (good through 2026).
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
import sys

import numpy as np
import polars as pl

from prediction import LEAN_FEATURES, MODELS_DIR
from prediction.loghar import load_loghar, predict_loghar
from theta.processing.events import FOMC_DATES
from theta.processing.rv import compute_rv_for_symbol

MIN_ROWS = 22  # longest RV window (rv_m / rq / semivariances)


# ── Data loading ────────────────────────────────────────────────────────

def load_prices_csv(path: str) -> pl.DataFrame:
    """Read a CSV with columns 'date' and 'close' into the price schema."""
    df = pl.read_csv(path, try_parse_dates=True)
    cols = {c.lower(): c for c in df.columns}
    if "date" not in cols or "close" not in cols:
        raise SystemExit(f"--prices CSV needs 'date' and 'close' columns; got {df.columns}")
    df = df.rename({cols["date"]: "date", cols["close"]: "underlying_price"})
    if df["date"].dtype != pl.Date:
        df = df.with_columns(pl.col("date").str.to_date())
    return df.select("date", pl.col("underlying_price").cast(pl.Float64)).drop_nulls().sort("date")


def fetch_prices_yahoo(symbol: str, period: str = "1y") -> pl.DataFrame:
    """Download daily closes from Yahoo Finance into the price schema."""
    try:
        import yfinance as yf
    except ImportError:
        raise SystemExit("yfinance not installed; pass --prices <file.csv> instead.")

    data = yf.download(symbol, period=period, interval="1d",
                       auto_adjust=True, progress=False)
    if data is None or len(data) == 0:
        raise SystemExit(f"No Yahoo price data for '{symbol}'. Check the ticker or pass --prices.")
    close = data["Close"]
    if hasattr(close, "columns"):          # MultiIndex single-ticker frame
        close = close.iloc[:, 0]
    return pl.DataFrame({
        "date": [d.date() for d in close.index],
        "underlying_price": close.to_numpy().astype("float64"),
    }).drop_nulls().sort("date")


# ── Event features ──────────────────────────────────────────────────────

def days_to_next_fomc(as_of: dt.date) -> int | None:
    future = [d for d in FOMC_DATES if d >= as_of]
    return (min(future) - as_of).days if future else None


def resolve_next_earnings(symbol: str, as_of: dt.date, supplied: str | None) -> int | None:
    """Return calendar days to next earnings, or None if unknown."""
    if supplied:
        days = (dt.date.fromisoformat(supplied) - as_of).days
        return days if days >= 0 else None  # past date -> treat as unknown
    try:
        import yfinance as yf
        eds = yf.Ticker(symbol).get_earnings_dates(limit=16)
        future = sorted(d.date() for d in eds.index if d.date() >= as_of)
        if future:
            return (future[0] - as_of).days
    except Exception:
        pass
    return None


# ── Core ────────────────────────────────────────────────────────────────

def build_rv_row(price_df: pl.DataFrame, symbol: str, as_of: dt.date | None) -> dict:
    """Compute RV features and return the as-of row as a dict of floats."""
    if len(price_df) < MIN_ROWS:
        raise SystemExit(
            f"Only {len(price_df)} price rows; need at least {MIN_ROWS} "
            f"trading days (use a few months of history)."
        )
    rv = compute_rv_for_symbol(price_df, symbol, truncate_warmup=False)

    if as_of is not None:
        row = rv.filter(pl.col("date") == as_of)
        if len(row) == 0:
            raise SystemExit(f"--as-of {as_of} not found in the price series.")
        row = row.row(0, named=True)
    else:
        row = rv.row(-1, named=True)  # latest date

    needed = ["rv_d", "rv_w", "rv_m", "rq", "rs_pos", "rs_neg"]
    if any(row[f] is None or not math.isfinite(row[f]) for f in needed):
        raise SystemExit(
            f"Insufficient history before {row['date']} to compute RV features "
            f"(need ~{MIN_ROWS} prior trading days)."
        )
    return row


def report(symbol, as_of, model, rv_pred, inputs):
    vol = math.sqrt(max(rv_pred, 0.0))
    print(f"\n  Symbol            : {symbol}")
    print(f"  As-of date        : {as_of}")
    print(f"  Model             : {model}")
    print(f"  Horizon           : 21 trading days forward")
    print(f"  Forecast RV       : {rv_pred:.4f}   (annualized variance)")
    print(f"  Implied annual vol: {vol:.4f}   ({vol * 100:.1f}%)")
    if inputs:
        print(f"  Inputs used       : {inputs}")
    print()


def run(args: argparse.Namespace) -> None:
    as_of = dt.date.fromisoformat(args.as_of) if args.as_of else None

    # 1. Prices -> RV features
    price_df = load_prices_csv(args.prices) if args.prices else fetch_prices_yahoo(args.symbol)
    row = build_rv_row(price_df, args.symbol, as_of)
    as_of = row["date"]

    if args.model == "loghar":
        coefs = load_loghar(MODELS_DIR / "loghar_coefs.json")
        rv_pred = predict_loghar(coefs, row["rv_d"], row["rv_w"], row["rv_m"])
        report(args.symbol, as_of, "LogHAR (price-only)", rv_pred,
               f"rv_d={row['rv_d']:.4f}, rv_w={row['rv_w']:.4f}, rv_m={row['rv_m']:.4f}")
        return

    # lightgbm: needs atm_iv + event features
    if args.atm_iv is None:
        raise SystemExit("--atm-iv is required for --model lightgbm "
                         "(annualized ATM implied vol as a decimal, e.g. 0.28).")

    d_fomc = days_to_next_fomc(as_of)
    if d_fomc is None:
        print(f"  [warn] {as_of} is beyond the built-in FOMC calendar; using neutral 21.",
              file=sys.stderr)
        d_fomc = 21
    d_earn = resolve_next_earnings(args.symbol, as_of, args.next_earnings)
    if d_earn is None:
        print("  [warn] next earnings date unknown; using neutral 45. "
              "Pass --next-earnings YYYY-MM-DD for accuracy.", file=sys.stderr)
        d_earn = 45

    feat = {
        "rv_d": row["rv_d"], "rv_w": row["rv_w"], "rv_m": row["rv_m"],
        "rq": row["rq"], "rs_pos": row["rs_pos"], "rs_neg": row["rs_neg"],
        "atm_iv": float(args.atm_iv),
        "days_to_earnings": float(d_earn), "days_to_fomc": float(d_fomc),
    }
    import lightgbm as lgb
    booster = lgb.Booster(model_file=str(MODELS_DIR / "lgbm_lean.txt"))
    X = np.array([[feat[c] for c in LEAN_FEATURES]], dtype=np.float64)
    rv_pred = float(np.clip(np.exp(booster.predict(X))[0], 1e-8, None))
    report(args.symbol, as_of, "LightGBM (lean, 9-feature)", rv_pred,
           f"atm_iv={args.atm_iv}, days_to_earnings={d_earn}, days_to_fomc={d_fomc}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Predict 21-day-forward realized volatility for a stock.")
    p.add_argument("--symbol", required=True, help="Ticker, e.g. AAPL")
    p.add_argument("--model", choices=["loghar", "lightgbm"], default="loghar")
    p.add_argument("--prices", help="CSV with 'date','close' columns (else Yahoo)")
    p.add_argument("--atm-iv", type=float, dest="atm_iv",
                   help="Annualized ATM implied vol, decimal (lightgbm only)")
    p.add_argument("--next-earnings", dest="next_earnings",
                   help="Next earnings date YYYY-MM-DD (lightgbm; else Yahoo)")
    p.add_argument("--as-of", dest="as_of",
                   help="Prediction date YYYY-MM-DD (default: latest price date)")
    run(p.parse_args())


if __name__ == "__main__":
    main()
