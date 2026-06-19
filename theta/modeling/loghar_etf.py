"""LogHAR walk-forward forecasts for the 5 basket ETFs.

Generates pooled LogHAR predictions on SPY/QQQ/IWM/GLD/TLT using the same
12 quarterly windows as the equity walk-forward (window_schedule.json).
Produces etf_predictions.parquet with the same schema as all_predictions.parquet
so the CSP backtest can consume it via `bt_data.load_wf_predictions(...)`.

ETFs are fit pooled (5 symbols stacked) because per-symbol fits on ~750 daily
rows are noisy; pooling matches the literature treatment of ETF RV forecasting.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from theta.modeling.baselines import predict_loghar
from theta.modeling.preprocessing import add_log_target
from theta.modeling.walk_forward import generate_windows

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PANEL_PATH = _PROJECT_ROOT / "data" / "processed" / "panel.parquet"
WF_DIR = _PROJECT_ROOT / "data" / "processed" / "evaluation" / "walk_forward"
OUTPUT_PATH = WF_DIR / "etf_predictions.parquet"

ETFS = ["SPY", "QQQ", "IWM", "GLD", "TLT"]


def run_etf_loghar() -> pl.DataFrame:
    panel = pl.read_parquet(PANEL_PATH)
    panel = add_log_target(panel)
    etf_panel = panel.filter(pl.col("symbol").is_in(ETFS))
    print(f"ETF panel: {len(etf_panel):,} rows across {etf_panel['symbol'].n_unique()} symbols")

    # Reuse the existing schedule so test windows align with the equity WF run.
    with open(WF_DIR / "window_schedule.json") as f:
        schedule = json.load(f)

    parts: list[pl.DataFrame] = []
    for w in schedule:
        wid = w["window_id"]
        train_end = w["train_end"]
        test_start = w["test_start"]
        test_end = w["test_end"]

        train_df = etf_panel.filter(pl.col("date") <= pl.lit(train_end).str.to_date())
        test_df = etf_panel.filter(
            (pl.col("date") >= pl.lit(test_start).str.to_date())
            & (pl.col("date") <= pl.lit(test_end).str.to_date())
        )
        if len(train_df) < 100 or len(test_df) == 0:
            print(f"  window {wid}: insufficient data, skip")
            continue

        preds = predict_loghar(train_df, test_df)
        preds = preds.with_columns(pl.lit(wid).alias("window_id"))
        parts.append(preds)
        print(
            f"  window {wid}: test {test_start}..{test_end} "
            f"train={len(train_df):,} test={len(test_df):,} preds={len(preds):,}"
        )

    out = pl.concat(parts)
    out.write_parquet(OUTPUT_PATH)
    print(f"\nWrote {OUTPUT_PATH} ({len(out):,} rows)")
    return out


if __name__ == "__main__":
    run_etf_loghar()
