"""Level 1: VRP signal quality.

Does `VRP > 0` predict that realized variance came in below implied?
And does the *ranking* add information on top of the binary signal?

For every (symbol, date) we compute three VRPs:
    VRP_lgbm   = atm_iv^2 - y_pred(LightGBM)     [the trading signal]
    VRP_loghar = atm_iv^2 - y_pred(LogHAR)       [sanity-check model]
    VRP_actual = atm_iv^2 - y_true                [ex-post truth]

A predicted signal "hits" when VRP_pred > 0 AND VRP_actual > 0 — we wouldn't
trade on VRP ≤ 0, so accuracy is defined conditional on a positive signal.

Plan thresholds (BACKTEST_PLAN.md §Level 1 metrics):
    Overall accuracy > 60% (gate at 55%)
    Accuracy by quintile: monotonically increasing
    Accuracy by daily rank: top-10 > top-50 > all
    LightGBM > LogHAR
    Calibration within 20%
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from theta.backtest import data as bt_data

OUTPUT_FILE = bt_data.OUTPUT_DIR / "vrp_signal_quality.parquet"


# ----- frame assembly ---------------------------------------------------

def build_level1_frame(include_etfs: bool = False) -> pl.DataFrame:
    """Per-(symbol, date) frame with both models' VRP, actual VRP, daily
    rank, quintile. Joined on common (symbol, date) pairs so the two
    models are evaluated over the same universe.
    """
    symbols = bt_data.list_symbols(include_etfs=include_etfs)

    preds = (
        pl.read_parquet(bt_data.WF_PREDS)
        .filter(pl.col("symbol").is_in(symbols))
        .filter(pl.col("model").is_in(["LightGBM", "LogHAR"]))
        .select("symbol", "date", "model", "y_pred", "y_true")
    )
    # Pivot to one row per (symbol, date) with both models' forecasts.
    lgbm = (
        preds.filter(pl.col("model") == "LightGBM")
        .select("symbol", "date", "y_true",
                pl.col("y_pred").alias("y_pred_lgbm"))
    )
    lh = (
        preds.filter(pl.col("model") == "LogHAR")
        .select("symbol", "date", pl.col("y_pred").alias("y_pred_loghar"))
    )
    paired = lgbm.join(lh, on=["symbol", "date"], how="inner")

    # Attach atm_iv from per-symbol features.
    feat_frames = [
        bt_data.load_features(s).select("symbol", "date", "atm_iv")
        for s in symbols
    ]
    feats = pl.concat(feat_frames)

    df = paired.join(feats, on=["symbol", "date"], how="inner").with_columns(
        (pl.col("atm_iv") ** 2 - pl.col("y_pred_lgbm")).alias("vrp_lgbm"),
        (pl.col("atm_iv") ** 2 - pl.col("y_pred_loghar")).alias("vrp_loghar"),
        (pl.col("atm_iv") ** 2 - pl.col("y_true")).alias("vrp_actual"),
    ).with_columns(
        (pl.col("vrp_actual") > 0).alias("was_overpriced"),
    )

    # Daily rank + quintile, using LightGBM VRP as the trading signal.
    df = df.sort(["date", "vrp_lgbm"], descending=[False, True]).with_columns(
        pl.col("vrp_lgbm").rank("ordinal", descending=True)
        .over("date").alias("rank_lgbm"),
        pl.col("vrp_loghar").rank("ordinal", descending=True)
        .over("date").alias("rank_loghar"),
    )
    # Quintile 1 = top 20% by VRP on that date.
    df = df.with_columns(
        pl.col("vrp_lgbm").qcut(5, labels=["Q5", "Q4", "Q3", "Q2", "Q1"])
        .over("date").alias("quintile_lgbm"),
    )
    return df


# ----- metrics ----------------------------------------------------------

def vrp_accuracy(df: pl.DataFrame, vrp_col: str = "vrp_lgbm") -> dict:
    """Overall accuracy on positive-signal rows: what % were right?"""
    positive = df.filter(pl.col(vrp_col) > 0)
    n_pos = positive.height
    n_correct = positive.filter(pl.col("was_overpriced")).height
    return {
        "n_positive_signals": n_pos,
        "n_correct": n_correct,
        "accuracy": float(n_correct / n_pos) if n_pos else 0.0,
    }


def accuracy_by_quintile(df: pl.DataFrame,
                         vrp_col: str = "vrp_lgbm") -> pl.DataFrame:
    """Hit rate per VRP quintile. Q1 = highest VRP (the trades we'd actually
    take). Monotonically increasing Q5→Q1 would confirm the ranking carries
    information beyond the binary sign.
    """
    return (
        df
        .filter(pl.col(vrp_col) > 0)
        .group_by("quintile_lgbm")
        .agg(
            pl.len().alias("n"),
            pl.col("was_overpriced").mean().alias("accuracy"),
            pl.col(vrp_col).mean().alias("mean_vrp"),
        )
        .sort("quintile_lgbm")
    )


def accuracy_by_rank(df: pl.DataFrame,
                     buckets: list[int] = [5, 10, 20, 50, 100]) -> pl.DataFrame:
    """Accuracy restricted to daily top-N — answers 'does rank matter?'.

    Only the top-N candidates per day contribute. Lower N = more selective.
    If top-5 accuracy > top-20 accuracy > overall accuracy, the ranking
    carries information that binary VRP > 0 does not.
    """
    rows = []
    # "all positive" reference
    all_pos = df.filter(pl.col("vrp_lgbm") > 0)
    rows.append({
        "bucket": "all_positive",
        "n": all_pos.height,
        "accuracy": float(all_pos["was_overpriced"].mean() or 0.0),
    })
    for n in buckets:
        top = df.filter((pl.col("rank_lgbm") <= n) & (pl.col("vrp_lgbm") > 0))
        rows.append({
            "bucket": f"top_{n}",
            "n": top.height,
            "accuracy": float(top["was_overpriced"].mean() or 0.0),
        })
    return pl.DataFrame(rows)


def model_comparison(df: pl.DataFrame) -> pl.DataFrame:
    """Side-by-side accuracy for LightGBM vs LogHAR, both on positive signals.
    Evaluated over the same (symbol, date) universe (df is already joined).
    """
    return pl.DataFrame([
        {"model": "LightGBM", **vrp_accuracy(df, "vrp_lgbm")},
        {"model": "LogHAR",   **vrp_accuracy(df, "vrp_loghar")},
    ])


def calibration(df: pl.DataFrame,
                vrp_col: str = "vrp_lgbm") -> dict:
    """Mean predicted vs mean actual VRP on positive-signal rows. Ratio
    within 20% of 1.0 is the plan's calibration target.
    """
    pos = df.filter(pl.col(vrp_col) > 0)
    if pos.height == 0:
        return {"n": 0, "mean_predicted": 0.0, "mean_actual": 0.0, "ratio": 0.0}
    mean_pred = float(pos[vrp_col].mean())
    mean_actual = float(pos["vrp_actual"].mean())
    return {
        "n": pos.height,
        "mean_predicted": mean_pred,
        "mean_actual": mean_actual,
        "ratio": mean_actual / mean_pred if mean_pred != 0 else 0.0,
    }


def rank_stability(df: pl.DataFrame, top_n: int = 10,
                   lag_days: int = 5) -> dict:
    """Jaccard overlap of daily top-N sets between t and t+lag. Too-high
    overlap (>0.7) = same symbols always win, low diversity. Plan wants
    rotation to be healthy.
    """
    dates = sorted(df["date"].unique().to_list())
    top_sets = {}
    for d in dates:
        day = df.filter((pl.col("date") == d) & (pl.col("rank_lgbm") <= top_n))
        top_sets[d] = set(day["symbol"].to_list())

    overlaps = []
    for i, d in enumerate(dates):
        if i + lag_days >= len(dates):
            break
        d2 = dates[i + lag_days]
        a, b = top_sets[d], top_sets[d2]
        union = a | b
        if union:
            overlaps.append(len(a & b) / len(union))
    if not overlaps:
        return {"lag_days": lag_days, "mean_overlap": 0.0, "n_pairs": 0}
    return {
        "lag_days": lag_days,
        "mean_overlap": sum(overlaps) / len(overlaps),
        "n_pairs": len(overlaps),
    }


# ----- orchestrator -----------------------------------------------------

def run_level1(include_etfs: bool = False,
               save: bool = True) -> dict[str, Any]:
    """Full Level 1 report. Returns summary dict; writes the per-row frame
    to vrp_signal_quality.parquet and the summary dict is caller's problem
    (print, save as JSON, inspect in notebook).
    """
    df = build_level1_frame(include_etfs=include_etfs)

    summary: dict[str, Any] = {
        "universe_size": df["symbol"].n_unique(),
        "n_days": df["date"].n_unique(),
        "n_rows": df.height,
        "overall": vrp_accuracy(df, "vrp_lgbm"),
        "by_quintile": accuracy_by_quintile(df).to_dicts(),
        "by_rank": accuracy_by_rank(df).to_dicts(),
        "model_comparison": model_comparison(df).to_dicts(),
        "calibration_lgbm": calibration(df, "vrp_lgbm"),
        "calibration_loghar": calibration(df, "vrp_loghar"),
        "rank_stability_5d": rank_stability(df, top_n=10, lag_days=5),
        "rank_stability_21d": rank_stability(df, top_n=10, lag_days=21),
    }

    if save:
        Path(bt_data.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        df.write_parquet(OUTPUT_FILE)

    return summary


# ----- gate -------------------------------------------------------------

def pass_gate(summary: dict) -> tuple[bool, str]:
    """BACKTEST_PLAN.md: 'If VRP accuracy < 55%, the signal is too noisy to
    trade. Stop here.'
    """
    acc = summary["overall"]["accuracy"]
    if acc < 0.55:
        return False, f"Level 1 FAILED: overall accuracy {acc:.1%} < 55% gate"
    return True, f"Level 1 PASSED: overall accuracy {acc:.1%} (>= 55%)"
