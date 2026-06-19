"""Ex-post Variance Risk Premium analysis.

Computes VRP_21d = atm_iv - sqrt(rv_21d_forward) for each (symbol, date),
then produces summary statistics and rankings.

Usage:
    python -m theta.analysis.vrp
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl


PANEL_PATH = Path("data/processed/panel.parquet")
OUTPUT_DIR = Path("data/processed/analysis")


def compute_expost_vrp(panel: pl.DataFrame) -> pl.DataFrame:
    """Add ex-post VRP columns to panel.

    vrp_expost: atm_iv - sqrt(rv_21d_forward)  (vol space, annualized)
    vrp_expost_var: atm_iv^2 - rv_21d_forward  (variance space, annualized)
    """
    return panel.with_columns(
        (pl.col("atm_iv") - pl.col("rv_21d_forward").sqrt()).alias("vrp_expost"),
        (pl.col("atm_iv").pow(2) - pl.col("rv_21d_forward")).alias("vrp_expost_var"),
    )


def symbol_summary(df: pl.DataFrame) -> pl.DataFrame:
    """Per-symbol VRP summary statistics."""
    return (
        df.group_by("symbol")
        .agg(
            pl.col("vrp_expost").mean().alias("mean_vrp"),
            pl.col("vrp_expost").median().alias("median_vrp"),
            pl.col("vrp_expost").std().alias("std_vrp"),
            (pl.col("vrp_expost") > 0).mean().alias("win_rate"),
            pl.col("vrp_expost").min().alias("min_vrp"),
            pl.col("vrp_expost").max().alias("max_vrp"),
            pl.len().alias("n_obs"),
        )
        .with_columns(
            (pl.col("mean_vrp") / pl.col("std_vrp")).alias("vrp_sharpe"),
        )
        .sort("mean_vrp", descending=True)
    )


def monthly_vrp(df: pl.DataFrame) -> pl.DataFrame:
    """Cross-sectional average VRP by month."""
    return (
        df.with_columns(pl.col("date").dt.truncate("1mo").alias("month"))
        .group_by("month")
        .agg(
            pl.col("vrp_expost").mean().alias("mean_vrp"),
            pl.col("vrp_expost").median().alias("median_vrp"),
            (pl.col("vrp_expost") > 0).mean().alias("win_rate"),
            pl.len().alias("n_obs"),
        )
        .sort("month")
    )


def extreme_days(df: pl.DataFrame, n: int = 20) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Top N highest and lowest VRP observations."""
    cols = ["symbol", "date", "atm_iv", "rv_21d_forward", "vrp_expost"]
    top = df.sort("vrp_expost", descending=True).head(n).select(cols)
    bottom = df.sort("vrp_expost").head(n).select(cols)
    return top, bottom


def run() -> dict[str, pl.DataFrame]:
    """Run full VRP analysis and save outputs."""
    print("Loading panel...")
    panel = pl.read_parquet(PANEL_PATH)
    print(f"  {panel.shape[0]:,} rows, {panel.shape[1]} cols")

    df = compute_expost_vrp(panel)

    # Summary stats
    overall_mean = df["vrp_expost"].mean()
    overall_median = df["vrp_expost"].median()
    overall_win = (df["vrp_expost"] > 0).mean()
    print(f"\nOverall ex-post VRP (vol space):")
    print(f"  Mean:    {overall_mean:.4f} ({overall_mean * 100:.1f} vol pts)")
    print(f"  Median:  {overall_median:.4f} ({overall_median * 100:.1f} vol pts)")
    print(f"  Win rate: {overall_win:.1%}")

    # Per-symbol
    sym = symbol_summary(df)
    print(f"\nTop 10 symbols by mean VRP:")
    for row in sym.head(10).iter_rows(named=True):
        print(f"  {row['symbol']:>5s}: mean={row['mean_vrp']:.4f}, "
              f"win={row['win_rate']:.1%}, sharpe={row['vrp_sharpe']:.2f}")

    print(f"\nBottom 10 symbols by mean VRP:")
    for row in sym.tail(10).iter_rows(named=True):
        print(f"  {row['symbol']:>5s}: mean={row['mean_vrp']:.4f}, "
              f"win={row['win_rate']:.1%}, sharpe={row['vrp_sharpe']:.2f}")

    # Monthly
    monthly = monthly_vrp(df)

    # Extremes
    top, bottom = extreme_days(df)
    print(f"\nBiggest VRP day: {top.row(0, named=True)['symbol']} "
          f"on {top.row(0, named=True)['date']} = {top.row(0, named=True)['vrp_expost']:.4f}")
    print(f"Worst VRP day:   {bottom.row(0, named=True)['symbol']} "
          f"on {bottom.row(0, named=True)['date']} = {bottom.row(0, named=True)['vrp_expost']:.4f}")

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.select("symbol", "date", "vrp_expost", "vrp_expost_var").write_parquet(
        OUTPUT_DIR / "vrp_expost.parquet"
    )
    sym.write_parquet(OUTPUT_DIR / "vrp_symbol_summary.parquet")
    monthly.write_parquet(OUTPUT_DIR / "vrp_monthly.parquet")
    print(f"\nSaved to {OUTPUT_DIR}/")

    return {"panel_vrp": df, "symbol_summary": sym, "monthly": monthly,
            "top": top, "bottom": bottom}


if __name__ == "__main__":
    run()
