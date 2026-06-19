"""Preprocessing pipeline for v3.0 modeling.

Produces train/val/test splits + log-transformed target + scaler stats.
All artifacts saved to data/processed/splits/.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterator

import numpy as np
import polars as pl

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PANEL_PATH = _PROJECT_ROOT / "data" / "processed" / "panel.parquet"
SPLITS_DIR = _PROJECT_ROOT / "data" / "processed" / "splits"

TARGET_COL = "rv_21d_forward"
LOG_TARGET_COL = "log_rv_21d_forward"
ID_COLS = {"symbol", "date"}


def get_feature_cols(df: pl.DataFrame) -> list[str]:
    """Return all columns except IDs and target columns."""
    exclude = ID_COLS | {TARGET_COL, LOG_TARGET_COL}
    return [c for c in df.columns if c not in exclude]


def split_panel(
    df: pl.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.10,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Split pooled panel by unique trading dates (temporal order preserved).

    All symbols for a given date land in the same split.
    """
    dates = df["date"].unique().sort()
    n = len(dates)
    train_end = dates[int(n * train_frac) - 1]
    val_end = dates[int(n * (train_frac + val_frac)) - 1]

    train = df.filter(pl.col("date") <= train_end)
    val = df.filter((pl.col("date") > train_end) & (pl.col("date") <= val_end))
    test = df.filter(pl.col("date") > val_end)

    return train, val, test


def add_log_target(df: pl.DataFrame) -> pl.DataFrame:
    """Add log(rv_21d_forward) column. Original column preserved."""
    return df.with_columns(
        pl.col(TARGET_COL).log(base=math.e).alias(LOG_TARGET_COL)
    )


def fit_scaler(
    train: pl.DataFrame,
    feature_cols: list[str],
) -> dict[str, tuple[float, float]]:
    """Compute (mean, std) per feature column from train set only."""
    stats = {}
    for col in feature_cols:
        mean = float(train[col].mean())
        std = float(train[col].std())
        stats[col] = (mean, std if std > 1e-10 else 1.0)
    return stats


def apply_scaler(
    df: pl.DataFrame,
    stats: dict[str, tuple[float, float]],
) -> pl.DataFrame:
    """Apply pre-fitted scaler. Non-feature columns pass through unchanged."""
    exprs = [
        ((pl.col(c) - mean) / std).alias(c)
        for c, (mean, std) in stats.items()
    ]
    return df.with_columns(exprs)


def purged_kfold(
    train: pl.DataFrame,
    n_splits: int = 5,
    embargo_days: int = 21,
) -> Iterator[tuple[pl.DataFrame, pl.DataFrame]]:
    """Yield (fold_train, fold_val) DataFrames with purge + embargo.

    Operates on unique trading dates in the training set. Embargo is
    21 positions in the sorted date array (= 21 trading days = duration
    of the forward label window).

    Source: Lopez de Prado (2018) Advances in Financial ML, Chapter 7.
    """
    dates = train["date"].unique().sort().to_numpy()
    n = len(dates)
    fold_size = n // n_splits

    for fold_idx in range(n_splits):
        val_start = fold_idx * fold_size
        val_end = val_start + fold_size if fold_idx < n_splits - 1 else n

        # Purge 21 dates before val + val window + 21 dates after
        purge_start = max(0, val_start - embargo_days)
        embargo_end = min(n, val_end + embargo_days)

        train_mask = np.ones(n, dtype=bool)
        train_mask[purge_start:embargo_end] = False

        fold_train_dates = dates[train_mask]
        fold_val_dates = dates[val_start:val_end]

        fold_train = train.filter(pl.col("date").is_in(fold_train_dates.tolist()))
        fold_val = train.filter(pl.col("date").is_in(fold_val_dates.tolist()))

        yield fold_train, fold_val


def run_preprocessing() -> None:
    """Run full preprocessing pipeline and save artifacts."""
    print("Loading panel...")
    df = pl.read_parquet(PANEL_PATH)
    print(f"  Panel: {len(df):,} rows x {len(df.columns)} cols")

    # Log-transform target
    df = add_log_target(df)

    # Split
    train, val, test = split_panel(df)
    print(f"  Train: {len(train):,} rows ({train['date'].min()} to {train['date'].max()})")
    print(f"  Val:   {len(val):,} rows ({val['date'].min()} to {val['date'].max()})")
    print(f"  Test:  {len(test):,} rows ({test['date'].min()} to {test['date'].max()})")
    print(f"  Total: {len(train) + len(val) + len(test):,}")

    # Scaler stats (train only)
    feature_cols = get_feature_cols(train)
    stats = fit_scaler(train, feature_cols)
    print(f"  Features: {len(feature_cols)}")

    # Train mean for R2_OOS (Gu/Kelly/Xiu)
    train_mean_rv = float(train[TARGET_COL].mean())
    print(f"  train_mean_rv: {train_mean_rv:.4f}")

    # Log target skewness
    from scipy import stats as sp_stats
    log_skew = float(sp_stats.skew(train[LOG_TARGET_COL].to_numpy()))
    print(f"  Log target train skewness: {log_skew:.2f}")

    # Save
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    train.write_parquet(SPLITS_DIR / "train.parquet")
    val.write_parquet(SPLITS_DIR / "val.parquet")
    test.write_parquet(SPLITS_DIR / "test.parquet")

    scaler_json = {col: list(vals) for col, vals in stats.items()}
    scaler_json["__train_mean_rv__"] = train_mean_rv
    with open(SPLITS_DIR / "scaler_stats.json", "w") as f:
        json.dump(scaler_json, f, indent=2)

    print(f"\nSaved to {SPLITS_DIR}/")
    print("  train.parquet, val.parquet, test.parquet, scaler_stats.json")


if __name__ == "__main__":
    run_preprocessing()
