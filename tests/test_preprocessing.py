"""Tests for theta.modeling.preprocessing."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from scipy import stats

from theta.modeling.preprocessing import (
    TARGET_COL,
    LOG_TARGET_COL,
    add_log_target,
    apply_scaler,
    fit_scaler,
    get_feature_cols,
    purged_kfold,
    split_panel,
)


@pytest.fixture
def synthetic_panel() -> pl.DataFrame:
    """Create a synthetic panel with 100 dates x 5 symbols = 500 rows."""
    np.random.seed(42)
    all_dates = pl.date_range(
        pl.date(2023, 1, 2), pl.date(2023, 6, 1), eager=True
    )
    # Filter to weekdays only (Mon-Fri: weekday 1-5 in polars)
    weekday_mask = all_dates.dt.weekday() < 6
    dates = all_dates.filter(weekday_mask)[:100]
    symbols = ["AAA", "BBB", "CCC", "DDD", "EEE"]

    rows = []
    for s in symbols:
        for d in dates:
            rows.append({
                "symbol": s,
                "date": d,
                "rv_21d_forward": float(np.random.lognormal(mean=-2, sigma=1.5)),
                "feat_a": float(np.random.uniform(0, 1000)),
                "feat_b": float(np.random.uniform(0, 0.01)),
                "is_event": float(np.random.choice([0.0, 1.0])),
            })
    return pl.DataFrame(rows)


# --- PREP-01: Sequential split ---

def test_split_panel_row_counts(synthetic_panel: pl.DataFrame):
    train, val, test = split_panel(synthetic_panel)
    n_dates = synthetic_panel["date"].n_unique()
    n_symbols = synthetic_panel["symbol"].n_unique()

    # 70/10/20 of dates, times 5 symbols
    assert abs(len(train) - int(n_dates * 0.70) * n_symbols) <= n_symbols
    assert abs(len(val) - int(n_dates * 0.10) * n_symbols) <= n_symbols
    assert abs(len(test) - int(n_dates * 0.20) * n_symbols) <= n_symbols


def test_split_no_date_overlap(synthetic_panel: pl.DataFrame):
    train, val, test = split_panel(synthetic_panel)
    train_dates = set(train["date"].to_list())
    val_dates = set(val["date"].to_list())
    test_dates = set(test["date"].to_list())

    assert len(train_dates & val_dates) == 0
    assert len(train_dates & test_dates) == 0
    assert len(val_dates & test_dates) == 0


def test_split_temporal_order(synthetic_panel: pl.DataFrame):
    train, val, test = split_panel(synthetic_panel)
    assert train["date"].max() < val["date"].min()
    assert val["date"].max() < test["date"].min()


def test_split_all_rows_accounted(synthetic_panel: pl.DataFrame):
    train, val, test = split_panel(synthetic_panel)
    assert len(train) + len(val) + len(test) == len(synthetic_panel)


# --- PREP-04: Log-transform ---

def test_log_target_reduces_skewness(synthetic_panel: pl.DataFrame):
    df = add_log_target(synthetic_panel)
    original_skew = abs(stats.skew(df[TARGET_COL].to_numpy()))
    log_skew = abs(stats.skew(df[LOG_TARGET_COL].to_numpy()))
    assert log_skew < original_skew
    assert log_skew < 1.0


def test_log_target_no_nulls(synthetic_panel: pl.DataFrame):
    df = add_log_target(synthetic_panel)
    assert df[LOG_TARGET_COL].null_count() == 0
    assert df[LOG_TARGET_COL].is_infinite().sum() == 0


# --- PREP-03: Feature scaling ---

def test_scaler_train_statistics(synthetic_panel: pl.DataFrame):
    train, _, _ = split_panel(synthetic_panel)
    feature_cols = get_feature_cols(train)
    scaler_stats = fit_scaler(train, feature_cols)
    scaled = apply_scaler(train, scaler_stats)

    for col in feature_cols:
        assert abs(float(scaled[col].mean())) < 0.01, f"{col} mean not ~0"
        assert abs(float(scaled[col].std()) - 1.0) < 0.05, f"{col} std not ~1"


def test_scaler_no_leakage():
    """Construct data where train and val have deliberately different distributions."""
    np.random.seed(99)
    # Train: mean=100, val: mean=500 — scaler fit on train should NOT center val
    train = pl.DataFrame({
        "symbol": ["A"] * 50,
        "date": pl.date_range(pl.date(2023, 1, 2), pl.date(2023, 3, 20), eager=True)[:50],
        "rv_21d_forward": np.random.uniform(0.01, 0.5, 50).tolist(),
        "feat_x": np.random.normal(100, 10, 50).tolist(),
    })
    val = pl.DataFrame({
        "symbol": ["A"] * 20,
        "date": pl.date_range(pl.date(2023, 4, 1), pl.date(2023, 5, 10), eager=True)[:20],
        "rv_21d_forward": np.random.uniform(0.01, 0.5, 20).tolist(),
        "feat_x": np.random.normal(500, 10, 20).tolist(),
    })

    feature_cols = ["feat_x"]
    scaler_stats = fit_scaler(train, feature_cols)
    scaled_val = apply_scaler(val, scaler_stats)

    # Val mean should be far from 0 (around (500-100)/10 = 40)
    val_mean = abs(float(scaled_val["feat_x"].mean()))
    assert val_mean > 5, f"Val mean too close to 0 ({val_mean:.2f}) — possible leakage"


# --- PREP-02: Purged k-fold ---

def test_purged_kfold_no_overlap(synthetic_panel: pl.DataFrame):
    train, _, _ = split_panel(synthetic_panel)
    for fold_train, fold_val in purged_kfold(train, n_splits=5, embargo_days=21):
        train_dates = set(fold_train["date"].to_list())
        val_dates = set(fold_val["date"].to_list())
        assert len(train_dates & val_dates) == 0, "Date overlap in fold"


def test_purged_kfold_embargo_gap(synthetic_panel: pl.DataFrame):
    train, _, _ = split_panel(synthetic_panel)
    all_dates = train["date"].unique().sort().to_list()
    date_to_idx = {d: i for i, d in enumerate(all_dates)}

    for fold_train, fold_val in purged_kfold(train, n_splits=5, embargo_days=21):
        ft_dates = fold_train["date"].unique().to_list()
        fv_dates = fold_val["date"].unique().to_list()
        train_indices = sorted(date_to_idx[d] for d in ft_dates)
        val_indices = sorted(date_to_idx[d] for d in fv_dates)

        if not train_indices or not val_indices:
            continue

        # Min distance between any train date and any val date
        min_gap = min(
            abs(t - v) for t in train_indices for v in val_indices
        )
        assert min_gap >= 21, f"Embargo gap {min_gap} < 21 trading days"


def test_purged_kfold_correct_count(synthetic_panel: pl.DataFrame):
    train, _, _ = split_panel(synthetic_panel)
    folds = list(purged_kfold(train, n_splits=5, embargo_days=21))
    assert len(folds) == 5
