"""Tests for walk-forward validation module."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from theta.modeling.walk_forward import (
    generate_windows,
    run_baselines_window,
    compute_window_metrics,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_dates():
    """1055 trading dates (approx 4.2 years)."""
    import datetime
    dates = []
    d = datetime.date(2022, 1, 3)
    while len(dates) < 1055:
        if d.weekday() < 5:  # skip weekends
            dates.append(d)
        d += datetime.timedelta(days=1)
    return np.array(dates)


@pytest.fixture
def small_panel():
    """Tiny synthetic panel: 3 symbols, 400 trading days, 4 features + target."""
    import datetime

    dates = []
    d = datetime.date(2022, 1, 3)
    while len(dates) < 400:
        if d.weekday() < 5:
            dates.append(d)
        d += datetime.timedelta(days=1)

    symbols = ["AAA", "BBB", "CCC"]
    rng = np.random.default_rng(42)

    rows = []
    for sym in symbols:
        for dt in dates:
            rv = rng.exponential(0.1)
            rows.append({
                "symbol": sym,
                "date": dt,
                "rv_21d_forward": rv,
                "log_rv_21d_forward": float(np.log(max(rv, 1e-10))),
                "rv_d": rng.exponential(0.1),
                "rv_w": rng.exponential(0.1),
                "rv_m": rng.exponential(0.1),
                "rv_bv": rng.exponential(0.1),
                "rs_pos": rng.exponential(0.05),
                "rs_neg": rng.exponential(0.05),
                "rq": rng.exponential(0.01),
            })

    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# generate_windows tests
# ---------------------------------------------------------------------------


def test_generate_windows_count(sample_dates):
    """Should produce ~12 windows for 1055 dates."""
    windows = generate_windows(sample_dates, min_train_days=252, test_days=63, step_days=63)
    assert 10 <= len(windows) <= 14, f"Expected ~12 windows, got {len(windows)}"


def test_generate_windows_no_overlap(sample_dates):
    """Test windows should not overlap."""
    windows = generate_windows(sample_dates)
    for i in range(1, len(windows)):
        prev_end = windows[i - 1]["test_end"]
        curr_start = windows[i]["test_start"]
        assert curr_start > prev_end, (
            f"Window {i} test_start {curr_start} overlaps with "
            f"window {i-1} test_end {prev_end}"
        )


def test_generate_windows_expanding(sample_dates):
    """Each window's training set should be larger than the previous."""
    windows = generate_windows(sample_dates)
    for i in range(1, len(windows)):
        assert windows[i]["n_train_dates"] > windows[i - 1]["n_train_dates"], (
            f"Window {i} train ({windows[i]['n_train_dates']}) not larger than "
            f"window {i-1} ({windows[i-1]['n_train_dates']})"
        )


def test_generate_windows_embargo(sample_dates):
    """21-day embargo gap between train end and test start."""
    windows = generate_windows(sample_dates, embargo_days=21)
    for w in windows:
        train_end_idx = np.searchsorted(sample_dates, w["train_end"])
        test_start_idx = np.searchsorted(sample_dates, w["test_start"])
        gap = test_start_idx - train_end_idx - 1
        assert gap >= 21, (
            f"Window {w['window_id']}: embargo gap is {gap}, expected >= 21"
        )


def test_generate_windows_min_train(sample_dates):
    """First window should have at least min_train_days training dates."""
    windows = generate_windows(sample_dates, min_train_days=252)
    assert windows[0]["n_train_dates"] >= 252


def test_generate_windows_empty():
    """Too few dates should produce no windows."""
    short = np.array([np.datetime64("2022-01-03") + np.timedelta64(i, "D") for i in range(100)])
    windows = generate_windows(short, min_train_days=252)
    assert len(windows) == 0


# ---------------------------------------------------------------------------
# run_baselines_window tests
# ---------------------------------------------------------------------------


def test_baselines_window_schema(small_panel):
    """Output should have expected columns including window_id."""
    dates = small_panel["date"].unique().sort().to_numpy()
    train_df = small_panel.filter(pl.col("date") <= dates[299])
    test_df = small_panel.filter(
        (pl.col("date") >= dates[320]) & (pl.col("date") <= dates[382])
    )
    preds = run_baselines_window(train_df, test_df, window_id=0)

    assert len(preds) > 0, "Expected predictions"
    expected_cols = {"symbol", "date", "model", "y_true", "y_pred", "window_id"}
    assert set(preds.columns) >= expected_cols, (
        f"Missing columns: {expected_cols - set(preds.columns)}"
    )


def test_baselines_window_models(small_panel):
    """Should produce predictions for 6 baselines (no GARCH)."""
    dates = small_panel["date"].unique().sort().to_numpy()
    train_df = small_panel.filter(pl.col("date") <= dates[299])
    test_df = small_panel.filter(
        (pl.col("date") >= dates[320]) & (pl.col("date") <= dates[382])
    )
    preds = run_baselines_window(train_df, test_df, window_id=0)

    models = set(preds["model"].unique().to_list())
    # AR5 may fail on very short data, but HAR family should work
    expected = {"HAR", "LogHAR", "SHAR", "HARQ"}
    assert models >= expected, f"Missing models: {expected - models}"


# ---------------------------------------------------------------------------
# compute_window_metrics tests
# ---------------------------------------------------------------------------


def test_compute_window_metrics():
    """Metrics should have correct schema and reasonable values."""
    rng = np.random.default_rng(42)
    y_true = rng.exponential(0.1, 100)
    y_pred = y_true * (1 + rng.normal(0, 0.1, 100))
    y_pred = np.clip(y_pred, 1e-8, None)

    preds_df = pl.DataFrame({
        "symbol": ["AAA"] * 100,
        "date": [f"2023-01-{i:02d}" for i in range(1, 101)],
        "model": ["TestModel"] * 100,
        "y_true": y_true,
        "y_pred": y_pred,
        "window_id": [0] * 100,
    })

    metrics = compute_window_metrics(preds_df, train_mean_rv=0.1)

    assert len(metrics) == 1
    row = metrics.row(0, named=True)
    assert row["model"] == "TestModel"
    assert row["QLIKE"] > 0
    assert row["MSE"] > 0
    assert row["n"] == 100


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_small_panel_walk_forward(small_panel):
    """End-to-end walk-forward on tiny synthetic data, baselines only."""
    dates = small_panel["date"].unique().sort().to_numpy()
    windows = generate_windows(dates, min_train_days=200, test_days=50, step_days=50, embargo_days=21)

    assert len(windows) >= 2, f"Expected at least 2 windows, got {len(windows)}"

    all_metrics = []
    for w in windows:
        train_df = small_panel.filter(pl.col("date") <= w["train_end"])
        test_df = small_panel.filter(
            (pl.col("date") >= w["test_start"]) & (pl.col("date") <= w["test_end"])
        )
        train_mean_rv = float(train_df["rv_21d_forward"].mean())

        preds = run_baselines_window(train_df, test_df, w["window_id"])
        if len(preds) > 0:
            metrics = compute_window_metrics(preds, train_mean_rv)
            metrics = metrics.with_columns(pl.lit(w["window_id"]).alias("window_id"))
            all_metrics.append(metrics)

    assert len(all_metrics) >= 2, "Expected metrics from at least 2 windows"
    combined = pl.concat(all_metrics)
    assert combined["window_id"].n_unique() >= 2
