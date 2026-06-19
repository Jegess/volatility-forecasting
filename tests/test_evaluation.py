"""Tests for theta.modeling.evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from theta.modeling.evaluation import (
    qlike,
    mse,
    mae,
    mape,
    r2_oos,
    diebold_mariano,
    load_all_predictions,
    get_train_mean_rv,
    compute_metrics,
    compute_dm_tests,
    run_evaluation,
    PREDICTIONS_DIR,
    SPLITS_DIR,
)

slow = pytest.mark.slow


# =====================================================================
# QLIKE
# =====================================================================

def test_qlike_perfect_prediction():
    """At y == yhat, QLIKE should be 0."""
    y = np.array([1.0, 2.0, 0.5])
    assert abs(qlike(y, y)) < 1e-10


def test_qlike_positive_for_different():
    """QLIKE > 0 for imperfect forecasts (Jensen's inequality)."""
    y = np.array([1.0, 2.0, 0.5])
    yhat = np.array([1.5, 1.5, 1.0])
    assert qlike(y, yhat) > 0


def test_qlike_manual_computation():
    """Verify against hand calculation."""
    y = np.array([1.0, 4.0])
    yhat = np.array([2.0, 2.0])
    # i=0: 1/2 - log(1/2) - 1 = 0.5 - (-0.6931) - 1 = 0.1931
    # i=1: 4/2 - log(4/2) - 1 = 2.0 - 0.6931 - 1 = 0.3069
    # mean = 0.25
    expected = np.mean(y / yhat - np.log(y / yhat) - 1)
    np.testing.assert_allclose(qlike(y, yhat), expected, rtol=1e-10)


def test_qlike_clips_near_zero_yhat():
    """Near-zero yhat should not produce inf/nan."""
    y = np.array([1.0])
    yhat = np.array([0.0])  # will be clipped to 1e-8
    result = qlike(y, yhat)
    assert np.isfinite(result)
    assert result > 0


def test_qlike_asymmetry():
    """QLIKE penalizes underprediction more than overprediction (Patton 2011)."""
    y = np.array([1.0])
    under = qlike(y, np.array([0.5]))  # yhat too low
    over = qlike(y, np.array([1.5]))   # yhat too high (same distance)
    assert under > over, "QLIKE should penalize underprediction more"


# =====================================================================
# MSE
# =====================================================================

def test_mse_perfect():
    y = np.array([1.0, 2.0])
    assert mse(y, y) == 0.0


def test_mse_manual():
    y = np.array([1.0, 3.0])
    yhat = np.array([2.0, 1.0])
    # (1-2)^2 + (3-1)^2 = 1 + 4 = 5, mean = 2.5
    np.testing.assert_allclose(mse(y, yhat), 2.5)


# =====================================================================
# MAE
# =====================================================================

def test_mae_perfect():
    y = np.array([1.0, 2.0])
    assert mae(y, y) == 0.0


def test_mae_manual():
    y = np.array([1.0, 3.0])
    yhat = np.array([2.0, 1.0])
    # |1-2| + |3-1| = 1 + 2 = 3, mean = 1.5
    np.testing.assert_allclose(mae(y, yhat), 1.5)


# =====================================================================
# MAPE
# =====================================================================

def test_mape_perfect():
    y = np.array([1.0, 2.0])
    assert mape(y, y) == 0.0


def test_mape_manual():
    y = np.array([2.0, 4.0])
    yhat = np.array([1.0, 3.0])
    # |2-1|/2 + |4-3|/4 = 0.5 + 0.25 = 0.75, *100 = 75/2 = 37.5%
    np.testing.assert_allclose(mape(y, yhat), 37.5)


def test_mape_near_zero_y():
    """Near-zero y should not produce inf (clipped at 1e-4)."""
    y = np.array([0.0])
    yhat = np.array([0.01])
    result = mape(y, yhat)
    assert np.isfinite(result)


# =====================================================================
# R2_OOS
# =====================================================================

def test_r2_oos_perfect():
    """Perfect prediction -> R2 = 1."""
    y = np.array([1.0, 2.0, 3.0])
    assert r2_oos(y, y, train_mean=2.0) == 1.0


def test_r2_oos_predict_mean():
    """Predicting train mean -> R2 = 0."""
    y = np.array([1.0, 2.0, 3.0])
    yhat = np.full(3, 2.0)
    np.testing.assert_allclose(r2_oos(y, yhat, train_mean=2.0), 0.0, atol=1e-10)


def test_r2_oos_worse_than_mean():
    """Bad predictions -> negative R2."""
    y = np.array([1.0, 2.0, 3.0])
    yhat = np.array([10.0, 10.0, 10.0])
    assert r2_oos(y, yhat, train_mean=2.0) < 0


def test_r2_oos_uses_train_mean_not_test():
    """Denominator should use train_mean, not test mean."""
    y = np.array([10.0, 20.0, 30.0])  # test mean = 20
    yhat = np.full(3, 5.0)  # train mean = 5
    # SS_res = (10-5)^2 + (20-5)^2 + (30-5)^2 = 25+225+625 = 875
    # SS_tot = (10-5)^2 + (20-5)^2 + (30-5)^2 = 875  (same because yhat == train_mean)
    r2 = r2_oos(y, yhat, train_mean=5.0)
    np.testing.assert_allclose(r2, 0.0, atol=1e-10)


# =====================================================================
# Diebold-Mariano test
# =====================================================================

def test_dm_identical_models():
    """Two identical forecasters -> DM stat near 0, p-value near 1."""
    np.random.seed(42)
    y = np.abs(np.random.randn(500)) + 0.1
    yhat = np.abs(np.random.randn(500)) + 0.1
    dm_stat, p_val = diebold_mariano(y, yhat, yhat, loss="qlike")
    assert abs(dm_stat) < 1e-10
    assert p_val > 0.9


def test_dm_better_model_positive():
    """When model B is clearly better, DM stat should be positive."""
    np.random.seed(42)
    y = np.abs(np.random.randn(1000)) + 0.1
    yhat_bad = y + np.abs(np.random.randn(1000)) * 0.5  # systematic overprediction
    yhat_good = y + np.random.randn(1000) * 0.01  # near perfect
    dm_stat, p_val = diebold_mariano(y, yhat_bad, yhat_good, loss="qlike")
    assert dm_stat > 0, "Better model B should yield positive DM stat"
    assert p_val < 0.05


def test_dm_mse_loss():
    """DM test with MSE loss should also work."""
    np.random.seed(42)
    y = np.abs(np.random.randn(500)) + 0.1
    yhat_a = y + 1.0
    yhat_b = y + 0.01
    dm_stat, p_val = diebold_mariano(y, yhat_a, yhat_b, loss="mse")
    assert dm_stat > 0
    assert p_val < 0.05


def test_dm_unknown_loss_raises():
    """Unknown loss function should raise ValueError."""
    y = np.array([1.0])
    with pytest.raises(ValueError, match="Unknown loss"):
        diebold_mariano(y, y, y, loss="huber")


def test_dm_hac_lag_effect():
    """Longer max_lag should change DM stat (HAC adjusts variance)."""
    np.random.seed(42)
    y = np.abs(np.random.randn(500)) + 0.1
    yhat_a = y + 0.5
    yhat_b = y + 0.1
    stat_short, _ = diebold_mariano(y, yhat_a, yhat_b, max_lag=1)
    stat_long, _ = diebold_mariano(y, yhat_a, yhat_b, max_lag=50)
    assert stat_short != stat_long


# =====================================================================
# Data loading
# =====================================================================

def test_load_all_predictions_schema():
    """Loaded predictions should have expected columns."""
    preds = load_all_predictions()
    assert set(preds.columns) >= {"symbol", "date", "model", "y_true", "y_pred"}


def test_load_all_predictions_models():
    """Should contain all expected models."""
    preds = load_all_predictions()
    models = set(preds["model"].unique().to_list())
    expected = {"HAR", "LogHAR", "LevHAR", "SHAR", "HARQ", "GARCH", "AR5",
                "LightGBM", "FNN", "LSTM"}
    assert models == expected


def test_load_all_predictions_positive_values():
    """All y_true and y_pred should be positive (level-space variance)."""
    preds = load_all_predictions()
    assert (preds["y_true"].to_numpy() > 0).all()
    assert (preds["y_pred"].to_numpy() > 0).all()


def test_get_train_mean_rv():
    """Train mean should be a reasonable positive float."""
    mean_rv = get_train_mean_rv()
    assert isinstance(mean_rv, float)
    assert 0.01 < mean_rv < 1.0  # annualized variance, should be ~0.17


# =====================================================================
# compute_metrics
# =====================================================================

def test_compute_metrics_shape():
    """Should return one row per model with expected columns."""
    preds = load_all_predictions()
    train_mean = get_train_mean_rv()
    metrics = compute_metrics(preds, train_mean)
    n_models = preds["model"].n_unique()
    assert len(metrics) == n_models
    assert set(metrics.columns) == {"model", "n", "QLIKE", "MSE", "MAE", "MAPE", "R2_OOS"}


def test_compute_metrics_sorted_by_qlike():
    """Results should be sorted by QLIKE ascending (best first)."""
    preds = load_all_predictions()
    metrics = compute_metrics(preds, get_train_mean_rv())
    qlikes = metrics["QLIKE"].to_list()
    assert qlikes == sorted(qlikes)


def test_compute_metrics_lstm_best():
    """LSTM should have the lowest QLIKE."""
    preds = load_all_predictions()
    metrics = compute_metrics(preds, get_train_mean_rv())
    best_model = metrics[0, "model"]
    assert best_model == "LSTM"


# =====================================================================
# compute_dm_tests
# =====================================================================

def test_dm_tests_excludes_benchmark():
    """DM tests should not include benchmark vs itself."""
    preds = load_all_predictions()
    dm = compute_dm_tests(preds, benchmark="LogHAR")
    models = dm["model"].to_list()
    assert "LogHAR" not in models


def test_dm_tests_all_models_present():
    """Should test every non-benchmark model."""
    preds = load_all_predictions()
    dm = compute_dm_tests(preds, benchmark="LogHAR")
    expected = {"HAR", "LevHAR", "SHAR", "HARQ", "GARCH", "AR5",
                "LightGBM", "FNN", "LSTM"}
    assert set(dm["model"].to_list()) == expected


def test_dm_tests_schema():
    """DM test output should have expected columns."""
    preds = load_all_predictions()
    dm = compute_dm_tests(preds, benchmark="LogHAR")
    assert set(dm.columns) == {"model", "vs", "dm_stat", "p_value", "n_common", "significant_5pct"}


def test_dm_tests_lstm_significant_vs_loghar():
    """LSTM should be significantly better than LogHAR."""
    preds = load_all_predictions()
    dm = compute_dm_tests(preds, benchmark="LogHAR")
    lstm_row = dm.filter(pl.col("model") == "LSTM")
    assert lstm_row[0, "dm_stat"] > 0  # positive = better than benchmark
    assert lstm_row[0, "p_value"] < 0.01


# =====================================================================
# Integration (slow)
# =====================================================================

@slow
def test_run_evaluation_produces_files(tmp_path, monkeypatch):
    """run_evaluation should produce all output files."""
    # Just verify it runs without error on real data
    run_evaluation()
    out_dir = Path("data/processed/evaluation")
    assert (out_dir / "metrics.parquet").exists()
    assert (out_dir / "metrics.json").exists()
    assert (out_dir / "dm_tests_vs_loghar.parquet").exists()
    assert (out_dir / "dm_tests_vs_lgbm.parquet").exists()

    # Verify JSON is valid
    with open(out_dir / "metrics.json") as f:
        data = json.load(f)
    assert "LSTM" in data
    assert "QLIKE" in data["LSTM"]
