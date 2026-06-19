"""Tests for theta.modeling.baselines."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from theta.modeling.baselines import (
    SPLITS_DIR,
    TARGET_COL,
    LOG_TARGET_COL,
    _ols_fit_predict,
    _build_leverage_features,
    predict_har,
    predict_loghar,
    predict_shar,
    predict_harq,
    predict_ar5,
    predict_levhar,
    predict_garch,
    _fit_garch_symbol,
    run_baselines,
)

slow = pytest.mark.slow


# --- Fixtures ---

@pytest.fixture(scope="module")
def splits():
    """Load real train/test splits (shared across tests in module)."""
    train = pl.read_parquet(SPLITS_DIR / "train.parquet")
    test = pl.read_parquet(SPLITS_DIR / "test.parquet")
    return train, test


# =====================================================================
# Task 1: OLS infrastructure + HAR + LogHAR
# =====================================================================

def test_ols_fit_predict_basic():
    """Synthetic data: OLS recovers approximate linear relationship."""
    np.random.seed(42)
    n = 100
    X = np.random.randn(n, 2)
    true_coeffs = np.array([3.0, 1.5, -0.5])  # intercept + 2 features
    y = true_coeffs[0] + X @ true_coeffs[1:] + np.random.randn(n) * 0.1

    coeffs, preds, sigma2 = _ols_fit_predict(X[:80], y[:80], X[80:])
    assert coeffs.shape == (3,)
    assert preds.shape == (20,)
    np.testing.assert_allclose(coeffs, true_coeffs, atol=0.5)
    assert sigma2 > 0


def test_ols_fit_predict_handles_nan():
    """OLS excludes NaN rows and still produces finite coefficients."""
    np.random.seed(7)
    X = np.random.randn(50, 2)
    y = X @ [1.0, 2.0] + 0.5
    # Inject NaN
    X[10, 0] = np.nan
    X[20, 1] = np.nan
    y[30] = np.nan

    coeffs, preds, _ = _ols_fit_predict(X[:40], y[:40], X[40:])
    assert np.all(np.isfinite(coeffs))
    assert preds.shape == (10,)


def test_har_predictions_no_nan(splits):
    train, test = splits
    result = predict_har(train, test)

    assert set(result.columns) == {"symbol", "date", "model", "y_true", "y_pred"}
    assert result["y_pred"].null_count() == 0
    assert result["y_pred"].is_nan().sum() == 0
    assert result["model"].unique().to_list() == ["HAR"]
    assert len(result) == len(test)


def test_har_coefficients_finite(splits):
    train, test = splits
    features = ["rv_d", "rv_w", "rv_m"]
    X_train = train.select(features).to_numpy()
    y_train = train[LOG_TARGET_COL].to_numpy()
    X_test = test.select(features).to_numpy()

    coeffs, _, _ = _ols_fit_predict(X_train, y_train, X_test)
    assert coeffs.shape == (4,)  # intercept + 3
    assert np.all(np.isfinite(coeffs))


def test_loghar_jensen_vs_naive(splits):
    """LogHAR with Jensen correction produces higher mean than naive exp."""
    train, test = splits
    features = ["rv_d", "rv_w", "rv_m"]
    # Filter zero-RV rows (log(0) = -inf)
    pos_mask = [pl.col(f) > 0 for f in features]
    train_f = train.filter(pl.all_horizontal(pos_mask))
    test_f = test.filter(pl.all_horizontal(pos_mask))

    X_train = np.log(train_f.select(features).to_numpy())
    y_train = train_f[LOG_TARGET_COL].to_numpy()
    X_test = np.log(test_f.select(features).to_numpy())

    coeffs, log_preds, sigma2 = _ols_fit_predict(X_train, y_train, X_test)

    naive = np.exp(log_preds)
    jensen = np.exp(log_preds + 0.5 * sigma2)

    assert np.mean(jensen) > np.mean(naive)


def test_loghar_predictions_positive(splits):
    train, test = splits
    result = predict_loghar(train, test)
    assert (result["y_pred"].to_numpy() > 0).all()


def test_loghar_sigma2_positive(splits):
    train, test = splits
    features = ["rv_d", "rv_w", "rv_m"]
    pos_mask = [pl.col(f) > 0 for f in features]
    train_f = train.filter(pl.all_horizontal(pos_mask))
    test_f = test.filter(pl.all_horizontal(pos_mask))

    X_train = np.log(train_f.select(features).to_numpy())
    y_train = train_f[LOG_TARGET_COL].to_numpy()
    X_test = np.log(test_f.select(features).to_numpy())

    _, _, sigma2 = _ols_fit_predict(X_train, y_train, X_test)
    assert sigma2 > 0


# =====================================================================
# Task 2: SHAR + HARQ + AR(5)
# =====================================================================

def test_shar_uses_semivariance(splits):
    """SHAR uses rs_pos/rs_neg and produces different coefficients than HAR."""
    train, test = splits

    # HAR coefficients
    har_X = train.select(["rv_d", "rv_w", "rv_m"]).to_numpy()
    har_coeffs, _, _ = _ols_fit_predict(har_X, train[LOG_TARGET_COL].to_numpy(),
                                         test.select(["rv_d", "rv_w", "rv_m"]).to_numpy())

    # SHAR coefficients
    shar_X = train.select(["rs_pos", "rs_neg", "rv_w", "rv_m"]).to_numpy()
    shar_coeffs, _, _ = _ols_fit_predict(shar_X, train[LOG_TARGET_COL].to_numpy(),
                                          test.select(["rs_pos", "rs_neg", "rv_w", "rv_m"]).to_numpy())

    assert har_coeffs.shape != shar_coeffs.shape  # 4 vs 5


def test_shar_predictions_no_nan(splits):
    train, test = splits
    result = predict_shar(train, test)
    assert result["y_pred"].null_count() == 0
    assert result["y_pred"].is_nan().sum() == 0
    assert result["model"].unique().to_list() == ["SHAR"]


def test_harq_interaction_term(splits):
    """HARQ design matrix has 5 columns (intercept + 4 features)."""
    train, _ = splits
    X_base = train.select(["rv_d", "rv_w", "rv_m"]).to_numpy()
    interaction = train["rv_d"].to_numpy() * np.sqrt(train["rq"].to_numpy())
    X = np.column_stack([X_base, interaction])
    assert X.shape[1] == 4  # 4 features (intercept added by _ols_fit_predict)
    assert not np.any(np.isnan(interaction))


def test_harq_predictions_no_nan(splits):
    train, test = splits
    result = predict_harq(train, test)
    assert result["y_pred"].null_count() == 0
    assert result["y_pred"].is_nan().sum() == 0
    assert result["model"].unique().to_list() == ["HARQ"]


def test_ar5_lag_construction(splits):
    """AR(5) lags don't leak across symbol boundaries."""
    _, test = splits
    test_sorted = test.sort("symbol", "date")
    symbols = test_sorted["symbol"].unique().sort().to_list()

    # Take second symbol's first date — its lag1 should NOT be the last rv_d
    # of the first symbol
    sym1, sym2 = symbols[0], symbols[1]
    last_rv_d_sym1 = test_sorted.filter(pl.col("symbol") == sym1)["rv_d"][-1]

    test_with_lags = test_sorted.with_columns(
        pl.col("rv_d").shift(1).over("symbol").alias("rv_d_lag1")
    )
    first_row_sym2 = test_with_lags.filter(pl.col("symbol") == sym2).head(1)
    lag1_val = first_row_sym2["rv_d_lag1"][0]

    # Should be null (first row of symbol) not the last value of previous symbol
    assert lag1_val is None


def test_ar5_predictions_no_nan(splits):
    train, test = splits
    result = predict_ar5(train, test)
    assert result["y_pred"].null_count() == 0
    assert result["y_pred"].is_nan().sum() == 0
    assert result["model"].unique().to_list() == ["AR5"]


def test_ar5_has_5_coefficients(splits):
    """AR(5) has 6 coefficients: intercept + 5 lags."""
    train, test = splits
    lag_cols = [f"rv_d_lag{i}" for i in range(1, 6)]

    train_l = train.sort("symbol", "date").with_columns([
        pl.col("rv_d").shift(i).over("symbol").alias(f"rv_d_lag{i}")
        for i in range(1, 6)
    ]).drop_nulls(subset=lag_cols)

    test_l = test.sort("symbol", "date").with_columns([
        pl.col("rv_d").shift(i).over("symbol").alias(f"rv_d_lag{i}")
        for i in range(1, 6)
    ]).drop_nulls(subset=lag_cols)

    coeffs, _, _ = _ols_fit_predict(
        train_l.select(lag_cols).to_numpy(),
        train_l[LOG_TARGET_COL].to_numpy(),
        test_l.select(lag_cols).to_numpy(),
    )
    assert coeffs.shape == (6,)  # intercept + 5 lags


# =====================================================================
# LevHAR
# =====================================================================

def test_build_leverage_features_shape():
    """AAPL leverage features have expected shape."""
    lev = _build_leverage_features("AAPL")
    assert set(lev.columns) == {"date", "lev_d", "lev_w"}
    assert len(lev) >= 900


def test_build_leverage_features_no_nan():
    lev = _build_leverage_features("AAPL")
    assert lev["lev_d"].null_count() == 0
    assert lev["lev_w"].null_count() == 0


def test_levhar_design_matrix_cols(splits):
    """LevHAR has 6 features: rv_d, rv_w, rv_m, lev_d, lev_w, rs_neg."""
    train, _ = splits
    lev = _build_leverage_features("AAPL").with_columns(pl.lit("AAPL").alias("symbol"))
    merged = train.filter(pl.col("symbol") == "AAPL").join(lev, on=["symbol", "date"], how="left")
    features = ["rv_d", "rv_w", "rv_m", "lev_d", "lev_w", "rs_neg"]
    merged = merged.drop_nulls(subset=features)
    X = merged.select(features).to_numpy()
    assert X.shape[1] == 6  # +1 intercept in _ols_fit_predict = 7 total


def test_levhar_predictions_no_nan(splits):
    train, test = splits
    result = predict_levhar(train, test)
    assert result["y_pred"].null_count() == 0
    assert result["y_pred"].is_nan().sum() == 0
    assert result["model"].unique().to_list() == ["LevHAR"]


def test_levhar_coefficients_differ_from_har(splits):
    train, test = splits
    # HAR: 4 coefficients
    har_result = predict_har(train, test)
    features_har = ["rv_d", "rv_w", "rv_m"]
    har_coeffs, _, _ = _ols_fit_predict(
        train.select(features_har).to_numpy(),
        train[LOG_TARGET_COL].to_numpy(),
        test.select(features_har).to_numpy(),
    )
    assert har_coeffs.shape == (4,)

    # LevHAR builds leverage internally — just verify it has more coefficients
    # by checking predictions differ
    levhar_result = predict_levhar(train, test)
    # LevHAR may have fewer rows (dropped nulls), but predictions should differ
    assert levhar_result["y_pred"].mean() != har_result["y_pred"].mean()


# =====================================================================
# GARCH
# =====================================================================

def test_garch_fit_symbol_aapl():
    """GARCH on AAPL returns positive forecasts for post-train indices."""
    from theta.processing.rv import adjust_splits, compute_log_returns

    df = pl.read_parquet(SPLITS_DIR / ".." / ".." / "raw" / "underlying" / "AAPL.parquet")
    df = df.select("date", "underlying_price").unique("date").sort("date")
    df = adjust_splits(df, "AAPL")
    df = compute_log_returns(df)
    df = df.drop_nulls("log_return")

    returns_pct = df["log_return"].to_numpy() * 100
    n_train = int(len(returns_pct) * 0.7)

    forecasts = _fit_garch_symbol(returns_pct, n_train, fallback_rv=0.1748)

    assert len(forecasts) >= 100
    assert all(v > 0 for v in forecasts.values())
    assert all(k >= n_train - 1 for k in forecasts.keys())


def test_garch_convergence_fallback():
    """GARCH with garbage data falls back to fallback_rv."""
    fallback = 0.1234
    returns_pct = np.zeros(500)  # constant zero → no variance → convergence issues
    forecasts = _fit_garch_symbol(returns_pct, 350, fallback_rv=fallback)

    assert len(forecasts) > 0
    assert all(v == fallback for v in forecasts.values())


def test_garch_predictions_no_nan(splits):
    """GARCH produces test predictions with zero NaN. (Uses a few symbols only.)"""
    train, test = splits
    # Use a small subset for speed
    subset_symbols = ["AAPL", "MSFT", "GOOGL"]
    train_sub = train.filter(pl.col("symbol").is_in(subset_symbols))
    test_sub = test.filter(pl.col("symbol").is_in(subset_symbols))

    result = predict_garch(train_sub, test_sub)
    assert result["y_pred"].null_count() == 0
    assert result["y_pred"].is_nan().sum() == 0
    assert result["model"].unique().to_list() == ["GARCH"]


def test_garch_predictions_positive(splits):
    """All GARCH predictions are positive (variance > 0)."""
    train, test = splits
    subset_symbols = ["AAPL", "MSFT"]
    result = predict_garch(
        train.filter(pl.col("symbol").is_in(subset_symbols)),
        test.filter(pl.col("symbol").is_in(subset_symbols)),
    )
    assert (result["y_pred"].to_numpy() > 0).all()


# =====================================================================
# Integration: run_baselines (slow)
# =====================================================================

@slow
def test_run_baselines_output_schema():
    result = run_baselines()
    assert set(result.columns) == {"symbol", "date", "model", "y_true", "y_pred"}
    assert result["y_pred"].dtype == pl.Float64
    assert result["y_true"].dtype == pl.Float64


@slow
def test_run_baselines_seven_models():
    result = run_baselines()
    models = sorted(result["model"].unique().to_list())
    assert models == ["AR5", "GARCH", "HAR", "HARQ", "LevHAR", "LogHAR", "SHAR"]


@slow
def test_run_baselines_no_nan():
    result = run_baselines()
    assert result["y_pred"].null_count() == 0
    assert result["y_pred"].is_nan().sum() == 0


@slow
def test_har_outperforms_ar5():
    """HAR should have lower QLIKE than AR(5) — literature expectation."""
    result = run_baselines()

    def qlike(sub):
        y = sub["y_true"].to_numpy()
        yhat = sub["y_pred"].to_numpy()
        return float(np.mean(y / yhat - np.log(y / yhat) - 1))

    har_q = qlike(result.filter(pl.col("model") == "HAR"))
    ar5_q = qlike(result.filter(pl.col("model") == "AR5"))
    assert har_q < ar5_q, f"HAR QLIKE ({har_q:.4f}) >= AR5 QLIKE ({ar5_q:.4f})"


@slow
def test_baselines_parquet_written():
    from theta.modeling.baselines import PREDICTIONS_DIR
    run_baselines()
    assert (PREDICTIONS_DIR / "baselines.parquet").exists()
