"""Tests for theta.modeling.lightgbm_model."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
import lightgbm as lgb

from theta.modeling.lightgbm_model import (
    qlike_objective_lgb,
    qlike_eval_lgb,
    qlike_score,
    make_optuna_objective,
    train_final_model,
    predict_lgbm,
    compute_shap_values,
    run_lgbm,
    SPLITS_DIR,
    PREDICTIONS_DIR,
    MODELS_DIR,
    TARGET_COL,
    LOG_TARGET_COL,
)

slow = pytest.mark.slow


# --- Fixtures ---

@pytest.fixture(scope="module")
def splits():
    """Load real train/test splits (shared across tests in module)."""
    train = pl.read_parquet(SPLITS_DIR / "train.parquet")
    test = pl.read_parquet(SPLITS_DIR / "test.parquet")
    return train, test


@pytest.fixture(scope="module")
def small_booster():
    """Train a tiny LightGBM model for fast tests."""
    np.random.seed(42)
    n, k = 200, 5
    X = np.random.randn(n, k)
    y = np.random.randn(n)  # log-space target
    feature_names = [f"feat_{i}" for i in range(k)]
    dtrain = lgb.Dataset(X, label=y, feature_name=feature_names)
    params = {"objective": qlike_objective_lgb, "verbosity": -1, "num_leaves": 8}
    model = lgb.train(params, dtrain, num_boost_round=10)
    return model, X, feature_names


# =====================================================================
# QLIKE objective
# =====================================================================

def test_qlike_objective_perfect_prediction():
    """At perfect prediction (y == yhat), gradient should be 0, hessian 1."""
    y_log = np.array([0.0, 0.0])  # exp(0) = 1.0
    y_pred = np.array([0.0, 0.0])
    ds = lgb.Dataset(np.zeros((2, 1)), label=y_log).construct()
    grad, hess = qlike_objective_lgb(y_pred, ds)
    np.testing.assert_allclose(grad, [0.0, 0.0], atol=1e-10)
    np.testing.assert_allclose(hess, [1.0, 1.0], atol=1e-10)


def test_qlike_objective_shape():
    """Gradient and hessian should have same shape as input."""
    n = 100
    y_log = np.random.randn(n)
    y_pred = np.random.randn(n)
    ds = lgb.Dataset(np.zeros((n, 1)), label=y_log).construct()
    grad, hess = qlike_objective_lgb(y_pred, ds)
    assert grad.shape == (n,)
    assert hess.shape == (n,)


def test_qlike_objective_hessian_positive():
    """Hessian should always be positive (convex loss)."""
    np.random.seed(7)
    n = 100
    y_log = np.random.randn(n)
    y_pred = np.random.randn(n)
    ds = lgb.Dataset(np.zeros((n, 1)), label=y_log).construct()
    _, hess = qlike_objective_lgb(y_pred, ds)
    assert (hess > 0).all()


def test_qlike_objective_gradient_direction():
    """When yhat < y, grad < 0 (push up). When yhat > y, grad > 0 (push down)."""
    # Underpredicting: yhat = exp(-1) ~ 0.37, y = exp(1) ~ 2.72
    y_log = np.array([1.0])
    y_pred_under = np.array([-1.0])
    ds = lgb.Dataset(np.zeros((1, 1)), label=y_log).construct()
    grad, _ = qlike_objective_lgb(y_pred_under, ds)
    assert grad[0] < 0, "Gradient should be negative when underpredicting"

    # Overpredicting: yhat = exp(2) ~ 7.39, y = exp(-1) ~ 0.37
    y_log2 = np.array([-1.0])
    y_pred_over = np.array([2.0])
    ds2 = lgb.Dataset(np.zeros((1, 1)), label=y_log2).construct()
    grad2, _ = qlike_objective_lgb(y_pred_over, ds2)
    assert grad2[0] > 0, "Gradient should be positive when overpredicting"


# =====================================================================
# QLIKE eval
# =====================================================================

def test_qlike_eval_returns_tuple():
    """Eval metric should return (name, value, is_higher_better)."""
    y_log = np.array([0.0, 0.5, -0.5])
    y_pred = np.array([0.1, 0.4, -0.3])
    ds = lgb.Dataset(np.zeros((3, 1)), label=y_log).construct()
    result = qlike_eval_lgb(y_pred, ds)
    assert isinstance(result, tuple)
    assert len(result) == 3
    assert result[0] == "qlike"
    assert isinstance(result[1], float)
    assert result[2] is False


def test_qlike_eval_perfect_is_zero():
    """At perfect prediction, QLIKE should be 0."""
    y_log = np.array([0.0, 1.0, -1.0])
    ds = lgb.Dataset(np.zeros((3, 1)), label=y_log).construct()
    name, value, _ = qlike_eval_lgb(y_log.copy(), ds)
    assert abs(value) < 1e-10


# =====================================================================
# QLIKE score (standalone)
# =====================================================================

def test_qlike_score_identical_is_zero():
    """Identical arrays -> QLIKE = 0."""
    arr = np.array([0.0, 1.0, -0.5, 2.0])
    assert abs(qlike_score(arr, arr)) < 1e-10


def test_qlike_score_positive_for_different():
    """Different arrays -> QLIKE > 0."""
    a = np.array([0.0, 1.0, -0.5])
    b = np.array([0.5, 0.5, 0.0])
    assert qlike_score(a, b) > 0


def test_qlike_score_manual_computation():
    """Verify QLIKE against manual computation."""
    y_true_log = np.log(np.array([1.0, 2.0]))
    y_pred_log = np.log(np.array([1.5, 1.5]))
    # y=[1,2], yhat=[1.5,1.5]
    # QLIKE_i = y/yhat - log(y/yhat) - 1
    # i=0: 1/1.5 - log(1/1.5) - 1 = 0.6667 - (-0.4055) - 1 = 0.0721
    # i=1: 2/1.5 - log(2/1.5) - 1 = 1.3333 - 0.2877 - 1 = 0.0457
    # mean = (0.0721 + 0.0457) / 2 = 0.0589
    expected = np.mean(
        np.array([1.0, 2.0]) / np.array([1.5, 1.5])
        - np.log(np.array([1.0, 2.0]) / np.array([1.5, 1.5]))
        - 1
    )
    result = qlike_score(y_true_log, y_pred_log)
    np.testing.assert_allclose(result, expected, rtol=1e-10)


# =====================================================================
# Optuna objective
# =====================================================================

def test_make_optuna_objective_returns_callable(splits):
    """Factory should return a callable."""
    train, _ = splits
    from theta.modeling.preprocessing import get_feature_cols
    feature_cols = get_feature_cols(train)
    obj = make_optuna_objective(train, feature_cols)
    assert callable(obj)


# =====================================================================
# Train / predict
# =====================================================================

def test_train_final_model_returns_booster():
    """train_final_model should return an lgb.Booster."""
    np.random.seed(42)
    n = 200
    feature_cols = [f"f{i}" for i in range(5)]
    data = {c: np.random.randn(n) for c in feature_cols}
    data[LOG_TARGET_COL] = np.random.randn(n)
    data[TARGET_COL] = np.exp(data[LOG_TARGET_COL])
    data["symbol"] = ["A"] * n
    data["date"] = pl.date_range(pl.date(2023, 1, 1), pl.date(2023, 1, 1), eager=True).extend_constant(
        pl.date(2023, 1, 1), n - 1
    ).to_list()
    df = pl.DataFrame(data)

    params = {
        "num_boost_round": 10,
        "num_leaves": 15,
        "learning_rate": 0.1,
        "max_depth": 4,
        "min_child_samples": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.01,
        "reg_lambda": 1.0,
    }
    model = train_final_model(df, feature_cols, params)
    assert isinstance(model, lgb.Booster)


def test_predict_lgbm_positive(small_booster):
    """All predictions should be positive (level space)."""
    model, X, feature_names = small_booster
    # Create a polars DataFrame for predict_lgbm
    data = {name: X[:, i] for i, name in enumerate(feature_names)}
    data[LOG_TARGET_COL] = np.zeros(len(X))
    data[TARGET_COL] = np.ones(len(X))
    data["symbol"] = ["A"] * len(X)
    data["date"] = [None] * len(X)
    df = pl.DataFrame(data)

    preds = predict_lgbm(model, df, feature_names)
    assert (preds > 0).all()


def test_predict_lgbm_correct_length(small_booster):
    """Prediction array length should match input."""
    model, X, feature_names = small_booster
    data = {name: X[:50, i] for i, name in enumerate(feature_names)}
    data[LOG_TARGET_COL] = np.zeros(50)
    data[TARGET_COL] = np.ones(50)
    data["symbol"] = ["A"] * 50
    data["date"] = [None] * 50
    df = pl.DataFrame(data)

    preds = predict_lgbm(model, df, feature_names)
    assert len(preds) == 50


# =====================================================================
# SHAP
# =====================================================================

def test_compute_shap_values_shape(small_booster):
    """SHAP values should be (n_samples, n_features)."""
    model, X, feature_names = small_booster
    shap_vals = compute_shap_values(model, X[:50], feature_names)
    assert shap_vals.shape == (50, len(feature_names))


# =====================================================================
# Integration (slow)
# =====================================================================

@slow
def test_run_lgbm_integration():
    """End-to-end: run_lgbm with minimal trials produces valid output."""
    result = run_lgbm(n_trials=2, n_splits=2)
    assert isinstance(result, pl.DataFrame)
    assert set(result.columns) == {"symbol", "date", "model", "y_true", "y_pred"}
    assert result["model"].unique().to_list() == ["LightGBM"]
    assert result["y_pred"].null_count() == 0
    assert (result["y_pred"].to_numpy() > 0).all()
    assert (PREDICTIONS_DIR / "lgbm.parquet").exists()
    assert (MODELS_DIR / "lgbm_best.txt").exists()
    assert (MODELS_DIR / "lgbm_best_params.json").exists()
