"""LightGBM volatility forecasting with custom QLIKE objective.

Trains LightGBM on all 44 features using log-transformed target,
tunes hyperparameters via Optuna with purged k-fold CV, and
extracts SHAP feature importances via TreeExplainer.

Output: data/processed/predictions/lgbm.parquet (same schema as baselines).
"""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import polars as pl
import shap

from theta.modeling.preprocessing import get_feature_cols, purged_kfold

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SPLITS_DIR = _PROJECT_ROOT / "data" / "processed" / "splits"
PREDICTIONS_DIR = _PROJECT_ROOT / "data" / "processed" / "predictions"
MODELS_DIR = _PROJECT_ROOT / "data" / "processed" / "models"

TARGET_COL = "rv_21d_forward"
LOG_TARGET_COL = "log_rv_21d_forward"


# ---------------------------------------------------------------------------
# Custom QLIKE objective & eval
# ---------------------------------------------------------------------------

def qlike_objective_lgb(
    y_pred: np.ndarray, dataset: lgb.Dataset
) -> tuple[np.ndarray, np.ndarray]:
    """Custom QLIKE objective for LightGBM in log space.

    LightGBM trains on log_rv_21d_forward. Gradient/hessian derived
    from QLIKE = y/yhat - log(y/yhat) - 1 with yhat = exp(z).

    CRITICAL: In LightGBM 4.6.0, this goes in params dict as
    'objective': qlike_objective_lgb  -- NOT as fobj= in lgb.train().
    """
    y_log = dataset.get_label()
    y_level = np.exp(y_log)
    yhat_level = np.clip(np.exp(y_pred), 1e-8, None)
    grad = 1.0 - y_level / yhat_level
    hess = y_level / yhat_level
    return grad, hess


def qlike_eval_lgb(
    y_pred: np.ndarray, dataset: lgb.Dataset
) -> tuple[str, float, bool]:
    """LightGBM eval metric callback returning (name, value, is_higher_better)."""
    y_log = dataset.get_label()
    y_level = np.exp(y_log)
    yhat_level = np.clip(np.exp(y_pred), 1e-8, None)
    qlike = float(np.mean(y_level / yhat_level - np.log(y_level / yhat_level) - 1))
    return "qlike", qlike, False


def qlike_score(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> float:
    """Standalone QLIKE for Optuna CV scoring (no dataset object needed)."""
    y_level = np.exp(y_true_log)
    yhat_level = np.clip(np.exp(y_pred_log), 1e-8, None)
    return float(np.mean(y_level / yhat_level - np.log(y_level / yhat_level) - 1))


# ---------------------------------------------------------------------------
# Optuna HP tuning
# ---------------------------------------------------------------------------

def make_optuna_objective(
    train_df: pl.DataFrame,
    feature_cols: list[str],
    n_splits: int = 5,
):
    """Factory returning an Optuna objective closure over training data."""

    def objective(trial: optuna.Trial) -> float:
        num_boost_round = trial.suggest_int("num_boost_round", 100, 1000)
        params = {
            "objective": qlike_objective_lgb,
            "num_leaves": trial.suggest_int("num_leaves", 20, 300),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "verbosity": -1,
            "n_jobs": -1,
        }

        fold_qlikes = []
        for fold_train, fold_val in purged_kfold(train_df, n_splits=n_splits, embargo_days=21):
            X_ft = fold_train.select(feature_cols).to_numpy()
            y_ft = fold_train[LOG_TARGET_COL].to_numpy()
            X_fv = fold_val.select(feature_cols).to_numpy()
            y_fv = fold_val[LOG_TARGET_COL].to_numpy()

            dtrain = lgb.Dataset(X_ft, label=y_ft)
            model = lgb.train(params, dtrain, num_boost_round=num_boost_round)
            y_pred = model.predict(X_fv)
            fold_qlikes.append(qlike_score(y_fv, y_pred))

        return float(np.mean(fold_qlikes))

    return objective


# ---------------------------------------------------------------------------
# Training & prediction
# ---------------------------------------------------------------------------

def train_final_model(
    train_df: pl.DataFrame,
    feature_cols: list[str],
    best_params: dict,
) -> lgb.Booster:
    """Train final LightGBM on full training set with best HP."""
    num_boost_round = best_params.pop("num_boost_round")
    params = {**best_params, "objective": qlike_objective_lgb, "verbosity": -1, "n_jobs": -1}
    X = train_df.select(feature_cols).to_numpy()
    y = train_df[LOG_TARGET_COL].to_numpy()
    dtrain = lgb.Dataset(X, label=y, feature_name=feature_cols)
    model = lgb.train(params, dtrain, num_boost_round=num_boost_round)
    best_params["num_boost_round"] = num_boost_round  # restore for logging
    return model


def predict_lgbm(
    model: lgb.Booster,
    test_df: pl.DataFrame,
    feature_cols: list[str],
) -> np.ndarray:
    """Predict on test set, return level-space RV predictions."""
    X_test = test_df.select(feature_cols).to_numpy()
    y_pred_log = model.predict(X_test)
    return np.clip(np.exp(y_pred_log), 1e-8, None)


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------

def compute_shap_values(
    model: lgb.Booster,
    X: np.ndarray,
    feature_cols: list[str] | None = None,
) -> np.ndarray:
    """Compute TreeSHAP values for a feature matrix."""
    explainer = shap.TreeExplainer(model)
    return explainer.shap_values(X)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_lgbm(n_trials: int = 50, n_splits: int = 5) -> pl.DataFrame:
    """Full pipeline: Optuna tuning -> final training -> prediction -> SHAP."""
    # Load splits
    train = pl.read_parquet(SPLITS_DIR / "train.parquet")
    test = pl.read_parquet(SPLITS_DIR / "test.parquet")
    feature_cols = get_feature_cols(train)
    print(f"Train: {len(train):,} rows, Test: {len(test):,} rows, Features: {len(feature_cols)}")

    # Optuna HP tuning
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", study_name="lgbm_qlike")
    study.optimize(make_optuna_objective(train, feature_cols, n_splits), n_trials=n_trials)

    best_params = dict(study.best_params)
    print(f"\nBest trial QLIKE: {study.best_value:.4f}")
    print(f"Best params: {json.dumps(best_params, indent=2)}")

    # Train final model on full train set
    model = train_final_model(train, feature_cols, best_params)

    # Save model + params
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODELS_DIR / "lgbm_best.txt"))
    with open(MODELS_DIR / "lgbm_best_params.json", "w") as f:
        json.dump(best_params, f, indent=2)

    # Predict on test set
    y_pred = predict_lgbm(model, test, feature_cols)

    # Build predictions DataFrame (same schema as baselines.parquet)
    preds_df = pl.DataFrame({
        "symbol": test["symbol"].to_numpy(),
        "date": test["date"].to_numpy(),
        "model": ["LightGBM"] * len(test),
        "y_true": test[TARGET_COL].to_numpy().astype(np.float64),
        "y_pred": y_pred.astype(np.float64),
    })
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    preds_df.write_parquet(PREDICTIONS_DIR / "lgbm.parquet")

    # QLIKE on test set
    y_true_log = test[LOG_TARGET_COL].to_numpy()
    y_pred_log = np.log(np.clip(y_pred, 1e-8, None))
    test_qlike = qlike_score(y_true_log, y_pred_log)
    print(f"\nTest QLIKE: {test_qlike:.4f} (LogHAR baseline: 0.0259)")

    # SHAP on 2000-row sample
    X_test = test.select(feature_cols).to_numpy()
    n_shap = min(2000, len(X_test))
    rng = np.random.default_rng(42)
    shap_idx = rng.choice(len(X_test), size=n_shap, replace=False)
    shap_values = compute_shap_values(model, X_test[shap_idx], feature_cols)
    np.save(str(MODELS_DIR / "lgbm_shap_values.npy"), shap_values)
    print(f"SHAP values saved: {shap_values.shape}")

    # Print top 10 features by mean |SHAP|
    importance = np.abs(shap_values).mean(axis=0)
    ranking = sorted(zip(feature_cols, importance), key=lambda x: -x[1])
    print("\nTop 10 features by SHAP:")
    for name, val in ranking[:10]:
        print(f"  {name:25s} {val:.4f}")

    return preds_df


if __name__ == "__main__":
    run_lgbm()
