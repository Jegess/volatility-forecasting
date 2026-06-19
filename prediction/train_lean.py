"""One-time builder for the deployable lean models.

Retrains a lean LightGBM on the 9 TreeSHAP-selected features and fits
LogHAR coefficients, both on the existing train split. Saves artifacts to
``prediction/models/`` and prints each model's test QLIKE so we can see
whether the lean LightGBM still beats LogHAR with so few features.

    python -m prediction.train_lean

Reuses the existing LightGBM training code (custom QLIKE objective, the
tuned hyperparameters from the full model) — no logic is duplicated.
"""

from __future__ import annotations

import json

import numpy as np
import polars as pl

from prediction import (
    LEAN_FEATURES, LOG_TARGET_COL, MODELS_DIR, SPLITS_DIR, TARGET_COL, _FULL_HP_PATH,
)
from prediction.loghar import fit_loghar, predict_loghar, save_loghar
from theta.modeling.lightgbm_model import predict_lgbm, qlike_score, train_final_model


def _qlike_level(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """QLIKE in level space: mean(y/yhat - log(y/yhat) - 1)."""
    yhat = np.clip(y_pred, 1e-8, None)
    y = np.clip(y_true, 1e-8, None)
    return float(np.mean(y / yhat - np.log(y / yhat) - 1))


def build() -> None:
    train = pl.read_parquet(SPLITS_DIR / "train.parquet")
    test = pl.read_parquet(SPLITS_DIR / "test.parquet")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Train: {len(train):,} rows  Test: {len(test):,} rows")
    print(f"Lean features ({len(LEAN_FEATURES)}): {LEAN_FEATURES}\n")

    # ── Lean LightGBM ───────────────────────────────────────────────────
    # Reuse the tuned hyperparameters from the full 44-feature model.
    best_params = json.loads(_FULL_HP_PATH.read_text())
    model = train_final_model(train, LEAN_FEATURES, dict(best_params))
    model.save_model(str(MODELS_DIR / "lgbm_lean.txt"))

    y_pred = predict_lgbm(model, test, LEAN_FEATURES)
    qlike_lgbm = qlike_score(
        test[LOG_TARGET_COL].to_numpy(), np.log(np.clip(y_pred, 1e-8, None))
    )
    print(f"[LightGBM lean] saved -> {MODELS_DIR / 'lgbm_lean.txt'}")
    print(f"[LightGBM lean] test QLIKE = {qlike_lgbm:.4f}  (full 44-feat: 0.0215)\n")

    # ── LogHAR ──────────────────────────────────────────────────────────
    coefs = fit_loghar(train)
    save_loghar(coefs, MODELS_DIR / "loghar_coefs.json")

    # Evaluate LogHAR on the test split (positive RV rows only).
    feats = coefs["features"]
    pos = pl.all_horizontal([pl.col(f) > 0 for f in feats]) & (pl.col(TARGET_COL) > 0)
    test_f = test.filter(pos)
    rv = {f: test_f[f].to_numpy() for f in feats}
    y_loghar = np.array([
        predict_loghar(coefs, rv["rv_d"][i], rv["rv_w"][i], rv["rv_m"][i])
        for i in range(len(test_f))
    ])
    qlike_loghar = _qlike_level(test_f[TARGET_COL].to_numpy(), y_loghar)
    print(f"[LogHAR] saved -> {MODELS_DIR / 'loghar_coefs.json'}")
    print(f"[LogHAR] test QLIKE = {qlike_loghar:.4f}  (full-panel LogHAR: 0.0259)\n")

    # ── Verdict ─────────────────────────────────────────────────────────
    winner = "LightGBM (lean)" if qlike_lgbm < qlike_loghar else "LogHAR"
    print(f"Lean verdict: {winner} wins on test "
          f"(LightGBM {qlike_lgbm:.4f} vs LogHAR {qlike_loghar:.4f}).")


if __name__ == "__main__":
    build()
