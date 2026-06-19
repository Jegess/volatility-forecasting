"""LogHAR mechanism: fit pooled OLS coefficients and predict 21-day RV.

LogHAR regresses log(rv_21d_forward) on log(rv_d), log(rv_w), log(rv_m),
pooled across all symbols, and applies the Jensen correction when mapping
back to level space. This mirrors ``theta.modeling.baselines.predict_loghar``
but is kept local so the prediction tool is self-contained.

Coefficients are fit once by ``train_lean.py`` and saved to
``prediction/models/loghar_coefs.json``; ``predict.py`` loads and applies them.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import polars as pl

from prediction import LOGHAR_FEATURES, LOG_TARGET_COL, TARGET_COL


def fit_loghar(train: pl.DataFrame) -> dict:
    """Fit pooled LogHAR coefficients on a training panel.

    Returns a dict: {features, intercept, betas{feature: coef}, sigma2}.
    Rows where any RV feature or the target is <= 0 (log undefined) are
    excluded, matching the baseline implementation.
    """
    feats = LOGHAR_FEATURES
    pos = pl.all_horizontal([pl.col(f) > 0 for f in feats]) & (pl.col(TARGET_COL) > 0)
    tf = train.filter(pos)

    X = np.log(tf.select(feats).to_numpy())
    y = tf[LOG_TARGET_COL].to_numpy()

    # Prepend intercept; drop any non-finite rows before lstsq.
    Xi = np.column_stack([np.ones(len(X)), X])
    mask = np.all(np.isfinite(Xi), axis=1) & np.isfinite(y)
    coeffs, _, _, _ = np.linalg.lstsq(Xi[mask], y[mask], rcond=None)

    resid = y[mask] - Xi[mask] @ coeffs
    sigma2 = float(np.var(resid))

    return {
        "features": feats,
        "intercept": float(coeffs[0]),
        "betas": {f: float(c) for f, c in zip(feats, coeffs[1:])},
        "sigma2": sigma2,
        "n_train": int(mask.sum()),
    }


def predict_loghar(coefs: dict, rv_d: float, rv_w: float, rv_m: float) -> float:
    """Predict level-space 21-day-forward annualized RV from RV features.

    Applies the Jensen correction: E[exp(X)] = exp(mu + 0.5*sigma^2).
    """
    vals = {"rv_d": rv_d, "rv_w": rv_w, "rv_m": rv_m}
    for f in coefs["features"]:
        if vals[f] <= 0:
            raise ValueError(f"LogHAR requires positive {f}; got {vals[f]}")
    log_pred = coefs["intercept"] + sum(
        coefs["betas"][f] * math.log(vals[f]) for f in coefs["features"]
    )
    return float(math.exp(log_pred + 0.5 * coefs["sigma2"]))


def save_loghar(coefs: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(coefs, indent=2))


def load_loghar(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())
