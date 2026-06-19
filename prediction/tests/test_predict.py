"""Fast, offline tests for the prediction tool (no network, no data splits).

These rely only on the committed artifacts in prediction/models/ and on
synthetic inputs, so they run in a clean checkout.
"""

from __future__ import annotations

import datetime as dt

import lightgbm as lgb
import numpy as np
import polars as pl
import pytest

from prediction import LEAN_FEATURES, LOGHAR_FEATURES, MODELS_DIR
from prediction.loghar import fit_loghar, load_loghar, predict_loghar
from prediction.predict import (
    build_rv_row, days_to_next_fomc, load_prices_csv, resolve_next_earnings,
)


# ── Feature contract ────────────────────────────────────────────────────

def test_lean_feature_contract():
    assert LEAN_FEATURES == [
        "rv_d", "rv_w", "rv_m", "rq", "rs_pos", "rs_neg",
        "atm_iv", "days_to_earnings", "days_to_fomc",
    ]
    assert LOGHAR_FEATURES == ["rv_d", "rv_w", "rv_m"]


# ── LogHAR mechanism ────────────────────────────────────────────────────

def _synthetic_panel(n: int = 500, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    rv_m = rng.uniform(0.02, 0.40, n)
    rv_w = rv_m * rng.uniform(0.5, 1.5, n)
    rv_d = rv_w * rng.uniform(0.5, 1.5, n)
    target = rv_m * rng.uniform(0.7, 1.3, n)  # forward RV tracks rv_m
    return pl.DataFrame(
        {"rv_d": rv_d, "rv_w": rv_w, "rv_m": rv_m, "rv_21d_forward": target}
    ).with_columns(pl.col("rv_21d_forward").log().alias("log_rv_21d_forward"))


def test_loghar_fit_and_predict():
    coefs = fit_loghar(_synthetic_panel())
    assert set(coefs["betas"]) == set(LOGHAR_FEATURES)
    pred = predict_loghar(coefs, 0.012, 0.041, 0.055)
    assert np.isfinite(pred) and pred > 0


def test_loghar_rejects_nonpositive():
    coefs = fit_loghar(_synthetic_panel())
    with pytest.raises(ValueError):
        predict_loghar(coefs, 0.0, 0.04, 0.05)


def test_loghar_artifact_loads_and_predicts():
    coefs = load_loghar(MODELS_DIR / "loghar_coefs.json")
    pred = predict_loghar(coefs, 0.01, 0.04, 0.05)
    assert np.isfinite(pred) and pred > 0


# ── Lean LightGBM artifact ──────────────────────────────────────────────

def test_lgbm_lean_artifact_predicts():
    booster = lgb.Booster(model_file=str(MODELS_DIR / "lgbm_lean.txt"))
    assert booster.num_feature() == len(LEAN_FEATURES)
    row = [0.012, 0.041, 0.055, 0.001, 0.02, 0.03, 0.25, 30.0, 10.0]
    rv = float(np.exp(booster.predict(np.array([row], dtype=np.float64)))[0])
    assert np.isfinite(rv) and rv > 0


# ── RV features from a raw price series ─────────────────────────────────

def test_build_rv_row_from_prices():
    rng = np.random.default_rng(1)
    n = 80
    rets = rng.normal(0, 0.012, n)
    prices = 100.0 * np.exp(np.cumsum(rets))
    start = dt.date(2025, 1, 1)
    dates = [start + dt.timedelta(days=i) for i in range(n)]
    price_df = pl.DataFrame({"date": dates, "underlying_price": prices})

    row = build_rv_row(price_df, "TEST", as_of=None)
    for f in ["rv_d", "rv_w", "rv_m", "rq", "rs_pos", "rs_neg"]:
        assert row[f] is not None and np.isfinite(row[f]) and row[f] >= 0
    assert row["rv_m"] > 0


def test_build_rv_row_too_short():
    dates = [dt.date(2025, 1, 1) + dt.timedelta(days=i) for i in range(10)]
    price_df = pl.DataFrame({"date": dates, "underlying_price": [100.0] * 10})
    with pytest.raises(SystemExit):
        build_rv_row(price_df, "TEST", as_of=None)


# ── CSV loader ──────────────────────────────────────────────────────────

def test_load_prices_csv(tmp_path):
    p = tmp_path / "px.csv"
    p.write_text("date,close\n2025-01-03,101.5\n2025-01-02,100.0\n")
    df = load_prices_csv(str(p))
    assert df.columns == ["date", "underlying_price"]
    assert df["date"].dtype == pl.Date
    assert df["date"].to_list() == [dt.date(2025, 1, 2), dt.date(2025, 1, 3)]  # sorted


# ── Event helpers ───────────────────────────────────────────────────────

def test_days_to_next_fomc():
    # Next FOMC after 2026-07-01 is 2026-07-29 (28 days).
    assert days_to_next_fomc(dt.date(2026, 7, 1)) == 28


def test_resolve_next_earnings_supplied():
    as_of = dt.date(2026, 6, 18)
    assert resolve_next_earnings("X", as_of, "2026-07-28") == 40
    # Past date -> unknown (None), so the CLI falls back to a neutral default.
    assert resolve_next_earnings("X", as_of, "2026-04-30") is None
