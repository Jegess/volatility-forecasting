"""Baseline volatility forecasting models.

Seven models in total:
  OLS-based (pooled): HAR, LogHAR, SHAR, HARQ, AR(5), LevHAR
  Per-symbol: GARCH(1,1)

All predict rv_21d_forward (annualized variance) on the test set.
Output: data/processed/predictions/baselines.parquet (long format).
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import polars as pl

from theta.processing.rv import adjust_splits, compute_log_returns


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SPLITS_DIR = _PROJECT_ROOT / "data" / "processed" / "splits"
PREDICTIONS_DIR = _PROJECT_ROOT / "data" / "processed" / "predictions"
UNDERLYING_DIR = _PROJECT_ROOT / "data" / "raw" / "underlying"

TARGET_COL = "rv_21d_forward"
LOG_TARGET_COL = "log_rv_21d_forward"


# ---------------------------------------------------------------------------
# OLS infrastructure
# ---------------------------------------------------------------------------

def _ols_fit_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit OLS via least squares and return (coeffs, predictions, sigma2_resid).

    Prepends an intercept column. Rows with any NaN in X_train or y_train
    are excluded from fitting.
    """
    # Prepend intercept
    X_train = np.column_stack([np.ones(len(X_train)), X_train])
    X_test = np.column_stack([np.ones(len(X_test)), X_test])

    # Exclude rows with NaN or Inf
    mask = (
        np.all(np.isfinite(X_train), axis=1)
        & np.isfinite(y_train)
    )
    coeffs, _, _, _ = np.linalg.lstsq(X_train[mask], y_train[mask], rcond=None)

    # Residual variance (for Jensen correction)
    resid = y_train[mask] - X_train[mask] @ coeffs
    sigma2 = float(np.var(resid))

    preds = X_test @ coeffs
    return coeffs, preds, sigma2


def _make_predictions_df(
    symbols: np.ndarray,
    dates: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
) -> pl.DataFrame:
    """Build standardised predictions DataFrame."""
    return pl.DataFrame({
        "symbol": symbols,
        "date": dates,
        "model": [model_name] * len(y_true),
        "y_true": y_true.astype(np.float64),
        "y_pred": y_pred.astype(np.float64),
    })


# ---------------------------------------------------------------------------
# HAR family models
# ---------------------------------------------------------------------------

def predict_har(train: pl.DataFrame, test: pl.DataFrame) -> pl.DataFrame:
    """HAR (Corsi 2009): rv_d, rv_w, rv_m → rv_21d_forward (OLS in levels)."""
    features = ["rv_d", "rv_w", "rv_m"]
    X_train = train.select(features).to_numpy()
    y_train = train[TARGET_COL].to_numpy()
    X_test = test.select(features).to_numpy()

    _, y_pred, _ = _ols_fit_predict(X_train, y_train, X_test)
    y_pred = np.clip(y_pred, 1e-8, None)

    return _make_predictions_df(
        test["symbol"].to_numpy(), test["date"].to_numpy(),
        test[TARGET_COL].to_numpy(), y_pred, "HAR",
    )


def predict_loghar(train: pl.DataFrame, test: pl.DataFrame) -> pl.DataFrame:
    """LogHAR: log(rv_d), log(rv_w), log(rv_m) → log target → Jensen correction.

    Rows with zero RV features (log → -inf) are excluded from train fitting
    and test predictions.
    """
    features = ["rv_d", "rv_w", "rv_m"]

    # Filter out rows where any feature is <= 0 (log undefined)
    pos_mask_expr = [pl.col(f) > 0 for f in features]
    train_f = train.filter(pl.all_horizontal(pos_mask_expr))
    test_f = test.filter(pl.all_horizontal(pos_mask_expr))

    X_train = np.log(train_f.select(features).to_numpy())
    y_train = train_f[LOG_TARGET_COL].to_numpy()
    X_test = np.log(test_f.select(features).to_numpy())

    coeffs, log_preds, sigma2 = _ols_fit_predict(X_train, y_train, X_test)
    # Jensen correction: E[exp(X)] = exp(mu + 0.5*sigma^2)
    y_pred = np.exp(log_preds + 0.5 * sigma2)

    return _make_predictions_df(
        test_f["symbol"].to_numpy(), test_f["date"].to_numpy(),
        test_f[TARGET_COL].to_numpy(), y_pred, "LogHAR",
    )


def predict_shar(train: pl.DataFrame, test: pl.DataFrame) -> pl.DataFrame:
    """SHAR (Patton & Sheppard): rs_pos, rs_neg, rv_w, rv_m → rv_21d_forward (OLS in levels)."""
    features = ["rs_pos", "rs_neg", "rv_w", "rv_m"]
    X_train = train.select(features).to_numpy()
    y_train = train[TARGET_COL].to_numpy()
    X_test = test.select(features).to_numpy()

    _, y_pred, _ = _ols_fit_predict(X_train, y_train, X_test)
    y_pred = np.clip(y_pred, 1e-8, None)

    return _make_predictions_df(
        test["symbol"].to_numpy(), test["date"].to_numpy(),
        test[TARGET_COL].to_numpy(), y_pred, "SHAR",
    )


def predict_harq(train: pl.DataFrame, test: pl.DataFrame) -> pl.DataFrame:
    """HARQ (Bollerslev et al.): rv_d, rv_w, rv_m, rv_d*sqrt(rq) → rv_21d_forward (OLS in levels)."""
    features = ["rv_d", "rv_w", "rv_m"]

    X_train_base = train.select(features).to_numpy()
    interaction_train = train["rv_d"].to_numpy() * np.sqrt(train["rq"].to_numpy())
    X_train = np.column_stack([X_train_base, interaction_train])

    X_test_base = test.select(features).to_numpy()
    interaction_test = test["rv_d"].to_numpy() * np.sqrt(test["rq"].to_numpy())
    X_test = np.column_stack([X_test_base, interaction_test])

    y_train = train[TARGET_COL].to_numpy()

    _, y_pred, _ = _ols_fit_predict(X_train, y_train, X_test)
    y_pred = np.clip(y_pred, 1e-8, None)

    return _make_predictions_df(
        test["symbol"].to_numpy(), test["date"].to_numpy(),
        test[TARGET_COL].to_numpy(), y_pred, "HARQ",
    )


def predict_ar5(train: pl.DataFrame, test: pl.DataFrame) -> pl.DataFrame:
    """AR(5): 5 lags of rv_d → rv_21d_forward (OLS in levels).

    Lags constructed with .over('symbol') to prevent cross-symbol leakage.
    """
    lag_cols = [f"rv_d_lag{i}" for i in range(1, 6)]

    def _add_lags(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns([
            pl.col("rv_d").shift(i).over("symbol").alias(f"rv_d_lag{i}")
            for i in range(1, 6)
        ])

    train_l = _add_lags(train.sort("symbol", "date"))
    test_l = _add_lags(test.sort("symbol", "date"))

    # Drop rows with NaN lags (first rows per symbol)
    train_l = train_l.drop_nulls(subset=lag_cols)
    test_l = test_l.drop_nulls(subset=lag_cols)

    X_train = train_l.select(lag_cols).to_numpy()
    y_train = train_l[TARGET_COL].to_numpy()
    X_test = test_l.select(lag_cols).to_numpy()

    _, y_pred, _ = _ols_fit_predict(X_train, y_train, X_test)
    y_pred = np.clip(y_pred, 1e-8, None)

    return _make_predictions_df(
        test_l["symbol"].to_numpy(), test_l["date"].to_numpy(),
        test_l[TARGET_COL].to_numpy(), y_pred, "AR5",
    )


# ---------------------------------------------------------------------------
# LevHAR (leverage from underlying files)
# ---------------------------------------------------------------------------

def _build_leverage_features(symbol: str) -> pl.DataFrame:
    """Derive daily and weekly leverage features from underlying prices.

    lev_d: annualised negative squared return (0 if return is positive)
    lev_w: 5-day rolling mean of lev_d
    """
    df = pl.read_parquet(UNDERLYING_DIR / f"{symbol}.parquet")
    df = df.select("date", "underlying_price").unique("date").sort("date")
    df = adjust_splits(df, symbol)
    df = compute_log_returns(df)

    df = df.with_columns(
        (
            pl.when(pl.col("log_return") < 0)
            .then(pl.col("log_return") ** 2)
            .otherwise(0.0)
            * 252
        ).alias("lev_d")
    )
    df = df.with_columns(
        pl.col("lev_d").rolling_mean(5).alias("lev_w")
    )
    return df.select("date", "lev_d", "lev_w").drop_nulls()


def predict_levhar(train: pl.DataFrame, test: pl.DataFrame) -> pl.DataFrame:
    """LevHAR: rv_d, rv_w, rv_m, lev_d, lev_w, rs_neg → rv_21d_forward (OLS in levels).

    Leverage features derived from underlying price files at fitting time.
    """
    symbols = sorted(
        set(train["symbol"].unique().to_list())
        | set(test["symbol"].unique().to_list())
    )
    all_lev = pl.concat([
        _build_leverage_features(s).with_columns(pl.lit(s).alias("symbol"))
        for s in symbols
    ])

    train_lev = train.join(all_lev, on=["symbol", "date"], how="left")
    test_lev = test.join(all_lev, on=["symbol", "date"], how="left")

    features = ["rv_d", "rv_w", "rv_m", "lev_d", "lev_w", "rs_neg"]
    train_lev = train_lev.drop_nulls(subset=features)
    test_lev = test_lev.drop_nulls(subset=features)

    X_train = train_lev.select(features).to_numpy()
    y_train = train_lev[TARGET_COL].to_numpy()
    X_test = test_lev.select(features).to_numpy()

    _, y_pred, _ = _ols_fit_predict(X_train, y_train, X_test)
    y_pred = np.clip(y_pred, 1e-8, None)

    return _make_predictions_df(
        test_lev["symbol"].to_numpy(), test_lev["date"].to_numpy(),
        test_lev[TARGET_COL].to_numpy(), y_pred, "LevHAR",
    )


# ---------------------------------------------------------------------------
# GARCH(1,1) per-symbol
# ---------------------------------------------------------------------------

def _fit_garch_symbol(
    returns_pct: np.ndarray,
    n_train: int,
    fallback_rv: float,
) -> dict[int, float]:
    """Fit GARCH(1,1) on percentage log returns, forecast 21-day variance.

    Returns dict mapping integer index (>= n_train) to annualised RV forecast.
    On convergence failure, all post-train indices get fallback_rv.
    """
    from arch import arch_model

    n = len(returns_pct)
    result_dict: dict[int, float] = {}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        am = arch_model(
            returns_pct, vol="Garch", p=1, q=1,
            dist="Normal", rescale=False, mean="Zero",
        )
        try:
            res = am.fit(disp="off", last_obs=n_train, options={"maxiter": 200})
        except Exception:
            return {i: fallback_rv for i in range(n_train, n)}

        if res.convergence_flag != 0:
            return {i: fallback_rv for i in range(n_train, n)}

        fixed = am.fix(res.params)
        fc = fixed.forecast(horizon=21, start=n_train - 1, reindex=False)
        var_df = fc.variance

        for idx in var_df.index:
            # Sum h.1..h.21, convert pct-squared → decimal, annualise
            rv = float(var_df.loc[idx].sum()) / 10000 * (252 / 21)
            result_dict[int(idx)] = rv

    return result_dict


def predict_garch(train: pl.DataFrame, test: pl.DataFrame) -> pl.DataFrame:
    """GARCH(1,1): per-symbol fitting on underlying percentage log returns.

    Convergence failures fall back to train_mean_rv.
    """
    # Load fallback
    with open(SPLITS_DIR / "scaler_stats.json") as f:
        train_mean_rv = json.load(f)["__train_mean_rv__"]

    train_end_date = train["date"].max()
    test_max_date = test["date"].max()

    symbols = test["symbol"].unique().sort().to_list()
    all_rows: list[tuple] = []

    for i, symbol in enumerate(symbols):
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  GARCH: {i + 1}/{len(symbols)} {symbol}")

        # Load underlying returns
        path = UNDERLYING_DIR / f"{symbol}.parquet"
        if not path.exists():
            # Missing underlying (e.g. XYZ) → fallback
            sub = test.filter(pl.col("symbol") == symbol)
            for row in sub.iter_rows(named=True):
                all_rows.append((symbol, row["date"], row[TARGET_COL], train_mean_rv))
            continue

        udf = pl.read_parquet(path)
        udf = udf.select("date", "underlying_price").unique("date").sort("date")
        udf = udf.filter(pl.col("date") <= test_max_date)
        udf = adjust_splits(udf, symbol)
        udf = compute_log_returns(udf)
        udf = udf.drop_nulls("log_return")

        dates_arr = udf["date"].to_list()
        returns_arr = udf["log_return"].to_numpy() * 100  # percentage

        # Find n_train: number of underlying dates <= train_end_date
        n_train_idx = sum(1 for d in dates_arr if d <= train_end_date)
        if n_train_idx < 100:
            # Too few training observations
            sub = test.filter(pl.col("symbol") == symbol)
            for row in sub.iter_rows(named=True):
                all_rows.append((symbol, row["date"], row[TARGET_COL], train_mean_rv))
            continue

        forecasts = _fit_garch_symbol(returns_arr, n_train_idx, train_mean_rv)

        # Build date→index mapping
        date_to_idx = {d: j for j, d in enumerate(dates_arr)}

        sub = test.filter(pl.col("symbol") == symbol)
        for row in sub.iter_rows(named=True):
            idx = date_to_idx.get(row["date"])
            if idx is not None and idx in forecasts:
                y_pred = forecasts[idx]
            else:
                y_pred = train_mean_rv
            all_rows.append((symbol, row["date"], row[TARGET_COL], y_pred))

    syms, dts, yts, yps = zip(*all_rows)
    return _make_predictions_df(
        np.array(syms), list(dts),
        np.array(yts, dtype=np.float64),
        np.array(yps, dtype=np.float64),
        "GARCH",
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_baselines() -> pl.DataFrame:
    """Run all 7 baseline models and save combined predictions."""
    train = pl.read_parquet(SPLITS_DIR / "train.parquet")
    test = pl.read_parquet(SPLITS_DIR / "test.parquet")

    results = []
    print("=== Baseline Models ===")

    print("Running HAR...")
    results.append(predict_har(train, test))

    print("Running LogHAR...")
    results.append(predict_loghar(train, test))

    print("Running LevHAR...")
    results.append(predict_levhar(train, test))

    print("Running SHAR...")
    results.append(predict_shar(train, test))

    print("Running HARQ...")
    results.append(predict_harq(train, test))

    print("Running AR(5)...")
    results.append(predict_ar5(train, test))

    print("Running GARCH(1,1)...")
    results.append(predict_garch(train, test))

    combined = pl.concat(results)

    # Validate
    assert combined["y_pred"].null_count() == 0, "NaN in predictions"
    assert combined["model"].n_unique() == 7, "Expected 7 models"

    # Save
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    combined.write_parquet(PREDICTIONS_DIR / "baselines.parquet")

    # QLIKE preview
    print("\n=== QLIKE Preview ===")
    for model_name in combined["model"].unique().sort().to_list():
        sub = combined.filter(pl.col("model") == model_name)
        y = sub["y_true"].to_numpy()
        yhat = sub["y_pred"].to_numpy()
        qlike = float(np.mean(y / yhat - np.log(y / yhat) - 1))
        print(f"  {model_name:8s}: QLIKE={qlike:.4f}, n={len(sub)}")

    print(f"\nSaved: {PREDICTIONS_DIR / 'baselines.parquet'}")
    print(f"Total rows: {len(combined):,}")
    return combined


if __name__ == "__main__":
    run_baselines()
