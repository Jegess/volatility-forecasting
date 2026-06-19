"""Model evaluation for volatility forecasting.

Computes QLIKE, MSE, MAE, MAPE, R²_OOS across all models.
Runs Diebold-Mariano pairwise significance tests with HAC standard errors.
Produces consolidated results table.

All metrics computed in LEVEL space (annualized variance) from saved
prediction parquets: {symbol, date, model, y_true, y_pred}.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats as sp_stats

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PREDICTIONS_DIR = _PROJECT_ROOT / "data" / "processed" / "predictions"
SPLITS_DIR = _PROJECT_ROOT / "data" / "processed" / "splits"


# ---------------------------------------------------------------------------
# Metrics (all level-space)
# ---------------------------------------------------------------------------


def qlike(y: np.ndarray, yhat: np.ndarray) -> float:
    """QLIKE = mean(y/ŷ - log(y/ŷ) - 1).  Lower is better.

    Patton (2011): most powerful loss function for variance forecasts.
    Both inputs must be positive (level-space annualized variance).
    """
    yhat = np.clip(yhat, 1e-8, None)
    ratio = y / yhat
    return float(np.mean(ratio - np.log(ratio) - 1))


def mse(y: np.ndarray, yhat: np.ndarray) -> float:
    """Mean Squared Error in level space."""
    return float(np.mean((y - yhat) ** 2))


def mae(y: np.ndarray, yhat: np.ndarray) -> float:
    """Mean Absolute Error in level space."""
    return float(np.mean(np.abs(y - yhat)))


def mape(y: np.ndarray, yhat: np.ndarray) -> float:
    """Mean Absolute Percentage Error (%).

    Guards against near-zero y by clipping denominator at 1e-4
    (annualized variance below 0.01% is effectively zero vol).
    """
    denom = np.clip(np.abs(y), 1e-4, None)
    return float(np.mean(np.abs(y - yhat) / denom) * 100)


def r2_oos(y: np.ndarray, yhat: np.ndarray, train_mean: float) -> float:
    """Out-of-sample R² per Gu/Kelly/Xiu (2020).

    R²_OOS = 1 - sum((y - ŷ)²) / sum((y - ȳ_train)²)
    Denominator uses TRAINING set mean, not test mean.
    Negative values mean the model is worse than predicting the train mean.
    """
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - train_mean) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1 - ss_res / ss_tot)


# ---------------------------------------------------------------------------
# Diebold-Mariano test
# ---------------------------------------------------------------------------


def diebold_mariano(
    y: np.ndarray,
    yhat_a: np.ndarray,
    yhat_b: np.ndarray,
    loss: str = "qlike",
    max_lag: int = 21,
) -> tuple[float, float]:
    """Diebold-Mariano test for equal predictive accuracy.

    H0: E[L(e_a) - L(e_b)] = 0
    Positive DM stat => model B is better (lower loss).

    Uses Newey-West HAC standard errors with max_lag bandwidth
    (21 = target horizon, accounts for overlapping forecast errors).

    Returns (dm_statistic, p_value) for two-sided test.
    """
    if loss == "qlike":
        yhat_a = np.clip(yhat_a, 1e-8, None)
        yhat_b = np.clip(yhat_b, 1e-8, None)
        loss_a = y / yhat_a - np.log(y / yhat_a) - 1
        loss_b = y / yhat_b - np.log(y / yhat_b) - 1
    elif loss == "mse":
        loss_a = (y - yhat_a) ** 2
        loss_b = (y - yhat_b) ** 2
    else:
        raise ValueError(f"Unknown loss: {loss}")

    d = loss_a - loss_b
    n = len(d)
    d_bar = np.mean(d)

    # Newey-West HAC variance estimator
    gamma_0 = np.mean((d - d_bar) ** 2)
    hac_var = gamma_0
    for lag in range(1, max_lag + 1):
        weight = 1 - lag / (max_lag + 1)  # Bartlett kernel
        gamma_k = np.mean((d[lag:] - d_bar) * (d[:-lag] - d_bar))
        hac_var += 2 * weight * gamma_k

    hac_var = max(hac_var, 1e-12)  # Guard against zero variance
    dm_stat = d_bar / np.sqrt(hac_var / n)
    p_value = float(2 * sp_stats.norm.sf(np.abs(dm_stat)))

    return float(dm_stat), p_value


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_all_predictions() -> pl.DataFrame:
    """Load and concatenate all prediction parquets into a single DataFrame."""
    dfs = []
    for f in sorted(PREDICTIONS_DIR.glob("*.parquet")):
        dfs.append(pl.read_parquet(f))
    return pl.concat(dfs)


def get_train_mean_rv() -> float:
    """Load train_mean_rv from scaler_stats.json."""
    with open(SPLITS_DIR / "scaler_stats.json") as f:
        raw = json.load(f)
    return float(raw["__train_mean_rv__"])


# ---------------------------------------------------------------------------
# Evaluation pipeline
# ---------------------------------------------------------------------------


def compute_metrics(preds: pl.DataFrame, train_mean: float) -> pl.DataFrame:
    """Compute all metrics for each model. Returns a tidy DataFrame."""
    rows = []
    for model_name in preds["model"].unique().sort().to_list():
        mdf = preds.filter(pl.col("model") == model_name)
        y = mdf["y_true"].to_numpy()
        yhat = mdf["y_pred"].to_numpy()

        rows.append({
            "model": model_name,
            "n": len(y),
            "QLIKE": qlike(y, yhat),
            "MSE": mse(y, yhat),
            "MAE": mae(y, yhat),
            "MAPE": mape(y, yhat),
            "R2_OOS": r2_oos(y, yhat, train_mean),
        })

    return pl.DataFrame(rows).sort("QLIKE")


def compute_dm_tests(
    preds: pl.DataFrame,
    benchmark: str = "LogHAR",
    loss: str = "qlike",
) -> pl.DataFrame:
    """Run DM tests for every model vs a benchmark on common (symbol, date) pairs.

    Returns DataFrame with columns: model, dm_stat, p_value, n_common, significant.
    """
    bench_df = preds.filter(pl.col("model") == benchmark)
    bench_keys = bench_df.select("symbol", "date")

    rows = []
    for model_name in preds["model"].unique().sort().to_list():
        if model_name == benchmark:
            continue

        model_df = preds.filter(pl.col("model") == model_name)

        # Align to common (symbol, date) pairs
        common = bench_keys.join(
            model_df.select("symbol", "date"), on=["symbol", "date"], how="inner"
        )
        bench_aligned = bench_df.join(common, on=["symbol", "date"], how="inner").sort("symbol", "date")
        model_aligned = model_df.join(common, on=["symbol", "date"], how="inner").sort("symbol", "date")

        y = bench_aligned["y_true"].to_numpy()
        yhat_bench = bench_aligned["y_pred"].to_numpy()
        yhat_model = model_aligned["y_pred"].to_numpy()

        dm_stat, p_val = diebold_mariano(y, yhat_bench, yhat_model, loss=loss)

        rows.append({
            "model": model_name,
            "vs": benchmark,
            "dm_stat": round(dm_stat, 4),
            "p_value": round(p_val, 6),
            "n_common": len(y),
            "significant_5pct": p_val < 0.05,
        })

    return pl.DataFrame(rows).sort("dm_stat", descending=True)


def run_evaluation() -> None:
    """Full evaluation pipeline. Prints results and saves to disk."""
    print("Loading predictions...")
    preds = load_all_predictions()
    train_mean = get_train_mean_rv()
    models = preds["model"].unique().sort().to_list()
    print(f"  {len(preds):,} rows, {len(models)} models: {models}")
    print(f"  train_mean_rv: {train_mean:.4f}")

    # --- Metrics table ---
    print("\n" + "=" * 80)
    print("METRICS TABLE")
    print("=" * 80)
    metrics = compute_metrics(preds, train_mean)
    # Print formatted
    print(f"\n{'Model':<12} {'n':>7} {'QLIKE':>8} {'MSE':>10} {'MAE':>8} {'MAPE%':>8} {'R²_OOS':>8}")
    print("-" * 70)
    for row in metrics.iter_rows(named=True):
        print(f"{row['model']:<12} {row['n']:>7,} {row['QLIKE']:>8.4f} {row['MSE']:>10.6f} "
              f"{row['MAE']:>8.4f} {row['MAPE']:>8.2f} {row['R2_OOS']:>8.4f}")

    # --- DM tests vs LogHAR ---
    print("\n" + "=" * 80)
    print("DIEBOLD-MARIANO TESTS (vs LogHAR, QLIKE loss, HAC lag=21)")
    print("=" * 80)
    print("Positive DM stat => model improves over LogHAR\n")
    dm_qlike = compute_dm_tests(preds, benchmark="LogHAR", loss="qlike")
    print(f"{'Model':<12} {'DM stat':>8} {'p-value':>10} {'n':>7} {'Sig 5%':>7}")
    print("-" * 50)
    for row in dm_qlike.iter_rows(named=True):
        sig = "***" if row["p_value"] < 0.01 else ("**" if row["p_value"] < 0.05 else "")
        print(f"{row['model']:<12} {row['dm_stat']:>8.4f} {row['p_value']:>10.6f} "
              f"{row['n_common']:>7,} {sig:>7}")

    # --- DM tests vs LightGBM (for LSTM comparison) ---
    print("\n" + "=" * 80)
    print("DIEBOLD-MARIANO TESTS (vs LightGBM, QLIKE loss, HAC lag=21)")
    print("=" * 80)
    dm_lgbm = compute_dm_tests(preds, benchmark="LightGBM", loss="qlike")
    print(f"{'Model':<12} {'DM stat':>8} {'p-value':>10} {'n':>7} {'Sig 5%':>7}")
    print("-" * 50)
    for row in dm_lgbm.iter_rows(named=True):
        sig = "***" if row["p_value"] < 0.01 else ("**" if row["p_value"] < 0.05 else "")
        print(f"{row['model']:<12} {row['dm_stat']:>8.4f} {row['p_value']:>10.6f} "
              f"{row['n_common']:>7,} {sig:>7}")

    # --- Save results ---
    out_dir = _PROJECT_ROOT / "data" / "processed" / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics.write_parquet(out_dir / "metrics.parquet")
    dm_qlike.write_parquet(out_dir / "dm_tests_vs_loghar.parquet")
    dm_lgbm.write_parquet(out_dir / "dm_tests_vs_lgbm.parquet")

    # Also save as readable JSON
    metrics_dict = {
        row["model"]: {k: v for k, v in row.items() if k != "model"}
        for row in metrics.iter_rows(named=True)
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics_dict, f, indent=2)

    print(f"\nSaved to {out_dir}/")
    print("  metrics.parquet, metrics.json")
    print("  dm_tests_vs_loghar.parquet, dm_tests_vs_lgbm.parquet")


if __name__ == "__main__":
    run_evaluation()
