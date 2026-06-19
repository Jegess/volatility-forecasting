"""Walk-forward validation for volatility forecasting models.

Expanding-window walk-forward with quarterly test windows.
Validates whether model advantages (esp. LSTM 38% over LogHAR)
hold across different time periods and market regimes.

All models (baselines, LightGBM, LSTM) retrained each window.
LightGBM and LSTM use fixed HP from original Optuna search.

Does NOT modify existing models, predictions, or artifacts.
Output: data/processed/evaluation/walk_forward/
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from theta.modeling.preprocessing import (
    TARGET_COL,
    LOG_TARGET_COL,
    add_log_target,
    fit_scaler,
    get_feature_cols,
)
from theta.modeling.baselines import (
    predict_har,
    predict_loghar,
    predict_shar,
    predict_harq,
    predict_ar5,
    predict_levhar,
)
from theta.modeling.lightgbm_model import (
    train_final_model,
    predict_lgbm,
)
from theta.modeling.evaluation import (
    qlike,
    mse,
    r2_oos,
    diebold_mariano,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PANEL_PATH = _PROJECT_ROOT / "data" / "processed" / "panel.parquet"
MODELS_DIR = _PROJECT_ROOT / "data" / "processed" / "models"
SPLITS_DIR = _PROJECT_ROOT / "data" / "processed" / "splits"
WF_OUTPUT_DIR = _PROJECT_ROOT / "data" / "processed" / "evaluation" / "walk_forward"


# ---------------------------------------------------------------------------
# Window generation
# ---------------------------------------------------------------------------


def generate_windows(
    all_dates: np.ndarray,
    min_train_days: int = 252,
    test_days: int = 63,
    step_days: int = 63,
    embargo_days: int = 21,
) -> list[dict]:
    """Generate expanding-window walk-forward schedule.

    Returns list of dicts with:
        window_id, train_start, train_end, test_start, test_end,
        n_train_dates, n_test_dates
    """
    n = len(all_dates)
    windows = []
    window_id = 0
    train_end_idx = min_train_days - 1

    while True:
        test_start_idx = train_end_idx + 1 + embargo_days
        test_end_idx = test_start_idx + test_days - 1

        if test_end_idx >= n:
            break

        windows.append({
            "window_id": window_id,
            "train_start": all_dates[0],
            "train_end": all_dates[train_end_idx],
            "test_start": all_dates[test_start_idx],
            "test_end": all_dates[test_end_idx],
            "n_train_dates": train_end_idx + 1,
            "n_test_dates": test_days,
        })
        window_id += 1
        train_end_idx += step_days

    return windows


# ---------------------------------------------------------------------------
# Per-window model runners
# ---------------------------------------------------------------------------


def run_baselines_window(
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    window_id: int,
) -> pl.DataFrame:
    """Run 6 HAR-family baselines on one window. GARCH excluded."""
    predictors = [
        ("HAR", predict_har),
        ("LogHAR", predict_loghar),
        ("SHAR", predict_shar),
        ("HARQ", predict_harq),
        ("AR5", predict_ar5),
        ("LevHAR", predict_levhar),
    ]
    parts = []
    for name, fn in predictors:
        try:
            preds = fn(train_df, test_df)
            parts.append(preds.with_columns(pl.lit(window_id).alias("window_id")))
        except Exception as e:
            print(f"  Warning: {name} failed on window {window_id}: {e}")
    if not parts:
        return pl.DataFrame()
    return pl.concat(parts)


def run_lgbm_window(
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    feature_cols: list[str],
    best_params: dict,
    window_id: int,
) -> pl.DataFrame:
    """Train LightGBM with fixed HP on one window and predict."""
    params = {**best_params}  # copy to avoid mutation
    model = train_final_model(train_df, feature_cols, params)
    y_pred = predict_lgbm(model, test_df, feature_cols)
    y_true = test_df[TARGET_COL].to_numpy()

    return pl.DataFrame({
        "symbol": test_df["symbol"].to_list(),
        "date": test_df["date"].to_list(),
        "model": ["LightGBM"] * len(test_df),
        "y_true": y_true,
        "y_pred": y_pred,
        "window_id": [window_id] * len(test_df),
    })


def run_lstm_retrained_window(
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    feature_cols: list[str],
    lstm_params: dict,
    window_id: int,
    n_seeds: int = 5,
    seq_len: int = 21,
) -> pl.DataFrame:
    """Train fresh LSTM ensemble on this window's training data, predict test.

    Uses fixed HP from Optuna (no re-search). Trains n_seeds models with
    different random seeds, averages predictions in log space.
    """
    import torch
    from torch.utils.data import DataLoader
    from theta.modeling.neural_models import (
        LSTMModel,
        VolatilitySequenceDataset,
        _predict_batched,
        _standardize_df,
        train_model,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Compute scaler stats from THIS window's training data
    window_scaler = {}
    for col in feature_cols:
        arr = train_df[col].to_numpy()
        window_scaler[col] = (float(np.mean(arr)), float(np.std(arr)))

    # Split training data into train/val (last 20% of dates for early stopping)
    train_dates = train_df["date"].unique().sort()
    n_dates = len(train_dates)
    val_cutoff = train_dates[int(n_dates * 0.8)]
    sub_train = train_df.filter(pl.col("date") <= val_cutoff)
    sub_val = train_df.filter(pl.col("date") > val_cutoff)

    # Standardize
    sub_train_scaled = _standardize_df(sub_train, feature_cols, window_scaler)
    sub_val_scaled = _standardize_df(sub_val, feature_cols, window_scaler)

    # Build datasets
    train_ds = VolatilitySequenceDataset(sub_train_scaled, feature_cols, LOG_TARGET_COL, seq_len)
    val_ds = VolatilitySequenceDataset(sub_val_scaled, feature_cols, LOG_TARGET_COL, seq_len)

    if len(train_ds) == 0 or len(val_ds) == 0:
        return pl.DataFrame()

    batch_size = lstm_params.get("batch_size", 256)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=True)

    # Train n_seeds models
    input_dim = len(feature_cols)
    models = []
    for seed in range(n_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = LSTMModel(
            input_dim=input_dim,
            hidden_size=lstm_params["hidden_size"],
            num_layers=lstm_params["num_layers"],
            dropout=lstm_params["dropout"],
        )
        model = train_model(
            model, train_loader, val_loader,
            lr=lstm_params["lr"],
            weight_decay=lstm_params["weight_decay"],
            max_epochs=50,
            patience=8,
            device=device,
        )
        models.append(model)

    # Build test dataset: prepend warmup from train for sequence context
    warmup_start = train_dates[-seq_len + 1] if len(train_dates) >= seq_len - 1 else train_dates[0]
    warmup_df = train_df.filter(pl.col("date") >= warmup_start)
    combined_df = pl.concat([warmup_df, test_df])
    combined_scaled = _standardize_df(combined_df, feature_cols, window_scaler)
    test_ds = VolatilitySequenceDataset(combined_scaled, feature_cols, LOG_TARGET_COL, seq_len)

    # Ensemble predict
    all_preds_log = []
    for model in models:
        preds_log = _predict_batched(model, test_ds, device)
        all_preds_log.append(preds_log)

    avg_log = np.mean(all_preds_log, axis=0)
    y_pred_level = np.clip(np.exp(avg_log), 1e-8, None)
    y_true_level = np.exp(test_ds.y.numpy())

    # Filter to test dates only
    test_date_set = set(test_df["date"].to_list())
    mask = [d in test_date_set for d in test_ds.dates]
    mask_arr = np.array(mask)

    if mask_arr.sum() == 0:
        return pl.DataFrame()

    return pl.DataFrame({
        "symbol": [s for s, m in zip(test_ds.symbols, mask) if m],
        "date": [d for d, m in zip(test_ds.dates, mask) if m],
        "model": ["LSTM"] * int(mask_arr.sum()),
        "y_true": y_true_level[mask_arr],
        "y_pred": y_pred_level[mask_arr],
        "window_id": [window_id] * int(mask_arr.sum()),
    })


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_window_metrics(
    preds_df: pl.DataFrame,
    train_mean_rv: float,
) -> pl.DataFrame:
    """Compute QLIKE, MSE, R2_OOS per model for one window."""
    rows = []
    for model_name in preds_df["model"].unique().sort().to_list():
        model_preds = preds_df.filter(pl.col("model") == model_name)
        y = model_preds["y_true"].to_numpy()
        yhat = model_preds["y_pred"].to_numpy()

        rows.append({
            "model": model_name,
            "QLIKE": qlike(y, yhat),
            "MSE": mse(y, yhat),
            "R2_OOS": r2_oos(y, yhat, train_mean_rv),
            "n": len(y),
        })

    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate_results(
    all_preds: pl.DataFrame,
    all_metrics: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Compute summary metrics and DM tests across all windows.

    Returns (summary_df, dm_tests_df).
    """
    # Summary: mean/std QLIKE per model
    summary = all_metrics.group_by("model").agg(
        pl.col("QLIKE").mean().alias("mean_QLIKE"),
        pl.col("QLIKE").std().alias("std_QLIKE"),
        pl.col("MSE").mean().alias("mean_MSE"),
        pl.col("R2_OOS").mean().alias("mean_R2_OOS"),
        pl.col("QLIKE").count().alias("n_windows"),
    ).sort("mean_QLIKE")

    # DM tests: all models vs LogHAR on concatenated predictions
    models = all_preds["model"].unique().sort().to_list()
    benchmark = "LogHAR"
    dm_rows = []

    if benchmark in models:
        bench_preds = all_preds.filter(pl.col("model") == benchmark)
        for model_name in models:
            if model_name == benchmark:
                continue
            model_preds = all_preds.filter(pl.col("model") == model_name)

            # Join on (symbol, date, window_id) for matched comparison
            joined = model_preds.join(
                bench_preds.select(["symbol", "date", "window_id", "y_true", "y_pred"]),
                on=["symbol", "date", "window_id"],
                suffix="_bench",
            )
            if len(joined) == 0:
                continue

            y = joined["y_true"].to_numpy()
            yhat_bench = joined["y_pred_bench"].to_numpy()
            yhat_model = joined["y_pred"].to_numpy()

            dm_stat, p_val = diebold_mariano(y, yhat_bench, yhat_model)
            dm_rows.append({
                "model": model_name,
                "benchmark": benchmark,
                "dm_stat": dm_stat,
                "p_value": p_val,
                "n": len(joined),
            })

    dm_df = pl.DataFrame(dm_rows) if dm_rows else pl.DataFrame()
    return summary, dm_df


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def run_walk_forward(
    skip_nn: bool = False,
    min_train_days: int = 252,
    test_days: int = 63,
    step_days: int = 63,
    embargo_days: int = 21,
) -> None:
    """Run full walk-forward validation and save results."""
    print("=" * 60)
    print("Walk-Forward Validation")
    print("=" * 60)

    # Load panel
    print("\nLoading panel...")
    panel = pl.read_parquet(PANEL_PATH)
    panel = add_log_target(panel)
    feature_cols = get_feature_cols(panel)
    print(f"  Panel: {len(panel):,} rows, {len(feature_cols)} features")

    # Generate windows
    all_dates = panel["date"].unique().sort().to_numpy()
    windows = generate_windows(all_dates, min_train_days, test_days, step_days, embargo_days)
    print(f"\n  Generated {len(windows)} windows:")
    for w in windows:
        print(f"    Window {w['window_id']}: "
              f"train {w['train_start']}..{w['train_end']} ({w['n_train_dates']}d) | "
              f"test {w['test_start']}..{w['test_end']} ({w['n_test_dates']}d)")

    # Load LightGBM best params
    lgbm_params_path = MODELS_DIR / "lgbm_best_params.json"
    with open(lgbm_params_path) as f:
        lgbm_best_params = json.load(f)
    print(f"\n  LightGBM HP loaded from {lgbm_params_path}")

    # Load LSTM best params (fixed HP, retrain weights each window)
    lstm_params_path = MODELS_DIR / "lstm_best_params.json"
    if not skip_nn and lstm_params_path.exists():
        with open(lstm_params_path) as f:
            lstm_best_params = json.load(f)
        print(f"  LSTM HP loaded from {lstm_params_path}")
    else:
        lstm_best_params = None

    # Run windows
    all_preds_parts: list[pl.DataFrame] = []
    all_metrics_parts: list[pl.DataFrame] = []

    for w in windows:
        wid = w["window_id"]
        print(f"\n{'-' * 50}")
        print(f"Window {wid}: test {w['test_start']}..{w['test_end']}")
        print(f"{'-' * 50}")

        # Split panel for this window
        train_df = panel.filter(pl.col("date") <= w["train_end"])
        test_df = panel.filter(
            (pl.col("date") >= w["test_start"]) & (pl.col("date") <= w["test_end"])
        )
        train_mean_rv = float(train_df[TARGET_COL].mean())
        print(f"  train: {len(train_df):,} rows, test: {len(test_df):,} rows, "
              f"train_mean_rv: {train_mean_rv:.4f}")

        window_preds = []

        # 1. Baselines
        print("  Running baselines...")
        baseline_preds = run_baselines_window(train_df, test_df, wid)
        if len(baseline_preds) > 0:
            window_preds.append(baseline_preds)
            n_models = baseline_preds["model"].n_unique()
            print(f"    {n_models} baselines done")

        # 2. LightGBM
        print("  Running LightGBM (fixed HP)...")
        lgbm_preds = run_lgbm_window(
            train_df, test_df, feature_cols, lgbm_best_params, wid
        )
        window_preds.append(lgbm_preds)
        print(f"    LightGBM done ({len(lgbm_preds):,} rows)")

        # 3. LSTM (retrained each window with fixed HP)
        if not skip_nn and lstm_best_params is not None:
            print("  Training LSTM (fixed HP, 5 seeds)...")
            lstm_preds = run_lstm_retrained_window(
                train_df, test_df, feature_cols, lstm_best_params, wid
            )
            if len(lstm_preds) > 0:
                window_preds.append(lstm_preds)
                print(f"    LSTM done ({len(lstm_preds):,} rows)")

        # Combine and compute metrics
        if window_preds:
            # Align schemas (baselines Float64, LSTM Float32; dict-built Int32 vs lit Int64)
            common_schema = {
                "symbol": pl.Utf8, "date": pl.Date, "model": pl.Utf8,
                "y_true": pl.Float64, "y_pred": pl.Float64, "window_id": pl.Int64,
            }
            preds_df = pl.concat(
                [p.cast(common_schema) for p in window_preds]
            )
            metrics_df = compute_window_metrics(preds_df, train_mean_rv)
            metrics_df = metrics_df.with_columns(
                pl.lit(wid).alias("window_id"),
                pl.lit(str(w["test_start"])).alias("test_start"),
                pl.lit(str(w["test_end"])).alias("test_end"),
                pl.lit(w["n_train_dates"]).alias("n_train_dates"),
            )
            all_preds_parts.append(preds_df)
            all_metrics_parts.append(metrics_df)

            # Print window summary
            print(f"  Window {wid} results:")
            for row in metrics_df.iter_rows(named=True):
                print(f"    {row['model']:>10s}: QLIKE={row['QLIKE']:.4f}, "
                      f"R²_OOS={row['R2_OOS']:.1%}, n={row['n']}")

    # Aggregate
    print(f"\n{'=' * 60}")
    print("Aggregating results...")
    all_preds = pl.concat(all_preds_parts)
    all_metrics = pl.concat(all_metrics_parts)

    summary, dm_tests = aggregate_results(all_preds, all_metrics)

    print("\nSummary (mean across windows):")
    for row in summary.iter_rows(named=True):
        print(f"  {row['model']:>10s}: QLIKE={row['mean_QLIKE']:.4f} "
              f"(±{row['std_QLIKE']:.4f}), "
              f"R²_OOS={row['mean_R2_OOS']:.1%}, "
              f"windows={row['n_windows']}")

    if len(dm_tests) > 0:
        print("\nDM tests vs LogHAR (concatenated):")
        for row in dm_tests.iter_rows(named=True):
            sig = "***" if row["p_value"] < 0.001 else "**" if row["p_value"] < 0.01 else "*" if row["p_value"] < 0.05 else ""
            print(f"  {row['model']:>10s}: DM={row['dm_stat']:+.2f}, "
                  f"p={row['p_value']:.4f} {sig}")

    # Save
    WF_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Window schedule
    schedule = []
    for w in windows:
        schedule.append({
            k: str(v) if isinstance(v, (date, np.datetime64)) else v
            for k, v in w.items()
        })
    with open(WF_OUTPUT_DIR / "window_schedule.json", "w") as f:
        json.dump(schedule, f, indent=2, default=str)

    all_preds.write_parquet(WF_OUTPUT_DIR / "all_predictions.parquet")
    all_metrics.write_parquet(WF_OUTPUT_DIR / "per_window_metrics.parquet")
    summary.write_parquet(WF_OUTPUT_DIR / "summary_metrics.parquet")
    if len(dm_tests) > 0:
        dm_tests.write_parquet(WF_OUTPUT_DIR / "dm_tests_wf.parquet")

    print(f"\nResults saved to {WF_OUTPUT_DIR}")
    print("Done.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    skip = "--skip-nn" in sys.argv
    run_walk_forward(skip_nn=skip)
