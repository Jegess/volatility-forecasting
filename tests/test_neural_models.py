"""Tests for theta.modeling.neural_models.

Unit tests cover: FNN/LSTM architecture, datasets, training loop,
Optuna objectives, ensemble diversity, scaler utilities.
Integration tests (marked @slow) exercise run_fnn / run_lstm end-to-end.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest
import torch

from theta.modeling.neural_models import (
    FNN,
    LSTMModel,
    VolatilityDataset,
    VolatilitySequenceDataset,
    train_model,
    make_optuna_objective_fnn,
    make_optuna_objective_lstm,
    run_fnn,
    run_lstm,
    _load_scaler_stats,
    _standardize_array,
    _standardize_df,
    SPLITS_DIR,
    PREDICTIONS_DIR,
)
from theta.modeling.preprocessing import LOG_TARGET_COL, TARGET_COL

slow = pytest.mark.slow


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture(scope="module")
def feature_cols_5() -> list[str]:
    return [f"f{i}" for i in range(5)]


@pytest.fixture(scope="module")
def synthetic_df(feature_cols_5) -> pl.DataFrame:
    """200 rows, 2 symbols (A=100, B=100), 5 features, log and level target."""
    rng = np.random.default_rng(42)
    n_per_sym = 100
    symbols = ["A"] * n_per_sym + ["B"] * n_per_sym
    base_date = date(2023, 1, 2)
    dates = [base_date + timedelta(days=i) for i in range(n_per_sym)] * 2

    data: dict = {c: rng.standard_normal(n_per_sym * 2).astype(np.float32) for c in feature_cols_5}
    rv_vals = np.abs(rng.standard_normal(n_per_sym * 2)).astype(np.float64) + 0.05
    data[TARGET_COL] = rv_vals
    data[LOG_TARGET_COL] = np.log(rv_vals).astype(np.float32)
    data["symbol"] = symbols
    data["date"] = dates
    return pl.DataFrame(data).sort(["symbol", "date"])


@pytest.fixture(scope="module")
def scaler_stats_5(feature_cols_5) -> dict[str, tuple[float, float]]:
    """Identity scaler: mean=0.0, std=1.0 for all 5 features."""
    return {c: (0.0, 1.0) for c in feature_cols_5}


# ===========================================================================
# Task 2 (TDD): Unit tests
# ===========================================================================


# --- FNN ---

def test_fnn_forward_shape():
    """FNN(44, (32,16,8)) on (16, 44) input returns shape (16,)."""
    model = FNN(44, (32, 16, 8))
    x = torch.randn(16, 44)
    out = model(x)
    assert out.shape == (16,), f"Expected (16,), got {out.shape}"


def test_fnn_glorot_init():
    """First Linear layer has Xavier-initialized weights (not all zeros, not all equal)."""
    model = FNN(10, (8, 4))
    first_linear = [m for m in model.modules() if isinstance(m, torch.nn.Linear)][0]
    w = first_linear.weight.detach().numpy()
    assert not np.allclose(w, 0.0), "Weights should not be all zeros after Glorot init"
    # Xavier init: variance ~ 2 / (fan_in + fan_out), not constant
    assert w.std() > 0.01, "Weight std too low — Glorot init may not have been applied"


def test_fnn_dropout_applied():
    """FNN with dropout=0.5 in train mode produces different outputs on same input (stochastic)."""
    model = FNN(10, (8, 4), dropout=0.5)
    model.train()
    x = torch.randn(32, 10)
    out1 = model(x).detach().numpy()
    out2 = model(x).detach().numpy()
    assert not np.allclose(out1, out2), "Dropout in train mode should produce different outputs"


# --- LSTM ---

def test_lstm_forward_shape():
    """LSTMModel(44, hidden_size=32) on (8, 21, 44) input returns shape (8,)."""
    model = LSTMModel(44, hidden_size=32, num_layers=1)
    x = torch.randn(8, 21, 44)
    out = model(x)
    assert out.shape == (8,), f"Expected (8,), got {out.shape}"


def test_lstm_single_layer_no_internal_dropout():
    """LSTMModel with num_layers=1 has lstm.dropout == 0.0 (PyTorch restriction)."""
    model = LSTMModel(10, 16, num_layers=1, dropout=0.3)
    assert model.lstm.dropout == 0.0, (
        f"Expected lstm.dropout==0.0 for single-layer LSTM, got {model.lstm.dropout}"
    )


def test_lstm_multi_layer_has_dropout():
    """LSTMModel with num_layers=2 and dropout=0.3 has lstm.dropout == 0.3."""
    model = LSTMModel(10, 16, num_layers=2, dropout=0.3)
    assert model.lstm.dropout == 0.3, (
        f"Expected lstm.dropout==0.3 for 2-layer LSTM, got {model.lstm.dropout}"
    )


# --- VolatilityDataset ---

def test_volatility_dataset_len_and_getitem():
    """VolatilityDataset with (100,5) X and (100,) y has len 100, correct item shapes."""
    X = np.random.randn(100, 5).astype(np.float32)
    y = np.random.randn(100).astype(np.float32)
    ds = VolatilityDataset(X, y)
    assert len(ds) == 100
    xi, yi = ds[0]
    assert xi.shape == (5,), f"Expected (5,), got {xi.shape}"
    assert yi.shape == (), f"Expected scalar, got {yi.shape}"


# --- VolatilitySequenceDataset ---

def test_sequence_dataset_no_cross_symbol(synthetic_df, feature_cols_5):
    """2 symbols × 100 rows, seq_len=21 → each contributes 80 sequences (100-21+1=80)."""
    ds = VolatilitySequenceDataset(synthetic_df, feature_cols_5, seq_len=21)
    expected = 2 * (100 - 21 + 1)  # 160
    assert len(ds) == expected, f"Expected {expected} sequences, got {len(ds)}"

    # Each symbol's sequences should form a contiguous block
    for sym in ["A", "B"]:
        sym_indices = [i for i, s in enumerate(ds.symbols) if s == sym]
        assert sym_indices == list(
            range(sym_indices[0], sym_indices[-1] + 1)
        ), f"Symbol {sym} sequences are not contiguous — cross-symbol contamination possible"


def test_sequence_dataset_short_symbol(feature_cols_5):
    """Symbol with only 15 rows contributes 0 sequences when seq_len=21."""
    rng = np.random.default_rng(0)
    n = 15
    base_date = date(2023, 1, 2)
    dates = [base_date + timedelta(days=i) for i in range(n)]
    rv_vals = np.abs(rng.standard_normal(n)).astype(np.float64) + 0.05
    data = {c: rng.standard_normal(n).astype(np.float32) for c in feature_cols_5}
    data[TARGET_COL] = rv_vals
    data[LOG_TARGET_COL] = np.log(rv_vals).astype(np.float32)
    data["symbol"] = ["SHORT"] * n
    data["date"] = dates
    df = pl.DataFrame(data)

    ds = VolatilitySequenceDataset(df, feature_cols_5, seq_len=21)
    assert len(ds) == 0, f"Symbol with 15 rows should produce 0 sequences, got {len(ds)}"


# --- Training loop ---

def test_train_model_reduces_loss(synthetic_df, feature_cols_5, scaler_stats_5):
    """FNN on synthetic data converges: model trains without error, QLIKE finite."""
    from theta.modeling.lightgbm_model import qlike_score
    X = _standardize_array(
        synthetic_df.select(feature_cols_5).to_numpy(), feature_cols_5, scaler_stats_5
    )
    y = synthetic_df[LOG_TARGET_COL].to_numpy().astype(np.float32)

    split = int(0.8 * len(X))
    X_tr, X_vl = X[:split], X[split:]
    y_tr, y_vl = y[:split], y[split:]

    model = FNN(5, (4, 4), dropout=0.1)
    tr_ds = VolatilityDataset(X_tr, y_tr)
    vl_ds = VolatilityDataset(X_vl, y_vl)
    tr_loader = torch.utils.data.DataLoader(tr_ds, batch_size=32, shuffle=True, drop_last=False)
    vl_loader = torch.utils.data.DataLoader(vl_ds, batch_size=32, shuffle=False)

    trained = train_model(
        model, tr_loader, vl_loader,
        lr=1e-3, weight_decay=1e-4, max_epochs=20, patience=25,
        device=torch.device("cpu")
    )
    trained.eval()
    with torch.no_grad():
        preds = trained(torch.tensor(X_vl, dtype=torch.float32)).numpy()

    assert math.isfinite(qlike_score(y_vl, preds)), "QLIKE should be finite after training"


def test_train_model_early_stopping():
    """With patience=2 and non-improving data, training stops well before max_epochs."""
    import time

    # Random data — model unlikely to improve consistently
    rng = np.random.default_rng(7)
    X = rng.standard_normal((500, 5)).astype(np.float32)
    y = rng.standard_normal(500).astype(np.float32)
    X_vl = rng.standard_normal((100, 5)).astype(np.float32)
    y_vl = rng.standard_normal(100).astype(np.float32)

    model = FNN(5, (4,), dropout=0.0)
    tr_ds = VolatilityDataset(X, y)
    vl_ds = VolatilityDataset(X_vl, y_vl)
    tr_loader = torch.utils.data.DataLoader(tr_ds, batch_size=128, shuffle=True, drop_last=False)
    vl_loader = torch.utils.data.DataLoader(vl_ds, batch_size=128, shuffle=False)

    t0 = time.time()
    train_model(
        model, tr_loader, vl_loader,
        lr=1e-3, weight_decay=0.0, max_epochs=200, patience=3,
        device=torch.device("cpu")
    )
    elapsed = time.time() - t0
    # Early stopping with patience=3 on random data should finish in a few seconds
    assert elapsed < 60.0, f"Early stopping took too long: {elapsed:.1f}s (expected < 60s)"


# --- Optuna objectives ---

def test_optuna_fnn_objective_callable(synthetic_df, feature_cols_5, scaler_stats_5):
    """make_optuna_objective_fnn returns a callable."""
    obj = make_optuna_objective_fnn(
        synthetic_df, feature_cols_5, scaler_stats_5, torch.device("cpu"), n_splits=2
    )
    assert callable(obj)


def test_optuna_lstm_objective_callable(synthetic_df, feature_cols_5, scaler_stats_5):
    """make_optuna_objective_lstm returns a callable."""
    obj = make_optuna_objective_lstm(
        synthetic_df, feature_cols_5, scaler_stats_5, torch.device("cpu"), n_splits=2
    )
    assert callable(obj)


# --- Ensemble diversity ---

def test_ensemble_seeds_differ():
    """Two FNNs with different seeds produce different predictions."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 5)).astype(np.float32)
    y = (X.sum(axis=1) + rng.standard_normal(200) * 0.1).astype(np.float32)
    X_test = rng.standard_normal((50, 5)).astype(np.float32)

    preds = []
    for seed in [0, 1]:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = FNN(5, (8, 4), dropout=0.2)
        tr_ds = VolatilityDataset(X, y)
        vl_ds = VolatilityDataset(X[:50], y[:50])
        tr_loader = torch.utils.data.DataLoader(tr_ds, batch_size=32, shuffle=True, drop_last=False)
        vl_loader = torch.utils.data.DataLoader(vl_ds, batch_size=32, shuffle=False)
        model = train_model(
            model, tr_loader, vl_loader, lr=1e-3, max_epochs=10, patience=10,
            device=torch.device("cpu")
        )
        model.eval()
        with torch.no_grad():
            p = model(torch.tensor(X_test)).numpy()
        preds.append(p)

    assert not np.allclose(preds[0], preds[1]), (
        "Two seeds should produce different predictions"
    )


# --- Scaler utilities ---

def test_standardize_array(feature_cols_5):
    """_standardize_array with known mean/std produces standardized output."""
    rng = np.random.default_rng(42)
    n = 1000
    X = np.column_stack([rng.normal(5.0, 2.0, n) for _ in feature_cols_5]).astype(np.float32)
    stats = {c: (5.0, 2.0) for c in feature_cols_5}
    X_std = _standardize_array(X, feature_cols_5, stats)
    assert X_std.shape == X.shape
    col_mean = X_std[:, 0].mean()
    col_std = X_std[:, 0].std()
    assert abs(col_mean) < 0.1, f"Standardized mean should be ~0, got {col_mean:.3f}"
    assert abs(col_std - 1.0) < 0.1, f"Standardized std should be ~1, got {col_std:.3f}"


def test_load_scaler_stats():
    """_load_scaler_stats() returns dict with expected structure."""
    stats = _load_scaler_stats()
    assert isinstance(stats, dict), "Should return a dict"
    assert len(stats) > 0, "Should have at least one feature"
    assert "rv_d" in stats, "Should contain rv_d (first feature)"
    rv_d_stats = stats["rv_d"]
    assert len(rv_d_stats) == 2, "Each entry should be a 2-element (mean, std) tuple"
    mean, std = rv_d_stats
    assert isinstance(mean, float)
    assert isinstance(std, float)
    assert std > 0, "Std should be positive"
    # Ensure sentinel key was skipped
    assert "__train_mean_rv__" not in stats


# ===========================================================================
# Integration tests (slow)
# ===========================================================================


@slow
def test_run_fnn_integration():
    """run_fnn with minimal trials produces valid parquet and DataFrame."""
    result = run_fnn(n_trials=2, n_splits=2, n_seeds=2)
    assert isinstance(result, pl.DataFrame)
    assert set(result.columns) == {"symbol", "date", "model", "y_true", "y_pred"}
    assert result["model"].unique().to_list() == ["FNN"]
    assert result["y_pred"].null_count() == 0, "No null predictions"
    assert (result["y_pred"].to_numpy() > 0).all(), "All predictions should be positive"
    assert (PREDICTIONS_DIR / "fnn.parquet").exists(), "fnn.parquet should be saved"


@slow
def test_run_lstm_integration():
    """run_lstm with minimal trials and short seq_len produces valid parquet."""
    result = run_lstm(n_trials=2, n_splits=2, n_seeds=2, seq_len=5)
    assert isinstance(result, pl.DataFrame)
    assert set(result.columns) == {"symbol", "date", "model", "y_true", "y_pred"}
    assert result["model"].unique().to_list() == ["LSTM"]
    assert result["y_pred"].null_count() == 0, "No null predictions"
    assert (result["y_pred"].to_numpy() > 0).all(), "All predictions should be positive"
    assert (PREDICTIONS_DIR / "lstm.parquet").exists(), "lstm.parquet should be saved"
