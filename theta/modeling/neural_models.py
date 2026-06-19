"""PyTorch neural network models for volatility forecasting.

Implements FNN (Feed-Forward Network) and LSTM models with:
- Custom QLIKE-aligned training via MSE loss on log-space targets
- Shared training loop with early stopping and gradient clipping
- Optuna HP tuning objectives using purged k-fold CV
- Seed ensemble with averaged log-space predictions
- Orchestrator functions run_fnn() and run_lstm() mirroring run_lgbm()

Output schema matches baselines.parquet and lgbm.parquet:
    {symbol, date, model, y_true, y_pred}  (both in LEVEL space)
    To compute QLIKE from saved parquet, log-transform both columns first.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import optuna
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from theta.modeling.preprocessing import (
    get_feature_cols,
    purged_kfold,
    LOG_TARGET_COL,
    TARGET_COL,
)
from theta.modeling.lightgbm_model import qlike_score

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SPLITS_DIR = _PROJECT_ROOT / "data" / "processed" / "splits"
PREDICTIONS_DIR = _PROJECT_ROOT / "data" / "processed" / "predictions"
MODELS_DIR = _PROJECT_ROOT / "data" / "processed" / "models"


# ---------------------------------------------------------------------------
# Model architectures
# ---------------------------------------------------------------------------


class FNN(nn.Module):
    """Feed-forward neural network for volatility forecasting.

    Architecture: per Gu/Kelly/Xiu (2020) — 3 hidden layers 32->16->8
    with BatchNorm, ReLU, Dropout after each hidden layer.
    Glorot (Xavier) initialization on all Linear weights.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] = (32, 16, 8),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_d = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_d, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_d = h
        layers.append(nn.Linear(in_d, 1))
        self.net = nn.Sequential(*layers)
        self._glorot_init()

    def _glorot_init(self) -> None:
        """Apply Xavier normal init to all Linear weights; zero out biases."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class LSTMModel(nn.Module):
    """LSTM model for volatility forecasting using sliding-window sequences.

    Takes (batch, seq_len, input_dim) input and returns (batch,) log-RV.
    Dropout is applied to LSTM inter-layer connections only when num_layers>1
    (PyTorch restriction), plus a standalone Dropout after the last LSTM step.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        # PyTorch LSTM only applies dropout between stacked layers
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)
        nn.init.xavier_normal_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(self.dropout(last)).squeeze(-1)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


class VolatilityDataset(Dataset):
    """Flat tabular dataset for FNN: each sample is one observation."""

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class VolatilitySequenceDataset(Dataset):
    """Sliding-window dataset for LSTM.

    Builds sequences PER SYMBOL to prevent cross-symbol contamination.
    Symbols with fewer rows than seq_len contribute zero sequences.
    """

    def __init__(
        self,
        df: pl.DataFrame,
        feature_cols: list[str],
        target_col: str = LOG_TARGET_COL,
        seq_len: int = 21,
    ) -> None:
        sequences: list[np.ndarray] = []
        targets: list[float] = []
        symbols: list[str] = []
        dates: list = []

        for symbol in df["symbol"].unique().sort().to_list():
            sym_df = df.filter(pl.col("symbol") == symbol).sort("date")
            X_sym = sym_df.select(feature_cols).to_numpy().astype(np.float32)
            y_sym = sym_df[target_col].to_numpy().astype(np.float32)
            dates_sym = sym_df["date"].to_list()

            for i in range(seq_len - 1, len(X_sym)):
                sequences.append(X_sym[i - seq_len + 1 : i + 1])
                targets.append(y_sym[i])
                symbols.append(symbol)
                dates.append(dates_sym[i])

        if sequences:
            self.X = torch.tensor(np.array(sequences), dtype=torch.float32)
            self.y = torch.tensor(np.array(targets), dtype=torch.float32)
        else:
            # Edge case: no symbol has enough rows
            n_features = len(feature_cols)
            self.X = torch.zeros((0, seq_len, n_features), dtype=torch.float32)
            self.y = torch.zeros((0,), dtype=torch.float32)

        self.symbols = symbols
        self.dates = dates

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


# ---------------------------------------------------------------------------
# Scaler utilities
# ---------------------------------------------------------------------------


def _load_scaler_stats() -> dict[str, tuple[float, float]]:
    """Load scaler_stats.json and return {feature: (mean, std)} dict.

    Skips the __train_mean_rv__ sentinel key.
    """
    with open(SPLITS_DIR / "scaler_stats.json") as f:
        raw = json.load(f)
    return {
        k: (float(v[0]), float(v[1]))
        for k, v in raw.items()
        if not k.startswith("__")
    }


def _standardize_array(
    X: np.ndarray,
    feature_cols: list[str],
    stats: dict[str, tuple[float, float]],
) -> np.ndarray:
    """Standardize feature matrix using per-feature (mean, std)."""
    means = np.array([stats[c][0] for c in feature_cols], dtype=np.float32)
    stds = np.array([stats[c][1] for c in feature_cols], dtype=np.float32)
    return ((X - means) / stds).astype(np.float32)


def _standardize_df(
    df: pl.DataFrame,
    feature_cols: list[str],
    stats: dict[str, tuple[float, float]],
) -> pl.DataFrame:
    """Return a new DataFrame with feature columns standardized."""
    exprs = [
        ((pl.col(c) - stats[c][0]) / stats[c][1]).alias(c)
        for c in feature_cols
        if c in stats
    ]
    return df.with_columns(exprs)


# ---------------------------------------------------------------------------
# Batched inference (avoids CUDA OOM on large sequence datasets)
# ---------------------------------------------------------------------------


@torch.no_grad()
def _predict_batched(
    model: nn.Module,
    dataset: Dataset,
    device: torch.device,
    batch_size: int = 512,
) -> np.ndarray:
    """Run inference in batches to avoid GPU OOM on large datasets."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    preds: list[np.ndarray] = []
    for X_batch, _ in loader:
        preds.append(model(X_batch.to(device)).cpu().numpy())
    return np.concatenate(preds)


# ---------------------------------------------------------------------------
# Shared training loop
# ---------------------------------------------------------------------------


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    max_epochs: int = 50,
    patience: int = 8,
    clip_norm: float = 1.0,
    device: torch.device = torch.device("cpu"),
) -> nn.Module:
    """Train model with AdamW, MSE loss, early stopping, and grad clipping.

    Validation QLIKE (log-space) is used as the early-stopping criterion.
    Best model weights are restored before returning.
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    best_val_qlike = float("inf")
    best_state: dict = {}
    epochs_without_improvement = 0

    for _epoch in range(max_epochs):
        # --- Train ---
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()

        # --- Validate ---
        model.eval()
        val_preds: list[np.ndarray] = []
        val_targets: list[np.ndarray] = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                preds = model(X_batch).cpu().numpy()
                val_preds.append(preds)
                val_targets.append(y_batch.numpy())

        val_preds_arr = np.concatenate(val_preds)
        val_targets_arr = np.concatenate(val_targets)
        val_qlike = qlike_score(val_targets_arr, val_preds_arr)

        # Guard against inf/nan from early-training garbage predictions
        if not np.isfinite(val_qlike):
            val_qlike = float("inf")

        if val_qlike < best_val_qlike:
            best_val_qlike = val_qlike
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    # Restore best weights
    if best_state:
        model.load_state_dict(best_state)

    return model


# ---------------------------------------------------------------------------
# Optuna objectives
# ---------------------------------------------------------------------------


def make_optuna_objective_fnn(
    train_df: pl.DataFrame,
    feature_cols: list[str],
    scaler_stats: dict[str, tuple[float, float]],
    device: torch.device,
    n_splits: int = 3,
):
    """Return Optuna objective closure for FNN HP tuning via purged k-fold CV.

    Fold datasets are pre-built once (data is identical across trials,
    only HPs change). This avoids rebuilding n_trials * n_splits times.
    """
    fold_data: list[tuple[VolatilityDataset, VolatilityDataset, np.ndarray]] = []
    for fold_train, fold_val in purged_kfold(train_df, n_splits=n_splits, embargo_days=21):
        X_ft = _standardize_array(
            fold_train.select(feature_cols).to_numpy(), feature_cols, scaler_stats
        )
        y_ft = fold_train[LOG_TARGET_COL].to_numpy().astype(np.float32)
        X_fv = _standardize_array(
            fold_val.select(feature_cols).to_numpy(), feature_cols, scaler_stats
        )
        y_fv = fold_val[LOG_TARGET_COL].to_numpy().astype(np.float32)

        fold_data.append((VolatilityDataset(X_ft, y_ft), VolatilityDataset(X_fv, y_fv), y_fv))

    def objective(trial: optuna.Trial) -> float:
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)

        fold_qlikes: list[float] = []
        for train_ds, val_ds, y_fv in fold_data:
            train_loader = DataLoader(
                train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
                pin_memory=True,
            )
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                    pin_memory=True)

            model = FNN(len(feature_cols), dropout=dropout)
            model = train_model(
                model, train_loader, val_loader, lr=lr, weight_decay=weight_decay, device=device
            )

            model.eval()
            y_pred = _predict_batched(model, val_ds, device, batch_size=batch_size)
            fold_qlikes.append(qlike_score(y_fv, y_pred))

        return float(np.mean(fold_qlikes))

    return objective


def make_optuna_objective_lstm(
    train_df: pl.DataFrame,
    feature_cols: list[str],
    scaler_stats: dict[str, tuple[float, float]],
    device: torch.device,
    n_splits: int = 3,
    seq_len: int = 21,
):
    """Return Optuna objective closure for LSTM HP tuning via purged k-fold CV.

    Standardization and sequence building are done once per fold upfront.
    Across trials only model weights and DataLoader iteration order change,
    so the cached datasets are safe to reuse (read-only tensors).
    """
    # Pre-standardize full training set once
    train_scaled = _standardize_df(train_df, feature_cols, scaler_stats)

    # Pre-build sequence datasets per fold
    fold_data: list[tuple[VolatilitySequenceDataset, VolatilitySequenceDataset]] = []
    print("  Pre-building LSTM fold datasets...")
    for i, (fold_train, fold_val) in enumerate(
        purged_kfold(train_df, n_splits=n_splits, embargo_days=21)
    ):
        t0 = time.time()
        train_dates = set(fold_train["date"].to_list())
        val_dates = set(fold_val["date"].to_list())

        fold_train_scaled = train_scaled.filter(pl.col("date").is_in(list(train_dates)))
        fold_val_scaled = train_scaled.filter(pl.col("date").is_in(list(val_dates)))

        train_ds = VolatilitySequenceDataset(fold_train_scaled, feature_cols, seq_len=seq_len)
        val_ds = VolatilitySequenceDataset(fold_val_scaled, feature_cols, seq_len=seq_len)
        fold_data.append((train_ds, val_ds))
        print(f"    Fold {i}: train={len(train_ds):,} val={len(val_ds):,} ({time.time()-t0:.1f}s)")

    def objective(trial: optuna.Trial) -> float:
        hidden_size = trial.suggest_int("hidden_size", 32, 128)
        num_layers = trial.suggest_int("num_layers", 1, 2)
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)

        fold_qlikes: list[float] = []
        for train_ds, val_ds in fold_data:
            if len(train_ds) == 0 or len(val_ds) == 0:
                continue

            train_loader = DataLoader(
                train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
                pin_memory=True,
            )
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                    pin_memory=True)

            model = LSTMModel(
                len(feature_cols),
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
            )
            model = train_model(
                model, train_loader, val_loader, lr=lr, weight_decay=weight_decay, device=device
            )

            model.eval()
            y_pred = _predict_batched(model, val_ds, device, batch_size=batch_size)
            y_fv = val_ds.y.numpy()
            fold_qlikes.append(qlike_score(y_fv, y_pred))
            torch.cuda.empty_cache()

        if not fold_qlikes:
            return float("inf")
        return float(np.mean(fold_qlikes))

    return objective


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_ensemble(
    path: Path,
    device: torch.device | None = None,
) -> list[nn.Module]:
    """Load a saved ensemble of FNN or LSTM models from a .pt checkpoint.

    Returns a list of models in eval mode, ready for inference.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(path, map_location=device, weights_only=False)
    models: list[nn.Module] = []
    for state in ckpt["seed_states"]:
        if ckpt["model_class"] == "FNN":
            model = FNN(
                ckpt["n_features"],
                dropout=ckpt["best_params"].get("dropout", 0.2),
            )
        elif ckpt["model_class"] == "LSTMModel":
            model = LSTMModel(
                ckpt["n_features"],
                hidden_size=ckpt["best_params"].get("hidden_size", 32),
                num_layers=ckpt["best_params"].get("num_layers", 1),
                dropout=ckpt["best_params"].get("dropout", 0.2),
            )
        else:
            raise ValueError(f"Unknown model class: {ckpt['model_class']}")
        model.load_state_dict(state)
        model.to(device).eval()
        models.append(model)
    return models


# ---------------------------------------------------------------------------
# Orchestrators
# ---------------------------------------------------------------------------


def run_fnn(
    n_trials: int = 10,
    n_splits: int = 3,
    n_seeds: int = 5,
) -> pl.DataFrame:
    """Full FNN pipeline: HP tuning -> seed ensemble -> predictions.

    1. Load train/test from SPLITS_DIR.
    2. Tune HP via Optuna with purged k-fold CV.
    3. Train n_seeds FNNs on full train set, average log-space predictions.
    4. Save fnn.parquet to PREDICTIONS_DIR.
    Returns predictions DataFrame.
    """
    # Load data
    train = pl.read_parquet(SPLITS_DIR / "train.parquet")
    test = pl.read_parquet(SPLITS_DIR / "test.parquet")
    feature_cols = get_feature_cols(train)
    print(f"Train: {len(train):,} rows | Test: {len(test):,} rows | Features: {len(feature_cols)}")

    scaler_stats = _load_scaler_stats()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Optuna tuning
    print(f"\nOptuna: {n_trials} trials x {n_splits} folds")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", study_name="fnn_qlike")

    def _fnn_trial_cb(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        print(f"  Trial {trial.number + 1}/{n_trials}: "
              f"QLIKE={trial.value:.4f} ({trial.duration.total_seconds():.1f}s)")

    study.optimize(
        make_optuna_objective_fnn(train, feature_cols, scaler_stats, device, n_splits),
        n_trials=n_trials,
        callbacks=[_fnn_trial_cb],
    )
    best_params = dict(study.best_params)
    print(f"\nBest trial QLIKE: {study.best_value:.4f}")
    print(f"Best params: {json.dumps(best_params, indent=2)}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODELS_DIR / "fnn_best_params.json", "w") as f:
        json.dump(best_params, f, indent=2)

    # Standardize full train and test
    X_train = _standardize_array(
        train.select(feature_cols).to_numpy(), feature_cols, scaler_stats
    )
    y_train = train[LOG_TARGET_COL].to_numpy().astype(np.float32)
    X_test = _standardize_array(
        test.select(feature_cols).to_numpy(), feature_cols, scaler_stats
    )

    # Val split for early stopping: last 10% of train dates
    train_dates = train["date"].unique().sort()
    val_cutoff = train_dates[int(len(train_dates) * 0.9)]
    train_mask = (train["date"] <= val_cutoff).to_numpy()
    val_mask = ~train_mask

    X_tr = X_train[train_mask]
    y_tr = y_train[train_mask]
    X_vl = X_train[val_mask]
    y_vl = y_train[val_mask]

    dropout = best_params.get("dropout", 0.2)
    lr = best_params.get("lr", 1e-3)
    batch_size = best_params.get("batch_size", 128)
    weight_decay = best_params.get("weight_decay", 1e-4)

    # Build datasets once (identical across seeds — only model init differs)
    tr_ds = VolatilityDataset(X_tr, y_tr)
    vl_ds = VolatilityDataset(X_vl, y_vl)

    # Seed ensemble
    print(f"\nSeed ensemble ({n_seeds} seeds):")
    y_test_log = test[LOG_TARGET_COL].to_numpy()
    all_seed_preds: list[np.ndarray] = []
    seed_states: list[dict] = []
    for seed in range(n_seeds):
        t0 = time.time()
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = FNN(len(feature_cols), dropout=dropout)
        tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,
                               drop_last=True, pin_memory=True)
        vl_loader = DataLoader(vl_ds, batch_size=batch_size, shuffle=False,
                               pin_memory=True)

        model = train_model(
            model, tr_loader, vl_loader, lr=lr, weight_decay=weight_decay, device=device
        )
        model.eval()
        with torch.no_grad():
            X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
            preds = model(X_test_t).cpu().numpy()
        all_seed_preds.append(preds)
        seed_states.append(model.cpu().state_dict())
        qlike = qlike_score(y_test_log, preds)
        print(f"  Seed {seed}: QLIKE = {qlike:.4f} ({time.time() - t0:.1f}s)")

    mean_log_pred = np.mean(all_seed_preds, axis=0)
    y_pred_level = np.clip(np.exp(mean_log_pred), 1e-8, None)

    # Save model weights
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_class": "FNN",
            "n_features": len(feature_cols),
            "best_params": best_params,
            "seed_states": seed_states,
        },
        MODELS_DIR / "fnn_ensemble.pt",
    )
    print(f"Saved FNN ensemble ({n_seeds} seeds) to {MODELS_DIR / 'fnn_ensemble.pt'}")

    # Build output DataFrame (y_true and y_pred in LEVEL space)
    preds_df = pl.DataFrame({
        "symbol": test["symbol"].to_numpy(),
        "date": test["date"].to_numpy(),
        "model": ["FNN"] * len(test),
        "y_true": test[TARGET_COL].to_numpy().astype(np.float64),
        "y_pred": y_pred_level.astype(np.float64),
    })

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    preds_df.write_parquet(PREDICTIONS_DIR / "fnn.parquet")

    test_qlike = qlike_score(y_test_log, mean_log_pred)
    print(f"\nTest QLIKE: {test_qlike:.4f} (LightGBM: 0.0215, LogHAR: 0.0259)")

    return preds_df


def run_lstm(
    n_trials: int = 10,
    n_splits: int = 3,
    n_seeds: int = 5,
    seq_len: int = 21,
) -> pl.DataFrame:
    """Full LSTM pipeline: HP tuning -> seed ensemble -> predictions.

    LSTM drops the first (seq_len - 1) observations per symbol.
    Output rows are tracked via VolatilitySequenceDataset.symbols / .dates.
    Returns predictions DataFrame.
    """
    # Load data
    train = pl.read_parquet(SPLITS_DIR / "train.parquet")
    test = pl.read_parquet(SPLITS_DIR / "test.parquet")
    feature_cols = get_feature_cols(train)
    print(f"Train: {len(train):,} rows | Test: {len(test):,} rows | Features: {len(feature_cols)}")

    scaler_stats = _load_scaler_stats()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Optuna tuning
    print(f"\nOptuna: {n_trials} trials x {n_splits} folds")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", study_name="lstm_qlike")

    def _lstm_trial_cb(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        print(f"  Trial {trial.number + 1}/{n_trials}: "
              f"QLIKE={trial.value:.4f} ({trial.duration.total_seconds():.1f}s)")

    study.optimize(
        make_optuna_objective_lstm(
            train, feature_cols, scaler_stats, device, n_splits, seq_len
        ),
        n_trials=n_trials,
        callbacks=[_lstm_trial_cb],
    )
    best_params = dict(study.best_params)
    print(f"\nBest trial QLIKE: {study.best_value:.4f}")
    print(f"Best params: {json.dumps(best_params, indent=2)}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODELS_DIR / "lstm_best_params.json", "w") as f:
        json.dump(best_params, f, indent=2)

    # Standardize
    train_scaled = _standardize_df(train, feature_cols, scaler_stats)
    test_scaled = _standardize_df(test, feature_cols, scaler_stats)

    # Val split for early stopping: last 10% of train dates
    train_dates = train["date"].unique().sort()
    val_cutoff = train_dates[int(len(train_dates) * 0.9)]
    tr_df = train_scaled.filter(pl.col("date") <= val_cutoff)
    vl_df = train_scaled.filter(pl.col("date") > val_cutoff)

    hidden_size = best_params.get("hidden_size", 32)
    num_layers = best_params.get("num_layers", 1)
    dropout = best_params.get("dropout", 0.2)
    lr = best_params.get("lr", 1e-3)
    batch_size = best_params.get("batch_size", 128)
    weight_decay = best_params.get("weight_decay", 1e-4)

    # Build all datasets once (identical across seeds)
    print("\nBuilding sequence datasets...")
    t0 = time.time()
    test_ds = VolatilitySequenceDataset(test_scaled, feature_cols, seq_len=seq_len)
    tr_ds = VolatilitySequenceDataset(tr_df, feature_cols, seq_len=seq_len)
    vl_ds = VolatilitySequenceDataset(vl_df, feature_cols, seq_len=seq_len)
    print(f"  train={len(tr_ds):,} val={len(vl_ds):,} test={len(test_ds):,} ({time.time()-t0:.1f}s)")

    # Seed ensemble
    print(f"\nSeed ensemble ({n_seeds} seeds):")
    all_seed_preds: list[np.ndarray] = []
    seed_states: list[dict] = []
    for seed in range(n_seeds):
        t0 = time.time()
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = LSTMModel(
            len(feature_cols),
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
        tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,
                               drop_last=True, pin_memory=True)
        vl_loader = DataLoader(vl_ds, batch_size=batch_size, shuffle=False,
                               pin_memory=True)

        model = train_model(
            model, tr_loader, vl_loader, lr=lr, weight_decay=weight_decay, device=device
        )
        model.eval()
        preds = _predict_batched(model, test_ds, device, batch_size=batch_size)
        all_seed_preds.append(preds)
        seed_states.append(model.cpu().state_dict())
        y_test_log = test_ds.y.numpy()
        qlike = qlike_score(y_test_log, preds)
        print(f"  Seed {seed}: QLIKE = {qlike:.4f} ({time.time() - t0:.1f}s)")
        torch.cuda.empty_cache()

    mean_log_pred = np.mean(all_seed_preds, axis=0)
    y_pred_level = np.clip(np.exp(mean_log_pred), 1e-8, None)

    # Save model weights
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_class": "LSTMModel",
            "n_features": len(feature_cols),
            "seq_len": seq_len,
            "best_params": best_params,
            "seed_states": seed_states,
        },
        MODELS_DIR / "lstm_ensemble.pt",
    )
    print(f"Saved LSTM ensemble ({n_seeds} seeds) to {MODELS_DIR / 'lstm_ensemble.pt'}")

    # y_true comes from VolatilitySequenceDataset targets (log-space), exp'd to level
    y_true_level = np.exp(test_ds.y.numpy()).astype(np.float64)

    # Build output DataFrame (y_true and y_pred in LEVEL space)
    preds_df = pl.DataFrame({
        "symbol": test_ds.symbols,
        "date": test_ds.dates,
        "model": ["LSTM"] * len(test_ds),
        "y_true": y_true_level,
        "y_pred": y_pred_level.astype(np.float64),
    })

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    preds_df.write_parquet(PREDICTIONS_DIR / "lstm.parquet")

    test_qlike = qlike_score(test_ds.y.numpy(), mean_log_pred)
    print(f"\nTest QLIKE: {test_qlike:.4f} (LightGBM: 0.0215, LogHAR: 0.0259)")

    return preds_df


if __name__ == "__main__":
    run_fnn()
