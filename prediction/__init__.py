"""Self-contained 21-day realized-volatility prediction tool.

Everything needed to *use* the trained models lives in this folder:
  - loghar.py     LogHAR fit/predict mechanism (price-only baseline)
  - train_lean.py one-time builder: lean LightGBM + LogHAR coefficients
  - predict.py    CLI forecaster
  - models/       all generated artifacts (lgbm_lean.txt, loghar_coefs.json)

Run from the repository root:
    python -m prediction.train_lean
    python -m prediction.predict --symbol AAPL --model loghar

The scripts reuse the existing ``theta`` package for feature computation;
no modeling logic is duplicated.
"""

from __future__ import annotations

from pathlib import Path

# ── Paths ───────────────────────────────────────────────────────────────
_PKG_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _PKG_DIR.parent

MODELS_DIR = _PKG_DIR / "models"                                   # our artifacts
SPLITS_DIR = _PROJECT_ROOT / "data" / "processed" / "splits"       # train/test (read-only)
_FULL_HP_PATH = _PROJECT_ROOT / "data" / "processed" / "models" / "lgbm_best_params.json"

# ── Feature contract (TreeSHAP-selected lean subset) ────────────────────
# Order matters: the lean LightGBM is trained with these names in this order.
PRICE_FEATURES = ["rv_d", "rv_w", "rv_m", "rq", "rs_pos", "rs_neg"]
LEAN_FEATURES = PRICE_FEATURES + ["atm_iv", "days_to_earnings", "days_to_fomc"]
LOGHAR_FEATURES = ["rv_d", "rv_w", "rv_m"]

# Target columns (as produced by theta preprocessing)
TARGET_COL = "rv_21d_forward"
LOG_TARGET_COL = "log_rv_21d_forward"

__all__ = [
    "MODELS_DIR", "SPLITS_DIR",
    "PRICE_FEATURES", "LEAN_FEATURES", "LOGHAR_FEATURES",
    "TARGET_COL", "LOG_TARGET_COL",
]
