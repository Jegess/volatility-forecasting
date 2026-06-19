# Predicting volatility

A small, self-contained tool that runs the trained models to forecast a
stock's **21-trading-day-forward realized volatility**. Everything it needs —
the scripts, the deployable models, and the files they create — lives in this
folder. It reuses the main `theta` package for feature computation; no
modelling logic is duplicated.

## Models

| Model | What it is | Test QLIKE |
|---|---|---|
| **LogHAR** | log-space HAR on `rv_d, rv_w, rv_m`, price-only baseline | 0.0259 |
| **Lean LightGBM** | gradient boosting on 9 TreeSHAP-selected features | **0.0221** |

The lean LightGBM is a **retrained 9-feature variant** of the published model.
TreeSHAP on the full model showed the top features hold ~82% of importance, so
the lean model (test QLIKE 0.0221) matches the full 44-feature model (0.0215)
while needing far less input data. The published dissertation results use the
full model; this lean one is the deployable tool.

The 9 lean features: `rv_d, rv_w, rv_m, rq, rs_pos, rs_neg` (from prices) +
`atm_iv` (one number you supply) + `days_to_earnings, days_to_fomc` (calendar).

## Data requirements

You provide the inputs; the script computes features and runs the model.

| Model | Inputs needed | Min price history | Any ticker? |
|---|---|---|---|
| LogHAR | daily closes only | ~22 trading days (use ≥40) | **Yes**, free |
| Lean LightGBM | daily closes + `atm_iv` (1 number) + next-earnings date | ~22 trading days | **Yes** (you supply `atm_iv`) |

History *length* is not the bottleneck — the longest feature window is the
22-day `rv_m`, so a couple of months of daily closes is plenty. This is
inference, not training.

## Usage

Run from the repository root.

```bash
# LogHAR from Yahoo prices (any ticker, nothing else needed)
python -m prediction.predict --symbol AAPL --model loghar

# LogHAR from your own CSV (columns: date, close)
python -m prediction.predict --symbol AAPL --model loghar --prices aapl.csv

# Lean LightGBM: prices + one ATM implied-vol number (+ optional earnings date)
python -m prediction.predict --symbol AAPL --model lightgbm \
    --atm-iv 0.28 --next-earnings 2026-07-28
```

Output reports the as-of date, the forecast annualized realized variance, and
the implied annualized volatility (its square root).

### Arguments

- `--symbol` ticker (required).
- `--model` `loghar` (default) or `lightgbm`.
- `--prices FILE.csv` daily closes (`date,close`); if omitted, fetched from Yahoo Finance.
- `--atm-iv VALUE` annualized ATM implied vol as a decimal, e.g. `0.28` (LightGBM only, required).
- `--next-earnings YYYY-MM-DD` next earnings date (LightGBM only; if omitted, tries Yahoo, else a neutral default).
- `--as-of YYYY-MM-DD` prediction date (default: latest available price date).

## Rebuilding the models

The two artifacts in `prediction/models/` (`lgbm_lean.txt`, `loghar_coefs.json`)
are committed so the tool works out of the box. To regenerate them from the
training split (requires `data/processed/splits/`):

```bash
python -m prediction.train_lean
```

It prints each model's test QLIKE and the head-to-head verdict.

## Notes & limits

- `--atm-iv` and the next-earnings date are inputs you supply for the LightGBM path.
- `days_to_fomc` uses a built-in FOMC calendar (good through 2026); extend
  `theta/processing/events.py:FOMC_DATES` for later dates.
- Yahoo prices are split/dividend-adjusted; for arbitrary historical dates of a
  new ticker the option-implied `atm_iv` is still your responsibility to source.

## Tests

```bash
python -m pytest prediction/tests -q
```
