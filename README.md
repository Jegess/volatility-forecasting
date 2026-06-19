# Theta: Volatility Forecasting for US Equities

Machine-learning models that forecast 21-day realized volatility for US mid- and large-cap
equities, benchmarked against the HAR family and GARCH, with an investigation of whether the
forecast edge is tradeable.

This is the codebase behind a master's dissertation, *"Machine Learning Approaches to
Volatility Forecasting: Evidence from US Equities."* It covers the full pipeline:
raw options data ingestion, feature engineering, model training and evaluation, and a trading
backtest that tests whether the statistical edge survives real-world frictions. (The written dissertation is
in Hungarian; an English version will be added later.)

> **Headline result:** the ML models beat the standard econometric baselines on forecast
> accuracy by a statistically significant margin, but the resulting trading edge does not
> survive transaction costs and tail risk at retail scale. The contribution lies in the forecast
> accuracy, and the negative trading result is reported in full.

---

## Results summary

The honest test of a forecasting model is **walk-forward validation** — retrain on an expanding
window, score the next out-of-sample block, repeat. Across 12 quarterly windows (2023-02 to
2026-02), **LightGBM is the robust winner**, posting the best score in all 12 windows; the LSTM
is the *worst* real model. Primary metric is **QLIKE** (Patton, 2011); lower is better.

### Walk-forward (primary, 12 expanding windows)

| Model | Mean QLIKE | vs. LogHAR | Note |
|-------|-----------:|-----------:|------|
| **LightGBM** | **0.0231** | −17.4% | Robust winner — best in all 12 windows, low variance |
| LogHAR | 0.0280 | — | Best baseline |
| HAR / SHAR / HARQ / LevHAR | ~0.0282 | ~0% | HAR family, clustered |
| LSTM | 0.0428 | +53% | Worst real model; high variance |
| AR(5) | 0.419 | — | Reference floor |

### Single static split (secondary — does not generalize)

A single 70/10/20 split (test 2025-05-15 to 2026-03-19) flatters the LSTM, which scores best
here. That result is genuine but **collapses under rolling retraining** (see above), so it is
reported only for completeness.

| Model | QLIKE | vs. LogHAR | R²_OOS |
|-------|------:|-----------:|-------:|
| LSTM *(does not generalize)* | 0.0160 | −38.2% | 96.5% |
| **LightGBM** | **0.0215** | −17.0% | 94.0% |
| LogHAR (best baseline) | 0.0259 | — | 93.1% |
| FNN | 0.0285 | +10.0% | 85.5% |
| GARCH(1,1) | 0.2240 | +765% | 17.2% |

All pairwise single-split differences are significant at p < 0.001 (Diebold–Mariano with
Newey–West HAC, lag = 21 to match the forecast horizon). The LSTM's static-split win is
consistent only with literature caution that LSTMs need far longer or higher-frequency data to
be stable. See [`EVALUATION.md`](EVALUATION.md).

---

## What this project does

1. **Uses** end-of-day US options chains, open interest, and underlying prices (193 symbols,
   2021–2026).
2. **Engineers 44 features** across five groups: realized-volatility (HAR/HARQ/SHAR
   decompositions), option-implied (ATM IV, variance risk premium, skew, term slope, BKM
   risk-neutral moments, informed-trading signals), technical, macro, and event features
   (distance-to-earnings/FOMC/CPI).
3. **Computes implied volatility** via European Black–Scholes on mid-quotes, with a
   literature-grounded filter chain (data integrity → liquidity → DTE → monthly expirations →
   delta/moneyness).
4. **Trains and tunes** a suite of models (HAR family, GARCH, AR, LightGBM, FNN, and LSTM)
   with a custom QLIKE objective, Optuna hyperparameter search, and **purged k-fold
   cross-validation with a 21-day embargo** (Lopez de Prado) to prevent leakage from
   overlapping targets.
5. **Evaluates** with QLIKE / MSE / MAE / R²_OOS and formal Diebold–Mariano significance tests,
   plus overfitting diagnostics (matched subsets, temporal stability, held-out symbols).
6. **Backtests** the forecast as a trading signal across several option and equity structures,
   and documents why none of them clears the bar at retail capital.

---

## Repository structure

```
theta/
  config.py          # pipeline configuration
  processing/        # join → filter → IV → features → panel
    join.py, filters.py, iv.py, compute_iv.py, rates.py
    rv.py, technical.py, option_features.py, compute_features.py
    macro.py, events.py, panel.py
  modeling/
    preprocessing.py      # splits, scaling, purged k-fold, log target
    baselines.py          # HAR, LogHAR, LevHAR, SHAR, HARQ, GARCH, AR
    lightgbm_model.py     # LightGBM + custom QLIKE objective + Optuna + SHAP
    neural_models.py      # FNN + LSTM (PyTorch), seed ensemble, early stopping
    walk_forward.py       # 12-window expanding walk-forward validation
    evaluation.py         # QLIKE/MSE/MAE/R²_OOS metrics + DM tests with HAC SEs
  analysis/
    vrp.py                # variance risk premium
    correlation/          # 193-symbol correlation, Marchenko–Pastur denoising, detoning
  backtest/               # VRP premium-selling, CSP, directional & long-short strategies

tests/                    # unit tests (run without the full dataset) + @slow integration
notebooks/                # rendered results: model comparison, SHAP, walk-forward, overfitting

STRATEGY.md               # full trading-strategy design (VRP premium selling)
EVALUATION.md             # model evaluation results and DM tests
BACKTEST_FINDINGS.md      # why no strategy proved tradeable
```

---

## Key findings

**Forecasting (the contribution).** Gradient boosting extracts real, significant signal beyond
the persistence that HAR models already capture, and — unlike the LSTM — it holds up under
walk-forward retraining, making **LightGBM the practical winner**. SHAP attribution shows
realized-volatility lags dominate (`rv_m` ≈ 51% of importance), but option-implied features
(ATM IV, vol-of-vol) and event features (distance-to-earnings, a top-10 feature) add
incremental value. The improvement holds on 30 entirely held-out symbols, so it is not
overfitting. The LSTM wins a single static split but is the worst model under walk-forward —
a cautionary, not a headline, result.

**Trading (the negative result).** The forecast ranks stocks correctly, producing a clean
monotonic decile gradient in subsequent returns, yet every tradeable structure fails at retail
scale:

| Structure | Sharpe | Verdict |
|-----------|-------:|---------|
| Bull-put spreads (VRP signal) | −0.68 | Negative Kelly, structural tail risk |
| Cash-secured puts on ETFs | 2.14 | Only +4% total; T-bills beat it |
| Long-only bottom-decile VRP | 0.60 | Beaten by SPY (1.43); the "edge" is just beta |
| Long-short market-neutral | 0.23 | Real alpha (~1.7%/yr) but too small for frictions |

The cross-sectional alpha, isolated from beta, is real but ~1.7%/yr net, only investable at
institutional scale where shorting and margin costs amortize. This matches the central
cautionary theme of Gu, Kelly & Xiu (2020): statistical predictability and economic tradability
are different things. Full reasoning in [`BACKTEST_FINDINGS.md`](BACKTEST_FINDINGS.md).

---

## Tech stack

- **Python 3.11+**, [Polars](https://pola.rs/) for data processing (no pandas in the hot path)
- **LightGBM** with a hand-written QLIKE objective and gradient
- **PyTorch** for the FNN and LSTM, with seed ensembling and batched GPU inference
- **Optuna** for hyperparameter search over purged k-fold CV
- **SHAP** for feature attribution, **statsmodels** for HAC standard errors
- `py_vollib_vectorized` for vectorized European BSM implied volatility
- Validation built from scratch: purged k-fold and walk-forward, no off-the-shelf splitters

---

## Running the code

```bash
pip install -e .          # installs deps from pyproject.toml
pytest -m "not slow"      # unit tests (fast, no large data needed)
```

**The dataset is not included.** The raw options data (~14 GB) came from a paid provider
(ThetaData, subscription since cancelled) and is not redistributable. The repository is
therefore intended to be **read and evaluated** (the code, the methodology, and the results in
the notebooks and markdown reports) rather than re-run end-to-end. Unit tests run on small
synthetic fixtures and exercise the core logic without the full panel.

The processed feature panel is 189,006 rows × 47 columns (193 symbols, zero nulls); the modeling
target is 21-day-forward realized volatility (annualized variance).

### Generating a forecast

To actually run a trained model on a stock, see [`prediction/`](prediction/README.md). It ships
two ready-to-use predictors — LogHAR (prices only, any ticker) and a lean 9-feature LightGBM —
with committed model artifacts, so no dataset is required:

```bash
python -m prediction.predict --symbol AAPL --model loghar
python -m prediction.predict --symbol AAPL --model lightgbm --atm-iv 0.28 --next-earnings 2026-07-28
```

---

## Limitations

- Close-to-close realized volatility (no intraday data); RV is a noisy proxy for the latent process.
- No stock-level fundamentals or volume features in the current panel.
- Test window (2023–2026) is a single, largely bullish regime, unfavorable to market-neutral
  structures and not a multi-regime stress test.
- IV from end-of-day mid-quotes, not real-time; live deployment would face distributional shift
  between data sources.

---

## Selected references

Corsi (2009) · Patton (2011) · Patton & Sheppard (2015) · Bollerslev, Tauchen & Zhou (2009) ·
Gu, Kelly & Xiu (2020) · Bali, Engle & Murray (2016) · Bakshi, Kapadia & Madan (2003) ·
Carr & Wu (2009) · Cremers & Weinbaum (2010) · Diebold & Mariano (1995) · Lopez de Prado (2018) ·
Sinclair, *Positional Option Trading* · Natenberg, *Option Volatility and Pricing*.

---

*Academic / research project. Nothing here is investment advice. The trading analysis concludes
that the strategies studied are not profitable at retail scale.*
