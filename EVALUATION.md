# Model Evaluation Results

**Generated:** 2026-03-26
**Primary metric:** QLIKE (Patton 2011), lower is better
**Benchmark:** LogHAR (best baseline)

## Walk-forward validation (primary)

The headline ranking comes from **12-window expanding walk-forward** (retrain on a growing
window, score the next quarter, 2023-02 to 2026-02). This is the honest out-of-sample test, and
it overturns the single-split ranking below: **LightGBM is the robust winner**, best in all 12
windows with low variance, while the **LSTM is the worst real model** and unstable.

| Model | Mean QLIKE | Std | vs. LogHAR | Note |
|-------|-----------:|----:|-----------:|------|
| **LightGBM** | **0.0231** | 0.0028 | −17.4% | Best in all 12 windows |
| LogHAR | 0.0280 | 0.0051 | — | Best baseline |
| SHAR | 0.0281 | 0.0038 | +0.6% | |
| HARQ | 0.0283 | 0.0045 | +1.2% | |
| LevHAR | 0.0287 | 0.0044 | +2.6% | |
| HAR | 0.0287 | 0.0045 | +2.7% | |
| LSTM | 0.0428 | 0.0133 | +53% | Worst real model; high variance |
| AR5 | 0.419 | 0.063 | — | Reference floor |

Source: `data/processed/evaluation/walk_forward/summary_metrics.parquet`;
see `notebooks/walk_forward_results.ipynb`.

## Single static split (secondary — does not generalize)

A single 70/10/20 split (test 2025-05-15 to 2026-03-19, 36,924 obs, 193 symbols) flatters the
LSTM. Its score here is genuine but **collapses under walk-forward retraining**, so this table is
reported for completeness, not as the ranking.

| Model | n | QLIKE | MSE | MAE | MAPE (%) | R2_OOS |
|-------|------:|------:|----:|----:|--------:|-------:|
| **LSTM** | 33,084 | **0.0160** | 0.001485 | 0.0106 | 8.39 | **96.5%** |
| LightGBM | 36,924 | 0.0215 | 0.002583 | 0.0164 | 13.22 | 94.0% |
| LogHAR | 36,811 | 0.0259 | 0.002994 | 0.0167 | 13.32 | 93.1% |
| HARQ | 36,924 | 0.0263 | 0.003024 | 0.0164 | 14.08 | 93.0% |
| SHAR | 36,924 | 0.0264 | 0.003019 | 0.0162 | 14.25 | 93.0% |
| LevHAR | 36,924 | 0.0265 | 0.003022 | 0.0162 | 14.38 | 93.0% |
| HAR | 36,924 | 0.0265 | 0.003018 | 0.0163 | 14.39 | 93.0% |
| FNN | 36,924 | 0.0285 | 0.006273 | 0.0239 | 16.41 | 85.5% |
| GARCH | 36,924 | 0.2240 | 0.035877 | 0.0869 | 80.99 | 17.2% |
| AR5 | 35,964 | 0.3691 | 0.028390 | 0.0954 | 148.42 | 33.7% |

## Improvement over LogHAR

| Model | QLIKE Change | Interpretation |
|-------|-------------|----------------|
| LSTM | **-38.2%** | Substantial improvement |
| LightGBM | -17.0% | Moderate improvement |
| LogHAR | --- | Baseline |
| HAR family | +1-2% | Within noise |
| FNN | +10.0% | Underperforms baseline |
| GARCH | +765% | Poor fit for cross-sectional panel |
| AR5 | +1325% | Not competitive |

## Diebold-Mariano Tests

All tests use QLIKE loss with Newey-West HAC standard errors (lag=21, matching the 21-day forecast horizon). Positive DM statistic means the model improves over the benchmark.

### vs LogHAR

| Model | DM stat | p-value | n | Sig |
|-------|--------:|--------:|------:|-----|
| LSTM | 11.99 | <0.001 | 32,992 | *** |
| LightGBM | 4.65 | <0.001 | 36,811 | *** |
| HARQ | -4.84 | <0.001 | 36,811 | *** |
| SHAR | -4.98 | <0.001 | 36,811 | *** |
| LevHAR | -5.32 | <0.001 | 36,811 | *** |
| HAR | -5.66 | <0.001 | 36,811 | *** |
| FNN | -5.87 | <0.001 | 36,811 | *** |
| GARCH | -20.25 | <0.001 | 36,811 | *** |
| AR5 | -35.73 | <0.001 | 35,855 | *** |

### vs LightGBM

| Model | DM stat | p-value | n | Sig |
|-------|--------:|--------:|------:|-----|
| LSTM | 9.91 | <0.001 | 33,084 | *** |
| LogHAR | -4.65 | <0.001 | 36,811 | *** |
| HARQ | -5.19 | <0.001 | 36,924 | *** |
| SHAR | -5.48 | <0.001 | 36,924 | *** |
| HAR | -5.60 | <0.001 | 36,924 | *** |
| LevHAR | -5.61 | <0.001 | 36,924 | *** |
| FNN | -9.10 | <0.001 | 36,924 | *** |
| GARCH | -20.72 | <0.001 | 36,924 | *** |
| AR5 | -35.76 | <0.001 | 35,964 | *** |

All pairwise differences are statistically significant at p < 0.001.

## Key Findings

1. **LightGBM is the robust winner.** On walk-forward (the primary test) it beats every model in all 12 windows (mean QLIKE 0.0231) with the lowest variance. SHAP shows RV features dominate (rv_m=0.507), but option-implied features (atm_iv, vol_of_vol) and event features (days_to_earnings) add incremental signal.

2. **LSTM is best on a single static split but fails walk-forward.** Its single-split score (QLIKE=0.0160, R2_OOS=96.5%) is genuine, but under expanding-window retraining it is the *worst* real model (mean QLIKE 0.0428, high variance). The static-split win does not generalize — consistent with the literature's caution that LSTMs need far longer or higher-frequency data to be stable. It is a cautionary result, not the headline.

3. **HAR family tightly clustered** (0.026x): volatility persistence is the dominant signal. Log-transform helps (LogHAR best baseline). Extensions (leverage, jumps) add marginal value.

4. **FNN underperforms baselines:** a small 3-layer network cannot match HAR's explicit lag structure on tabular data. Consistent with Gu/Kelly/Xiu finding that tabular NNs struggle vs trees.

5. **GARCH and AR5 not competitive:** designed for single-series, not cross-sectional panels.

6. **Static-split overfitting diagnostics (not walk-forward):** within the single split, the LSTM passes all 3 — matched subset (gap unchanged), temporal stability across the test quarters, held-out symbols (QLIKE=0.0145 on 30 unseen stocks). These probe leakage *within* the static split and do not contradict finding #2: the LSTM still fails when retrained walk-forward, which is the harder, primary test.

## Methodology Notes

- **QLIKE** = mean(y/yhat - log(y/yhat) - 1), the highest statistical power for variance forecasts (Patton 2011)
- **R2_OOS** = 1 - SS_res / SS_tot where SS_tot uses training mean (Gu/Kelly/Xiu 2020)
- **DM test** uses Bartlett kernel HAC with lag=21 to account for overlapping 21-day forecast errors
- **LSTM n=33,084** < 36,924 due to seq_len=21 warmup per symbol; DM tests align on common (symbol, date) pairs
- All metrics computed in level space (annualized variance)
- **Deployable model:** these results use the full 44-feature models. A lean 9-feature LightGBM (TreeSHAP-selected) reaches test QLIKE 0.0221 — essentially matching the full model — and ships as the runnable forecaster in [`prediction/`](prediction/README.md).
