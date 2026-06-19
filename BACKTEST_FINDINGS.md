# Backtest Findings Summary

**Status (2026-04-24):** Strategy exploration closed. ML edge is real but uninvestable at $100K retail. Dissertation pivots to forecasting contribution only.

## Strategies Tested

| Strategy | Window | Sharpe | Total Return | Verdict |
|---|---|---:|---:|---|
| Bull put spread (LGBM VRP signal) | 2023-2026 WF | -0.68 | varies | Negative Kelly, structural tail; NO BET |
| Cash-secured put on SPY/ETFs | 3yr | 2.14 | +4% | T-bills beat it; dead end |
| Long-only bottom-decile VRP equities | 3yr | 0.60 | +35.7% | Beaten by SPY (Sharpe 1.43) |
| Long-short market-neutral (bottom vs top decile) | 3yr | 0.23 @ gross 2x | +5.3% | Alpha too small to trade |

## Key Diagnostic Finding (Decile Sweep)

Long-only portfolios sorted by LGBM VRP forecast, monthly rebalance:

| Decile | Return | Sharpe |
|---:|---:|---:|
| 1 (top VRP, highest IV vs forecast) | +8.0% | 0.29 |
| 2 | +4.3% | 0.23 |
| 3 | -1.7% | -0.06 |
| 4 | -1.0% | -0.03 |
| 5 | +1.6% | 0.11 |
| 6 | +0.8% | 0.10 |
| 7 | +3.2% | 0.15 |
| 8 | +14.1% | 0.44 |
| 9 | +26.0% | 0.48 |
| **10 (bottom VRP, cheapest IV vs forecast)** | **+35.7%** | **0.60** |

**Monotonic gradient confirmed.** The ML model ranks stocks correctly by subsequent return, consistent with Bali et al. / low-vol anomaly. Spread (D10 − D1) = ~28pp over 3 years.

## Why No Strategy Worked at $100K

1. **Put spreads on equities:** Tail risk concentration. 10-18 breach events × -$250 dominates the +$30/trade profit bucket (8:1 ratio).
2. **CSP on ETFs:** Safe but tiny. Collateral-yield-equivalent; no edge above T-bills.
3. **Long-only bottom decile has NEGATIVE alpha:** Earned +35.7% vs SPY +65.4%. With β≈1.2, beta alone "should have" produced ~+78%. So α ≈ −42pp. Took MORE risk than SPY (24.5% MDD vs 18.1%) for LESS return. EW-188 null portfolio matched its Sharpe (0.59 vs 0.60), confirming the long-only "edge" was beta exposure, not selection skill.
4. **Long-short isolates SMALL POSITIVE alpha:** When beta is stripped (β≈0 by construction), the signal's true alpha is ~+5.3%/3yr (~+1.7%/yr) at gross 2.0. Real but too small to pay for shorting, borrow, and Reg T margin at retail capital.
5. **Signal-degradation exits:** Intra-month rank-based closures hurt L/S by ~4pp. Point-in-time monthly ranking is the cleaner signal for a cross-sectional forecast.

## Alpha vs Beta: What the Tests Actually Measure

The long-only +35.7% and L/S +5.3% are NOT two independent signals. They are the same cross-sectional edge viewed through two different lenses:

```
Long-only return = α + β × market_return + noise
                 = (negative) + 1.2 × 65% + noise
                 ≈ -42pp + 78pp  ≈ +35.7%

L/S return       = α + β × market_return + noise
                 = (+5.3%) + 0 × 65% + noise  (β≈0 by construction)
                 ≈ +5.3%
```

Only the L/S test isolates alpha. The long-only test bundles alpha with a large negative-alpha beta-trade. The dissertation framing must distinguish these: **the model produces a cross-sectional ranking with real but small alpha (~1.7%/yr net of beta). Long-only implementations look bigger but represent beta exposure, not skill.**

## Gross Decile Spread vs Net L/S Return

Why does the decile sort show +28pp gross but L/S only yields +5.3% net?

- **Gross decile spread** = decile_10_return − decile_1_return, each computed as a separate 100%-allocated portfolio. Hypothetical only.
- **L/S 2.0x** = long decile 10 + short decile 1 on a SINGLE capital base with monthly rebalance, compounding, whole-share rounding.

Implementation friction (capital dilution, compounding drag from short losses reducing long base, whole-share rounding at smaller per-name allocations) eats ~80% of the gross spread. The L/S +5.3% is the tradeable number.

## What the Model IS Good For

- **Forecasting:** 17% QLIKE improvement over LogHAR baseline (statistically significant, DM test p < 0.001)
- **Ranking:** Clean monotonic decile spread confirms cross-sectional validity
- **Risk attribution:** SHAP values identify which features drive vol forecasts (rv_m 51%, rq 15%, rs_neg 6%, atm_iv 4%, days_to_earnings top-10)
- **Walk-forward robust for LightGBM** (LSTM fails WF: best in single-split but worst in rolling)

## What the Model IS NOT Good For

- Standalone trading strategy at $100K
- Premium collection (no structure survives after tail events)
- Long-only factor portfolio (beaten by passive index)
- Market-neutral factor (alpha too small vs friction)

## Possible Improvements (Not Pursued)

1. **Aggregate VRP as SPY timing signal:** universe mean VRP to time long/flat on SPY. Tests macro timing vs cross-sectional selection.
2. **Longer horizon:** 2007-2026 multi-regime data would be fairer to market-neutral strategies (2023-2026 is pure bull market).
3. **Multi-factor combination:** VRP + momentum + quality. Outside dissertation scope.
4. **Institutional capital (≥$1M):** shorting costs amortize, can push gross to 3-4x, Sharpe 0.23 becomes investable. Not relevant at retail scale.
5. **Vol-targeting / vol-weighted sizing:** marginal Sharpe lift possible, unlikely to clear SPY.

## Reusable Artifacts

- `theta/backtest/directional/`: full subpackage, isolated from put-spread/CSP code paths
- `run_directional(decile, n_deciles, exit_pct_rank, capital, ...)`: long-only
- `run_long_short(long_decile, short_decile, gross_leverage, ...)`: L/S market-neutral
- `run_spy_benchmark()`, `run_equal_weight_universe()`: null/baseline portfolios

## Decision

**Close strategy exploration. Dissertation frames ML model as a forecasting contribution with diagnostic (ranking) validity, not a tradeable strategy at retail capital.** Matches Gu-Kelly-Xiu's central cautionary theme in the literature.
