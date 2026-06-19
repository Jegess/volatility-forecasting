# Backtest Plan: VRP-Based Premium Selling Strategy

## Purpose

Validate whether the VRP signal (powered by walk-forward LightGBM forecasts) translates into profitable risk-defined option trades on historical data, before risking real capital.

This plan follows two principles from the literature:
- **Lopez de Prado:** "Backtesting is not a research tool." All strategy rules (VRP signal, filters, exits, sizing) are fixed BEFORE running any backtest. No parameter will be tuned after seeing results.
- **Sinclair:** Options backtests must use conservative fill assumptions (bid/ask, never mid-price) and account for the negative skewness inherent in short premium positions.

## Data Inventory

### Walk-Forward LightGBM Predictions (Primary Signal Source)

```
File: data/processed/evaluation/walk_forward/all_predictions.parquet
Columns: symbol, date, model, y_true, y_pred, window_id
LightGBM rows: 138,629
Period: 2023-02-07 to 2026-02-19 (756 trading days)
Symbols: 193
Windows: 12 quarterly expanding-window retrains
```

These predictions are **out-of-sample by construction**: each window was trained only on data preceding the test period. This eliminates look-ahead bias in the RV forecast.

### Option Chains with IV and Delta

```
Directory: data/processed/options_iv/ (193 files)
Columns: symbol, date, expiration, strike, right, bid, ask, mid_quote,
         volume, close, open_interest, underlying_price,
         dte, moneyness, relative_spread, rate, t_years, iv, delta
Period: 2021-01-04 to 2026-03-06
```

These are post-filter options: monthly expirations only, delta-filtered (puts -0.50 to -0.05, calls 0.15 to 0.50), liquidity and integrity checks applied. Exactly the universe we would trade live.

Coverage in test period: ~170-210 trading days per symbol, with tradeable puts (delta -0.10 to -0.25, DTE 14-45) available on 98%+ of trading days.

Contracts can be tracked day-by-day (same symbol/strike/expiration appears across multiple dates with updated bid/ask/delta).

### Raw EOD Option Data (Fallback for Position Tracking)

```
Directory: data/raw/eod/ (193 files)
Columns: symbol, created, expiration, strike, right, bid, ask,
         volume, close, open_interest, ...
Period: 2021-01-04 to 2026-03-06
```

Used when a held contract drops below the delta filter threshold and disappears from `options_iv/`. The raw data has no filters; any contract that existed on that date appears here.

### Feature Data (For Filters and VRP Calculation)

```
Directory: data/processed/features/ (193 files)
Key columns: symbol, date, atm_iv, vrp, days_to_earnings,
             is_earnings_week, is_fomc_week, rsi_14, mom_5d, ...
Period: 2022-01-19 to 2026-03-06
```

```
File: data/processed/macro/macro.parquet
Key columns: date, vix, vvix, term_spread, credit_spread, ...
Period: 2021-2026
```

### LogHAR Predictions (Sanity Check)

```
File: data/processed/evaluation/walk_forward/all_predictions.parquet
Filter: model == 'LogHAR'
Same period and symbols as LightGBM.
```

## Pre-Committed Strategy Parameters

These are FIXED before any backtest is run. No parameter will be changed based on backtest results (Lopez de Prado: "researching under the influence of a backtest is like drinking and driving").

### Signal

| Parameter | Value | Source |
|-----------|-------|--------|
| VRP formula | `ATM_IV² - RV_forecast` | Sinclair, STRATEGY.md |
| RV forecast model | Walk-forward LightGBM | Walk-forward validation |
| Sanity check model | LogHAR | Baseline comparison |
| Model disagreement threshold | 50% relative difference | STRATEGY.md |

### Filters

| Filter | Rule | Source |
|--------|------|--------|
| VRP positive | `VRP > 0` | Sinclair |
| VRP strength | `VRP > symbol's historical median VRP` | STRATEGY.md |
| VIX regime | `14 < VIX < 30` | Sinclair endorses "mid-level VIX" qualitatively; specific bounds are own choice |
| No earnings | `days_to_earnings > DTE + 5` | STRATEGY.md |
| No FOMC week | `is_fomc_week == 0` | STRATEGY.md |
| Liquidity | `relative_spread < 0.15` at target strikes | STRATEGY.md |
| Open interest | `open_interest > 100` at target strikes | STRATEGY.md |

### Spread Construction

| Parameter | Value | Source |
|-----------|-------|--------|
| Strategy | Bull put spread (primary) | Sinclair calls short spreads "most conservative directional strategy" (his own simulation uses 1y LEAPS ATM, which we are not following) |
| Short strike delta | Closest to -0.15 to -0.20 | STRATEGY.md (own choice, not Sinclair) |
| Wing width | $2 default ($1 if stock < $30, $3 if stock > $80) | STRATEGY.md |
| Minimum premium | ≥ 30% of width | STRATEGY.md |
| Target DTE at entry | 21-30 days | Matches 21-day forecast horizon |
| Expiration type | Monthly only (3rd Friday) | STRATEGY.md |

### Fill Assumptions

| Action | Price Used | Rationale |
|--------|-----------|-----------|
| Sell short put (entry) | **Bid** | Sinclair: "always value against the bid if selling" |
| Buy long put (entry) | **Ask** | Sinclair: "always value against the offer if buying" |
| Buy back short put (exit) | **Ask** | Crossing the spread to close |
| Sell long put (exit) | **Bid** | Crossing the spread to close |
| Entry premium | `short_bid - long_ask` | Conservative: worst-case fills on both legs |
| Exit cost | `short_ask - long_bid` | Conservative: worst-case fills on both legs |

Sinclair warns that mid-price fills are a delusion for anything above "infinitesimal size." By using bid/ask throughout, the backtest shows what a real trader would experience.

### Exit Rules

| Trigger | Action | Source |
|---------|--------|--------|
| Profit ≥ 50% of max profit | Close | Tastytrade convention (not Sinclair) |
| DTE ≤ 10 | Close | Natenberg gamma/gap risk warnings (his strongest case is ATM; ours is OTM so weaker) |
| Underlying breaches short strike | Close | Standard risk mgmt |
| Loss > 2× premium collected | Close | Own choice |
| VIX crosses 30 | Close ALL positions | Own regime exit |

### Position Management

| Parameter | Value |
|-----------|-------|
| Max concurrent positions | 3 |
| Max same sector | 2 |
| Max same symbol | 1 |
| Commission per leg | $0.65 (IBKR standard) |
| Commission per vertical round-trip | $2.60 (4 legs × $0.65) |

### Capital & Sizing

| Parameter | Value |
|-----------|-------|
| Starting capital | $10,750 (EUR 10,000) |
| Risky subaccount | 44% = $4,730 |
| Cash reserve | 56% = $6,020 |
| Kelly fraction (estimated) | 0.17 |
| Max risk per trade | `risky_capital × 0.17` |
| Trailing stop | 43% from peak of risky subaccount |

## Backtest Levels

Four levels of increasing realism and complexity. Each level gates the next; if a level fails, there is no point building the next one.

### Level 1: VRP Signal Quality

**Question:** Does `VRP > 0` actually predict that IV was too high, i.e. that `IV² > RV_actual`? And does the daily VRP *ranking* contain information: are top-ranked symbols better trades?

**Method:**
```
For each trading day in the WF prediction period:
  1. Compute VRP for all symbols: VRP_i = atm_iv_i² - rv_forecast_i
     (using LightGBM y_pred and ATM_IV from features/)
  2. Also compute LogHAR VRP: VRP_loghar_i = atm_iv_i² - rv_loghar_i
  3. Rank all symbols by VRP descending for this day
  4. Compute actual_VRP_i = atm_iv_i² - rv_actual_i (using y_true)
  5. For each (symbol, date), record:
     - VRP predicted, VRP actual
     - Was VRP > 0? Was actual_VRP > 0? (hit/miss)
     - Daily VRP rank (1 = highest VRP that day)
     - VRP quintile (top 20%, 20-40%, etc.)
```

**Metrics:**

| Metric | What It Measures | Pass Threshold |
|--------|-----------------|----------------|
| VRP accuracy | % of VRP > 0 signals where actual IV² > RV_actual | > 60% |
| VRP accuracy by quintile | Same, split by VRP magnitude | Monotonically increasing |
| **VRP accuracy by daily rank** | **Top-10 vs top-50 vs all, does ranking help?** | **Top-10 > top-50 > all** |
| Edge vs naive | Does LightGBM VRP beat LogHAR VRP in accuracy? | LightGBM > LogHAR |
| VRP calibration | Mean predicted VRP vs mean actual VRP | Within 20% |
| **Rank stability** | **Do the same symbols dominate the top, or does the ranking rotate?** | **Document, rotation is healthy** |

**Output:** Summary table, calibration plot (predicted VRP vs actual VRP), accuracy-by-rank curve.

**The accuracy-by-rank curve is the key output.** If accuracy is 62% across all symbols but 75% for daily top-10, the ranking is adding substantial value, and since Level 2 only trades the top-ranked symbols (limited by position slots), the effective accuracy is higher than the overall number suggests.

**Gate:** If VRP accuracy < 55%, the signal is too noisy to trade. Stop here. If accuracy is decent overall but flat across ranks (top-10 ≈ bottom-10), the ranking adds no value; the model helps with direction but not selection.

**Effort:** Half day.

### Level 1.5: Terminal P&L Distribution (Unhedged)

**Question:** If we enter spreads based on VRP signals and hold to expiration (no management), what does the P&L distribution look like?

**Rationale (Sinclair):** Understanding the terminal distribution reveals the raw risk profile, especially the negative skewness, before exit rules improve it. "Even if a position is actively adjusted, it is instantaneously subject to the exact same risks."

**Important:** Level 1.5 deliberately ignores portfolio constraints (max positions, sector limits). Every qualifying signal that can produce a valid spread becomes a trade. This isolates the raw edge of the signal+spread combination. Portfolio-level filtering is added in Level 2.

**Method:**
```
For each trading day in the WF prediction period:
  1. RANK: Compute VRP for all 193 symbols (ATM_IV² - RV_forecast)
  2. FILTER: Remove symbols failing any filter (VRP ≤ 0, VIX outside 14-30,
     earnings within DTE+5, FOMC week, etc.)
  3. SORT: Rank surviving symbols by VRP descending
  4. For EACH qualifying symbol (no position limit in Level 1.5):
     a. In options_iv/, find puts at DTE 21-30 for this symbol+date
     b. Select the put closest to delta -0.175 (midpoint of -0.15 to -0.20)
     c. Find the wing: next available strike $2 below (or $1/$3 per stock price)
     d. Check liquidity: both strikes must have OI > 100
     e. Compute entry premium = short_bid - long_ask
     f. If premium < 30% of width → skip this symbol
     g. If no valid expiration/strikes found → skip this symbol
  5. For each opened trade, simulate hold-to-expiration:
     a. Look up underlying_price on expiration date
     b. Compute terminal P&L:
        - If underlying > short_strike: P&L = +premium (max profit)
        - If underlying < long_strike: P&L = -(width - premium) (max loss)
        - If between: P&L = premium - (short_strike - underlying_price)
     c. Subtract commission ($2.60)
  6. Record: entry date, symbol, VRP rank, premium, terminal P&L, holding period
```

**Why no position limits here:** Level 1.5 answers "does the signal+spread produce positive expected P&L?" If we limit to 3 positions, we conflate signal quality with portfolio management. By testing every qualifying trade, we get a clean P&L distribution with enough samples for statistical confidence. Level 2 adds the realistic constraints.

**Metrics:**

| Metric | What It Measures | Pass Threshold |
|--------|-----------------|----------------|
| Mean terminal P&L | Average unmanaged outcome | > $0 (profitable after commissions) |
| Median terminal P&L | Typical outcome (less sensitive to tails) | > $0 |
| Win rate | % of trades with positive terminal P&L | > 60% |
| Max loss frequency | % of trades hitting max loss | < 25% (Sinclair: butterflies hit max loss 26% of time) |
| Skewness | Negative skew of P&L distribution | Document, don't threshold; it will be negative |
| 5th percentile P&L | Tail risk per trade | > -$200 |

**Output:** P&L histogram, cumulative equity curve (unmanaged), skewness stats.

**Gate:** If mean terminal P&L < $0 after commissions, the strategy loses money even without management overhead. Stop here.

**Effort:** 1 day (includes building the contract matcher).

### Level 2: Managed Spread Simulation

**Question:** How much do the exit rules (50% profit, DTE ≤ 10, breach) improve over holding to expiration? And what happens when realistic portfolio constraints are applied?

**This is the realistic simulation.** It mirrors the actual daily workflow: rank all symbols, select the best ones within position limits, manage open positions, track capital.

**Method:**
```
STATE: portfolio = {open_positions: [], risky_capital: $4,730, peak_capital: $4,730}

For each trading day in the WF prediction period:

  === PHASE A: MANAGE EXISTING POSITIONS ===

  For each open position in portfolio:
    1. Look up both contracts' bid/ask for today:
       - First try options_iv/ (filtered data)
       - If contract missing (fell below delta filter): use raw EOD data
       - If contract missing from both (expired, no quote): mark for settlement
    2. Compute current spread value = short_ask - long_bid (cost to close)
    3. Compute current P&L = entry_premium - current_spread_value
    4. Check exit triggers (first matching trigger wins):
       a. VIX > 30 → close ALL positions (regime exit)
       b. DTE ≤ 10 → close (gamma risk)
       c. underlying_price ≤ short_strike → close (breach)
       d. Loss > 2 × premium → close (stop loss)
       e. P&L ≥ 50% of max profit → close (profit target)
    5. If triggered:
       - Exit P&L = entry_premium - exit_spread_value - $2.60 commission
       - Update risky_capital += exit P&L
       - Remove from open_positions
       - Log: exit date, exit reason, gross P&L, net P&L, holding days
    6. If at expiration with no prior trigger:
       - Settle at terminal value (same as Level 1.5 calculation)
       - Update risky_capital, remove from open_positions

  === PHASE B: CHECK PORTFOLIO STOPS ===

  7. Update peak_capital = max(peak_capital, risky_capital)
  8. If risky_capital < peak_capital × 0.57 (43% trailing stop):
     → HALT: no new trades until manually reset. Continue managing existing.
  9. If monthly drawdown > 8% of total capital:
     → PAUSE: no new trades for 5 trading days.

  === PHASE C: OPEN NEW POSITIONS (Daily Ranking) ===

  10. Compute available_slots = 3 - len(open_positions)
  11. If available_slots ≤ 0 or HALTED or PAUSED → skip to next day

  12. RANK: Compute VRP for all 193 symbols (ATM_IV² - RV_forecast)
  13. FILTER: Remove symbols failing any filter:
      - VRP ≤ 0
      - VRP ≤ symbol's historical median (computed on expanding window up to today)
      - VIX outside 14-30
      - days_to_earnings ≤ DTE + 5
      - is_fomc_week == 1
      - Already holding this symbol
  14. SORT: Rank surviving symbols by VRP descending

  15. Walk down the VRP ranking until slots are filled:
      For each candidate symbol (highest VRP first):
        a. Check sector limit: < 2 positions in same sector → skip if violated
        b. Check model disagreement: if LightGBM vs LogHAR differ > 50% → skip
        c. Find monthly expiration at DTE 21-30
        d. Find put closest to delta -0.175, check OI > 100
        e. Find wing strike $2 below (or $1/$3 per stock price), check OI > 100
        f. Compute entry premium = short_bid - long_ask
        g. If premium < 30% of width → skip to next candidate
        h. Compute max_loss = width - premium
        i. If max_loss > risky_capital × kelly_fraction → skip (can't afford)
        j. OPEN POSITION:
           - Deduct commission: risky_capital -= $1.30 (entry half of round-trip)
           - Add to open_positions with: symbol, entry_date, short_strike,
             long_strike, expiration, entry_premium, max_loss, VRP, VRP_rank
           - Decrement available_slots
        k. If available_slots == 0 → stop searching

  === END OF DAY ===
  16. Log daily snapshot: date, risky_capital, n_positions, daily_pnl
```

**Key difference from Level 1.5:** The daily ranking means only the TOP VRP symbols get traded on any given day. If 25 symbols qualify but only 1 slot is open, only the #1-ranked symbol gets a spread. This is how the strategy actually works; the model's primary value is in *ranking*, not just binary signal generation.

**Metrics:**

| Metric | What It Measures | Pass Threshold |
|--------|-----------------|----------------|
| Net P&L per trade | Average after commissions | > $15 |
| Win rate | % of trades closed profitably | > 65% |
| Avg holding period | Days from entry to exit | 7-18 days |
| Sharpe ratio (annualized) | Risk-adjusted return | > 0.5 (Sinclair: 0.4-1.0 typical) |
| Max drawdown | Peak-to-trough of risky subaccount | < 43% (Sinclair trailing stop) |
| Profit factor | Gross wins / gross losses | > 1.5 |
| Commission ratio | Total commissions / total gross P&L | < 15% |
| Exit rule value-add | Level 2 Sharpe vs Level 1.5 Sharpe | Level 2 > Level 1.5 |
| 50% profit target hit rate | How often the early exit fires | Document |
| Avg P&L by exit reason | Which exit rules help, which hurt | Document |

**Comparison to Level 1.5:**
- Does management improve mean P&L? (Should: exits cut losses, capture profits early)
- Does management reduce max drawdown? (Should: breach exits prevent full max loss)
- Does management reduce negative skewness? (Should: early exits cap left tail)
- Does ranking+portfolio selection improve per-trade P&L vs Level 1.5's unfiltered trades? (Should: only the best signals get capital)

**Output:** Trade log, daily equity curve, drawdown chart, monthly P&L breakdown, exit reason breakdown, daily portfolio snapshot.

**Gate:** If Sharpe < 0.3 or max drawdown > 50%, the strategy is not viable at this capital level.

**Effort:** 2 days (daily ranking loop, position state machine, contract tracking across days, portfolio rules, capital accounting).

### Level 3: Robustness Checks

**Question:** Is the Level 2 result real, or a product of the specific historical path we tested?

#### 3A. Combinatorial Purged Cross-Validation (CPCV)

**Rationale (Lopez de Prado):** Walk-forward tests only one historical path. CPCV generates multiple backtest paths to produce a distribution of Sharpe ratios instead of a single number.

**Method:**
```
1. Partition the 756-day WF prediction period into N=6 chronological groups
   (~126 trading days each, roughly 6 months)
2. For each combination of k=2 test groups (C(6,2) = 15 combinations):
   a. Train set = remaining 4 groups
   b. Test set = selected 2 groups
   c. PURGE: remove train observations whose 21-day target overlaps any test date
   d. EMBARGO: add ~7-day buffer after each test block (LdP rule: h ≈ 0.01*T,
      with T=756 → ~7-8 days). This is separate from purging.
   e. Run the full Level 2 backtest on the test set
   f. Record Sharpe ratio for this combination
3. Stitch test groups into phi(6,2) = 5 complete backtest paths
4. Compute Sharpe ratio for each path
```

**Output:** Distribution of 5 Sharpe ratios (histogram + mean + std).

**Pass criterion:** Mean Sharpe > 0.3 across all paths, and no path has Sharpe < 0.

#### 3B. Deflated Sharpe Ratio (DSR)

**Rationale (Lopez de Prado):** The raw Sharpe from a backtest is inflated by selection bias, non-normal returns, and track record length. DSR corrects for all three.

**Method:**
```
Inputs:
  SR_hat  = observed Sharpe ratio from Level 2
  T       = number of return observations (trading days)
  gamma_3 = skewness of strategy returns
  gamma_4 = kurtosis of strategy returns
  N       = number of strategy variants tested (N=1 for us, pre-committed)
  V(SR)   = variance of Sharpe ratios across trials

Compute:
  SR_star = adjusted benchmark (function of N, V(SR))
  DSR     = PSR(SR_hat, SR_star, T, gamma_3, gamma_4)
```

**Pass criterion:** DSR > 0.95 (standard 5% significance level). Since we test only N=1 pre-committed strategy, the deflation penalty is minimal, but the skewness and kurtosis corrections matter for short premium strategies.

#### 3C. Regime Decomposition

Split Level 2 results by VIX regime:

| Regime | VIX Range | Expected Behavior |
|--------|-----------|------------------|
| Low vol | 14-18 | Lower returns, but should still be positive |
| Normal | 18-25 | Best returns (Sinclair: mid-level optimal) |
| Elevated | 25-30 | Wider spreads but more risk; should still work |

**Pass criterion:** Positive mean P&L in at least 2 of 3 regimes. Strategy should not depend on a single regime.

#### 3D. Sector Decomposition

Split results by GICS sector. Check for:
- Any sector where the strategy consistently loses
- Over-concentration in one sector driving results

#### 3E. Synthetic Stress Test (Optional)

**Rationale (Lopez de Prado):** "History is just one random path. Test on synthetic data to ensure robustness."

**Method:** Bootstrap the daily returns of the underlying stocks (block bootstrap to preserve autocorrelation), re-run Level 2 on 100 synthetic paths, check that median Sharpe remains positive.

**Effort for all Level 3:** 2-3 days.

## Biases and Mitigations

| Bias | Risk to Our Backtest | Mitigation |
|------|---------------------|------------|
| **Look-ahead** | Using future data for decisions | Walk-forward predictions are out-of-sample by construction. Features (atm_iv, VIX, earnings) are observable at the decision date. |
| **Survivorship** | All 193 symbols survived to 2026 | Acknowledged. Slight positive bias. Could test on subset excluding symbols added in 2026 expansion. |
| **Selection bias** | Optimizing parameters after seeing results | ALL parameters pre-committed (this document). No tuning after any level runs. |
| **Fill assumption** | Assuming better fills than reality | Use bid for selling, ask for buying. Never mid-price. (Sinclair: "if still profitable after crossing the spread, it is a robust trade.") |
| **Timing cost** | Market moves between decision and fill | Not modeled explicitly. Mitigated by using EOD prices (signal computed overnight, executed at open). Residual risk acknowledged. |
| **Market impact** | Our order moves the price | At 1-contract size on liquid names (OI > 100), market impact is negligible. |
| **Commission model** | Understating transaction costs | $0.65/contract (IBKR standard). $2.60 round-trip per vertical. Applied on every entry AND exit. |
| **Regime bias** | Strategy works only in 2023-2025 conditions | Level 3C regime decomposition explicitly tests this. |
| **Overfitting exit rules** | Exit rules tuned to this dataset | Exit rules come from Sinclair (50% profit target) and Natenberg (DTE gamma risk), not from data. |

## Implementation Sequence

| Step | Level | What | Depends On | Effort |
|------|-------|------|-----------|--------|
| 1 | 1 | VRP signal quality + daily ranking analysis | WF predictions + features | Half day |
| 2 | — | Contract matcher function (find best spread for symbol+date) | options_iv data | Half day |
| 3 | 1.5 | Terminal P&L distribution (all qualifying signals, no portfolio limits) | Steps 1-2 | Half day |
| 4 | — | Position state machine (entry → daily track → exit triggers) | Raw EOD data fallback | 1 day |
| 5 | — | Daily ranking loop + portfolio manager (slots, sectors, capital) | Steps 2-4 | 1 day |
| 6 | 2 | Full managed backtest with daily workflow | Steps 2-5 | 1 day |
| 7 | — | Reporting notebook (equity curve, metrics, charts) | Step 6 | Half day |
| 8 | 3A | CPCV (multiple backtest paths) | Step 6 | 1 day |
| 9 | 3B | Deflated Sharpe Ratio | Step 6 | Half day |
| 10 | 3C-D | Regime + sector decomposition | Step 6 | Half day |
| 11 | 3E | Synthetic stress test (optional) | Step 6 | 1 day |
| **Total** | | | | **~8 days** |

## Decision Framework

After all levels complete:

```
Level 1 VRP accuracy < 55%?
  → STOP. Signal has no edge. Diagnose model or feature set.

Level 1.5 mean terminal P&L < $0?
  → STOP. Strategy loses money even without management.
     Check: are commissions the problem? Are fills too conservative?

Level 2 Sharpe < 0.3?
  → STOP. Strategy is not viable at EUR 10,000.
     Consider: higher capital? Different spread types?

Level 2 max drawdown > 50%?
  → STOP. Tail risk too high. Tighten filters or reduce position count.

Level 3A mean CPCV Sharpe < 0.1?
  → STOP. Result is path-dependent, not robust.

Level 3B DSR < 0.95?
  → STOP. Result is not statistically significant after corrections.

Level 3C loses money in 2+ regimes?
  → CAUTION. Strategy only works in specific conditions.
     Add regime filter or reduce scope.

ALL LEVELS PASS?
  → Proceed to paper trading (2-3 months, 30+ trades).
  → Then live trading with EUR 2,000.
```

## Output Artifacts

| File | Level | Contents |
|------|-------|---------|
| `data/processed/backtest/vrp_signal_quality.parquet` | 1 | Per-signal VRP accuracy |
| `data/processed/backtest/terminal_pnl.parquet` | 1.5 | Unmanaged spread P&L |
| `data/processed/backtest/trade_log.parquet` | 2 | Full trade log with entries, exits, P&L |
| `data/processed/backtest/daily_equity.parquet` | 2 | Daily portfolio value |
| `data/processed/backtest/metrics.json` | 2 | Summary metrics (Sharpe, win rate, etc.) |
| `data/processed/backtest/cpcv_sharpes.json` | 3A | Distribution of Sharpe ratios |
| `data/processed/backtest/dsr.json` | 3B | Deflated Sharpe Ratio + inputs |
| `notebooks/backtest_results.ipynb` | All | Visualizations, charts, analysis |

## Literature References

- **Lopez de Prado, M.** *Advances in Financial Machine Learning* (2018): CPCV, DSR, PBO, purging, embargo, backtesting sins, "backtesting is not a research tool"
- **Sinclair, E.** *Positional Option Trading* (2020): Terminal P&L simulation, bid/ask fill assumptions, transaction cost modeling, unhedged backtesting, timing cost, mid-price fallacy
- **Natenberg, S.** *Option Volatility and Pricing*: Gamma/theta dynamics informing DTE exit rule, spread mechanics
