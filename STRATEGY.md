# Trading Strategy: VRP-Based Option Premium Selling

## Overview

Harvest the Variance Risk Premium (VRP), the empirical tendency for implied volatility to systematically exceed realized volatility, using risk-defined option spreads on US equities. Use a walk-forward-validated LightGBM model to estimate 21-day realized volatility more accurately than traditional baselines, improving VRP measurement and trade selection.

Sinclair defines the variance premium as "the dominant force" in option trading, analogous to "what evolution is to biology." It persists because of structural demand for portfolio insurance, compensation for jump risk, trading restrictions on retail short-selling, and market-maker inventory needs. This is the edge. The ML model does not create the edge; it measures the premium more accurately.

> **Note:** All capital and position-size figures in this document are illustrative, based on a notional small retail account. They are worked examples of the sizing methodology, not a statement of actual funds.

**Account type:** retail margin account (developed against the Interactive Brokers API)
**Data cost:** $0/month (free sources only, see Section 4)

## 1. The Edge: What We're Capturing

### Variance Risk Premium (Sinclair)

The VRP is the difference between what the options market implies about future volatility and what actually materializes:

```
VRP = IV² - RV_realized
```

This premium exists because:
- **Insurance demand:** Investors overpay for OTM puts (downside protection) and calls (FOMO protection)
- **Jump risk compensation:** Dynamic delta-hedging cannot replicate an option's payoff through price gaps; options provide unique jump protection, making them expensive
- **Trading restrictions:** Many participants cannot sell options, forcing a net-long-volatility bias in the market
- **Market-maker inventory:** MMs need to be net long options as business insurance, buying even when systematically overpriced

The premium is not a market inefficiency that will be arbitraged away; it is structural compensation for bearing risk. Our ML model helps us measure it more precisely.

### What Our Model Actually Does

Sinclair warns: "Use models for sizing, not signals." The VRP is the signal. Our LightGBM model provides a better estimate of RV_forecast, which gives us:
- **Better VRP measurement:** avoiding trades where VRP appears positive but isn't
- **Better symbol ranking:** identifying where the premium is genuinely richest
- **Better position sizing:** more accurate edge estimates feed into Kelly calculations

The model does NOT discover hidden alpha. If the VRP phenomenon disappeared, the model would be worthless.

## 2. Model Hierarchy (Post Walk-Forward Validation)

Walk-forward validation (12 quarterly windows, expanding train) revealed the true robustness ranking:

| Model | WF Mean QLIKE | WF Std | Range | Verdict |
|-------|--------------|--------|-------|---------|
| **LightGBM** | **0.0231** | 0.003 | 0.019-0.030 | **Robust winner, use this** |
| LogHAR | 0.0280 | 0.005 | 0.021-0.040 | Reliable sanity check |
| LSTM | 0.0428 | 0.013 | 0.030-0.079 | **Failed, do not use live** |

### Decision Rules

- **Primary forecast:** LightGBM (17% better than LogHAR across all market regimes)
- **Sanity check:** LogHAR (if LightGBM and LogHAR disagree by >50%, reduce confidence)
- **No ensemble:** Averaging a robust model with an unstable one degrades the robust model
- **LSTM is retired** from live trading; single-split performance (QLIKE=0.0160) was genuine but non-generalizable with quarterly retraining on this data history

## 3. Capital Management: Sinclair's Subaccount Method

### Why Not Half-Kelly

Sinclair warns that even at half-Kelly, there is a **25% chance of overbetting**. Short options have negative skewness (capped upside, large downside), which makes standard Kelly approximations dangerously optimistic. For some short-vol strategies, there is a 7% probability the true Kelly fraction is actually below zero.

### Subaccount Structure

Split capital into two portions (Sinclair, *Positional Option Trading*):

| Portion | Amount | Purpose |
|---------|--------|---------|
| **Risky subaccount** | 44% = **EUR 880** (~$950) | Active trading, apply full Kelly here |
| **Cash reserve** | 56% = **EUR 1,120** (~$1,200) | Never traded, structural protection |

- Apply **full Kelly criterion to the risky subaccount only**
- This retains aggressive growth on the traded portion while structurally capping total portfolio drawdown
- **Trailing stop at 43%:** If the risky subaccount drops 43% from its peak value, stop all trading and reassess
- When the risky subaccount grows, ratchet the trailing stop up (percentage-based, tracks peak)

### Practical Sizing at EUR 880 Risky Capital

```
max_risk_per_trade = risky_capital * kelly_fraction
```

With estimated win rate ~70%, avg win/loss ratio ~0.4 (premium selling):
- Kelly fraction ≈ 0.17
- Max risk per trade = 880 * 0.17 = ~EUR 150 ($162)
- At $2 wing width: 1 contract per trade
- **Max concurrent positions: 2-3** (total risk < risky subaccount)

As the risky subaccount grows, position sizes scale proportionally.

## 4. Data Sources (Zero Cost)

**ThetaData subscription: CANCELLED.** At $80/month, the data cost is a prohibitive annual drag on a small retail account, more than any realistic strategy return.

The models are already trained. For live inference we need:

| Data | Source | Cost | Update Frequency |
|------|--------|------|-----------------|
| Daily close prices (193 symbols) | Yahoo Finance or IBKR API | Free | Daily |
| VIX, VVIX | Yahoo Finance / CBOE | Free | Daily |
| Macro (term spread, credit spread, EPU, ADS) | FRED API (free key) | Free | Daily/weekly |
| T-bill rate | FRED (DTB3) | Free | Daily |
| Earnings calendar | Yahoo Finance / SEC EDGAR | Free | Quarterly |
| FOMC / CPI dates | Federal Reserve / BLS calendars | Free | Annual (static) |
| Option chains (for IV check + trade execution) | IBKR API | Free with account | Real-time |

### Feature Strategy: Start Lite, Upgrade Later

**SHAP importance by feature group:**

| Group | Features | Cumulative SHAP | Source |
|-------|----------|----------------|--------|
| RV | 6 | ~0.81 | Free (daily prices) |
| Technical | 9 | ~0.06 | Free (daily prices) |
| Macro | 8 | ~0.04 | Free (FRED/Yahoo) |
| Events | 6 | ~0.03 | Free (calendars) |
| **Option-implied** | **15** | **~0.06** | **Broker API** |

`rv_m` alone (SHAP 0.507) contributes more than all 15 option-implied features combined.

**Phase 1 (start here):** Retrain LightGBM on 29 free features (drop option-implied). Test QLIKE loss. If <10% degradation, use this model.

**Phase 2 (upgrade):** Compute option-implied features from IBKR option chain snapshots at market close. Use full 44-feature model. Risk: distributional shift from ThetaData quotes vs IBKR quotes; validate before trusting.

## 5. VRP Signal Construction

### Daily Calculation

```
VRP_i = ATM_IV_i² - RV_forecast_i
```

- `ATM_IV`: current at-the-money implied volatility from broker option chain (annualized, squared to variance)
- `RV_forecast`: 21-day forward realized volatility predicted by LightGBM (already in annualized variance space)

### Ranking

Each trading day:
1. Compute features for all symbols in universe
2. Run LightGBM inference → RV_forecast per symbol
3. Also run LogHAR → RV_loghar as sanity check
4. Pull current ATM IV from IBKR for top candidates
5. Compute VRP, rank descending
6. Top quintile (~38 symbols) are initial candidates

### Model Disagreement Flag

```
If |RV_lgbm - RV_loghar| / RV_loghar > 0.50:
    → Model disagreement. Halve position size or skip entirely.
```

## 6. Trade Filters (All Must Pass)

| # | Filter | Rule | Rationale |
|---|--------|------|-----------|
| 1 | **VRP positive** | `VRP > 0` | No edge if options are cheap |
| 2 | **VRP strength** | `VRP > median(historical VRP for symbol)` | Don't trade marginal premium |
| 3 | **VIX regime** | `14 < VIX < 30` | Sinclair: mid-level VIX is optimal. Low VIX = poor returns/margin + expensive wings ("ultimate sucker bets"). High VIX = dangerous risk profile. |
| 4 | **No earnings** | `days_to_earnings > DTE + 5` | Gap risk is unhedgeable |
| 5 | **No FOMC week** | `is_fomc_week == 0` for macro-sensitive names | Vol crush is priced in |
| 6 | **Liquidity** | Bid-ask spread < 15% of mid at target strikes | Slippage kills small accounts |
| 7 | **Open interest** | OI > 100 at target strikes | Must be able to exit |
| 8 | **No duplication** | No existing position in this symbol | No doubling down |
| 9 | **Sector limit** | < 2 positions in same sector | Correlation protection |

## 7. Strategy Selection

### Tier 1: Bull Put Spreads (PRIMARY, ~70% of trades)

**When:** VRP positive + no strong bearish signal.

Sinclair calls it "the most conservative directional strategy": high winning percentage, high median return, capped downside. Also "the safest way to collect the implied skew premium" without naked tail risk.

- Sell OTM put at 15-20 delta
- Buy protective put $1-3 below (wing)
- Width: $1-2 on stocks under $50, $2-3 on stocks $50-150
- **Premium target: collect ≥ 30% of width**
- Max loss: width minus premium collected
- 2 legs = low commission ($2.60 round trip at IBKR)

### Tier 2: Bear Call Spreads (SECONDARY, ~20% of trades)

**When:** VRP positive + stock has bearish momentum, near resistance, or RSI > 70.

Same mechanics as bull put spread, inverted:
- Sell OTM call at 15-20 delta
- Buy protective call $1-3 above
- Same premium and width rules

### Tier 3: Iron Condors (SELECTIVE, ~10% of trades)

**When:** VRP very high (top quartile) + no directional signal + stock range-bound.

Sinclair warns: iron condors realize maximum possible loss "a significant amount of the time (26% in simulations)." Use selectively.

- Sell OTM put spread + OTM call spread
- Short strikes at 15-20 delta each side
- **Premium target: collect ≥ 25% of total width**
- 4 legs = double commissions ($5.20 round trip)
- Only when the extra premium justifies the extra cost and max-loss frequency

### Future Consideration: Calendar Spreads

Sinclair says calendars (sell 30-day, buy 60-day straddle at same strike) capture the steep short-term variance premium. But this is a **long vega** position, so it underperforms if overall IV drops. Use only when:
- VRP is large (short-term premium rich)
- Overall IV is relatively low (won't decline further)
- Term structure is steep (`iv_term_slope` strongly positive)

Defer until Phase 2 (full 44-feature model with term slope data).

### What We Don't Trade

| Strategy | Why Not |
|----------|---------|
| Naked puts/calls | Unlimited risk on EUR 2,000; never |
| Ratio spreads | Naked exposure on one side |
| Straddles/strangles | Undefined risk without broker-dealer margin (Sinclair: ~$100K margin for 1-lot on $100 stock) |
| Christmas trees / broken-wing butterflies | Too complex to systematize at this scale |
| Deep OTM "teenies" | Sinclair: "the ultimate sucker bets", highest IV but catastrophic tail risk |
| Dynamic delta-hedging | Sinclair: "most traders who are not broker-dealers should probably avoid it", transaction costs destroy the premium |

## 8. Strike & Expiration Selection

### Expiration (DTE)

**Target: 21-30 DTE.** Rationale:

- Sinclair: variance premium is highest in short-dated options, concentrated in "the few days immediately preceding expiration"
- But short-dated = dangerous gamma risk: "vega wounds but gamma kills"
- Natenberg: ATM theta accelerates from -0.03/day at 3 months to -0.16/day at 3 days, so the last week is the most dangerous
- Our model forecasts 21-day RV, so align entry with the forecast horizon
- 21-30 DTE captures accelerating theta decay while leaving margin before gamma explosion

**Practical rules:**
- Use monthly expirations (3rd Friday), which are more liquid than weeklies
- If 3rd Friday is 15-20 DTE: acceptable but be ready to exit early
- If 3rd Friday is < 14 DTE: skip to next month
- If 3rd Friday is > 35 DTE: acceptable, but theta decay is slower

### Strikes

- **Short strike:** 15-20 delta (from broker option chain)
- **Long strike (wing):** $1-3 below/above depending on underlying price
  - Stocks < $30: $1 wide
  - Stocks $30-80: $2 wide
  - Stocks > $80: $2.50-3 wide (but check max loss vs sizing limit)

**Natenberg's rule for high-IV environments:** focus on selling the ATM option and buying OTM protection. The ATM option has the most absolute sensitivity to volatility drops.

**Sinclair's warning on OTM put selection:** Don't chase the highest implied volatility in deep OTM puts. Find a strike that balances the skew premium with acceptable risk; maximize risk-adjusted return, not raw expected value.

### Premium Check

Before executing, verify:
- Vertical spread: premium ≥ 30% of width
- Iron condor: premium ≥ 25% of total width
- If not → strikes too far OTM, or IV too low → skip this trade

## 9. Greeks Check (Before Execution)

### Natenberg's Efficiency Metric: Gamma/Theta Ratio

For short premium positions, the key metric is **|gamma/theta|**: you want low gamma (risk) relative to high theta (reward).

| Greek | What to Check | Red Flag |
|-------|--------------|----------|
| **Gamma/Theta** | Efficiency of the spread | Ratio worsening (gamma growing faster than theta), meaning risk is outpacing reward |
| **Theta** (spread) | Daily time decay in your favor | < $0.50/day → not worth the commission |
| **Vega** (spread) | Sensitivity to IV change | If vega > 2× theta → you're making an IV bet, not a theta trade |
| **Delta** (spread) | Net directional exposure | > ±0.15 for "neutral" trades → adjust or skip |
| **Max loss** | Width - premium | > Kelly-sized risk for risky subaccount → reduce width or skip |

### Natenberg on Delta Management

For defined-risk spreads (our case), Natenberg offers four approaches:
1. **Adjust at predetermined delta limit:** if net delta drifts beyond ±0.25, consider closing
2. **Do not adjust:** for defined-risk spreads, accept the directional risk within the defined loss
3. **If adjustment needed, use underlying stock:** doesn't alter gamma/theta/vega profile
4. **Never adjust by selling more options:** increases total position size and magnifies blowup risk

For our strategy: **do not delta-hedge.** The wing IS the hedge. If directional risk becomes uncomfortable, close the trade entirely.

## 10. Complete Position Opening Checklist

### Step 1: Daily Screening (before market open, ~15 min)

- [ ] Pull last 252 days of close prices for all symbols → Yahoo Finance
- [ ] Compute RV features (rv_d, rv_w, rv_m, rq, rs_pos, rs_neg)
- [ ] Compute technical features (momentum, RSI, MA crossovers)
- [ ] Pull macro data (VIX, VVIX, term spread, credit spread, EPU, tbill, ADS, HSI vol) → FRED/Yahoo
- [ ] Compute event features (days_to_fomc/earnings/cpi, is_X_week)
- [ ] Run LightGBM inference → RV_forecast per symbol
- [ ] Run LogHAR → RV_loghar per symbol

### Step 2: VRP Ranking (~5 min)

- [ ] Pull current ATM IV for top ~50 candidates from IBKR
- [ ] Compute `VRP = ATM_IV² - RV_forecast`
- [ ] Rank by VRP descending
- [ ] Flag model disagreements (LightGBM vs LogHAR > 50%)

### Step 3: Apply Filters (pass/fail per candidate)

- [ ] VRP > 0
- [ ] VRP > historical median for this symbol
- [ ] 14 < VIX < 30
- [ ] No earnings within DTE + 5 days
- [ ] Not FOMC week (for macro-sensitive names)
- [ ] Bid-ask spread < 15% of mid
- [ ] OI > 100 at target strikes
- [ ] No existing position in this symbol
- [ ] Sector limit not exceeded

### Step 4: Select Strategy

- [ ] Check RSI, momentum, support/resistance for directional lean
- [ ] Choose: bull put (default) / bear call (bearish) / iron condor (strong VRP + neutral)

### Step 5: Select Strikes & Expiration

- [ ] Find monthly expiration at 21-30 DTE
- [ ] Identify short strike at 15-20 delta
- [ ] Set wing width ($1-3 based on stock price)
- [ ] Verify premium ≥ 30% of width (verticals) or ≥ 25% (IC)

### Step 6: Greeks Check

- [ ] Theta > $0.50/day
- [ ] Vega < 2× theta
- [ ] Net delta < ±0.15 (for neutral trades)
- [ ] Max loss within Kelly-sized risk limit

### Step 7: Size & Execute

- [ ] Calculate position size: `n_contracts = floor(kelly_risk / max_loss_per_contract)`
- [ ] Enter as **spread order** (not individual legs) on IBKR
- [ ] **Limit orders only:** start at mid-price, walk toward natural if not filled in 5 min
- [ ] Execute within first 2 hours of market open (peak liquidity)

### Step 8: Log the Trade

Record: symbol, date, strategy, strikes, DTE, premium, max loss, VRP at entry, RV_forecast, LogHAR forecast, VIX at entry, sector.

## 11. Exit Rules

### Mandatory Exits

| Trigger | Action | Rationale |
|---------|--------|-----------|
| **Profit ≥ 50% of max** | Close immediately | Sinclair: take profits early; waiting for 100% exposes you to reversal risk for diminishing marginal gain |
| **DTE ≤ 10** | Close regardless of P&L | Natenberg: gamma accelerates dramatically in final days, theta from -0.03 to -0.16 per day. Not worth the risk. |
| **Underlying breaches short strike** | Close | Directional risk has taken over; expected value turns negative |
| **Loss > 2× premium collected** | Close | You're wrong. Preserve capital. |
| **Earnings announced within remaining DTE** | Close immediately | Even if profitable, gap risk is unhedgeable |
| **VRP flips negative** (new model run) | Close or tighten | Your thesis is invalidated |
| **VIX crosses 30** | Close ALL positions | Regime shift; model unreliable above training range |
| **VIX drops below 14** | No new trades; manage existing | Low-vol regime: poor returns, expensive wings |

### Never Do

- Don't "roll" losing positions hoping for recovery; closing and re-entering is a new trade decision
- Don't add to losing positions (Natenberg: "never adjust by selling more options")
- Don't hold through earnings even if the position is profitable
- Don't override the 50% profit target; greed is the enemy of premium sellers

## 12. Risk Management

### Hard Stops

| Trigger | Action |
|---------|--------|
| Risky subaccount drops 43% from peak | **Stop all trading.** Reassess model, strategy, market regime. |
| Monthly drawdown > 8% of total capital (EUR 160) | Pause 5 trading days. Review every closed trade. |
| Any single trade loss > 7% of risky subaccount (~EUR 62) | Close. Reduce sizing for 2 weeks. |
| 3 consecutive losses | Pause 1 week. Compare model forecasts vs actual RV. |
| Win rate < 55% over 20+ trades | Stop live trading. Diagnose: is the model degraded, or are exits too slow? |

### Correlation Risk

With 2-3 concurrent positions in US equities:
- A broad market selloff hits ALL put spreads simultaneously
- **Max theoretical loss: ~EUR 450** (3 positions × EUR 150 max risk) = 51% of risky subaccount
- This would trigger the 43% trailing stop → forced pause

Mitigation:
- Max 2 positions in same sector
- Prefer symbols with low correlation to each other
- If holding 3 positions and VIX starts rising above 25, close the weakest one

### What Can Blow Up

1. **Correlated selloff:** All put spreads breached simultaneously. Survivable within defined risk but painful.
2. **Vol regime shift:** VIX jumps 18→35. Model was trained on VIX 12-25. Forecasts degrade. Hard stop at VIX 30 protects against this.
3. **Surprise earnings pre-announcement:** Filter says "no earnings" but company announces early. Check news before market open.
4. **Commission death spiral:** Making $25/winning trade, paying $2.60/trade in commissions. If win rate drops below 65%, commissions alone cause losses.
5. **Liquidity trap:** Enter a spread, underlying gaps, can't exit at reasonable price. OI > 100 filter helps but doesn't guarantee.
6. **Model degradation:** LightGBM was trained on 2022-2024 data. Market structure changes. Monitor forecast accuracy continuously.

## 13. Return Expectations

### Sinclair's Benchmarks

- Short premium strategies on equity indices: Sharpe ratio **0.4 to 1.0**
- Short straddle on S&P 500 historically profitable **78%** of the time
- With correct vol forecast: ~78% win rate. With no edge: 57%. When wrong: 25%.
- More complex implementations (fundamental sorts): monthly returns 3-5%, Sharpe 0.6-2.0

### Realistic Expectations (Notional Small Account)

| Scenario | Monthly Return | Annual | Notes |
|----------|---------------|--------|-------|
| **Good** (70% WR, avg +$25 net) | ~$50-75 | ~$600-900 | 3-4 trades/month, disciplined exits |
| **Base** (65% WR, avg +$20 net) | ~$25-40 | ~$300-480 | Commission drag, some losing months |
| **Bad** (55% WR, avg +$10 net) | ~$5-15 | ~$60-180 | Barely covers commissions |
| **Failure** (<55% WR) | Negative | Negative | Stop, diagnose, paper trade |

**This is proof-of-concept sizing.** The goal of the exercise is to validate the strategy mechanics at small scale before considering any larger allocation.

## 14. Daily Workflow

```
MORNING (before market open, 08:00-09:30 ET):
1. Run daily data pipeline (prices, macro, events)
2. Compute features for all symbols
3. Run LightGBM + LogHAR inference
4. Compute VRP, rank, filter
5. Identify 0-2 candidates

MARKET OPEN (09:30-11:30 ET):
6. Pull option chains for candidates from IBKR
7. Verify liquidity, premium, Greeks
8. Execute spread orders (limit only)

DAILY CHECK (any time, 5 min):
9. Check P&L on all open positions
10. Apply exit rules (50% profit, DTE, breach)
11. Log any closes
12. Check VIX level

WEEKLY REVIEW (Friday close):
13. Update performance tracker
14. Compare RV_forecast vs RV_actual for closed trades
15. Check win rate, commission drag
16. Decide if position count should change
```

## 15. Performance Tracking

### Per-Trade Log

| Field | Description |
|-------|-------------|
| Date open / close | Entry and exit dates |
| Symbol, sector | What and where |
| Strategy | Bull put / bear call / iron condor |
| Strikes, width, DTE at entry | Position details |
| Premium collected | Gross income |
| P&L (gross) | Actual profit/loss |
| Commissions | IBKR fees |
| P&L (net) | Gross minus commissions |
| VRP at entry | Signal strength |
| RV_forecast | Model prediction |
| RV_actual (21d after entry) | What actually happened |
| VIX at entry | Regime context |
| Exit reason | Profit target / DTE / breach / manual |

### Monthly Metrics

- Win rate (target: >65%)
- Average net P&L per trade
- Sharpe ratio (annualized, target: >0.5)
- Max drawdown of risky subaccount
- Forecast accuracy: mean |RV_forecast - RV_actual| / RV_actual
- Commission ratio: total commissions / total gross P&L (target: <15%)

### Model Health Check (Monthly)

- Is LightGBM still beating LogHAR on live forecasts?
- Has forecast error increased over the last 3 months?
- Any sector or symbol where model consistently fails?
- If model degrades: retrain on latest data (walk-forward style)

## 16. Paper Trading Phase (MANDATORY)

Before risking real capital:

1. **Build the daily pipeline** (data fetch → features → model → VRP → filter)
2. **Run for 2-3 months** generating daily signals
3. **Paper trade every signal:** record exactly what you would have done
4. **Track at least 30 paper trades** before going live
5. **Validate:** Is win rate >60%? Is forecast accuracy reasonable? Are there systematic failures?

If paper trading shows win rate <55% or systematic model failures → debug before going live.

## 17. Implementation Roadmap

| Step | Effort | Priority |
|------|--------|----------|
| Retrain LightGBM on 29 free features, compare QLIKE | 1 hour | **First, validates lite model** |
| Daily data fetcher (Yahoo + FRED + earnings) | 1-2 days | High |
| Feature computation pipeline (reuse theta/processing/) | 1 day | High |
| LightGBM inference + VRP scoring script | Half day | High |
| Position tracker (parquet log) | Half day | Medium |
| IBKR account setup + API connection | 1 day | Medium |
| Paper trading period (2-3 months, 30+ trades) | 0 code | **Critical** |
| Option chain fetcher via IBKR API | 1-2 days | Phase 2 |
| Full 44-feature model via IBKR data | 1 day | Phase 2 |
| Live order execution via IBKR API | 2-3 days | Last |

## 18. Literature References

- **Sinclair, E.** *Positional Option Trading*: VRP definition, subaccount sizing, Kelly warnings, spread selection, regime guidance, return expectations, transaction cost analysis
- **Natenberg, S.** *Option Volatility and Pricing*: Greeks management, gamma/theta efficiency, delta adjustment rules, theta decay curve, spread mechanics
- **Gu, Kelly & Xiu (2020):** ML asset pricing framework, R2_OOS metric, pooled panel approach
- **Corsi (2009):** HAR model (LogHAR baseline)
- **Patton (2011):** QLIKE loss function
- **Lopez de Prado (2018):** Purged k-fold CV, walk-forward validation
- **Bali, Engle & Murray (2016):** Option-implied feature taxonomy
- **Carr & Wu (2009):** Variance risk premium empirics
- **Bakshi, Kapadia & Madan (2003):** Risk-neutral moments, VRP decomposition
