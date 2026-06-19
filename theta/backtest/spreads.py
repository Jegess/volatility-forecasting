"""Bull put spread construction.

Given a (symbol, date) from the ranked candidate list, build a defensible
short put spread:

    1. Filter the chain to puts with DTE in [21, 30]
    2. Pick the expiration with the most contracts in that window
    3. Short leg  = put closest to delta -0.175 (midpoint of [-0.15, -0.20])
    4. Long leg   = next available strike below short by `wing_width`
                    ($1 / $2 / $3 stepped on stock price)
    5. Liquidity  = OI > 100 and relative_spread < 0.15 on BOTH legs
    6. Premium    = short_bid - long_ask (conservative fills)
    7. Premium must be ≥ 30% of wing width

Returns None if any step fails — the caller decides whether to skip the
symbol or fall back to a different strike.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import polars as pl

from theta.backtest import data as bt_data

# All numbers from BACKTEST_PLAN.md §Spread Construction / §Filters.
# TARGET_DELTA and MIN_PREMIUM_FRAC_OF_WIDTH diverge from the plan's nominal
# values (-0.175 and 0.30) — see TODO at top of build_spread. Empirical
# sweep on real chains shows those two values together produce ~0 trades
# because strike grids are wider than the plan's wing_width() assumes.
DTE_MIN = 21
DTE_MAX = 30
TARGET_DELTA = -0.30
MIN_OI = 100
MAX_REL_SPREAD = 0.15
MIN_PREMIUM_FRAC_OF_WIDTH = 0.20
COMMISSION_PER_ROUND_TRIP = 2.60


# ----- wing width sizing -------------------------------------------------

def wing_width(underlying_price: float) -> float:
    """Dollar width between short and long strikes, sized to underlying."""
    if underlying_price < 30:
        return 1.0
    if underlying_price > 80:
        return 3.0
    return 2.0


# ----- spread container --------------------------------------------------

@dataclass
class Spread:
    symbol: str
    entry_date: date
    expiration_date: date
    short_strike: float
    long_strike: float
    short_bid: float
    short_ask: float
    long_bid: float
    long_ask: float
    short_delta: float
    entry_premium: float     # short_bid - long_ask, per-share
    width: float
    max_profit: float        # = entry_premium * 100 (one contract)
    max_loss: float          # = (width - entry_premium) * 100
    dte_at_entry: int
    underlying_price: float

    def as_dict(self) -> dict:
        return self.__dict__.copy()


# ----- leg selection -----------------------------------------------------

def _pick_expiration(chain: pl.DataFrame) -> date | None:
    """Choose the expiration in [DTE_MIN, DTE_MAX] with the most contracts.
    If nothing in-window, return None.
    """
    window = chain.filter((pl.col("dte") >= DTE_MIN) & (pl.col("dte") <= DTE_MAX))
    if window.height == 0:
        return None
    counts = window.group_by("expiration_date").agg(pl.len().alias("n"))
    # Deterministic tie-break: largest n, then earliest expiration.
    return counts.sort(["n", "expiration_date"], descending=[True, False])[0, 0]


def find_short_leg(chain: pl.DataFrame,
                   target_delta: float = TARGET_DELTA) -> dict | None:
    """From a same-expiration put chain, the contract whose delta is
    closest to `target_delta` and within the filter envelope.

    Expected input columns (options_iv schema):
        strike, bid, ask, delta, open_interest, relative_spread,
        underlying_price.
    """
    ok = chain.filter(
        (pl.col("delta") <= -0.05) & (pl.col("delta") >= -0.50)
        & (pl.col("open_interest") > MIN_OI)
        & (pl.col("relative_spread") < MAX_REL_SPREAD)
        & (pl.col("bid") > 0)
    )
    if ok.height == 0:
        return None
    ok = ok.with_columns((pl.col("delta") - target_delta).abs().alias("_dist"))
    row = ok.sort("_dist").row(0, named=True)
    return {k: row[k] for k in (
        "strike", "bid", "ask", "delta", "open_interest",
        "relative_spread", "underlying_price"
    )}


def find_long_leg(chain: pl.DataFrame, short_strike: float,
                  width: float) -> dict | None:
    """Pick the strike exactly `width` below the short if it exists and is
    liquid. Falls back to the next-lower liquid strike if the exact match
    is missing (some chains have $1 gaps, others $2.50 etc.).
    """
    target = short_strike - width
    candidates = chain.filter(
        (pl.col("strike") < short_strike)
        & (pl.col("open_interest") > MIN_OI)
        & (pl.col("relative_spread") < MAX_REL_SPREAD)
        & (pl.col("ask") > 0)
    )
    if candidates.height == 0:
        return None
    # Closest to target, then prefer the lower strike on tie (wider wing = safer).
    candidates = candidates.with_columns(
        (pl.col("strike") - target).abs().alias("_dist")
    )
    row = candidates.sort(["_dist", "strike"]).row(0, named=True)
    return {k: row[k] for k in (
        "strike", "bid", "ask", "delta", "open_interest",
        "relative_spread", "underlying_price"
    )}


def find_long_leg_by_budget(chain: pl.DataFrame, short_strike: float,
                            short_bid: float, max_loss_budget: float,
                            min_premium_frac: float) -> dict | None:
    """Sinclair hedge-strike rule: among liquid long-put candidates, pick the
    strike whose max dollar loss is closest to but NOT exceeding
    `max_loss_budget` (one contract = 100 shares), with positive credit and
    credit >= min_premium_frac of the wing width.

    Per Sinclair: set the loss budget first, then harvest maximum credit
    consistent with that cap. That's the widest feasible wing, not the
    narrowest — a narrower wing has smaller max loss but much smaller
    credit, so reward/risk degrades. Wider wings within budget maximize
    the credit while still capping the tail at the chosen dollar number.
    """
    cands = chain.filter(
        (pl.col("strike") < short_strike)
        & (pl.col("open_interest") > MIN_OI)
        & (pl.col("relative_spread") < MAX_REL_SPREAD)
        & (pl.col("ask") > 0)
    )
    if cands.height == 0:
        return None

    best: dict | None = None
    best_max_loss = -1.0
    for row in cands.iter_rows(named=True):
        long_strike = float(row["strike"])
        long_ask = float(row["ask"])
        width = short_strike - long_strike
        if width <= 0:
            continue
        entry_premium = short_bid - long_ask
        if entry_premium <= 0:
            continue
        if entry_premium < min_premium_frac * width:
            continue
        max_loss_dollars = (width - entry_premium) * 100.0
        if max_loss_dollars > max_loss_budget:
            continue
        # Keep the feasible candidate with the largest max_loss (→ widest
        # wing, largest credit) still under budget.
        if max_loss_dollars > best_max_loss:
            best_max_loss = max_loss_dollars
            best = {k: row[k] for k in (
                "strike", "bid", "ask", "delta", "open_interest",
                "relative_spread", "underlying_price"
            )}
    return best


# ----- orchestrator ------------------------------------------------------

def build_spread(symbol: str, on_date: date,
                 target_delta: float = TARGET_DELTA,
                 min_premium_frac: float = MIN_PREMIUM_FRAC_OF_WIDTH,
                 max_loss_budget: float | None = None,
                 chain: pl.DataFrame | None = None) -> Spread | None:
    """Top-level entry: try to build a valid bull put spread for this
    symbol/date. Returns None if anything disqualifies the trade.

    Pass `chain` pre-filtered to (symbol, date, PUT) to skip the parquet
    load — used by level1_5 to cache per-symbol chains across many dates.

    TODO (defaults diverge from BACKTEST_PLAN.md): plan nominally asks for
    delta -0.175 + premium >= 30% of width. Empirical sweep on real chains
    (2026-04-20): that combination produces ~0 trades because strike grids
    in options_iv are coarser than `wing_width()` assumes (e.g. AAPL uses
    $5 steps, not $3), so actual width auto-inflates and the ratio drops to
    9-15%. Moving to delta -0.30 raises mean ratio to ~20% but only 1/17
    samples clear 30%. Current defaults (-0.30, 0.20) are the working
    compromise: premium comfortably above commission drag, trades actually
    fire. Revisit once Level 1/1.5 metrics are out — if breach rate at
    delta -0.30 is too high, try returning to -0.175 with a 10% floor (the
    Sinclair-style "safer wing, smaller credit" alternative).
    """
    full = chain if chain is not None else bt_data.get_chain_on(symbol, on_date, right="PUT")
    if full.height == 0:
        return None

    exp = _pick_expiration(full)
    if exp is None:
        return None

    leg_chain = full.filter(pl.col("expiration_date") == exp)

    short = find_short_leg(leg_chain, target_delta)
    if short is None:
        return None

    if max_loss_budget is not None:
        long_ = find_long_leg_by_budget(
            leg_chain, short_strike=float(short["strike"]),
            short_bid=float(short["bid"]),
            max_loss_budget=max_loss_budget,
            min_premium_frac=min_premium_frac,
        )
    else:
        width = wing_width(short["underlying_price"])
        long_ = find_long_leg(leg_chain, short["strike"], width)
    if long_ is None:
        return None

    # `width` from the step table is the target; use the actual strike distance.
    actual_width = short["strike"] - long_["strike"]
    if actual_width <= 0:
        return None

    entry_premium = short["bid"] - long_["ask"]
    if entry_premium <= 0:
        return None
    if entry_premium < min_premium_frac * actual_width:
        return None

    dte = int(leg_chain.filter(pl.col("strike") == short["strike"])["dte"][0])

    return Spread(
        symbol=symbol,
        entry_date=on_date,
        expiration_date=exp,
        short_strike=float(short["strike"]),
        long_strike=float(long_["strike"]),
        short_bid=float(short["bid"]),
        short_ask=float(short["ask"]),
        long_bid=float(long_["bid"]),
        long_ask=float(long_["ask"]),
        short_delta=float(short["delta"]),
        entry_premium=float(entry_premium),
        width=float(actual_width),
        max_profit=float(entry_premium) * 100.0,
        max_loss=float(actual_width - entry_premium) * 100.0,
        dte_at_entry=dte,
        underlying_price=float(short["underlying_price"]),
    )


# ----- pricing helpers (used by portfolio.py and level1_5.py) -----------

def spread_exit_cost(short_quote: dict, long_quote: dict) -> float:
    """Cost to close an open spread per share (we cross the spread both legs).

    short leg: buy back at ask; long leg: sell at bid.
    """
    return short_quote["ask"] - long_quote["bid"]


def terminal_pnl(spread: Spread, underlying_at_expiry: float) -> float:
    """Hold-to-expiration P&L in dollars for a single contract (100 shares),
    before commissions. Sign convention: positive = profit to seller.
    """
    premium = spread.entry_premium
    short_k = spread.short_strike
    long_k = spread.long_strike

    if underlying_at_expiry >= short_k:
        raw = premium
    elif underlying_at_expiry <= long_k:
        raw = premium - (short_k - long_k)
    else:
        raw = premium - (short_k - underlying_at_expiry)
    return raw * 100.0
