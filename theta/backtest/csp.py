"""Cash-secured put construction.

Given a (symbol, date), pick a single short put on a monthly expiration:

    1. Filter the chain to puts with DTE in [DTE_MIN, DTE_MAX]
    2. Pick the expiration with the most contracts in that window
    3. Short leg = put closest to `target_delta` (default -0.30), satisfying
       delta window, OI, and relative-spread filters
    4. Premium = bid (we sell at bid — conservative fill)

Linear payoff below the strike: there is no long hedge leg. Capital is
reserved as `strike × 100` in cash (IBKR CSP margin rule).

Returns None if any step fails.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import polars as pl

from theta.backtest import data as bt_data

# Defaults — CSPs on index ETFs sit in the 30-60 DTE sweet spot (more theta
# per day of exposure than 21-30 without the pre-expiry gamma of 14 DTE).
DTE_MIN = 30
DTE_MAX = 60
TARGET_DELTA = -0.30
MIN_OI = 100
MAX_REL_SPREAD = 0.15
# Minimum credit as a % of strike — a CSP's "return on cash tied up". We
# refuse trades where theta doesn't justify the notional lock-up.
MIN_PREMIUM_YIELD = 0.003   # 0.3% of strike, per contract (~1% monthly ann.)
COMMISSION_PER_ROUND_TRIP = 1.30   # IBKR single-leg round trip (2 × $0.65)


# ----- contract container ------------------------------------------------

@dataclass
class CspContract:
    symbol: str
    entry_date: date
    expiration_date: date
    strike: float
    bid: float
    ask: float
    delta: float
    entry_premium: float           # per share = bid (we sell at bid)
    max_profit: float              # = entry_premium * 100 (one contract)
    notional_margin: float         # = strike * 100 (cash reserved)
    dte_at_entry: int
    underlying_price: float

    def as_dict(self) -> dict:
        return self.__dict__.copy()


# ----- expiration selection ----------------------------------------------

def _pick_expiration(chain: pl.DataFrame) -> date | None:
    """Expiration with the most contracts in [DTE_MIN, DTE_MAX]."""
    window = chain.filter((pl.col("dte") >= DTE_MIN) & (pl.col("dte") <= DTE_MAX))
    if window.height == 0:
        return None
    counts = window.group_by("expiration_date").agg(pl.len().alias("n"))
    return counts.sort(["n", "expiration_date"], descending=[True, False])[0, 0]


def find_short_put(chain: pl.DataFrame,
                   target_delta: float = TARGET_DELTA) -> dict | None:
    """Put closest to `target_delta` passing liquidity filters."""
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
        "relative_spread", "underlying_price", "dte",
    )}


# ----- orchestrator -------------------------------------------------------

def build_csp(symbol: str, on_date: date,
              target_delta: float = TARGET_DELTA,
              min_premium_yield: float = MIN_PREMIUM_YIELD,
              chain: pl.DataFrame | None = None) -> CspContract | None:
    """Build a single cash-secured put for `symbol` on `on_date`.

    Returns None if the chain is empty, no expiration fits the DTE window,
    no liquid put is near `target_delta`, or the premium yield is too thin
    to justify the notional reserved.
    """
    full = chain if chain is not None else bt_data.get_chain_on(symbol, on_date, right="PUT")
    if full.height == 0:
        return None

    exp = _pick_expiration(full)
    if exp is None:
        return None

    leg_chain = full.filter(pl.col("expiration_date") == exp)
    short = find_short_put(leg_chain, target_delta)
    if short is None:
        return None

    strike = float(short["strike"])
    bid = float(short["bid"])
    if bid <= 0:
        return None

    # Yield gate: bid / strike must clear min_premium_yield.
    if (bid / strike) < min_premium_yield:
        return None

    return CspContract(
        symbol=symbol,
        entry_date=on_date,
        expiration_date=exp,
        strike=strike,
        bid=bid,
        ask=float(short["ask"]),
        delta=float(short["delta"]),
        entry_premium=bid,
        max_profit=bid * 100.0,
        notional_margin=strike * 100.0,
        dte_at_entry=int(short["dte"]),
        underlying_price=float(short["underlying_price"]),
    )


# ----- pricing helpers ---------------------------------------------------

def csp_exit_cost(quote: dict) -> float:
    """Per-share cost to buy back the short put (we cross the spread → ask)."""
    return float(quote["ask"])


def terminal_pnl(contract: CspContract, underlying_at_expiry: float) -> float:
    """Hold-to-expiration P&L in dollars for one contract (100 shares),
    before commissions. Positive = profit to seller.

    Above strike: keep full premium.
    Below strike: premium − (strike − spot), per share; linear loss floor
    is (premium − strike) × 100 at spot=0.
    """
    premium = contract.entry_premium
    strike = contract.strike
    if underlying_at_expiry >= strike:
        raw = premium
    else:
        raw = premium - (strike - underlying_at_expiry)
    return raw * 100.0
