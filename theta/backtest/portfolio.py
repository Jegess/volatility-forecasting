"""Portfolio state machine for Level 2.

The Level 2 daily loop hands this module:
    1. Live quotes for every open position (mark_to_market)
    2. Today's VIX (check_exits)
    3. Approved new spreads to add (add_position)

This module owns:
    - cash accounting (risky_capital, peak_capital)
    - exit triggers in priority order (plan §Phase A step 4)
    - portfolio-level guards: halt at 43% trailing DD, pause at 8% monthly DD
    - capacity checks: max 3 concurrent, max 1 per symbol, max 2 per sector

All I/O (quote lookup, date iteration) lives in level2.py — this module is
pure state transitions so it can be unit-tested with synthetic quotes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum
from typing import Optional

from theta.backtest.spreads import Spread, terminal_pnl

# Parameters from BACKTEST_PLAN.md §Position Management / §Capital & Sizing.
MAX_POSITIONS = 3
MAX_PER_SECTOR = 2
COMMISSION_HALF = 1.30      # one entry OR one exit (IBKR 2 legs × $0.65)
COMMISSION_ROUND_TRIP = 2.60
TRAILING_STOP_FRAC = 0.43   # halt if risky_capital < peak × (1 - 0.43)
MONTHLY_DD_PAUSE_FRAC = 0.08
PAUSE_LENGTH_DAYS = 5
DTE_EXIT_THRESHOLD = 10
VIX_REGIME_EXIT = 30.0


class ExitReason(str, Enum):
    VIX_REGIME = "vix_regime"
    DTE_CLOSE = "dte_close"
    BREACH = "breach"
    STOP_LOSS = "stop_loss"
    PROFIT_TARGET = "profit_target"
    EXPIRATION = "expiration"


# ----- Position ----------------------------------------------------------

@dataclass
class Position:
    """One open bull put spread, tracked across days.

    `mark()` updates the per-day fields in-place; the caller feeds fresh
    quotes each day. `current_pnl_dollars` is per CONTRACT (100 shares),
    NOT per share, to match the trade log convention.
    """
    spread: Spread
    sector: str = "UNKNOWN"

    # Per-day state — None until first mark_to_market.
    as_of: Optional[date] = None
    short_quote: Optional[dict] = None
    long_quote: Optional[dict] = None
    underlying_price: Optional[float] = None
    exit_cost_per_share: Optional[float] = None   # short_ask - long_bid
    current_pnl_dollars: Optional[float] = None

    @property
    def symbol(self) -> str:
        return self.spread.symbol

    @property
    def entry_date(self) -> date:
        return self.spread.entry_date

    @property
    def expiration_date(self) -> date:
        return self.spread.expiration_date

    def dte_on(self, today: date) -> int:
        return (self.expiration_date - today).days

    def mark(self, today: date, short_quote: dict, long_quote: dict) -> None:
        """Record today's quotes and compute current P&L per contract."""
        self.as_of = today
        self.short_quote = short_quote
        self.long_quote = long_quote
        self.underlying_price = short_quote.get("underlying_price") \
            or long_quote.get("underlying_price")
        self.exit_cost_per_share = short_quote["ask"] - long_quote["bid"]
        self.current_pnl_dollars = (
            (self.spread.entry_premium - self.exit_cost_per_share) * 100.0
        )


# ----- Portfolio ---------------------------------------------------------

@dataclass
class Portfolio:
    """Mutable state container: capital, positions, guards.

    Start with `risky_capital` = $4,730 (44% of $10,750) per BACKTEST_PLAN.

    `stop_loss_premium_mult` controls the early stop: close if unrealized
    loss exceeds this multiple of the entry credit. Plan default is 2.0
    (tastytrade). Pass `None` to disable (breach + DTE + profit target only).
    """
    risky_capital: float
    stop_loss_premium_mult: Optional[float] = 2.0
    peak_capital: float = field(init=False)
    open_positions: list[Position] = field(default_factory=list)
    halted: bool = False
    paused_until: Optional[date] = None
    trade_log: list[dict] = field(default_factory=list)
    _monthly_anchor_date: Optional[date] = None
    _monthly_anchor_capital: Optional[float] = None

    def __post_init__(self) -> None:
        self.peak_capital = self.risky_capital

    # ----- capacity queries ---------------------------------------------

    @property
    def available_slots(self) -> int:
        return max(0, MAX_POSITIONS - len(self.open_positions))

    def can_open(self, symbol: str, sector: str, today: date) -> bool:
        """Capacity gate: slots free, not halted/paused, no duplicate
        symbol, sector limit respected.
        """
        if self.halted or self.available_slots == 0:
            return False
        if self.paused_until is not None and today < self.paused_until:
            return False
        if any(p.symbol == symbol for p in self.open_positions):
            return False
        # "UNKNOWN" is the stub sector before a GICS mapping is wired in.
        # Skip the cap for UNKNOWN so it stays a true no-op — otherwise
        # every symbol shares one pseudo-sector and the cap becomes a
        # surprise 2-position hard limit.
        if sector == "UNKNOWN":
            return True
        same_sector = sum(1 for p in self.open_positions if p.sector == sector)
        return same_sector < MAX_PER_SECTOR

    # ----- entry ---------------------------------------------------------

    def add_position(self, spread: Spread, sector: str) -> Position:
        """Open a new position. Pays entry half-commission. Caller is
        expected to have already run `can_open()`; this method does not
        re-check (keeps Level 2 loop explicit about rejection reasons).
        """
        self.risky_capital -= COMMISSION_HALF
        pos = Position(spread=spread, sector=sector)
        self.open_positions.append(pos)
        return pos

    # ----- exit triggers ------------------------------------------------

    def _check_triggers(self, pos: Position, today: date,
                        vix: float) -> Optional[ExitReason]:
        """Exit reasons in priority order (first match wins). Assumes
        `pos.mark(...)` has been called for `today`.
        """
        if vix > VIX_REGIME_EXIT:
            return ExitReason.VIX_REGIME
        if pos.dte_on(today) <= DTE_EXIT_THRESHOLD:
            return ExitReason.DTE_CLOSE
        if pos.underlying_price is not None \
                and pos.underlying_price <= pos.spread.short_strike:
            return ExitReason.BREACH
        # Loss > mult × premium → stop. Disabled when stop_loss_premium_mult
        # is None — Level 2 uses this to turn off the premium-based stop for
        # thin-credit / low-delta setups where it fires on normal volatility.
        if self.stop_loss_premium_mult is not None:
            max_loss_stop = -self.stop_loss_premium_mult * pos.spread.max_profit
            if pos.current_pnl_dollars is not None \
                    and pos.current_pnl_dollars <= max_loss_stop:
                return ExitReason.STOP_LOSS
        # P&L ≥ 50% of max profit → take.
        if pos.current_pnl_dollars is not None \
                and pos.current_pnl_dollars >= 0.50 * pos.spread.max_profit:
            return ExitReason.PROFIT_TARGET
        return None

    def check_exits(self, today: date, vix: float) -> list[tuple[Position, ExitReason]]:
        """List of (position, reason) for every position triggered today."""
        out = []
        for p in self.open_positions:
            r = self._check_triggers(p, today, vix)
            if r is not None:
                out.append((p, r))
        return out

    # ----- close ---------------------------------------------------------

    def close_position(self, pos: Position, exit_date: date,
                       reason: ExitReason) -> dict:
        """Close an open position at its currently-marked quotes.

        Net P&L (dollars per contract) = gross_pnl - round-trip commission.
        Adds to trade_log and updates risky_capital (exit half-commission
        and gross P&L; entry half was already charged at open).
        """
        assert pos.exit_cost_per_share is not None, "mark() before closing"
        exit_cost = pos.exit_cost_per_share
        gross_pnl = (pos.spread.entry_premium - exit_cost) * 100.0
        net_pnl = gross_pnl - COMMISSION_ROUND_TRIP
        self.risky_capital += gross_pnl - COMMISSION_HALF   # only exit half left

        entry = {
            "symbol": pos.symbol,
            "sector": pos.sector,
            "entry_date": pos.entry_date,
            "exit_date": exit_date,
            "short_strike": pos.spread.short_strike,
            "long_strike": pos.spread.long_strike,
            "expiration_date": pos.spread.expiration_date,
            "entry_premium": pos.spread.entry_premium,
            "exit_value": exit_cost,
            "gross_pnl": gross_pnl,
            "commission": COMMISSION_ROUND_TRIP,
            "net_pnl": net_pnl,
            "exit_reason": reason.value,
            "holding_days": (exit_date - pos.entry_date).days,
            "underlying_at_exit": pos.underlying_price,
            "short_delta_at_entry": pos.spread.short_delta,
        }
        self.trade_log.append(entry)
        self.open_positions.remove(pos)
        return entry

    def settle_expiration(self, pos: Position,
                          underlying_at_expiry: float) -> dict:
        """Hold-to-expiration settlement. No market quotes needed — terminal
        value is determined by spot vs strikes. Charges exit half-commission
        to stay consistent with close_position accounting.
        """
        gross_pnl = terminal_pnl(pos.spread, underlying_at_expiry)
        net_pnl = gross_pnl - COMMISSION_ROUND_TRIP
        self.risky_capital += gross_pnl - COMMISSION_HALF

        entry = {
            "symbol": pos.symbol,
            "sector": pos.sector,
            "entry_date": pos.entry_date,
            "exit_date": pos.expiration_date,
            "short_strike": pos.spread.short_strike,
            "long_strike": pos.spread.long_strike,
            "expiration_date": pos.spread.expiration_date,
            "entry_premium": pos.spread.entry_premium,
            "exit_value": 0.0,  # settled at expiry, no close trade
            "gross_pnl": gross_pnl,
            "commission": COMMISSION_ROUND_TRIP,
            "net_pnl": net_pnl,
            "exit_reason": ExitReason.EXPIRATION.value,
            "holding_days": (pos.expiration_date - pos.entry_date).days,
            "underlying_at_exit": underlying_at_expiry,
            "short_delta_at_entry": pos.spread.short_delta,
        }
        self.trade_log.append(entry)
        self.open_positions.remove(pos)
        return entry

    # ----- portfolio-level guards ---------------------------------------

    def apply_portfolio_stops(self, today: date) -> None:
        """Update peak capital, check 43% trailing stop (halt), and 8%
        monthly drawdown (5-day pause). Called after all Phase A activity.
        """
        if self.risky_capital > self.peak_capital:
            self.peak_capital = self.risky_capital

        # 43% trailing stop from peak — latch on permanently.
        if self.risky_capital < self.peak_capital * (1.0 - TRAILING_STOP_FRAC):
            self.halted = True

        # Monthly drawdown anchor: reset on month change.
        if (self._monthly_anchor_date is None
                or today.month != self._monthly_anchor_date.month
                or today.year != self._monthly_anchor_date.year):
            self._monthly_anchor_date = today
            self._monthly_anchor_capital = self.risky_capital
            return

        assert self._monthly_anchor_capital is not None
        anchor = self._monthly_anchor_capital
        if anchor > 0 and (anchor - self.risky_capital) / anchor > MONTHLY_DD_PAUSE_FRAC:
            # Already paused? Don't extend — paused is a 5-day cooldown not a sliding window.
            if self.paused_until is None or today >= self.paused_until:
                self.paused_until = today + timedelta(days=PAUSE_LENGTH_DAYS)

    # ----- snapshot -----------------------------------------------------

    def snapshot(self, today: date) -> dict:
        """One-line daily equity-curve row."""
        return {
            "date": today,
            "risky_capital": self.risky_capital,
            "peak_capital": self.peak_capital,
            "n_positions": len(self.open_positions),
            "halted": self.halted,
            "paused": self.paused_until is not None and today < self.paused_until,
        }
