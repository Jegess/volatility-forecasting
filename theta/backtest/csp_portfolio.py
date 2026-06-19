"""Portfolio state machine for Level 2 CSP runs.

Parallel to `portfolio.py` (bull put spread version) but adapted for single-
strike, cash-secured short puts. The key structural difference is
**notional margin accounting**: every open CSP reserves `strike × 100` in
cash, and new entries are hard-capped by available cash, not just by a
Kelly fraction on max loss.

Exit triggers and the drawdown / pause / halt logic mirror `portfolio.py`
so results are directly comparable between strategies.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

from theta.backtest.csp import CspContract, terminal_pnl
from theta.backtest.portfolio import ExitReason  # reused enum

MAX_POSITIONS = 5
COMMISSION_HALF = 0.65                # single leg, IBKR — CSP is 1 leg
COMMISSION_ROUND_TRIP = 1.30
TRAILING_STOP_FRAC = 0.43
MONTHLY_DD_PAUSE_FRAC = 0.08
PAUSE_LENGTH_DAYS = 5
DTE_EXIT_THRESHOLD = 10
VIX_REGIME_EXIT = 30.0
PROFIT_TARGET_FRAC = 0.50             # close at 50% of max profit
STOP_LOSS_PREMIUM_MULT = 2.0          # close if unrealized loss > 2x credit


@dataclass
class CspPosition:
    contract: CspContract

    as_of: Optional[date] = None
    quote: Optional[dict] = None
    underlying_price: Optional[float] = None
    exit_cost_per_share: Optional[float] = None
    current_pnl_dollars: Optional[float] = None

    @property
    def symbol(self) -> str:
        return self.contract.symbol

    @property
    def entry_date(self) -> date:
        return self.contract.entry_date

    @property
    def expiration_date(self) -> date:
        return self.contract.expiration_date

    def dte_on(self, today: date) -> int:
        return (self.expiration_date - today).days

    def mark(self, today: date, quote: dict) -> None:
        self.as_of = today
        self.quote = quote
        self.underlying_price = quote.get("underlying_price")
        # Close the short put by buying at ask.
        self.exit_cost_per_share = float(quote["ask"])
        self.current_pnl_dollars = (
            (self.contract.entry_premium - self.exit_cost_per_share) * 100.0
        )


@dataclass
class CspPortfolio:
    """CSP portfolio state.

    Invariant: `notional_reserved = sum(pos.contract.notional_margin)` for
    all open positions. `available_cash` = `capital - notional_reserved`.
    """
    capital: float                                # total account cash
    stop_loss_premium_mult: Optional[float] = STOP_LOSS_PREMIUM_MULT
    profit_target_frac: Optional[float] = PROFIT_TARGET_FRAC
    peak_capital: float = field(init=False)
    notional_reserved: float = 0.0
    open_positions: list[CspPosition] = field(default_factory=list)
    halted: bool = False
    paused_until: Optional[date] = None
    trade_log: list[dict] = field(default_factory=list)
    _monthly_anchor_date: Optional[date] = None
    _monthly_anchor_capital: Optional[float] = None

    def __post_init__(self) -> None:
        self.peak_capital = self.capital

    # ----- queries -------------------------------------------------------

    @property
    def available_cash(self) -> float:
        return self.capital - self.notional_reserved

    @property
    def available_slots(self) -> int:
        return max(0, MAX_POSITIONS - len(self.open_positions))

    def can_open(self, contract: CspContract, today: date) -> bool:
        """All gates: halt, pause, slot, duplicate-symbol, cash."""
        if self.halted or self.available_slots == 0:
            return False
        if self.paused_until is not None and today < self.paused_until:
            return False
        if any(p.symbol == contract.symbol for p in self.open_positions):
            return False
        # Hard cash gate — the CSP-specific constraint.
        if contract.notional_margin > self.available_cash:
            return False
        return True

    # ----- entry ---------------------------------------------------------

    def add_position(self, contract: CspContract) -> CspPosition:
        self.capital -= COMMISSION_HALF
        self.notional_reserved += contract.notional_margin
        pos = CspPosition(contract=contract)
        self.open_positions.append(pos)
        return pos

    # ----- exits ---------------------------------------------------------

    def _check_triggers(self, pos: CspPosition, today: date,
                        vix: float) -> Optional[ExitReason]:
        if vix > VIX_REGIME_EXIT:
            return ExitReason.VIX_REGIME
        if pos.dte_on(today) <= DTE_EXIT_THRESHOLD:
            return ExitReason.DTE_CLOSE
        if pos.underlying_price is not None \
                and pos.underlying_price <= pos.contract.strike:
            return ExitReason.BREACH
        if self.stop_loss_premium_mult is not None:
            max_loss_stop = -self.stop_loss_premium_mult * pos.contract.max_profit
            if pos.current_pnl_dollars is not None \
                    and pos.current_pnl_dollars <= max_loss_stop:
                return ExitReason.STOP_LOSS
        if self.profit_target_frac is not None \
                and pos.current_pnl_dollars is not None \
                and pos.current_pnl_dollars >= self.profit_target_frac * pos.contract.max_profit:
            return ExitReason.PROFIT_TARGET
        return None

    def close_position(self, pos: CspPosition, exit_date: date,
                       reason: ExitReason) -> dict:
        assert pos.exit_cost_per_share is not None, "mark() before closing"
        exit_cost = pos.exit_cost_per_share
        gross_pnl = (pos.contract.entry_premium - exit_cost) * 100.0
        net_pnl = gross_pnl - COMMISSION_ROUND_TRIP
        # Release notional and credit gross PnL; exit-half commission now.
        self.notional_reserved -= pos.contract.notional_margin
        self.capital += gross_pnl - COMMISSION_HALF

        entry = {
            "symbol": pos.symbol,
            "entry_date": pos.entry_date,
            "exit_date": exit_date,
            "strike": pos.contract.strike,
            "expiration_date": pos.contract.expiration_date,
            "entry_premium": pos.contract.entry_premium,
            "exit_value": exit_cost,
            "gross_pnl": gross_pnl,
            "commission": COMMISSION_ROUND_TRIP,
            "net_pnl": net_pnl,
            "exit_reason": reason.value,
            "holding_days": (exit_date - pos.entry_date).days,
            "underlying_at_exit": pos.underlying_price,
            "delta_at_entry": pos.contract.delta,
            "notional_margin": pos.contract.notional_margin,
        }
        self.trade_log.append(entry)
        self.open_positions.remove(pos)
        return entry

    def settle_expiration(self, pos: CspPosition,
                          underlying_at_expiry: float) -> dict:
        gross_pnl = terminal_pnl(pos.contract, underlying_at_expiry)
        net_pnl = gross_pnl - COMMISSION_ROUND_TRIP
        self.notional_reserved -= pos.contract.notional_margin
        self.capital += gross_pnl - COMMISSION_HALF

        entry = {
            "symbol": pos.symbol,
            "entry_date": pos.entry_date,
            "exit_date": pos.expiration_date,
            "strike": pos.contract.strike,
            "expiration_date": pos.contract.expiration_date,
            "entry_premium": pos.contract.entry_premium,
            "exit_value": 0.0,
            "gross_pnl": gross_pnl,
            "commission": COMMISSION_ROUND_TRIP,
            "net_pnl": net_pnl,
            "exit_reason": ExitReason.EXPIRATION.value,
            "holding_days": (pos.expiration_date - pos.entry_date).days,
            "underlying_at_exit": underlying_at_expiry,
            "delta_at_entry": pos.contract.delta,
            "notional_margin": pos.contract.notional_margin,
        }
        self.trade_log.append(entry)
        self.open_positions.remove(pos)
        return entry

    # ----- guards --------------------------------------------------------

    def apply_portfolio_stops(self, today: date) -> None:
        if self.capital > self.peak_capital:
            self.peak_capital = self.capital
        if self.capital < self.peak_capital * (1.0 - TRAILING_STOP_FRAC):
            self.halted = True

        if (self._monthly_anchor_date is None
                or today.month != self._monthly_anchor_date.month
                or today.year != self._monthly_anchor_date.year):
            self._monthly_anchor_date = today
            self._monthly_anchor_capital = self.capital
            return

        assert self._monthly_anchor_capital is not None
        anchor = self._monthly_anchor_capital
        if anchor > 0 and (anchor - self.capital) / anchor > MONTHLY_DD_PAUSE_FRAC:
            if self.paused_until is None or today >= self.paused_until:
                self.paused_until = today + timedelta(days=PAUSE_LENGTH_DAYS)

    # ----- snapshot ------------------------------------------------------

    def snapshot(self, today: date) -> dict:
        return {
            "date": today,
            "capital": self.capital,
            "peak_capital": self.peak_capital,
            "notional_reserved": self.notional_reserved,
            "available_cash": self.available_cash,
            "n_positions": len(self.open_positions),
            "halted": self.halted,
            "paused": self.paused_until is not None and today < self.paused_until,
        }
