"""Long-equity portfolio for the directional backtest.

Holds whole-share long positions, equal-weighted at each rebalance.
Tracks cash, positions, daily equity curve, and trade log. No shorting,
no leverage, no options.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date


@dataclass
class EquityPosition:
    symbol: str
    entry_date: date
    entry_price: float
    shares: int

    def value(self, mark_price: float) -> float:
        return self.shares * mark_price

    def pnl(self, mark_price: float) -> float:
        return (mark_price - self.entry_price) * self.shares


@dataclass
class EquityPortfolio:
    starting_capital: float
    cash: float = field(init=False)
    positions: dict[str, EquityPosition] = field(default_factory=dict)
    equity_curve: list[dict] = field(default_factory=list)  # {date, equity, cash, n_pos}
    trade_log: list[dict] = field(default_factory=list)

    def __post_init__(self):
        self.cash = self.starting_capital

    # ----- valuation ------------------------------------------------------

    def total_equity(self, prices: dict[str, float]) -> float:
        pos_value = sum(
            p.value(prices[s]) for s, p in self.positions.items() if s in prices
        )
        return self.cash + pos_value

    def mark_to_market(self, prices: dict[str, float], on_date: date) -> None:
        """Append one equity-curve point."""
        equity = self.total_equity(prices)
        self.equity_curve.append({
            "date": on_date,
            "equity": equity,
            "cash": self.cash,
            "n_positions": len(self.positions),
        })

    # ----- mechanics ------------------------------------------------------

    def _close(self, symbol: str, price: float, on_date: date, reason: str) -> None:
        pos = self.positions.pop(symbol)
        proceeds = pos.shares * price
        self.cash += proceeds
        self.trade_log.append({
            "date": on_date, "symbol": symbol, "action": "SELL",
            "shares": pos.shares, "price": price, "proceeds": proceeds,
            "entry_date": pos.entry_date, "entry_price": pos.entry_price,
            "pnl": (price - pos.entry_price) * pos.shares,
            "reason": reason,
        })

    def _open(self, symbol: str, price: float, on_date: date,
              target_dollars: float) -> None:
        if price <= 0 or target_dollars <= 0:
            return
        shares = int(target_dollars // price)
        if shares <= 0:
            return
        cost = shares * price
        if cost > self.cash + 1e-6:
            return
        self.cash -= cost
        self.positions[symbol] = EquityPosition(
            symbol=symbol, entry_date=on_date, entry_price=price, shares=shares,
        )
        self.trade_log.append({
            "date": on_date, "symbol": symbol, "action": "BUY",
            "shares": shares, "price": price, "proceeds": -cost,
            "entry_date": on_date, "entry_price": price, "pnl": 0.0,
            "reason": "REBALANCE",
        })

    # ----- strategy actions ----------------------------------------------

    def rebalance(self, target_symbols: set[str], prices: dict[str, float],
                  on_date: date) -> None:
        """Hard replace: sell all non-targets, buy all new targets equal-weight.

        Targets with no price on this date are skipped (logged as dropped).
        Sizing is computed AFTER closures so the full equity is redeployable.
        """
        # close non-targets
        to_close = [s for s in self.positions if s not in target_symbols]
        for sym in to_close:
            if sym in prices:
                self._close(sym, prices[sym], on_date, reason="REBALANCE")
            # no price: carry the position; it'll be closed next time a quote appears

        # compute equal weight from current total equity (cash + retained positions)
        retained_value = sum(
            p.value(prices[s]) for s, p in self.positions.items() if s in prices
        )
        total = self.cash + retained_value
        tradeable_targets = [s for s in target_symbols if s in prices]
        if not tradeable_targets:
            return
        per_name_target = total / len(tradeable_targets)

        # rebalance retained positions toward the target (simple version:
        # close then reopen — gives exact equal-weight without partial-share math)
        for sym in list(self.positions):
            if sym in tradeable_targets:
                self._close(sym, prices[sym], on_date, reason="REBALANCE")

        for sym in tradeable_targets:
            self._open(sym, prices[sym], on_date, target_dollars=per_name_target)

    def exit_on_signal(self, symbols_to_close: set[str],
                       prices: dict[str, float], on_date: date) -> None:
        """Close named positions due to signal degradation. Freed cash idles."""
        for sym in list(symbols_to_close):
            if sym in self.positions and sym in prices:
                self._close(sym, prices[sym], on_date, reason="SIGNAL_DEGRADATION")
