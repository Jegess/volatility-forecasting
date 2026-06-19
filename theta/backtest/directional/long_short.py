"""Long-short market-neutral portfolio.

Dollar-neutral construction:
- Half of starting capital goes long the "bull" decile (bottom VRP by default)
- Half goes short the "bear" decile (top VRP by default)
- Daily P&L = long_leg_pnl + short_leg_pnl; equity = starting_capital + cumulative_pnl

Margin / borrow costs NOT modeled in v1. Reg T supports this easily (gross
exposure ≤ 200% of equity). Borrow cost for large caps is typically <50 bps/yr
for non-hard-to-borrow names; document as a limitation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import polars as pl

from theta.backtest.directional import loaders, ranking
from theta.backtest.directional.exits import signal_degradation_exits
from theta.backtest.directional.runner import rebalance_dates, _prices_on


@dataclass
class LSPosition:
    symbol: str
    side: str  # "LONG" or "SHORT"
    entry_date: date
    entry_price: float
    shares: int  # always positive; `side` encodes direction

    def pnl(self, mark_price: float) -> float:
        if self.side == "LONG":
            return (mark_price - self.entry_price) * self.shares
        return (self.entry_price - mark_price) * self.shares


@dataclass
class LongShortPortfolio:
    starting_capital: float
    long_book: dict[str, LSPosition] = field(default_factory=dict)
    short_book: dict[str, LSPosition] = field(default_factory=dict)
    equity_curve: list[dict] = field(default_factory=list)
    trade_log: list[dict] = field(default_factory=list)

    def total_pnl(self, prices: dict[str, float]) -> float:
        long_pnl = sum(p.pnl(prices[s]) for s, p in self.long_book.items() if s in prices)
        short_pnl = sum(p.pnl(prices[s]) for s, p in self.short_book.items() if s in prices)
        return long_pnl + short_pnl

    def equity(self, prices: dict[str, float]) -> float:
        return self.starting_capital + self.total_pnl(prices)

    def mark_to_market(self, prices: dict[str, float], on_date: date) -> None:
        self.equity_curve.append({
            "date": on_date,
            "equity": self.equity(prices),
            "n_long": len(self.long_book),
            "n_short": len(self.short_book),
        })

    def _close(self, book: dict, symbol: str, price: float, on_date: date,
               reason: str) -> None:
        pos = book.pop(symbol)
        pnl = pos.pnl(price)
        self.trade_log.append({
            "date": on_date, "symbol": symbol, "side": pos.side, "action": "CLOSE",
            "shares": pos.shares, "price": price,
            "entry_date": pos.entry_date, "entry_price": pos.entry_price,
            "pnl": pnl, "reason": reason,
        })

    def _open(self, book: dict, side: str, symbol: str, price: float,
              on_date: date, target_dollars: float) -> None:
        if price <= 0 or target_dollars <= 0:
            return
        shares = int(target_dollars // price)
        if shares <= 0:
            return
        book[symbol] = LSPosition(
            symbol=symbol, side=side, entry_date=on_date,
            entry_price=price, shares=shares,
        )
        self.trade_log.append({
            "date": on_date, "symbol": symbol, "side": side, "action": "OPEN",
            "shares": shares, "price": price,
            "entry_date": on_date, "entry_price": price,
            "pnl": 0.0, "reason": "REBALANCE",
        })

    def rebalance(self, long_targets: set[str], short_targets: set[str],
                  prices: dict[str, float], on_date: date,
                  gross_leverage: float = 1.0) -> None:
        """Dollar-neutral hard replace.

        `gross_leverage` = 1.0 means long book = 0.5 * equity, short book = 0.5
        * equity, gross exposure = equity. Use 2.0 for 100% long + 100% short.
        """
        current_equity = self.equity(prices)

        # close non-targets on both sides
        for sym in list(self.long_book):
            if sym not in long_targets and sym in prices:
                self._close(self.long_book, sym, prices[sym], on_date, "REBALANCE")
        for sym in list(self.short_book):
            if sym not in short_targets and sym in prices:
                self._close(self.short_book, sym, prices[sym], on_date, "REBALANCE")

        # close any retained targets too — simple clean rebalance
        for sym in list(self.long_book):
            if sym in prices:
                self._close(self.long_book, sym, prices[sym], on_date, "REBALANCE")
        for sym in list(self.short_book):
            if sym in prices:
                self._close(self.short_book, sym, prices[sym], on_date, "REBALANCE")

        # allocate fresh
        long_tradeable = [s for s in long_targets if s in prices]
        short_tradeable = [s for s in short_targets if s in prices]

        long_notional = current_equity * (gross_leverage / 2.0)
        short_notional = current_equity * (gross_leverage / 2.0)

        if long_tradeable:
            per_long = long_notional / len(long_tradeable)
            for sym in long_tradeable:
                self._open(self.long_book, "LONG", sym, prices[sym], on_date, per_long)
        if short_tradeable:
            per_short = short_notional / len(short_tradeable)
            for sym in short_tradeable:
                self._open(self.short_book, "SHORT", sym, prices[sym], on_date, per_short)

    def exit_on_signal(self, long_to_close: set[str], short_to_close: set[str],
                       prices: dict[str, float], on_date: date) -> None:
        for sym in list(long_to_close):
            if sym in self.long_book and sym in prices:
                self._close(self.long_book, sym, prices[sym], on_date,
                            "SIGNAL_DEGRADATION")
        for sym in list(short_to_close):
            if sym in self.short_book and sym in prices:
                self._close(self.short_book, sym, prices[sym], on_date,
                            "SIGNAL_DEGRADATION")


def _short_degradation_exits(held_shorts: set[str],
                             ranked_today: pl.DataFrame,
                             exit_pct_rank: float = 0.30) -> set[str]:
    """Mirror of long-side exit: close shorts whose VRP rank has fallen
    out of the top 3 deciles (pct_rank > exit_pct_rank means the name is
    no longer among the highest-VRP names).
    """
    if not held_shorts:
        return set()
    degraded = ranked_today.filter(
        pl.col("symbol").is_in(list(held_shorts))
        & (pl.col("vrp_pct_rank") > exit_pct_rank)
    )
    return set(degraded["symbol"].to_list())


def run_long_short(
    long_decile: str | int = "bottom",
    short_decile: str | int = "top",
    n_deciles: int = 10,
    gross_leverage: float = 1.0,
    long_exit_pct_rank: float = 0.70,
    short_exit_pct_rank: float = 0.30,
    capital: float = 100_000.0,
    start: date | None = None,
    end: date | None = None,
    signals: pl.DataFrame | None = None,
    closes: pl.DataFrame | None = None,
) -> dict:
    """Dollar-neutral long-short backtest.

    Defaults: long bottom decile (low VRP), short top decile (high VRP),
    gross=1.0 (50% long + 50% short). Set gross=2.0 for full 100/100.
    """
    if signals is None:
        signals = loaders.load_signals()

    ranked = ranking.rank_daily(ranking.compute_vrp_frame(signals))

    if start is not None:
        ranked = ranked.filter(pl.col("date") >= start)
    if end is not None:
        ranked = ranked.filter(pl.col("date") <= end)

    universe = sorted(ranked["symbol"].unique().to_list())
    if closes is None:
        closes = loaders.load_underlying_closes(universe)

    trading_days = sorted(ranked["date"].unique().to_list())
    rebal_days = set(rebalance_dates(trading_days))

    portfolio = LongShortPortfolio(starting_capital=capital)

    for day in trading_days:
        prices = _prices_on(closes, day)
        ranked_today = ranked.filter(pl.col("date") == day)

        if day in rebal_days:
            long_chosen = ranking.select_decile(ranked_today, decile=long_decile,
                                                n_deciles=n_deciles)
            short_chosen = ranking.select_decile(ranked_today, decile=short_decile,
                                                 n_deciles=n_deciles)
            portfolio.rebalance(
                long_targets=set(long_chosen["symbol"].to_list()),
                short_targets=set(short_chosen["symbol"].to_list()),
                prices=prices, on_date=day,
                gross_leverage=gross_leverage,
            )
        else:
            long_close = signal_degradation_exits(
                held=set(portfolio.long_book),
                ranked_today=ranked_today,
                exit_pct_rank=long_exit_pct_rank,
            ) if long_exit_pct_rank > 0 else set()
            short_close = _short_degradation_exits(
                held_shorts=set(portfolio.short_book),
                ranked_today=ranked_today,
                exit_pct_rank=short_exit_pct_rank,
            ) if short_exit_pct_rank < 1.0 else set()
            if long_close or short_close:
                portfolio.exit_on_signal(long_close, short_close, prices, day)

        portfolio.mark_to_market(prices, day)

    return {
        "equity_curve": pl.DataFrame(portfolio.equity_curve),
        "trade_log": pl.DataFrame(portfolio.trade_log) if portfolio.trade_log else pl.DataFrame(),
        "rebalance_dates": sorted(rebal_days),
        "config": {
            "long_decile": long_decile, "short_decile": short_decile,
            "n_deciles": n_deciles, "gross_leverage": gross_leverage,
            "long_exit_pct_rank": long_exit_pct_rank,
            "short_exit_pct_rank": short_exit_pct_rank,
            "capital": capital, "start": start, "end": end,
        },
    }
