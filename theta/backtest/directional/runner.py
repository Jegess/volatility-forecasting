"""Directional backtest orchestrator.

Daily loop from `start` to `end`:
  1. Mark-to-market portfolio on today's closes
  2. If today is a rebalance date: rank universe, pick decile, hard-replace
  3. Else: check signal degradation, close degraded positions
"""
from __future__ import annotations

from datetime import date

import polars as pl

from theta.backtest.directional import loaders, ranking
from theta.backtest.directional.exits import signal_degradation_exits
from theta.backtest.directional.portfolio import EquityPortfolio


def rebalance_dates(trading_days: list[date]) -> list[date]:
    """First trading day of each (year, month) in `trading_days`."""
    seen: set[tuple[int, int]] = set()
    out: list[date] = []
    for d in trading_days:
        key = (d.year, d.month)
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out


def _prices_on(closes: pl.DataFrame, on_date: date) -> dict[str, float]:
    row = closes.filter(pl.col("date") == on_date)
    return dict(zip(row["symbol"].to_list(), row["close"].to_list()))


def run_directional(
    decile: str | int = "bottom",
    n_deciles: int = 10,
    exit_pct_rank: float = 0.70,
    capital: float = 100_000.0,
    start: date | None = None,
    end: date | None = None,
    signals: pl.DataFrame | None = None,
    closes: pl.DataFrame | None = None,
) -> dict:
    """Run a single directional backtest.

    Args:
        decile: "bottom", "top", or 1..n_deciles.
        n_deciles: how many slices.
        exit_pct_rank: intra-month exit if held name's vrp_pct_rank drops
            below this threshold. Set to 0 to disable signal-degradation exits.
        capital: starting capital in USD.
        start, end: clip trading window. Defaults to full signal range.
        signals, closes: optional pre-loaded frames (skip disk I/O when
            running many deciles back-to-back in a notebook).

    Returns:
        dict with keys:
            equity_curve: pl.DataFrame (date, equity, cash, n_positions)
            trade_log: pl.DataFrame
            ranked: pl.DataFrame (full VRP ranking — for diagnostics)
            rebalance_dates: list[date]
            config: dict of inputs
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
    if not trading_days:
        raise ValueError("No trading days in the signal window")

    rebal_days = set(rebalance_dates(trading_days))

    portfolio = EquityPortfolio(starting_capital=capital)

    for day in trading_days:
        prices = _prices_on(closes, day)
        ranked_today = ranked.filter(pl.col("date") == day)

        if day in rebal_days:
            chosen = ranking.select_decile(ranked_today, decile=decile,
                                           n_deciles=n_deciles)
            targets = set(chosen["symbol"].to_list())
            portfolio.rebalance(targets, prices, day)
        elif exit_pct_rank > 0:
            to_close = signal_degradation_exits(
                held=set(portfolio.positions),
                ranked_today=ranked_today,
                exit_pct_rank=exit_pct_rank,
            )
            if to_close:
                portfolio.exit_on_signal(to_close, prices, day)

        portfolio.mark_to_market(prices, day)

    return {
        "equity_curve": pl.DataFrame(portfolio.equity_curve),
        "trade_log": pl.DataFrame(portfolio.trade_log) if portfolio.trade_log else pl.DataFrame(),
        "ranked": ranked,
        "rebalance_dates": sorted(rebal_days),
        "config": {
            "decile": decile, "n_deciles": n_deciles,
            "exit_pct_rank": exit_pct_rank, "capital": capital,
            "start": start, "end": end,
        },
    }


def run_spy_benchmark(capital: float = 100_000.0,
                      start: date | None = None,
                      end: date | None = None) -> pl.DataFrame:
    """SPY buy-and-hold equity curve aligned to the same window."""
    spy = loaders.load_spy_closes()
    if start is not None:
        spy = spy.filter(pl.col("date") >= start)
    if end is not None:
        spy = spy.filter(pl.col("date") <= end)
    spy = spy.sort("date")

    if spy.height == 0:
        raise ValueError("No SPY closes in the requested window")

    entry = spy["close"][0]
    shares = int(capital // entry)
    residual_cash = capital - shares * entry
    return spy.with_columns(
        (pl.col("close") * shares + residual_cash).alias("equity"),
    ).select("date", "equity")


def run_equal_weight_universe(capital: float = 100_000.0,
                              signals: pl.DataFrame | None = None,
                              closes: pl.DataFrame | None = None,
                              start: date | None = None,
                              end: date | None = None) -> dict:
    """Null strategy: equal-weight all 188 universe names, monthly rebalance."""
    if signals is None:
        signals = loaders.load_signals()

    if start is not None:
        signals = signals.filter(pl.col("date") >= start)
    if end is not None:
        signals = signals.filter(pl.col("date") <= end)

    universe = sorted(signals["symbol"].unique().to_list())
    if closes is None:
        closes = loaders.load_underlying_closes(universe)

    trading_days = sorted(signals["date"].unique().to_list())
    rebal_days = set(rebalance_dates(trading_days))

    portfolio = EquityPortfolio(starting_capital=capital)

    for day in trading_days:
        prices = _prices_on(closes, day)
        if day in rebal_days:
            available = set(signals.filter(pl.col("date") == day)["symbol"].to_list())
            portfolio.rebalance(available, prices, day)
        portfolio.mark_to_market(prices, day)

    return {
        "equity_curve": pl.DataFrame(portfolio.equity_curve),
        "trade_log": pl.DataFrame(portfolio.trade_log) if portfolio.trade_log else pl.DataFrame(),
        "rebalance_dates": sorted(rebal_days),
    }
