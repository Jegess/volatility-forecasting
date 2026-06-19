"""Null benchmarks for the ETF CSP strategy.

Three variants that strip the VRP ranking from the pipeline while keeping
everything else identical (capital, gates, exits, sizing). If any of them
matches the VRP strategy, the VRP signal is not the source of the edge.

    (a) round-robin  — every day, rotate the rank order through the 5 ETFs
                       (ignores VRP). Filters still applied: hist-median
                       gate DISABLED, VIX/FOMC gates KEPT.
    (b) random       — every day, assign ranks uniformly at random (seeded).
    (c) spy-only     — universe shrunk to SPY. On any candidate day that
                       passes VIX/FOMC, open one CSP if no SPY position is
                       already open. No VRP filter.
"""
from __future__ import annotations

import numpy as np
from datetime import date
from pathlib import Path
from typing import Any, Literal

import polars as pl

from theta.backtest import data as bt_data
from theta.backtest import signal as bt_signal
from theta.backtest import csp as bt_csp
from theta.backtest.csp_portfolio import COMMISSION_HALF, CspPortfolio, ExitReason
from theta.backtest.level2_csp import (
    ETF_UNIVERSE, STARTING_CAPITAL, _ChainCache, _daily_signals_etf,
    _phase_a_manage, _summary_stats,
)


Mode = Literal["round_robin", "random", "spy_only"]


def _build_candidates_null(mode: Mode,
                           vix_min: float = 14.0,
                           vix_max: float = 30.0,
                           seed: int = 0) -> pl.DataFrame:
    """Build candidates without the VRP edge filter.

    Keeps the VIX + FOMC gates (same regime filters as the real strategy).
    Drops the `vrp > hist_median_vrp` gate — otherwise the universe would
    still be pre-selected by VRP.
    """
    daily = _daily_signals_etf().pipe(bt_signal.add_vrp)

    daily = daily.filter(
        (pl.col("vix") >= vix_min)
        & (pl.col("vix") <= vix_max)
        & (pl.col("is_fomc_week") == 0)
    )

    if mode == "spy_only":
        return daily.filter(pl.col("symbol") == "SPY").with_columns(
            pl.lit(1).alias("vrp_rank")
        ).sort(["date", "symbol"])

    if mode == "round_robin":
        # Cycle the lead ETF by day number. Vectorized: build per-day offsets
        # in numpy, convert symbol to universe index, compute rank.
        sym_to_idx = {s: i for i, s in enumerate(ETF_UNIVERSE)}
        N = len(ETF_UNIVERSE)
        syms = daily["symbol"].to_numpy()
        days = daily["date"].to_numpy().astype("datetime64[D]").astype(int)
        sidx = np.array([sym_to_idx[s] for s in syms])
        offsets = days % N
        ranks = ((sidx - offsets) % N) + 1
        return daily.with_columns(
            pl.Series("vrp_rank", ranks, dtype=pl.Int32)
        ).sort(["date", "vrp_rank"])

    if mode == "random":
        # Vectorized: one random value per (symbol, date), then rank within date.
        rng = np.random.default_rng(seed)
        noise = rng.random(daily.height)
        return (
            daily.with_columns(pl.Series("_noise", noise))
            .with_columns(
                pl.col("_noise").rank("ordinal").over("date").cast(pl.Int32).alias("vrp_rank")
            )
            .drop("_noise")
            .sort(["date", "vrp_rank"])
        )

    raise ValueError(f"unknown mode: {mode}")


def _phase_c_enter_null(portfolio, today, day_candidates, cache,
                        target_delta, min_premium_yield) -> int:
    if portfolio.available_slots == 0:
        return 0
    opened = 0
    for row in day_candidates.sort("vrp_rank").iter_rows(named=True):
        if portfolio.available_slots == 0:
            break
        sym = row["symbol"]
        day_puts = cache.puts_on(sym, today)
        if day_puts.height == 0:
            continue
        contract = bt_csp.build_csp(
            sym, today, chain=day_puts,
            target_delta=target_delta,
            min_premium_yield=min_premium_yield,
        )
        if contract is None:
            continue
        if not portfolio.can_open(contract, today):
            continue
        if portfolio.available_cash <= COMMISSION_HALF:
            continue
        portfolio.add_position(contract)
        opened += 1
    return opened


def run_null(mode: Mode,
             starting_capital: float = STARTING_CAPITAL,
             target_delta: float = bt_csp.TARGET_DELTA,
             min_premium_yield: float = bt_csp.MIN_PREMIUM_YIELD,
             stop_loss_mult: float | None = 2.0,
             vix_min: float = 14.0,
             vix_max: float = 30.0,
             seed: int = 0) -> dict[str, Any]:
    candidates = _build_candidates_null(
        mode, vix_min=vix_min, vix_max=vix_max, seed=seed
    )
    trading_days: list[date] = sorted(candidates["date"].unique().to_list())
    vix_by_date = {
        r["date"]: float(r["vix"])
        for r in bt_data.load_macro().select("date", "vix").iter_rows(named=True)
    }
    portfolio = CspPortfolio(
        capital=starting_capital, stop_loss_premium_mult=stop_loss_mult
    )
    cache = _ChainCache()
    equity_rows: list[dict] = []

    for today in trading_days:
        vix = vix_by_date.get(today)
        if vix is None:
            continue
        _phase_a_manage(portfolio, today, vix, cache)
        portfolio.apply_portfolio_stops(today)
        day_candidates = candidates.filter(pl.col("date") == today)
        opened = _phase_c_enter_null(
            portfolio, today, day_candidates, cache,
            target_delta=target_delta, min_premium_yield=min_premium_yield,
        )
        snap = portfolio.snapshot(today)
        snap["n_opened"] = opened
        snap["n_candidates"] = day_candidates.height
        snap["vix"] = vix
        equity_rows.append(snap)

    last_day = trading_days[-1]
    for pos in list(portfolio.open_positions):
        spot = cache.underlying_on(pos.symbol, pos.expiration_date) \
            or cache.underlying_on(pos.symbol, last_day) \
            or pos.underlying_price
        if spot is None:
            portfolio.notional_reserved -= pos.contract.notional_margin
            portfolio.open_positions.remove(pos)
            continue
        portfolio.settle_expiration(pos, float(spot))

    trade_log = pl.DataFrame(portfolio.trade_log) if portfolio.trade_log else pl.DataFrame()
    equity = pl.DataFrame(equity_rows)
    return {
        "mode": mode,
        "n_trades": trade_log.height,
        "final_capital": portfolio.capital,
        "peak_capital": portfolio.peak_capital,
        "trade_log": trade_log,
        "equity_curve": equity,
        "summary_stats": _summary_stats(trade_log, equity, starting_capital),
    }


if __name__ == "__main__":
    rows = []
    # Real VRP strategy numbers (from prior run at $100K)
    rows.append(("vrp", 131, 70.2, 2.14, 3.0, 113_105))

    for mode in ("round_robin", "random", "spy_only"):
        print(f"running {mode}...", flush=True)
        r = run_null(mode, seed=0)
        s = r["summary_stats"]
        rows.append((mode, s["n"], s["win_rate"]*100,
                     s["sharpe"], s["max_drawdown"]*100,
                     r["final_capital"]))

    print("\n" + "=" * 72)
    print(f"{'strategy':<14} {'n':>6} {'win%':>7} {'Sharpe':>8} {'MDD%':>7} {'final $':>12}")
    print("-" * 72)
    for name, n, wr, sr, mdd, fc in rows:
        print(f"{name:<14} {n:>6} {wr:>6.1f}% {sr:>+8.2f} {mdd:>6.1f}% ${fc:>11,.0f}")
    print("=" * 72)
