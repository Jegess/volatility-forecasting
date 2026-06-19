"""Performance metrics for the backtest.

Inputs:
    trade_log     — one row per closed trade (see schema below)
    daily_equity  — one row per trading day (see schema below)

trade_log schema:
    symbol, entry_date, exit_date, short_strike, long_strike,
    entry_premium, exit_value, gross_pnl, commission, net_pnl,
    exit_reason, holding_days, vrp, vrp_rank

daily_equity schema:
    date, risky_capital, n_positions, daily_pnl

All thresholds from BACKTEST_PLAN.md §"Metrics" at Level 1.5 / Level 2.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import polars as pl

TRADING_DAYS_PER_YEAR = 252


# ----- primitives --------------------------------------------------------

def sharpe_annualized(returns: pl.Series | np.ndarray,
                      periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """Annualized Sharpe on a series of period returns (zero risk-free rate).

    Degenerate inputs (fewer than 2 observations or zero std) return 0.0
    rather than NaN so downstream gates don't need to special-case them.
    """
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    if r.size < 2:
        return 0.0
    std = r.std(ddof=1)
    if std == 0.0:
        return 0.0
    return float(r.mean() / std * math.sqrt(periods_per_year))


def max_drawdown(equity: pl.Series | np.ndarray) -> float:
    """Largest peak-to-trough drop as a positive fraction of the peak.

    A flat or monotonically rising curve returns 0.0.
    """
    e = np.asarray(equity, dtype=float)
    if e.size == 0:
        return 0.0
    running_peak = np.maximum.accumulate(e)
    # Peaks of 0 or negative would give a meaningless ratio — clamp.
    safe_peak = np.where(running_peak > 0, running_peak, np.nan)
    drawdowns = (running_peak - e) / safe_peak
    dd = np.nanmax(drawdowns)
    return float(dd) if np.isfinite(dd) else 0.0


def profit_factor(pnl: pl.Series | np.ndarray) -> float:
    """Gross wins / gross losses. Returns inf if there are no losses."""
    p = np.asarray(pnl, dtype=float)
    wins = p[p > 0].sum()
    losses = -p[p < 0].sum()
    if losses == 0.0:
        return float("inf") if wins > 0 else 0.0
    return float(wins / losses)


def win_rate(pnl: pl.Series | np.ndarray) -> float:
    """Fraction of trades with strictly positive P&L."""
    p = np.asarray(pnl, dtype=float)
    if p.size == 0:
        return 0.0
    return float((p > 0).mean())


# ----- breakdowns --------------------------------------------------------

def exit_reason_breakdown(trade_log: pl.DataFrame) -> pl.DataFrame:
    """Per exit-reason: trade count, mean net P&L, total net P&L, win rate."""
    if trade_log.height == 0:
        return pl.DataFrame({
            "exit_reason": [], "n_trades": [], "mean_pnl": [],
            "total_pnl": [], "win_rate": [],
        })
    return (
        trade_log
        .group_by("exit_reason")
        .agg(
            pl.len().alias("n_trades"),
            pl.col("net_pnl").mean().alias("mean_pnl"),
            pl.col("net_pnl").sum().alias("total_pnl"),
            (pl.col("net_pnl") > 0).mean().alias("win_rate"),
        )
        .sort("total_pnl", descending=True)
    )


# ----- aggregate ---------------------------------------------------------

def compute_all_metrics(trade_log: pl.DataFrame,
                        daily_equity: pl.DataFrame) -> dict[str, Any]:
    """One-stop metrics dict for BACKTEST_PLAN.md §Metrics gates.

    Returned keys:
        n_trades, win_rate, mean_net_pnl, median_net_pnl,
        total_gross_pnl, total_net_pnl, total_commissions,
        profit_factor, commission_ratio,
        avg_holding_days, sharpe_annualized, max_drawdown,
        exit_reason_breakdown (list of dicts).
    """
    out: dict[str, Any] = {}

    # Trade-level
    out["n_trades"] = int(trade_log.height)
    if trade_log.height > 0:
        pnl = trade_log["net_pnl"]
        gross = trade_log["gross_pnl"]
        comm = trade_log["commission"]
        out["win_rate"] = win_rate(pnl)
        out["mean_net_pnl"] = float(pnl.mean())
        out["median_net_pnl"] = float(pnl.median())
        out["total_gross_pnl"] = float(gross.sum())
        out["total_net_pnl"] = float(pnl.sum())
        out["total_commissions"] = float(comm.sum())
        out["profit_factor"] = profit_factor(pnl)
        gross_sum = out["total_gross_pnl"]
        out["commission_ratio"] = (
            float(out["total_commissions"] / gross_sum) if gross_sum > 0 else float("inf")
        )
        out["avg_holding_days"] = float(trade_log["holding_days"].mean())
    else:
        for k in (
            "win_rate", "mean_net_pnl", "median_net_pnl",
            "total_gross_pnl", "total_net_pnl", "total_commissions",
            "profit_factor", "commission_ratio", "avg_holding_days",
        ):
            out[k] = 0.0

    # Equity-curve level
    if daily_equity.height >= 2:
        eq = daily_equity.sort("date")["risky_capital"].to_numpy()
        # Daily returns on equity — avoid divide-by-zero if capital hits 0.
        prev = eq[:-1]
        daily_ret = np.where(prev > 0, np.diff(eq) / prev, 0.0)
        out["sharpe_annualized"] = sharpe_annualized(daily_ret)
        out["max_drawdown"] = max_drawdown(eq)
    else:
        out["sharpe_annualized"] = 0.0
        out["max_drawdown"] = 0.0

    out["exit_reason_breakdown"] = (
        exit_reason_breakdown(trade_log).to_dicts() if trade_log.height > 0 else []
    )

    return out
