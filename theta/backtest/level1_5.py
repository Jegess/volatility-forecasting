"""Level 1.5: hold-to-expiration unmanaged spread backtest.

Every qualifying (symbol, date) from `signal.build_candidates()` is turned
into a bull put spread via `spreads.build_spread()` and held to expiration.
No portfolio limits, no exit management, no position cap — the point is to
isolate the raw edge of VRP + spread construction before Level 2 layers in
managed exits and sizing.

Per-trade terminal P&L uses the expiration-day underlying price from
`options_iv/` (any contract's `underlying_price` for that symbol/date).
Trades whose expiration falls past the data horizon are skipped.

Gate: mean net P&L > $0 after $2.60 round-trip commission. If negative,
the signal+spread combination is unprofitable in the cleanest possible
setup and no amount of portfolio management can rescue it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from theta.backtest import data as bt_data
from theta.backtest import signal as bt_signal
from theta.backtest import spreads as bt_spreads

OUTPUT_FILE = bt_data.OUTPUT_DIR / "level1_5_trades.parquet"


# ----- trade log construction -------------------------------------------

def _build_trade_log(candidates: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, int]]:
    """Iterate candidates in (symbol, date) order, build spreads, compute
    terminal P&L. Per-symbol option chains are cached so each parquet is
    read at most once per backtest run.
    """
    chain_cache: dict[str, pl.DataFrame] = {}
    counts = {
        "attempted": 0,
        "no_chain_on_entry": 0,
        "no_spread_built": 0,
        "no_expiry_underlying": 0,
        "traded": 0,
    }
    rows: list[dict] = []

    # Sorting by symbol keeps cache hot and makes the run reproducible.
    ordered = candidates.sort(["symbol", "date"])

    for row in ordered.iter_rows(named=True):
        counts["attempted"] += 1
        sym = row["symbol"]
        entry_date = row["date"]

        if sym not in chain_cache:
            chain_cache[sym] = bt_data.load_options_iv(sym)
        sym_chain = chain_cache[sym]

        day_puts = sym_chain.filter(
            (pl.col("date") == entry_date) & (pl.col("right") == "PUT")
        )
        if day_puts.height == 0:
            counts["no_chain_on_entry"] += 1
            continue

        spread = bt_spreads.build_spread(sym, entry_date, chain=day_puts)
        if spread is None:
            counts["no_spread_built"] += 1
            continue

        # Expiration-day underlying: any contract for (symbol, expiration_date)
        # carries the spot price. If the expiration falls past our data
        # horizon, `sym_chain` won't have that date and we skip.
        exp_rows = sym_chain.filter(pl.col("date") == spread.expiration_date)
        if exp_rows.height == 0:
            counts["no_expiry_underlying"] += 1
            continue
        underlying_expiry = float(exp_rows["underlying_price"][0])

        gross = bt_spreads.terminal_pnl(spread, underlying_expiry)
        net = gross - bt_spreads.COMMISSION_PER_ROUND_TRIP

        rows.append({
            "symbol": sym,
            "entry_date": entry_date,
            "expiration_date": spread.expiration_date,
            "holding_days": (spread.expiration_date - entry_date).days,
            "dte_at_entry": spread.dte_at_entry,
            "short_strike": spread.short_strike,
            "long_strike": spread.long_strike,
            "short_delta": spread.short_delta,
            "entry_premium": spread.entry_premium,
            "width": spread.width,
            "max_profit": spread.max_profit,
            "max_loss": spread.max_loss,
            "underlying_at_entry": spread.underlying_price,
            "underlying_at_expiry": underlying_expiry,
            "gross_pnl": gross,
            "commission": bt_spreads.COMMISSION_PER_ROUND_TRIP,
            "net_pnl": net,
            "hit_max_loss": underlying_expiry <= spread.long_strike,
            "breached": underlying_expiry < spread.short_strike,
            "vrp": float(row["vrp"]),
            "vrp_rank": int(row["vrp_rank"]),
        })
        counts["traded"] += 1

    return pl.DataFrame(rows), counts


# ----- distribution summaries -------------------------------------------

def pnl_stats(trade_log: pl.DataFrame) -> dict[str, Any]:
    """Distribution summary matching BACKTEST_PLAN.md §Level 1.5 Metrics.

    Skewness is Fisher-Pearson (third standardized moment). Negative skew
    is the structural cost of short premium — document, don't threshold.
    """
    if trade_log.height == 0:
        return {
            "n": 0, "mean_net_pnl": 0.0, "median_net_pnl": 0.0,
            "std_net_pnl": 0.0, "win_rate": 0.0, "max_loss_rate": 0.0,
            "breach_rate": 0.0, "p5_net_pnl": 0.0, "skewness": 0.0,
        }
    net = trade_log["net_pnl"].to_numpy().astype(float)
    mean = float(net.mean())
    std = float(net.std(ddof=1)) if net.size > 1 else 0.0
    skew = float(((net - mean) ** 3).mean() / std ** 3) if std > 0 else 0.0
    return {
        "n": int(trade_log.height),
        "mean_net_pnl": mean,
        "median_net_pnl": float(np.median(net)),
        "std_net_pnl": std,
        "win_rate": float((net > 0).mean()),
        "max_loss_rate": float(trade_log["hit_max_loss"].mean()),
        "breach_rate": float(trade_log["breached"].mean()),
        "p5_net_pnl": float(np.percentile(net, 5)),
        "skewness": skew,
    }


def pnl_by_quintile(trade_log: pl.DataFrame) -> pl.DataFrame:
    """Per-entry-day VRP quintile breakdown. Q1 = top 20% by VRP on that
    day. Monotonic Q5 → Q1 confirms VRP ordinality survives into P&L.
    """
    if trade_log.height == 0:
        return pl.DataFrame()
    # qcut per day so "top quintile" is relative to that day's opportunity set.
    labeled = trade_log.with_columns(
        pl.col("vrp").qcut(5, labels=["Q5", "Q4", "Q3", "Q2", "Q1"],
                           allow_duplicates=True)
        .over("entry_date").alias("quintile")
    )
    return (
        labeled
        .group_by("quintile")
        .agg(
            pl.len().alias("n"),
            pl.col("net_pnl").mean().alias("mean_net_pnl"),
            pl.col("net_pnl").median().alias("median_net_pnl"),
            (pl.col("net_pnl") > 0).mean().alias("win_rate"),
            pl.col("breached").mean().alias("breach_rate"),
        )
        .sort("quintile")
    )


def pnl_by_rank_bucket(trade_log: pl.DataFrame,
                       buckets: list[int] | None = None) -> pl.DataFrame:
    """Top-N daily-rank cutoffs. Shows whether Sinclair's 'ranking beats
    bingo' intuition holds: tighter top-N should have higher mean P&L.
    """
    buckets = buckets or [5, 10, 20, 50, 100]
    if trade_log.height == 0:
        return pl.DataFrame()
    rows = [{
        "bucket": "all",
        "n": trade_log.height,
        "mean_net_pnl": float(trade_log["net_pnl"].mean()),
        "win_rate": float((trade_log["net_pnl"] > 0).mean()),
    }]
    for n in buckets:
        bk = trade_log.filter(pl.col("vrp_rank") <= n)
        if bk.height == 0:
            continue
        rows.append({
            "bucket": f"top_{n}",
            "n": bk.height,
            "mean_net_pnl": float(bk["net_pnl"].mean()),
            "win_rate": float((bk["net_pnl"] > 0).mean()),
        })
    return pl.DataFrame(rows)


# ----- orchestrator -----------------------------------------------------

def run_level1_5(include_etfs: bool = False,
                 save: bool = True) -> dict[str, Any]:
    """Full Level 1.5 report. Returns a summary dict; writes the trade log
    to level1_5_trades.parquet for downstream notebooks.
    """
    daily = bt_data.daily_signals(model="LightGBM", include_etfs=include_etfs)
    candidates = bt_signal.build_candidates(daily)

    trade_log, counts = _build_trade_log(candidates)

    summary: dict[str, Any] = {
        "universe_size": daily["symbol"].n_unique(),
        "n_candidates": int(candidates.height),
        "counts": counts,
        "n_symbols_with_trades": (
            int(trade_log["symbol"].n_unique()) if trade_log.height else 0
        ),
        "pnl_stats": pnl_stats(trade_log),
        "pnl_by_quintile": (
            pnl_by_quintile(trade_log).to_dicts() if trade_log.height else []
        ),
        "pnl_by_rank_bucket": (
            pnl_by_rank_bucket(trade_log).to_dicts() if trade_log.height else []
        ),
    }

    if save and trade_log.height > 0:
        Path(bt_data.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        trade_log.write_parquet(OUTPUT_FILE)

    return summary


# ----- gate -------------------------------------------------------------

def pass_gate(summary: dict) -> tuple[bool, str]:
    """Plan gate: mean net P&L must be strictly positive after commissions.
    A zero-or-negative mean means the signal doesn't monetize even without
    management overhead."""
    n = summary["pnl_stats"]["n"]
    mean = summary["pnl_stats"]["mean_net_pnl"]
    if n == 0:
        return False, "Level 1.5 FAILED: zero trades cleared the spread filter"
    if mean <= 0:
        return False, (
            f"Level 1.5 FAILED: mean net P&L ${mean:.2f} "
            f"≤ $0 after commissions (n={n})"
        )
    return True, f"Level 1.5 PASSED: mean net P&L ${mean:.2f} > $0 (n={n})"
