"""Level 2: managed spread simulation with portfolio state.

The realistic simulation. Each trading day in the WF prediction period runs
three phases (BACKTEST_PLAN.md §Level 2):

    Phase A — MANAGE: mark every open position to today's quotes, fire exit
              triggers in priority order (VIX > 30, DTE <= 10, breach, stop
              loss, 50% profit), settle anything that reached expiration.
    Phase B — GUARD:  update peak capital, latch the 43% trailing halt,
              pause for 5 days on an 8% monthly drawdown.
    Phase C — ENTER:  rank the day's candidates by VRP, walk the ranking
              until slots fill or candidates exhaust, skipping duplicates,
              sector-capped symbols, and LightGBM/LogHAR disagreements > 50%.

All cash accounting and exit logic live in `portfolio.py`; this module only
orchestrates the daily loop and handles I/O (quote lookups, trade log /
equity curve writing). That separation makes the state machine unit-testable
with synthetic quotes and keeps the loop readable.

Sector mapping is stubbed to "UNKNOWN" for every symbol until a GICS file
is wired in — see the TODO on `get_sector`. With all sectors "UNKNOWN" the
2-per-sector cap becomes a no-op (flagged in Open Questions, memory).
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

from theta.backtest import data as bt_data
from theta.backtest import signal as bt_signal
from theta.backtest import spreads as bt_spreads
from theta.backtest.portfolio import (
    COMMISSION_HALF,
    ExitReason,
    Portfolio,
    Position,
)

# Starting capital split (BACKTEST_PLAN §Capital — EUR 10K @ 1.075 USD):
#   total    = $10,750
#   risky    = $4,730 (44%)    ← Level 2 state
#   reserve  = $6,020 (56%)   ← not touched in backtest
STARTING_RISKY_CAPITAL = 4_730.0

# Model disagreement gate: drop a candidate if LightGBM and LogHAR point
# forecasts diverge by more than 50% of the LogHAR prediction. Stronger
# than either model alone — both have to "agree" the next 21d is quiet.
MODEL_DISAGREEMENT_FRAC = 0.50

# Half-Kelly sizing ceiling: reject a spread whose max_loss would exceed
# this fraction of risky_capital. At $4,730 starting capital and typical
# max_loss $200-$400, the cap rarely binds — but it's the correct guard
# as capital drifts up/down across the backtest.
KELLY_FRACTION = 0.25

TRADE_LOG_FILE = bt_data.OUTPUT_DIR / "level2_trade_log.parquet"
EQUITY_FILE = bt_data.OUTPUT_DIR / "level2_daily_equity.parquet"


# ----- sector map (stub) -------------------------------------------------

# TODO: replace with a real GICS mapping (yfinance info.sector or a static
# file). Until then, every symbol is "UNKNOWN" and the 2-per-sector cap is
# effectively inactive. Flagged in memory/project_backtest_implementation.md.
_SECTOR_MAP: dict[str, str] = {}


def get_sector(symbol: str) -> str:
    return _SECTOR_MAP.get(symbol, "UNKNOWN")


# ----- candidate assembly with both models ------------------------------

def _build_candidates_with_loghar(include_etfs: bool,
                                  vix_min: float = 14.0,
                                  vix_max: float = 30.0,
                                  earnings_buffer_days: int | None = None) -> pl.DataFrame:
    """Ranked candidates (LightGBM-based VRP) augmented with the LogHAR
    forecast so Phase C can apply the model-disagreement filter.

    Output cols: symbol, date, y_pred (lgbm), y_pred_loghar, atm_iv,
    vix, vrp, vrp_rank, days_to_earnings, is_fomc_week, hist_median_vrp.
    """
    lgbm_daily = bt_data.daily_signals(model="LightGBM", include_etfs=include_etfs)
    candidates = bt_signal.build_candidates(
        lgbm_daily, vix_min=vix_min, vix_max=vix_max,
        earnings_buffer_days=earnings_buffer_days,
    )

    loghar = (
        bt_data.load_wf_predictions("LogHAR")
        .select("symbol", "date", pl.col("y_pred").alias("y_pred_loghar"))
    )
    return candidates.join(loghar, on=["symbol", "date"], how="left")


def _disagrees(y_pred_lgbm: float, y_pred_loghar: float | None) -> bool:
    """True if the two models disagree by more than MODEL_DISAGREEMENT_FRAC.
    A missing LogHAR forecast is treated as disagreement (fail safe).
    """
    if y_pred_loghar is None:
        return True
    if y_pred_loghar <= 0:
        return True   # protects the ratio; LogHAR forecasts should be > 0
    return abs(y_pred_lgbm - y_pred_loghar) / y_pred_loghar > MODEL_DISAGREEMENT_FRAC


# ----- chain cache --------------------------------------------------------

class _ChainCache:
    """Lazy per-symbol options_iv cache. Every parquet is read at most once
    per backtest run. Same pattern as `level1_5._build_trade_log`.
    """
    def __init__(self) -> None:
        self._by_symbol: dict[str, pl.DataFrame] = {}

    def chain(self, symbol: str) -> pl.DataFrame:
        if symbol not in self._by_symbol:
            self._by_symbol[symbol] = bt_data.load_options_iv(symbol)
        return self._by_symbol[symbol]

    def puts_on(self, symbol: str, on_date: date) -> pl.DataFrame:
        return self.chain(symbol).filter(
            (pl.col("date") == on_date) & (pl.col("right") == "PUT")
        )

    def quote(self, symbol: str, on_date: date, strike: float,
              expiration_date: date) -> dict | None:
        """Single-contract quote from the cached chain, with raw-EOD
        fallback via `bt_data.get_quote` when a contract has dropped out
        of options_iv (delta moved outside the -0.50 to -0.05 window).

        Returns None only if the contract has no row anywhere — that
        means no market on that day (the caller carries the position).
        """
        df = self.chain(symbol).filter(
            (pl.col("date") == on_date)
            & (pl.col("strike") == strike)
            & (pl.col("expiration_date") == expiration_date)
            & (pl.col("right") == "PUT")
        )
        if df.height > 0:
            row = df.row(0, named=True)
            return {
                "bid": row["bid"], "ask": row["ask"],
                "underlying_price": row["underlying_price"],
                "delta": row["delta"], "dte": row["dte"],
            }

        # Fallback: raw EOD parquet. `get_quote` already does prefix-match
        # on the ISO `created` timestamp and handles the yyyymmdd
        # expiration string format.
        return bt_data.get_quote(
            symbol, on_date, strike, expiration_date, right="PUT",
            fallback_to_raw=True,
        )

    def underlying_on(self, symbol: str, on_date: date) -> float | None:
        df = self.chain(symbol).filter(pl.col("date") == on_date)
        if df.height == 0:
            return None
        return float(df["underlying_price"][0])


# ----- phase helpers ------------------------------------------------------

def _phase_a_manage(portfolio: Portfolio, today: date, vix: float,
                    cache: _ChainCache) -> None:
    """Mark every open position and fire exits. Any position whose
    expiration has arrived (or passed) is settled at the terminal
    underlying price.

    Iterating over `list(portfolio.open_positions)` lets `close_position`
    / `settle_expiration` mutate the list safely inside the loop.
    """
    for pos in list(portfolio.open_positions):
        # Expired — settle on terminal underlying.
        if today >= pos.expiration_date:
            spot = cache.underlying_on(pos.symbol, pos.expiration_date)
            if spot is None:
                # No data on expiration day (data horizon / delisting).
                # Use last known mark as best-effort settlement. If we have
                # no mark either, skip — position stays open until data
                # returns (should be rare).
                spot = pos.underlying_price
                if spot is None:
                    continue
            portfolio.settle_expiration(pos, float(spot))
            continue

        short_q = cache.quote(
            pos.symbol, today, pos.spread.short_strike, pos.spread.expiration_date
        )
        long_q = cache.quote(
            pos.symbol, today, pos.spread.long_strike, pos.spread.expiration_date
        )
        if short_q is None or long_q is None:
            # Contract(s) missing today — carry position to tomorrow without
            # firing exit triggers (we can't evaluate them without quotes).
            continue

        pos.mark(today, short_q, long_q)

    # Fire any triggers after all marking is done. Only evaluate positions
    # that were freshly marked today — otherwise check_exits would judge
    # on stale state (or None), and close_position would fail its mark()
    # precondition.
    marked = [p for p in portfolio.open_positions if p.as_of == today]
    for pos in marked:
        reason = portfolio._check_triggers(pos, today, vix)
        if reason is None:
            continue
        if reason == ExitReason.EXPIRATION:
            portfolio.settle_expiration(pos, float(pos.underlying_price or 0))
        else:
            portfolio.close_position(pos, today, reason)


def _phase_c_enter(portfolio: Portfolio, today: date,
                   day_candidates: pl.DataFrame,
                   cache: _ChainCache,
                   target_delta: float,
                   min_premium_frac: float,
                   max_loss_budget: float | None,
                   kelly_fraction: float = KELLY_FRACTION) -> int:
    """Walk today's VRP ranking top-down, opening spreads until slots fill.
    Returns the number of positions opened today.
    """
    if portfolio.available_slots == 0:
        return 0

    opened = 0
    # Already ordered by vrp_rank from build_candidates; explicit sort
    # guards against callers passing unsorted frames.
    for row in day_candidates.sort("vrp_rank").iter_rows(named=True):
        if portfolio.available_slots == 0:
            break

        sym = row["symbol"]
        sector = get_sector(sym)
        if not portfolio.can_open(sym, sector, today):
            continue

        if _disagrees(float(row["y_pred"]), row.get("y_pred_loghar")):
            continue

        day_puts = cache.puts_on(sym, today)
        if day_puts.height == 0:
            continue

        spread = bt_spreads.build_spread(
            sym, today, chain=day_puts,
            target_delta=target_delta, min_premium_frac=min_premium_frac,
            max_loss_budget=max_loss_budget,
        )
        if spread is None:
            continue

        # Half-Kelly sizing ceiling — a single max_loss can't exceed 25%
        # of the risky subaccount. Plus the trivial survival check that
        # we have room for the entry commission.
        if spread.max_loss > portfolio.risky_capital * kelly_fraction:
            continue
        if portfolio.risky_capital <= COMMISSION_HALF:
            continue

        portfolio.add_position(spread, sector)
        opened += 1

    return opened


# ----- orchestrator -------------------------------------------------------

def run_level2(include_etfs: bool = False,
               starting_capital: float = STARTING_RISKY_CAPITAL,
               target_delta: float = bt_spreads.TARGET_DELTA,
               min_premium_frac: float = bt_spreads.MIN_PREMIUM_FRAC_OF_WIDTH,
               max_loss_budget: float | None = None,
               stop_loss_mult: float | None = 2.0,
               vix_min: float = 14.0,
               vix_max: float = 30.0,
               earnings_buffer_days: int | None = None,
               kelly_fraction: float = KELLY_FRACTION,
               save: bool = True,
               verbose: bool = False) -> dict[str, Any]:
    """Full Level 2 simulation over the WF prediction period.

    Returns a summary dict with trade counts, final capital, and distribution
    stats. Writes `level2_trade_log.parquet` and `level2_daily_equity.parquet`
    for downstream notebook analysis.
    """
    candidates = _build_candidates_with_loghar(
        include_etfs=include_etfs,
        vix_min=vix_min, vix_max=vix_max,
        earnings_buffer_days=earnings_buffer_days,
    )

    # Trading calendar = every date that appears in the candidate frame
    # OR has an open position. Using the candidate frame's dates as the
    # driver is fine because WF predictions exist every trading day in
    # the evaluation window (756 days).
    trading_days: list[date] = sorted(candidates["date"].unique().to_list())

    # Per-date VIX (we already have it on candidates; macro join is cheap
    # even for days with zero candidates, e.g. everything filtered out).
    vix_by_date: dict[date, float] = {
        r["date"]: float(r["vix"])
        for r in bt_data.load_macro().select("date", "vix").iter_rows(named=True)
    }

    portfolio = Portfolio(
        risky_capital=starting_capital,
        stop_loss_premium_mult=stop_loss_mult,
    )
    cache = _ChainCache()
    equity_rows: list[dict] = []
    opened_per_day = 0

    for i, today in enumerate(trading_days):
        vix = vix_by_date.get(today)
        if vix is None:
            # No VIX on a candidate date would be a data bug. Be loud.
            raise KeyError(f"No VIX for {today}")

        _phase_a_manage(portfolio, today, vix, cache)
        portfolio.apply_portfolio_stops(today)

        day_candidates = candidates.filter(pl.col("date") == today)
        opened_per_day = _phase_c_enter(
            portfolio, today, day_candidates, cache,
            target_delta=target_delta, min_premium_frac=min_premium_frac,
            max_loss_budget=max_loss_budget, kelly_fraction=kelly_fraction,
        )

        snap = portfolio.snapshot(today)
        snap["n_opened"] = opened_per_day
        snap["n_candidates"] = day_candidates.height
        snap["vix"] = vix
        equity_rows.append(snap)

        if verbose and (i % 50 == 0 or i == len(trading_days) - 1):
            print(
                f"[{today}] cap=${portfolio.risky_capital:,.0f} "
                f"peak=${portfolio.peak_capital:,.0f} "
                f"open={len(portfolio.open_positions)} "
                f"halt={portfolio.halted} trades={len(portfolio.trade_log)}"
            )

    # Force-close anything still open on the last trading day at the last
    # available mark. Leaves no phantom equity on the books.
    last_day = trading_days[-1]
    for pos in list(portfolio.open_positions):
        spot = cache.underlying_on(pos.symbol, pos.expiration_date) \
            or cache.underlying_on(pos.symbol, last_day) \
            or pos.underlying_price
        if spot is None:
            # Nothing we can do; leave the position unclosed but drop it
            # from the open list so capital isn't locked.
            portfolio.open_positions.remove(pos)
            continue
        portfolio.settle_expiration(pos, float(spot))

    trade_log = pl.DataFrame(portfolio.trade_log) if portfolio.trade_log else pl.DataFrame()
    equity = pl.DataFrame(equity_rows)

    if save:
        Path(bt_data.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        if trade_log.height > 0:
            trade_log.write_parquet(TRADE_LOG_FILE)
        equity.write_parquet(EQUITY_FILE)

    return {
        "n_trading_days": len(trading_days),
        "n_trades": trade_log.height,
        "starting_capital": starting_capital,
        "final_capital": portfolio.risky_capital,
        "peak_capital": portfolio.peak_capital,
        "halted": portfolio.halted,
        "trade_log": trade_log,
        "equity_curve": equity,
        "summary_stats": _summary_stats(trade_log, equity, starting_capital),
    }


# ----- stats --------------------------------------------------------------

def _summary_stats(trade_log: pl.DataFrame, equity: pl.DataFrame,
                   starting_capital: float) -> dict[str, Any]:
    """Headline metrics matching BACKTEST_PLAN §Level 2 Metrics. Sharpe is
    annualized from daily equity changes; 252 trading-day convention.
    """
    if trade_log.height == 0:
        return {
            "n": 0, "mean_net_pnl": 0.0, "win_rate": 0.0, "sharpe": 0.0,
            "max_drawdown": 0.0, "profit_factor": 0.0,
        }

    net = trade_log["net_pnl"].to_numpy()
    gross_wins = float(trade_log.filter(pl.col("net_pnl") > 0)["net_pnl"].sum())
    gross_losses = float(-trade_log.filter(pl.col("net_pnl") < 0)["net_pnl"].sum())

    # Daily equity return series → Sharpe.
    eq = equity["risky_capital"].to_numpy()
    daily_ret = (eq[1:] - eq[:-1]) / eq[:-1]
    sharpe = 0.0
    if daily_ret.size > 1 and daily_ret.std(ddof=1) > 0:
        sharpe = float(daily_ret.mean() / daily_ret.std(ddof=1) * (252 ** 0.5))

    # Max drawdown from the risky_capital curve.
    peaks = 0.0
    mdd = 0.0
    for x in eq:
        peaks = max(peaks, x)
        if peaks > 0:
            mdd = max(mdd, (peaks - x) / peaks)

    # Mean holding period and exit-reason breakdown for quick eyeballing.
    exit_reason_counts = (
        trade_log.group_by("exit_reason")
        .agg(pl.len().alias("n"), pl.col("net_pnl").mean().alias("mean_net_pnl"))
        .sort("n", descending=True)
        .to_dicts()
    )

    return {
        "n": int(trade_log.height),
        "mean_net_pnl": float(net.mean()),
        "median_net_pnl": float(trade_log["net_pnl"].median()),
        "win_rate": float((net > 0).mean()),
        "avg_holding_days": float(trade_log["holding_days"].mean()),
        "sharpe": sharpe,
        "max_drawdown": mdd,
        "profit_factor": (gross_wins / gross_losses) if gross_losses > 0 else float("inf"),
        "total_commissions": float(trade_log["commission"].sum()),
        "exit_reason_counts": exit_reason_counts,
    }


# ----- gate ---------------------------------------------------------------

def pass_gate(summary: dict) -> tuple[bool, str]:
    """Plan gate: Sharpe > 0.3 AND max drawdown < 50%. Below either threshold
    and the strategy isn't viable at this capital level — stop before CPCV.
    """
    s = summary["summary_stats"]
    if s["n"] == 0:
        return False, "Level 2 FAILED: no trades executed"
    if s["sharpe"] <= 0.3:
        return False, f"Level 2 FAILED: Sharpe {s['sharpe']:.2f} ≤ 0.3"
    if s["max_drawdown"] >= 0.50:
        return False, f"Level 2 FAILED: max drawdown {s['max_drawdown']:.1%} ≥ 50%"
    return True, (
        f"Level 2 PASSED: Sharpe={s['sharpe']:.2f}, "
        f"MDD={s['max_drawdown']:.1%}, n={s['n']}, "
        f"mean P&L=${s['mean_net_pnl']:.2f}"
    )
