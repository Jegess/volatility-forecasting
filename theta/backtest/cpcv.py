"""CPCV, Deflated Sharpe, and Strategy Risk diagnostics.

Lopez de Prado (Advances in Financial Machine Learning, Ch. 12, 14, 15):

    - Combinatorial Purged CV: re-sample N groups of trading days into
      C(N, k) alternative backtest paths. Reports distribution of Sharpe
      across paths, not a single point estimate.
    - Deflated Sharpe Ratio: adjusts an observed Sharpe for the number of
      parameter trials already run (our 7-config sweep + the 3 new ones).
    - Strategy Risk: given an asymmetric payout profile, the minimum win
      rate p* required to hit a target Sharpe. Bootstrap the historical
      win rate to estimate P(realized p < p*).

The CPCV here runs on the daily-equity curve of a *completed* backtest,
not by re-simulating the strategy per fold. That is a simplification
(Lopez prescribes re-training per path) — but our strategy has no
training step, just simulation on signals, so the fold-level distinction
is the exposure window, which we capture by slicing daily returns.
"""
from __future__ import annotations

from itertools import combinations
from math import erf, log, pi, sqrt
from pathlib import Path

import numpy as np
import polars as pl

from theta.backtest import data as bt_data

EQUITY_FILE = bt_data.OUTPUT_DIR / "level2_daily_equity.parquet"
TRADE_LOG_FILE = bt_data.OUTPUT_DIR / "level2_trade_log.parquet"


# ----- helpers -----------------------------------------------------------

def _annualized_sharpe(daily_ret: np.ndarray) -> float:
    if daily_ret.size < 2 or daily_ret.std(ddof=1) == 0:
        return 0.0
    return float(daily_ret.mean() / daily_ret.std(ddof=1) * sqrt(252))


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2)))


# ----- CPCV --------------------------------------------------------------

def cpcv_paths(daily_equity: pl.DataFrame, n_groups: int = 10,
               n_test: int = 2, embargo_days: int = 21
               ) -> list[dict]:
    """Partition the equity curve into `n_groups` chronological bins; for
    every combination of `n_test` test bins, compute Sharpe over just those
    bins' daily returns with a trailing `embargo_days` purge.

    Returns one dict per path: {path_id, bins, n_days, sharpe, mean_ret, std}.
    """
    eq = daily_equity.sort("date")["risky_capital"].to_numpy()
    if eq.size < n_groups * 3:
        raise ValueError(f"Equity too short for {n_groups} groups")

    # Daily returns (length = eq.size - 1)
    daily_ret = (eq[1:] - eq[:-1]) / np.maximum(eq[:-1], 1e-9)
    n = daily_ret.size
    bin_edges = np.linspace(0, n, n_groups + 1, dtype=int)
    bins = [(int(bin_edges[i]), int(bin_edges[i + 1])) for i in range(n_groups)]

    paths: list[dict] = []
    for path_id, test_bins in enumerate(combinations(range(n_groups), n_test)):
        mask = np.zeros(n, dtype=bool)
        for b in test_bins:
            lo, hi = bins[b]
            # Embargo: drop the first `embargo_days` of each test bin so we
            # don't inherit the tail of a prior (adjacent) train bin's
            # still-open positions. Same idea Lopez prescribes for CV folds.
            lo_eff = min(lo + embargo_days, hi)
            mask[lo_eff:hi] = True
        selected = daily_ret[mask]
        paths.append({
            "path_id": path_id,
            "bins": list(test_bins),
            "n_days": int(mask.sum()),
            "sharpe": _annualized_sharpe(selected),
            "mean_ret": float(selected.mean()) if selected.size else 0.0,
            "std_ret": float(selected.std(ddof=1)) if selected.size > 1 else 0.0,
        })
    return paths


def cpcv_summary(paths: list[dict]) -> dict:
    sh = np.array([p["sharpe"] for p in paths])
    return {
        "n_paths": len(paths),
        "sharpe_mean": float(sh.mean()),
        "sharpe_std": float(sh.std(ddof=1)) if sh.size > 1 else 0.0,
        "sharpe_median": float(np.median(sh)),
        "sharpe_min": float(sh.min()),
        "sharpe_max": float(sh.max()),
        "frac_positive": float((sh > 0).mean()),
        "p05": float(np.quantile(sh, 0.05)),
        "p95": float(np.quantile(sh, 0.95)),
    }


# ----- Deflated Sharpe ----------------------------------------------------

# Euler-Mascheroni constant — used in the False Strategy Theorem.
_EULER = 0.5772156649015329


def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    """Expected maximum of `n_trials` IID Sharpe ratios drawn from N(0, V).
    Lopez de Prado (2018) False Strategy Theorem, Eq. 12:

        E[max SR] = sqrt(V) * ((1-gamma)*Z^{-1}(1 - 1/n) + gamma*Z^{-1}(1 - 1/(n*e)))

    where gamma = Euler-Mascheroni, Z^{-1} is the inverse standard normal.
    """
    if n_trials < 2:
        return 0.0
    from scipy.stats import norm
    a = norm.ppf(1.0 - 1.0 / n_trials)
    b = norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return sqrt(sr_variance) * ((1.0 - _EULER) * a + _EULER * b)


def deflated_sharpe_ratio(observed_sr: float, sr_variance_across_trials: float,
                           n_trials: int, n_obs: int,
                           skew: float = 0.0, kurt: float = 3.0) -> float:
    """Deflated Sharpe Ratio p-value (probability observed SR exceeds the
    multiple-testing adjusted benchmark). Lopez de Prado 2014.

    Returns PSR against SR* — probability the strategy's true SR > 0
    after adjusting for selection bias across `n_trials`.
    """
    if n_obs < 2:
        return 0.0
    sr_star = expected_max_sharpe(n_trials, sr_variance_across_trials)
    # Non-annualized SR for the PSR calc — convert from annualized assuming
    # 252 trading-day year.
    sr_nonann = observed_sr / sqrt(252)
    sr_star_nonann = sr_star / sqrt(252)
    num = (sr_nonann - sr_star_nonann) * sqrt(n_obs - 1)
    den = sqrt(1.0 - skew * sr_nonann + (kurt - 1.0) / 4.0 * sr_nonann ** 2)
    if den <= 0:
        return 0.0
    return _norm_cdf(num / den)


# ----- Strategy Risk bootstrap -------------------------------------------

def implied_precision_for_sharpe(target_sr: float, win_payout: float,
                                 loss_payout: float,
                                 trades_per_year: float = 252.0) -> float:
    """Solve for minimum win-rate p* such that a binary payout strategy
    achieves `target_sr` annualized. Lopez de Prado Ch. 15, Eq. 15.4.

    Payouts are per-trade dollars (positive win, negative loss).
    Returns p in [0, 1] or nan if unreachable.
    """
    pi_h = win_payout
    pi_l = loss_payout  # negative
    n = trades_per_year
    # mu(p) = p*pi_h + (1-p)*pi_l
    # var(p) = p*(pi_h - mu)^2 + (1-p)*(pi_l - mu)^2
    # Sharpe_ann = sqrt(n) * mu / sqrt(var) — solve for p.
    # Closed form gets ugly; use bisection.
    def sharpe_at(p: float) -> float:
        mu = p * pi_h + (1 - p) * pi_l
        var = p * (pi_h - mu) ** 2 + (1 - p) * (pi_l - mu) ** 2
        if var <= 0:
            return float("inf") if mu > 0 else float("-inf")
        return sqrt(n) * mu / sqrt(var)

    lo, hi = 1e-6, 1 - 1e-6
    if sharpe_at(hi) < target_sr:
        return float("nan")  # unreachable even at 100% win rate
    if sharpe_at(lo) > target_sr:
        return 0.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if sharpe_at(mid) < target_sr:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def strategy_risk_bootstrap(trade_pnls: np.ndarray, target_sr: float,
                            n_boot: int = 10_000, seed: int = 42
                            ) -> dict:
    """Bootstrap the observed trade P&L to estimate P(realized SR < target).

    - Win/loss payouts = mean positive / mean negative trade.
    - Observed precision p_obs = fraction of positive trades.
    - Derive p* from the implied-precision formula.
    - Bootstrap precision at same trade count → P(p_boot < p*).
    """
    wins = trade_pnls[trade_pnls > 0]
    losses = trade_pnls[trade_pnls < 0]
    if wins.size == 0 or losses.size == 0:
        return {"error": "no wins or no losses"}
    pi_h = float(wins.mean())
    pi_l = float(losses.mean())
    p_obs = float((trade_pnls > 0).mean())

    n_trades = trade_pnls.size
    # trades/year scaled from observed count over the 3-year backtest
    # (roughly 252 * 3 = 756 trading days).
    trades_per_year = n_trades / 3.0

    p_star = implied_precision_for_sharpe(target_sr, pi_h, pi_l, trades_per_year)

    # Bootstrap observed precision.
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_trades, size=(n_boot, n_trades))
    sampled_wins = (trade_pnls[idx] > 0).mean(axis=1)

    if np.isnan(p_star):
        prob_fail = 1.0
    else:
        prob_fail = float((sampled_wins < p_star).mean())

    return {
        "n_trades": int(n_trades),
        "win_payout": pi_h,
        "loss_payout": pi_l,
        "observed_precision": p_obs,
        "target_sharpe": target_sr,
        "required_precision": float(p_star) if not np.isnan(p_star) else None,
        "prob_sharpe_below_target": prob_fail,
        "trades_per_year": trades_per_year,
    }


# ----- orchestrator -------------------------------------------------------

def run_diagnostics(equity_path: Path = EQUITY_FILE,
                    trade_log_path: Path = TRADE_LOG_FILE,
                    n_groups: int = 10, n_test: int = 2,
                    prior_trials: int = 10,
                    target_sr: float = 0.3) -> dict:
    """End-to-end diagnostic on the most recent Level 2 run.

    - CPCV path distribution of Sharpe.
    - Deflated Sharpe adjusted for `prior_trials` configs tested.
    - Strategy Risk: probability of failing the Sharpe gate given the
      trade-level asymmetry.
    """
    equity = pl.read_parquet(equity_path)
    trades = pl.read_parquet(trade_log_path)

    paths = cpcv_paths(equity, n_groups=n_groups, n_test=n_test)
    cpcv = cpcv_summary(paths)

    # Deflated Sharpe: variance of SR across CPCV paths is our empirical
    # estimate of sr_variance_across_trials.
    observed_sr = _annualized_sharpe(
        (equity.sort("date")["risky_capital"].diff().drop_nulls() /
         equity.sort("date")["risky_capital"].shift(1).drop_nulls()).to_numpy()
    )
    daily_ret_var = float(np.var([p["sharpe"] for p in paths], ddof=1))
    n_obs_days = equity.height - 1
    dsr = deflated_sharpe_ratio(
        observed_sr=observed_sr,
        sr_variance_across_trials=daily_ret_var,
        n_trials=prior_trials,
        n_obs=n_obs_days,
    )

    # Strategy Risk
    trade_pnls = trades["net_pnl"].to_numpy()
    strat_risk = strategy_risk_bootstrap(trade_pnls, target_sr=target_sr)

    return {
        "observed_sharpe": observed_sr,
        "cpcv": cpcv,
        "cpcv_paths": paths,
        "deflated_sharpe_pvalue": dsr,
        "prior_trials": prior_trials,
        "strategy_risk": strat_risk,
    }
