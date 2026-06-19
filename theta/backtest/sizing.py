"""Skew-adjusted fractional Kelly sizing (Sinclair, Volatility Trading Ch. 9).

Standard Kelly fraction for a binary payout:

    f* = p / |L|  -  (1 - p) / W

where p = win rate, W = average win (positive), L = average loss (positive
magnitude). This is the fraction of the RISKY subaccount to put at risk per
trade; a negative value means no bet.

Sinclair's adjustments for short-vol payout profiles:

    1. Skew adjustment. Negative skewness inflates standard Kelly above the
       true optimum. A rough correction is to divide the raw Kelly by (1 +
       |skew|) or simply halve it when skew is materially negative. We use
       Sinclair's conservative "half-Kelly if skew < -0.5" rule.

    2. Uncertainty adjustment (fractional Kelly). Because p, W, L are all
       estimated with error, the raw value overstates edge. Sinclair and
       Lopez de Prado both recommend f_used = 0.25 x f_raw (quarter Kelly)
       or f_used = 0.5 x f_raw (half) to buy margin against estimation
       error. We default to 0.5 and expose as a parameter.

The output is `kelly_fraction` in [0, 0.5] — plugs directly into
level2.run_level2(kelly_fraction=...), which caps per-trade max_loss at
kelly_fraction * risky_capital.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class KellySizing:
    p_win: float
    avg_win: float
    avg_loss_abs: float
    skew: float
    raw_kelly: float
    skew_penalty: float
    uncertainty_penalty: float
    adjusted_kelly: float
    recommended_fraction: float  # clamped to [0, 0.5]
    verdict: str


def compute_kelly(trade_pnls: np.ndarray,
                  uncertainty_penalty: float = 0.5,
                  hard_cap: float = 0.25) -> KellySizing:
    """From a trade P&L vector, produce the Sinclair-recommended fraction
    of risky capital to put at risk per trade.

    Args:
        trade_pnls: per-trade net P&L in dollars (wins positive, losses
            negative). Caller is expected to filter out zeros / settlements.
        uncertainty_penalty: fractional Kelly coefficient. 0.5 = half
            Kelly (Sinclair default), 0.25 = quarter Kelly (Lopez default
            for genuinely uncertain edges).
        hard_cap: upper clamp on recommended fraction. A useful safety
            rail — even Kelly-optimal may suggest sizes that blow up on
            a single adverse path.
    """
    wins = trade_pnls[trade_pnls > 0]
    losses = trade_pnls[trade_pnls < 0]
    if wins.size == 0 or losses.size == 0:
        return KellySizing(
            p_win=0.0, avg_win=0.0, avg_loss_abs=0.0, skew=0.0,
            raw_kelly=0.0, skew_penalty=1.0,
            uncertainty_penalty=uncertainty_penalty,
            adjusted_kelly=0.0, recommended_fraction=0.0,
            verdict="insufficient data (need both wins and losses)",
        )

    p = float((trade_pnls > 0).mean())
    W = float(wins.mean())
    L = float(-losses.mean())  # positive magnitude

    raw = p / L - (1.0 - p) / W  # fraction per unit bankroll (binary Kelly)

    # Sample skewness
    x = trade_pnls - trade_pnls.mean()
    s = trade_pnls.std(ddof=1)
    skew = float((x ** 3).mean() / (s ** 3)) if s > 0 else 0.0

    # Skew penalty: half-Kelly if skew materially negative.
    skew_penalty = 0.5 if skew < -0.5 else 1.0

    adjusted = raw * skew_penalty * uncertainty_penalty

    if adjusted <= 0:
        recommended = 0.0
        verdict = (
            f"NO BET. Raw Kelly = {raw:+.4f} (negative edge). "
            f"Strategy loses money per trade in expectation — no amount of "
            f"sizing recovers a negative-EV payout."
        )
    else:
        recommended = float(min(adjusted, hard_cap))
        verdict = (
            f"Risk {recommended:.1%} of risky subaccount per trade. "
            f"raw Kelly = {raw:.4f}, after skew ({skew_penalty:.1f}x) + "
            f"uncertainty ({uncertainty_penalty:.1f}x) and hard cap "
            f"({hard_cap:.1%})."
        )

    return KellySizing(
        p_win=p, avg_win=W, avg_loss_abs=L, skew=skew,
        raw_kelly=raw, skew_penalty=skew_penalty,
        uncertainty_penalty=uncertainty_penalty,
        adjusted_kelly=adjusted, recommended_fraction=recommended,
        verdict=verdict,
    )
