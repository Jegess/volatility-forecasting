"""Directional VRP backtest — long bottom-decile equities.

Uses LightGBM VRP predictions as a cross-sectional equity ranking signal.
Each rebalance selects the bottom-decile VRP names (cheapest implied vol
relative to ML-forecast realized vol) and holds equal-weight long until
next rebalance. Intra-month exit on signal degradation.

Isolated from theta.backtest.level1/level2/csp — those harvest premium;
this one takes directional equity exposure.
"""
from theta.backtest.directional.runner import run_directional

__all__ = ["run_directional"]
