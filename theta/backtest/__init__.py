"""VRP-based premium-selling backtest pipeline.

See BACKTEST_PLAN.md for the methodology.

Levels:
    1   — VRP signal quality
    1.5 — Terminal P&L distribution (no portfolio limits)
    2   — Managed daily simulation (the realistic one)
    3   — Robustness: CPCV, DSR, regime/sector decomposition

Run via `python -m theta.backtest.run_all` or call individual level functions.
"""
