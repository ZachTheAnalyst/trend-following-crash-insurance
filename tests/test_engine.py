"""Validation tests for the backtest engine (synthetic data, exact match).

Pre-registered in Arnold_PreRegistration_2026-09-28.pdf, "Code validation".
These must pass before any real-data test is run.

Engine contract assumed by these tests (src/engine.py):

    backtest(ret, rf, signal, cost, initial_weight=1.0) -> Result

    ret            daily stock total returns (pd.Series)
    rf             daily risk-free returns, same index (pd.Series)
    signal         target stock weight decided at the close of day t (pd.Series)
    cost           one-way cost as a fraction (0.001 = 10 bp)
    initial_weight stock weight held before day 0, so day 0 pays no entry cost

    The weight held on day t is signal[t-1] (one-day lag; initial_weight on day 0).
    Result has:
        .net_ret    pd.Series, w*ret + (1-w)*rf - cost*|w_t - w_{t-1}|
        .weight     pd.Series, weight held each day
        .turnover   pd.Series, |w_t - w_{t-1}| each day
"""

import numpy as np
import pandas as pd

from src.engine import backtest


def _synthetic(n=500, seed=0):
    """Random daily stock and T-bill returns on a business-day index."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2000-01-03", periods=n)
    ret = pd.Series(rng.normal(0.0004, 0.012, n), index=idx, name="ret")
    rf = pd.Series(np.full(n, 0.00008), index=idx, name="rf")
    return ret, rf


def test_always_long_equals_buy_and_hold_exactly():
    """Signal always long, starting invested: strategy == buy-and-hold, no cost."""
    ret, rf = _synthetic()
    signal = pd.Series(1.0, index=ret.index)

    # Cost is deliberately large. If the engine charges an entry trade on day 0,
    # or any trade later, this test fails.
    res = backtest(ret, rf, signal, cost=0.01, initial_weight=1.0)

    pd.testing.assert_series_equal(
        res.net_ret, ret, check_exact=True, check_names=False
    )
    assert res.turnover.sum() == 0.0