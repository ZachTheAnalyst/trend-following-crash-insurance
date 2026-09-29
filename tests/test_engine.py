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


def test_always_cash_equals_tbill_exactly():
    """Signal always cash, starting in cash: strategy == T-bill series, no cost."""
    ret, rf = _synthetic()
    signal = pd.Series(0.0, index=ret.index)

    res = backtest(ret, rf, signal, cost=0.01, initial_weight=0.0)

    pd.testing.assert_series_equal(
        res.net_ret, rf, check_exact=True, check_names=False
    )
    assert res.turnover.sum() == 0.0


def test_ten_day_hand_computed_example_with_two_trades():
    """10 days, 2 trades (enter, exit), checked against arithmetic done by hand.

    signal (decided at close of day t):  0 1 1 1 1 0 0 0 0 0
    weight held on day t = signal[t-1]:  0 0 1 1 1 1 0 0 0 0   (day 0 = initial 0)
    trades: day 2 (0 -> 1) and day 6 (1 -> 0), cost c = 0.001 each
    """
    idx = pd.bdate_range("2000-01-03", periods=10)
    ret = pd.Series(
        [0.01, -0.02, 0.03, 0.01, -0.01, 0.02, -0.03, 0.01, 0.00, 0.02], index=idx
    )
    rf = pd.Series(0.0001, index=idx)
    signal = pd.Series([0, 1, 1, 1, 1, 0, 0, 0, 0, 0], index=idx, dtype=float)

    res = backtest(ret, rf, signal, cost=0.001, initial_weight=0.0)

    # Day by day:
    #   days 0,1,7,8,9 (cash):      rf                    = 0.0001
    #   day 2 (buy):    0.03 - 0.001                      = 0.029
    #   days 3,4,5 (invested):      0.01, -0.01, 0.02
    #   day 6 (sell):   rf - 0.001 = 0.0001 - 0.001       = -0.0009
    expected = pd.Series(
        [0.0001, 0.0001, 0.029, 0.01, -0.01, 0.02, -0.0009, 0.0001, 0.0001, 0.0001],
        index=idx,
    )
    pd.testing.assert_series_equal(
        res.net_ret, expected, check_exact=False, atol=1e-10, rtol=0, check_names=False
    )

    # Compounded over the 10 days: prod(1 + r) - 1
    assert abs((1 + res.net_ret).prod() - 1 - 0.04905488458296836) < 1e-10

    assert int((res.turnover > 0).sum()) == 2
    assert abs(res.turnover.sum() - 2.0) < 1e-10


def test_cost_accounting_total_cost_equals_c_times_sum_abs_dw():
    """Total cost paid = c * sum(|change in weight|), including fractional weights."""
    ret, rf = _synthetic(n=300, seed=1)
    rng = np.random.default_rng(2)
    signal = pd.Series(rng.uniform(0.0, 1.0, len(ret)), index=ret.index)
    c = 0.0025
    w0 = 0.4

    res = backtest(ret, rf, signal, cost=c, initial_weight=w0)

    # Independent weight path: initial weight, then signal lagged one day.
    w = np.r_[w0, signal.to_numpy()[:-1]]
    dw = np.abs(np.diff(np.r_[w0, w]))
    gross = w * ret.to_numpy() + (1 - w) * rf.to_numpy()

    np.testing.assert_allclose(res.turnover.to_numpy(), dw, rtol=0, atol=1e-12)
    total_cost_paid = (gross - res.net_ret.to_numpy()).sum()
    np.testing.assert_allclose(total_cost_paid, c * dw.sum(), rtol=0, atol=1e-12)


def test_no_lookahead_position_on_day_t_ignores_day_t_onward():
    """Changing signals or returns from day t onward must not change the position
    held on day t or earlier."""
    ret, rf = _synthetic(n=300, seed=3)
    rng = np.random.default_rng(4)
    signal = pd.Series(rng.integers(0, 2, len(ret)).astype(float), index=ret.index)
    base = backtest(ret, rf, signal, cost=0.001, initial_weight=0.0)

    for t in (1, 50, 150, 299):
        # Shuffle returns from day t onward.
        ret2 = ret.copy()
        ret2.iloc[t:] = rng.permutation(ret.iloc[t:].to_numpy())
        # Scramble signals from day t onward (signal[t] is only used on day t+1).
        sig2 = signal.copy()
        sig2.iloc[t:] = rng.integers(0, 2, len(ret) - t).astype(float)

        alt = backtest(ret2, rf, sig2, cost=0.001, initial_weight=0.0)

        pd.testing.assert_series_equal(
            alt.weight.iloc[: t + 1], base.weight.iloc[: t + 1], check_exact=True
        )