"""Unit tests for the equal-weighted industry portfolio, on hand-built data."""

import numpy as np
import pandas as pd
import pytest

from src import data, rules
from src.engine import backtest
from src.portfolio import evaluation_frame, signals_50_200, strategy_portfolio


def _hand_example():
    idx = pd.bdate_range("2000-01-03", periods=6)
    nan = np.nan
    ret = pd.DataFrame(
        {"A": [0.02, 0.01, -0.03, 0.02, 0.01, 0.00], "B": [nan, nan, 0.01, 0.02, nan, 0.03]}, index=idx
    )
    rf = pd.Series(0.001, index=idx)
    signal = pd.DataFrame({"A": [1, 1, 0, 0, 1, 1], "B": [1, 0, 1, 1, 0, 1]}, index=idx, dtype=float)
    active = pd.DataFrame(
        {
            "A": [False, True, True, True, True, True],  # A joins on day 1
            "B": [False, False, True, True, False, True],  # B joins on day 2, missing on day 4
        },
        index=idx,
    )
    return ret, rf, signal, active


def test_hand_computed_two_industry_portfolio_with_late_entry_and_a_gap():
    ret, rf, signal, active = _hand_example()
    out = strategy_portfolio(ret, rf, signal, active, cost=0.01)
    # Day 0 has no active industry and is dropped.
    assert list(out.index) == list(ret.index[1:])
    # Weights held: A = signal shifted one day; B likewise, on its active days only.
    # A: 1, 1, 0, 0, 1 (days 1-5). B: 0, 1, -, 0 (days 2, 3, 5).
    # Costs (c = 1%): A pays on day 3 (1 -> 0) and day 5 (0 -> 1). B pays on day 3 (0 -> 1) and
    # day 5, measured against its last ACTIVE weight (1 -> 0) across the missing day 4.
    # Neither pays an entry trade on its first active day.
    expected_net = [0.01, (-0.03 + 0.001) / 2, (-0.009 + 0.01) / 2, 0.001, (-0.01 - 0.009) / 2]
    assert out["net"].to_numpy() == pytest.approx(expected_net, abs=1e-15)
    assert out["bh"].to_numpy() == pytest.approx([0.01, -0.01, 0.02, 0.01, 0.015], abs=1e-15)
    assert list(out["n_active"]) == [1, 2, 2, 1, 2]
    assert out["turnover"].to_numpy() == pytest.approx([0, 0, 1, 0, 1], abs=1e-15)
    assert (out["rf"] == 0.001).all()


def test_buy_and_hold_pays_no_cost_and_ignores_inactive_industries():
    ret, rf, signal, active = _hand_example()
    always_long = signal * 0 + 1.0
    out = strategy_portfolio(ret, rf, always_long, active, cost=0.05)
    # Always long: strategy equals buy-and-hold exactly and pays nothing (no entry trade).
    assert out["net"].to_numpy() == pytest.approx(out["bh"].to_numpy(), abs=1e-15)
    assert (out["turnover"] == 0).all()


def test_always_cash_earns_the_tbill_rate():
    ret, rf, signal, active = _hand_example()
    out = strategy_portfolio(ret, rf, signal * 0.0, active, cost=0.05)
    assert out["net"].to_numpy() == pytest.approx(np.full(5, 0.001), abs=1e-15)


def test_single_industry_matches_the_engine_exactly():
    rng = np.random.default_rng(1)
    n = 800
    idx = pd.bdate_range("2000-01-03", periods=n)
    ret = pd.DataFrame({"X": rng.normal(0.0004, 0.012, n)}, index=idx)
    rf = pd.Series(0.0001, index=idx)
    sig = signals_50_200(ret)
    active = pd.DataFrame({"X": np.arange(n) >= 300}, index=idx)  # like the 273-day rule
    out = strategy_portfolio(ret, rf, sig, active, cost=0.001)

    k = 300
    s = sig["X"].iloc[k - 1 :]  # signal from the close before the first active day
    res = backtest(ret["X"].iloc[k - 1 :], rf.iloc[k - 1 :], s, 0.001, initial_weight=float(s.iloc[0]))
    ref = res.net_ret.iloc[1:]
    assert out["net"].to_numpy() == pytest.approx(ref.to_numpy(), abs=1e-15)


def test_signals_are_the_family1_flagship_per_industry():
    rng = np.random.default_rng(2)
    idx = pd.bdate_range("2000-01-03", periods=500)
    ret = pd.DataFrame(rng.normal(0.0003, 0.01, (500, 3)), index=idx, columns=list("ABC"))
    sig = signals_50_200(ret)
    for c in ret.columns:
        expected = rules.family1_sma(data.total_return_index(ret[c]), 50, 200, 0.0)
        pd.testing.assert_series_equal(sig[c], expected, check_names=False)
    assert sig.iloc[:199].isna().all().all()


def test_position_on_day_t_ignores_returns_from_day_t_onward():
    """The plan's look-ahead test at portfolio level."""
    rng = np.random.default_rng(3)
    n = 900
    idx = pd.bdate_range("2000-01-03", periods=n)
    ret = pd.DataFrame(rng.normal(0.0003, 0.012, (n, 4)), index=idx, columns=list("ABCD"))
    rf = pd.Series(0.0001, index=idx)
    active = pd.DataFrame(True, index=idx, columns=ret.columns)
    active.iloc[:273] = False
    base_sig = signals_50_200(ret)
    for t0 in range(400, 850, 41):
        r2 = ret.copy()
        r2.iloc[t0:] = rng.normal(0.0, 0.05, (n - t0, 4))
        alt_sig = signals_50_200(r2)
        # weight held on day t is signal[t-1]: compare weights through day t0
        w_base = base_sig.shift(1).iloc[: t0 + 1]
        w_alt = alt_sig.shift(1).iloc[: t0 + 1]
        pd.testing.assert_frame_equal(w_base, w_alt)


def test_evaluation_frame_starts_on_the_given_date():
    ret, rf, signal, active = _hand_example()
    out = strategy_portfolio(ret, rf, signal, active, cost=0.01)
    assert evaluation_frame(out, pd.Timestamp(ret.index[3])).index[0] == ret.index[3]
