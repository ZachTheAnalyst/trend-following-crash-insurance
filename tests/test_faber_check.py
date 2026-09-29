"""Unit tests for the Faber check helpers, on synthetic series only."""

import numpy as np
import pandas as pd
import pytest

from src.faber_check import (
    ann_sd,
    cagr,
    monthly_from_daily,
    sma_signal,
    timing_returns,
)


def _months(n, start="2000-01"):
    return pd.period_range(start, periods=n, freq="M")


def test_monthly_from_daily_compounds_within_each_month():
    idx = pd.to_datetime(["2000-01-03", "2000-01-04", "2000-02-01"])
    mkt = pd.Series([0.10, 0.10, 0.05], index=idx)
    rf = pd.Series([0.001, 0.001, 0.002], index=idx)
    m = monthly_from_daily(mkt, rf)
    assert m.loc[pd.Period("2000-01"), "mkt"] == pytest.approx(1.1 * 1.1 - 1)
    assert m.loc[pd.Period("2000-01"), "rf"] == pytest.approx(1.001**2 - 1)
    assert m.loc[pd.Period("2000-02"), "mkt"] == pytest.approx(0.05)


def test_sma_signal_first_value_is_at_month_10_and_needs_10_months():
    r = pd.Series(0.01, index=_months(12))
    sig = sma_signal(r, window=10)
    assert sig.iloc[:9].isna().all()  # months 1-9: not enough history
    assert sig.iloc[9] == True  # noqa: E712  rising index is above its average


def test_steady_rise_means_always_invested_after_warmup():
    n = 40
    m = pd.DataFrame({"mkt": np.full(n, 0.01), "rf": np.full(n, 0.001)}, index=_months(n))
    t = timing_returns(m)
    assert t.iloc[:10].isna().all()  # signal at month 10 decides month 11
    assert (t.iloc[10:] == 0.01).all()


def test_steady_fall_means_always_in_cash_after_warmup():
    n = 40
    m = pd.DataFrame({"mkt": np.full(n, -0.01), "rf": np.full(n, 0.001)}, index=_months(n))
    t = timing_returns(m)
    assert (t.iloc[10:] == 0.001).all()


def test_no_lookahead_month_t_plus_1_shock_cannot_change_the_position_in_t_plus_1():
    n = 40
    rng = np.random.default_rng(0)
    mkt = rng.normal(0.005, 0.04, n)
    rf = np.full(n, 0.002)
    base = timing_returns(pd.DataFrame({"mkt": mkt, "rf": rf}, index=_months(n)))
    for k in (12, 20, 30):
        mkt2 = mkt.copy()
        mkt2[k] = -0.5  # a crash in month k
        alt = timing_returns(pd.DataFrame({"mkt": mkt2, "rf": rf}, index=_months(n)))
        # The position held in month k was decided at the end of month k-1, so
        # whether month k earned the market or cash cannot depend on month k.
        held_base = base.iloc[k] == mkt[k]
        held_alt = alt.iloc[k] == mkt2[k]
        assert held_base == held_alt


def test_cagr_and_annualized_sd_on_known_series():
    r = pd.Series(np.full(24, 0.01))
    assert cagr(r) == pytest.approx(1.01**12 - 1)
    assert ann_sd(r) == pytest.approx(0.0)
    r2 = pd.Series([0.01, -0.01] * 12)
    assert ann_sd(r2) == pytest.approx(np.std(r2, ddof=1) * np.sqrt(12))
