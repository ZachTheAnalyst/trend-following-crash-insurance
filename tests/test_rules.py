"""Unit tests for the trend rules (families 1-5), on synthetic price series.

Includes the plan's sixth validation test: a flat price series gives zero
trades and zero cost for every trend rule.
"""

import itertools

import numpy as np
import pandas as pd
import pytest

from src.data import total_return_index
from src.engine import backtest
from src.rules import (
    ema,
    family1_sma,
    family2_price_vs_monthly_sma,
    family3_ema,
    family4_tsmom,
    family5_breakout,
    month_end_mask,
)

N_DAYS = 420  # more than the longest rule needs (250 days, or 12 x 21 = 252)
DAYS = pd.bdate_range("2000-01-03", periods=N_DAYS)
RF = pd.Series(0.0001, index=DAYS)


def _catalog():
    """Every rule of families 1-5 in the locked grid: (name, function of level)."""
    rules = []
    for short, long, band in itertools.product([1, 5, 10, 20, 50], [50, 100, 150, 200, 250], [0.0, 0.01]):
        if short < long:
            rules.append((f"f1 {short}/{long} b{band}", lambda x, s=short, l=long, b=band: family1_sma(x, s, l, b)))
    for n in (6, 8, 10, 12):
        rules.append((f"f2 N{n}", lambda x, n=n: family2_price_vs_monthly_sma(x, n)))
    for s, l in itertools.product([10, 20, 50], [100, 150, 200]):
        rules.append((f"f3 {s}/{l}", lambda x, s=s, l=l: family3_ema(x, s, l)))
    for L in (1, 3, 6, 9, 12):
        rules.append((f"f4 L{L}", lambda x, L=L: family4_tsmom(x, RF, L)))
    for n, band in itertools.product([50, 100, 150, 200, 250], [0.0, 0.01]):
        rules.append((f"f5 N{n} b{band}", lambda x, n=n, b=band: family5_breakout(x, n, b)))
    return rules


CATALOG = _catalog()


def test_catalog_has_the_planned_number_of_trend_rules():
    # 48 + 4 + 9 + 5 + 10 (families 1-5 in the plan's catalog table)
    assert len(CATALOG) == 76


# --- the plan's sixth validation test -------------------------------------

@pytest.mark.parametrize("flat_level", [1.0, 1.1])
@pytest.mark.parametrize("name,rule", CATALOG, ids=[n for n, _ in CATALOG])
def test_flat_price_means_zero_trades_and_zero_cost(name, rule, flat_level):
    # Index is flat at 1.0, or flat at 1.1 after one first-day move.
    ret = pd.Series(0.0, index=DAYS)
    ret.iloc[0] = flat_level - 1.0
    level = total_return_index(ret)
    sig = rule(level).dropna()
    assert len(sig) > 0  # the rule does produce signals on 420 days
    assert sig.nunique() == 1  # never changes
    out = backtest(ret.loc[sig.index], RF.loc[sig.index], sig, cost=0.01, initial_weight=sig.iloc[0])
    assert out.turnover.sum() == 0.0
    assert (out.turnover * 0.01).sum() == 0.0


# --- no look-ahead ----------------------------------------------------------

@pytest.mark.parametrize("name,rule", CATALOG, ids=[n for n, _ in CATALOG])
def test_signal_on_day_t_ignores_prices_after_day_t(name, rule):
    rng = np.random.default_rng(1)
    ret = pd.Series(rng.normal(0.0004, 0.01, N_DAYS), index=DAYS)
    base = rule(total_return_index(ret))
    for k in range(260, 400, 7):
        ret2 = ret.copy()
        ret2.iloc[k + 1:] = rng.normal(0.0, 0.05, N_DAYS - k - 1)  # different future
        ret2.iloc[k + 1] = 0.5 if k % 2 else -0.5  # a big shock right after day k
        alt = rule(total_return_index(ret2))
        pd.testing.assert_series_equal(base.iloc[: k + 1], alt.iloc[: k + 1])


# --- monthly rules only change at month-ends -------------------------------

@pytest.mark.parametrize(
    "rule",
    [lambda x: family2_price_vs_monthly_sma(x, 6), lambda x: family4_tsmom(x, RF, 3)],
    ids=["family2", "family4"],
)
def test_monthly_signal_changes_only_on_month_end_days(rule):
    rng = np.random.default_rng(2)
    level = total_return_index(pd.Series(rng.normal(0.0, 0.015, N_DAYS), index=DAYS))
    sig = rule(level).dropna()
    changed = sig.index[sig.diff().fillna(0.0) != 0.0]
    assert len(changed) > 0
    assert month_end_mask(level.index).loc[changed].all()


# --- trending series: always long when rising, always cash when falling ------

@pytest.mark.parametrize("name,rule", CATALOG, ids=[n for n, _ in CATALOG])
def test_steady_rise_is_long_and_steady_fall_is_cash(name, rule):
    up = pd.Series(1.02 ** np.arange(N_DAYS), index=DAYS)
    down = pd.Series(0.98 ** np.arange(N_DAYS), index=DAYS)
    assert (rule(up).dropna() == 1.0).all()
    assert (rule(down).dropna() == 0.0).all()


# --- hand-computed examples -------------------------------------------------

def _series(values, dates=None):
    idx = pd.to_datetime(dates) if dates is not None else pd.bdate_range("2000-01-03", periods=len(values))
    return pd.Series(values, index=idx, dtype=float)


def test_family1_band_holds_the_position_inside_the_band():
    # short = price (1), long = 3-day SMA. SMA3 from day 2: 10, 10.667, 10.833, 11.
    level = _series([10, 10, 10, 12, 10.5, 10.5])
    no_band = family1_sma(level, 1, 3, band=0.0).dropna()
    band = family1_sma(level, 1, 3, band=0.1).dropna()
    # Day 2: tie, starts in cash. Day 3: 12 enters. Day 4: 10.5 is below the SMA.
    assert list(no_band) == [0, 1, 0, 0]  # no band: exits at once
    assert list(band) == [0, 1, 1, 1]  # 10.5 is not below 0.9 x SMA: holds


def test_family1_band_entry_and_exit_on_hand_series():
    level = _series([10, 10, 10, 10, 10, 13, 13, 13, 7, 7])
    sig = family1_sma(level, 1, 3, band=0.1)
    assert sig.iloc[:2].isna().all()
    # SMA3: day 2..9 = 10, 10, 10, 11, 12, 13, 11, 9. Enter day 5 (13 > 12.1),
    # hold days 6-7, exit day 8 (7 < 9.9).
    assert list(sig.iloc[2:]) == [0, 0, 0, 1, 1, 1, 0, 0]


def test_family2_uses_month_end_prices_and_holds_between_them():
    dates = ["2000-01-31", "2000-02-15", "2000-02-29", "2000-03-15", "2000-03-31", "2000-04-14", "2000-04-28"]
    level = _series([1, 9, 2, 9, 1.5, 9, 1.6], dates)  # the 9s are mid-month and must not matter
    sig = family2_price_vs_monthly_sma(level, 2)
    # Month-ends 1, 2, 1.5, 1.6 -> SMA2: -, 1.5, 1.75, 1.55 -> signals: -, 1, 0, 1
    expected = [np.nan, np.nan, 1, 1, 0, 0, 1]
    np.testing.assert_array_equal(sig.to_numpy(), np.array(expected))


def test_family4_compares_past_return_with_tbill_over_the_same_days():
    dates = ["2000-01-31", "2000-02-01", "2000-02-29", "2000-03-31"]
    level = _series([1.0, 1.0, 1.1, 1.0], dates)
    rf = _series([0.01] * 4, dates)
    sig = family4_tsmom(level, rf, L=1, month=2)  # "one month" = 2 days here
    # Month-ends: days 0, 2, 3. Day 2: 1.1 / 1.0 - 1 = 10% > 1.01^2 - 1 = 2.01%: long.
    # Day 3: 1.0 / 1.0 - 1 = 0% < 2.01%: cash.
    np.testing.assert_array_equal(sig.to_numpy(), np.array([np.nan, np.nan, 1, 0]))


def test_family5_enters_on_new_high_exits_on_new_low_holds_otherwise():
    level = _series([5, 6, 7, 8, 7.5, 6, 9])
    sig = family5_breakout(level, N=3, band=0.0)
    assert sig.iloc[:3].isna().all()
    # Day 3: 8 > prior 3-day high 7: enter. Day 4: 7.5 inside 6..8: hold.
    # Day 5: 6 < prior 3-day low 7: exit. Day 6: 9 > 8: enter.
    assert list(sig.iloc[3:]) == [1, 1, 0, 1]


def test_ema_matches_hand_values_and_waits_for_span_observations():
    out = ema(_series([1.0, 2.0, 3.0]), span=3)  # alpha = 0.5
    assert out.iloc[:2].isna().all()
    assert out.iloc[2] == pytest.approx(2.25)  # 1 -> 1.5 -> 2.25


def test_ema_of_a_constant_is_exactly_that_constant():
    out = ema(_series([1.1] * 500), span=200).dropna()
    assert (out == 1.1).all()


def test_short_window_must_be_below_long_window():
    level = _series([1.0] * 10)
    with pytest.raises(ValueError):
        family1_sma(level, 50, 50)
    with pytest.raises(ValueError):
        family3_ema(level, 100, 50)
