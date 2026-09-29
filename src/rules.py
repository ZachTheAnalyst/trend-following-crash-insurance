"""Trend signals for families 1-5 of the strategy catalog (pre-registration,
"Strategy specification" and "Strategy catalog").

Every function takes the total-return index of ONE industry (`level`, a Series
indexed by date) and returns a daily Series of 1.0 (long), 0.0 (cash) or NaN
(not enough history yet). The engine does the one-day lag: the weight on day
t+1 is the signal at the close of day t (`engine.backtest` shifts by one day).

Monthly rules (families 2 and 4) decide at the last close of the month. Their
signal is held flat from that close until the next month-end, so it changes
only on month-end days, and the engine then trades on the first trading day of
the next month, as the plan says.

Implementation choices where the plan is silent (documented in the README):

* One month = 21 trading days (the plan's definition). Family 4 uses it for
  its L-month lookback. Family 2 averages the last N month-end prices.
* Ties count as "not long": every entry test is a strict ">", every exit test
  a strict "<".
* A band or channel rule starts in cash on its first valid day, unless its
  entry condition is already true that day.
* Channel breakout uses closing prices only (the data has no highs or lows):
  "prior N-day high" is the highest close of the N days before today.
* EMA: alpha = 2 / (span + 1), started at the first index value (pandas
  `ewm(adjust=False)`), and valid only after `span` observations.
"""

import numpy as np
import pandas as pd

TRADING_MONTH = 21  # trading days, from the plan


def sma(x, window):
    """Simple moving average of the last `window` values, including today."""
    return x.rolling(window, min_periods=window).mean()


def ema(x, span):
    """Exponential moving average with alpha = 2 / (span + 1).

    NaN until `span` valid observations have been seen.
    """
    return x.ewm(alpha=2.0 / (span + 1.0), adjust=False, min_periods=span).mean()


def month_end_mask(index):
    """True on the last trading day of each calendar month in `index`."""
    days = index.to_series()
    last = days.groupby(index.to_period("M")).transform("max")
    return days == last


def _hold_between_bands(enter, exit_, valid):
    """State machine: 1 after `enter`, 0 after `exit_`, otherwise keep the state.

    Starts in cash on the first valid day. NaN where `valid` is False.
    """
    raw = np.where(enter, 1.0, np.where(exit_, 0.0, np.nan))
    state = pd.Series(raw, index=valid.index).ffill()
    state = state.where(valid)
    return state.where(~(valid & state.isna()), 0.0)


def _month_end_signal(daily_condition, valid, level_index):
    """Sample a condition at month-ends and hold it until the next month-end."""
    at_close = pd.Series(np.where(valid, daily_condition.astype(float), np.nan), index=level_index)
    return at_close.where(month_end_mask(level_index)).ffill()


def family1_sma(level, short, long, band=0.0):
    """Long if short SMA > long SMA x (1 + band); exit if short < long x (1 - band).

    short = 1 means the price itself. Between the two triggers the position is held.
    """
    if not short < long:
        raise ValueError("short window must be smaller than long window")
    s = level if short == 1 else sma(level, short)
    lg = sma(level, long)
    valid = lg.notna()
    enter = s > lg * (1.0 + band)
    exit_ = s < lg * (1.0 - band)
    return _hold_between_bands(enter, exit_, valid)


def family2_price_vs_monthly_sma(level, n_months):
    """At month-end, long if the price is above the SMA of the last N month-end prices."""
    mask = month_end_mask(level.index)
    me = level[mask]
    avg = sma(me, n_months)
    sig = pd.Series(np.where(avg.notna(), (me > avg).astype(float), np.nan), index=me.index)
    return sig.reindex(level.index).ffill()


def family3_ema(level, short_span, long_span):
    """Long if the short EMA is above the long EMA (daily)."""
    if not short_span < long_span:
        raise ValueError("short span must be smaller than long span")
    s, lg = ema(level, short_span), ema(level, long_span)
    valid = s.notna() & lg.notna()
    enter = s > lg
    exit_ = ~enter
    return _hold_between_bands(enter, exit_, valid)


def family4_tsmom(level, rf, L, month=TRADING_MONTH):
    """At month-end, long if the past L-month return beats the T-bill over the same days.

    `rf` is the daily risk-free return (decimal), on the same index as `level`.
    """
    k = L * month
    ret = level / level.shift(k) - 1.0
    rf_index = (1.0 + rf).cumprod()
    rf_ret = rf_index / rf_index.shift(k) - 1.0
    valid = ret.notna() & rf_ret.notna()
    return _month_end_signal(ret > rf_ret, valid, level.index)


def family5_breakout(level, N, band=0.0):
    """Enter on close > prior N-day high x (1 + band); exit on close < prior N-day low x (1 - band).

    Otherwise keep the current position. Closing prices only.
    """
    prior = level.shift(1)
    high = prior.rolling(N, min_periods=N).max()
    low = prior.rolling(N, min_periods=N).min()
    valid = high.notna()
    enter = level > high * (1.0 + band)
    exit_ = level < low * (1.0 - band)
    return _hold_between_bands(enter, exit_, valid)
