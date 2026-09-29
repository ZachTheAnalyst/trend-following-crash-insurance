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


def catalog_families_1_to_5():
    """The locked grid of families 1-5 as (name, function(level, rf)). 76 rules."""
    import itertools

    out = []
    for short, long, band in itertools.product([1, 5, 10, 20, 50], [50, 100, 150, 200, 250], [0.0, 0.01]):
        if short < long:
            out.append((f"f1 {short}/{long} b{band}", lambda x, rf, s=short, l=long, b=band: family1_sma(x, s, l, b)))
    for n in (6, 8, 10, 12):
        out.append((f"f2 N{n}", lambda x, rf, n=n: family2_price_vs_monthly_sma(x, n)))
    for s, l in itertools.product([10, 20, 50], [100, 150, 200]):
        out.append((f"f3 {s}/{l}", lambda x, rf, s=s, l=l: family3_ema(x, s, l)))
    for L in (1, 3, 6, 9, 12):
        out.append((f"f4 L{L}", lambda x, rf, L=L: family4_tsmom(x, rf, L)))
    for n, band in itertools.product([50, 100, 150, 200, 250], [0.0, 0.01]):
        out.append((f"f5 N{n} b{band}", lambda x, rf, n=n, b=band: family5_breakout(x, n, b)))
    return out


def first_day_all_rules_valid(index):
    """First date on which every rule of families 1-5 has a valid signal.

    Depends only on the trading calendar, not on prices, so it is computed on a
    flat price series: no return of any strategy is used. Families 6-8 need at
    most 273 trading days (family 8: 12 months + the skipped month, at 21 days
    each), which the 273-day rule already covers.
    """
    level = pd.Series(1.0, index=index)
    rf = pd.Series(0.0, index=index)
    latest = None
    for _, rule in catalog_families_1_to_5():
        first = rule(level, rf).first_valid_index()
        if first is None:
            raise ValueError("calendar is too short for the longest rule")
        latest = first if latest is None else max(latest, first)
    return latest


def evaluation_start(index, first_273):
    """Plan: evaluation starts on the first day every rule has a valid signal.

    `first_273` is the first day an industry has 273 days of history
    (data.first_evaluation_date). The evaluation start is the later of the two.
    """
    return max(first_273, first_day_all_rules_valid(index))
