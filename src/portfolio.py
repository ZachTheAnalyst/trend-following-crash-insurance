"""Equal-weighted industry portfolios: the strategy and its buy-and-hold benchmark
(pre-registration, "Strategy specification": Portfolio, Costs).

Rules from the plan:

* The rule runs on each industry separately; the portfolio is the equal-weighted
  average of the industries active that day, rebalanced daily.
* An industry is active on a day when its return is valid and it has at least
  273 valid returns before that day (`data.active_mask`). An industry that
  enters later joins the strategy and the buy-and-hold portfolio on the same day.
* One-way cost c is charged on every change in an industry's stock weight,
  c x |w_t - w_(t-1)|. In cash the strategy earns the daily T-bill rate.
* Buy-and-hold is always fully invested, rebalanced daily, and pays no cost.

Implementation choices where the plan is silent (documented in the README):

* An industry's first active day is not an entry trade: the strategy simply holds
  the weight its signal gives, as the engine does with `initial_weight`.
* After a missing day the industry is not traded, so the weight change is
  measured against the last active day's weight.
* The 50/200 signal is computed on the industry's total-return index, in which a
  missing day leaves the index unchanged (`data.total_return_index`).
"""

import numpy as np
import pandas as pd

from src import data, rules

COST = 0.0010  # 10 bp one-way


def signals_50_200(ret):
    """50/200-day SMA crossover signal (family 1 flagship, no band) for each industry."""
    out = {}
    for name in ret.columns:
        level = data.total_return_index(ret[name])
        out[name] = rules.family1_sma(level, 50, 200, 0.0)
    return pd.DataFrame(out, index=ret.index)


def strategy_portfolio(ret, rf, signal, active, cost=COST):
    """Daily net return of the equal-weighted strategy portfolio, and its benchmark.

    ret     daily industry returns (date x industry), NaN where missing
    rf      daily T-bill rate (Series on the same dates)
    signal  target stock weight decided at each close (date x industry)
    active  True where an industry may be held that day
    Returns a DataFrame with columns net, bh, rf, n_active and turnover (the
    average absolute weight change across active industries).
    """
    w = signal.shift(1).where(active)  # one-day lag: weight held on day t is signal[t-1]
    prev = w.ffill().shift(1)  # weight on the industry's previous active day
    prev = prev.where(prev.notna(), w)  # first active day: no entry trade
    turnover = (w - prev).abs()
    rfm = pd.DataFrame(
        np.repeat(rf.to_numpy()[:, None], ret.shape[1], axis=1), index=ret.index, columns=ret.columns
    )
    net_i = w * ret + (1.0 - w) * rfm - cost * turnover  # NaN where inactive
    out = pd.DataFrame(
        {
            "net": net_i.mean(axis=1, skipna=True),
            "bh": ret.where(active).mean(axis=1, skipna=True),
            "rf": rf,
            "n_active": active.sum(axis=1),
            "turnover": turnover.mean(axis=1, skipna=True),
        }
    )
    return out[out["n_active"] > 0]


def evaluation_frame(portfolio, start):
    """Portfolio rows from the evaluation start date on."""
    return portfolio.loc[start:]
