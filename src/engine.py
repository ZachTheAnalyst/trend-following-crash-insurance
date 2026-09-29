"""Core backtest engine.

Turns a target-weight signal into net daily strategy returns.

Timing (pre-registration, "Signal and execution"): the signal is decided at
the close of day t and earns the return of day t+1, so the weight held on
day t is signal[t-1]. On day 0 the strategy holds `initial_weight`.

Costs: one-way cost c is charged on every change in stock weight,
cost_t = c * |w_t - w_{t-1}|, so fractional-weight strategies pay in
proportion. The weight before day 0 is `initial_weight`, so day 0 pays no
entry cost.
"""

from dataclasses import dataclass

import pandas as pd


@dataclass
class Result:
    net_ret: pd.Series  # daily net strategy return
    weight: pd.Series  # stock weight held each day
    turnover: pd.Series  # |w_t - w_{t-1}| each day


def backtest(
    ret: pd.Series,
    rf: pd.Series,
    signal: pd.Series,
    cost: float,
    initial_weight: float = 1.0,
) -> Result:
    """Run one long-only backtest.

    ret            daily stock total returns
    rf             daily risk-free (T-bill) returns, same index as ret
    signal         target stock weight decided at the close of each day
    cost           one-way trading cost as a fraction (0.001 = 10 bp)
    initial_weight stock weight held before day 0
    """
    if not (ret.index.equals(rf.index) and ret.index.equals(signal.index)):
        raise ValueError("ret, rf and signal must share the same index")

    # One-day lag: the weight held on day t is the signal from the close of t-1.
    weight = signal.shift(1)
    weight.iloc[0] = initial_weight

    prev = weight.shift(1)
    prev.iloc[0] = initial_weight
    turnover = (weight - prev).abs()

    net_ret = weight * ret + (1.0 - weight) * rf - cost * turnover
    return Result(net_ret=net_ret, weight=weight, turnover=turnover)