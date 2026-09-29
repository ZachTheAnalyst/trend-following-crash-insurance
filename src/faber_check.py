"""Replication check 2: Faber (2007) 10-month SMA rule (pre-registration,
"Code validation", replication check 2).

    python -m src.faber_check data/raw/2026-09-28

Faber reports, for the S&P 500 over 1972-2005 (July 2006 working paper, Table 5):

                        CAGR     Standard deviation
    buy-and-hold        11.24%   17.47%
    10-month SMA timing 11.18%   14.00%

French's market return is not the S&P 500, so the check compares the
timing-minus-buy-and-hold differences, -0.06 points of CAGR and -3.47 points of
standard deviation, and they must match within 0.5 percentage points.

Faber's conventions are used for this check only: total returns, trades at the
month-end close, no trading costs. His cash rate is 90-day commercial paper;
here it is the T-bill rate from the French file, and the paper explains any gap.

Rule: at each month-end, hold the market for the next month if the month-end
index is above the average of the last 10 month-end index values (including
this one), otherwise hold cash for the next month.

The numbers above must be checked against the published 2007 version before
this check is run on real data (pre-registration).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src import data

START, END = "1972-01", "2005-12"
WINDOW = 10
TOL_PP = 0.5
TARGET_DCAGR_PP = -0.06
TARGET_DSD_PP = -3.47


def monthly_from_daily(mkt_daily, rf_daily):
    """Compound daily market and risk-free returns into calendar months."""
    per = mkt_daily.index.to_period("M")
    mkt = (1.0 + mkt_daily).groupby(per).prod() - 1.0
    rf = (1.0 + rf_daily).groupby(per).prod() - 1.0
    return pd.DataFrame({"mkt": mkt, "rf": rf})


def sma_signal(returns, window=WINDOW):
    """True at month-end t when the index is above its `window`-month average.

    The index is the cumulative total return. The average includes month t and
    the window-1 months before it, so the first signal is at month `window`.
    """
    level = (1.0 + returns).cumprod()
    sma = level.rolling(window).mean()
    return (level > sma).where(sma.notna())


def timing_returns(monthly, window=WINDOW):
    """Monthly return of the rule. The signal at month-end t decides month t+1."""
    signal = sma_signal(monthly["mkt"], window)
    held = signal.shift(1)  # decided one month earlier: no look-ahead
    out = monthly["mkt"].where(held == True, monthly["rf"])  # noqa: E712
    return out.where(held.notna())


def cagr(monthly_returns):
    """Annualized geometric return."""
    r = monthly_returns.dropna()
    return (1.0 + r).prod() ** (12.0 / len(r)) - 1.0


def ann_sd(monthly_returns):
    """Annualized standard deviation of monthly returns."""
    return monthly_returns.dropna().std(ddof=1) * np.sqrt(12.0)


def run(raw_dir):
    raw = Path(raw_dir)
    ff3 = data.load_ff3(raw / "F-F_Research_Data_Factors_daily_CSV.zip")
    monthly = monthly_from_daily(data.market_return(ff3), ff3["RF"])
    timing = timing_returns(monthly)
    m = monthly.loc[START:END]
    t = timing.loc[START:END]
    assert t.notna().all() and len(m) == 408, "1972-01 to 2005-12 must be 408 full months"

    bh_cagr, bh_sd = 100 * cagr(m["mkt"]), 100 * ann_sd(m["mkt"])
    tm_cagr, tm_sd = 100 * cagr(t), 100 * ann_sd(t)
    d_cagr, d_sd = tm_cagr - bh_cagr, tm_sd - bh_sd

    print("REPLICATION CHECK 2: Faber 10-month SMA, French market return, 1972-2005")
    print(f"                  CAGR %   SD %")
    print(f"  buy-and-hold   {bh_cagr:7.2f} {bh_sd:6.2f}")
    print(f"  10-month SMA   {tm_cagr:7.2f} {tm_sd:6.2f}")
    print(f"  difference     {d_cagr:7.2f} {d_sd:6.2f}   (Faber: {TARGET_DCAGR_PP:.2f}, {TARGET_DSD_PP:.2f})")
    gap_c, gap_s = abs(d_cagr - TARGET_DCAGR_PP), abs(d_sd - TARGET_DSD_PP)
    ok = gap_c <= TOL_PP and gap_s <= TOL_PP
    print(f"  gap to Faber   {gap_c:7.2f} {gap_s:6.2f}   tolerance {TOL_PP} pp each")
    print(f"  pre-registered criterion: {'MET' if ok else 'NOT MET'}")
    return ok


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python -m src.faber_check <raw data folder>")
    run(sys.argv[1])
