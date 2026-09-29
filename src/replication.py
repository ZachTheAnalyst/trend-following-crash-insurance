"""Replication check 1 (pre-registration, "Code validation").

The plan: daily industry returns compounded over a calendar year should match
French's own annual industry file to within 0.1% per year.

    python -m src.replication data/raw/2026-09-28

This is a report, not a gate. It prints the result against the pre-registered
tolerance, breaks it down by era, and runs two diagnostics that show where any
gap comes from (annual vs monthly file, daily vs monthly file).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src import data

TOL = 0.001  # 0.1 percentage point, as pre-registered
FIRST_YEAR, LAST_YEAR = 1927, 2025  # French's annual table starts in 1927
ERAS = [(1927, 1945), (1946, 1962), (1963, 1989), (1990, 2025)]


def compound(daily, by):
    """Compound daily returns within each group. Also returns a mask that is
    True where the group had no missing days."""
    grouped = (1.0 + daily).groupby(by)
    comp = grouped.prod(min_count=1) - 1.0
    complete = daily.isna().groupby(by).sum() == 0
    return comp, complete


def abs_diff(computed, reference, complete):
    """Absolute gap, only where both values exist and the daily data is complete."""
    ok = computed.notna() & reference.notna() & complete
    return (computed - reference).abs().where(ok)


def summarize(diff, tol=TOL):
    x = diff.to_numpy(dtype=float)
    x = x[~np.isnan(x)]
    return {
        "n": len(x),
        "within_tol_pct": 100.0 * (x <= tol).mean(),
        "median_pp": 100.0 * np.median(x),
        "p95_pp": 100.0 * np.percentile(x, 95),
        "max_pp": 100.0 * x.max(),
    }


def run(raw_dir):
    raw = Path(raw_dir)
    tabs = data.parse_french_tables(data.read_text(raw / "49_Industry_Portfolios_CSV.zip"))
    monthly = tabs["Average Value Weighted Returns -- Monthly"]
    annual = tabs["Average Value Weighted Returns -- Annual"]
    annual.index = annual.index.year
    daily = data.load_industry(raw / "49_Industry_Portfolios_daily_CSV.zip", "vw")

    # Check 1 as pre-registered: daily compounded by year vs French's annual file.
    comp, complete = compound(daily, daily.index.year)
    rows = slice(FIRST_YEAR, LAST_YEAR)
    diff = abs_diff(comp.loc[rows], annual.loc[rows], complete.loc[rows])
    s = summarize(diff)
    n_out = int((diff > TOL).sum().sum())
    print("CHECK 1: daily returns compounded by year vs French annual file")
    print(f"  industry-years compared: {s['n']}")
    print(f"  within 0.1 point: {s['within_tol_pct']:.1f}%   (exceeding: {n_out})")
    print(f"  median gap {s['median_pp']:.3f} pp, 95th pct {s['p95_pp']:.3f} pp, max {s['max_pp']:.2f} pp")
    print(f"  pre-registered criterion (all within 0.1 point): {'MET' if n_out == 0 else 'NOT MET'}")
    print("\n  by era:")
    print(f"  {'period':<10}{'n':>6}{'% within':>10}{'median pp':>11}{'95th pp':>9}{'max pp':>8}")
    for lo, hi in ERAS:
        e = summarize(diff.loc[lo:hi])
        print(
            f"  {lo}-{hi:<5}{e['n']:>6}{e['within_tol_pct']:>10.1f}"
            f"{e['median_pp']:>11.3f}{e['p95_pp']:>9.3f}{e['max_pp']:>8.2f}"
        )
    worst = diff.stack().sort_values(ascending=False).head(5)
    print("\n  five largest gaps (pp):")
    for (year, ind), v in worst.items():
        print(f"    {year} {ind}: {100 * v:.2f}")

    # Diagnostic A: French's annual file vs his own monthly file.
    months_per_year = monthly.groupby(monthly.index.year).size()
    full_years = [y for y in months_per_year[months_per_year == 12].index if FIRST_YEAR <= y <= LAST_YEAR]
    m_comp = (1.0 + monthly).groupby(monthly.index.year).prod(min_count=12) - 1.0
    a = summarize(abs_diff(m_comp.loc[full_years], annual.loc[full_years], pd.DataFrame(True, index=full_years, columns=annual.columns)))
    print("\nDIAGNOSTIC A: French annual file vs his monthly file compounded")
    print(f"  compared {a['n']}, within 0.1 point {a['within_tol_pct']:.1f}%, max {a['max_pp']:.3f} pp")

    # Diagnostic B: daily file compounded within each month vs his monthly file.
    dm, dcomplete = compound(daily, daily.index.to_period("M"))
    monthly.index = monthly.index.to_period("M")
    common = dm.index.intersection(monthly.index)
    b = summarize(abs_diff(dm.loc[common], monthly.loc[common], dcomplete.loc[common]))
    print("\nDIAGNOSTIC B: daily file compounded by month vs his monthly file")
    print(f"  compared {b['n']}, within 0.1 point {b['within_tol_pct']:.1f}%, median {b['median_pp']:.3f} pp, max {b['max_pp']:.2f} pp")
    return n_out


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python -m src.replication <raw data folder>")
    run(sys.argv[1])
