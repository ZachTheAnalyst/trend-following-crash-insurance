"""Unit tests for the replication-check helpers, on tiny hand-made series."""

import numpy as np
import pandas as pd
import pytest

from src.replication import abs_diff, compound, summarize


def test_compound_by_year_and_completeness_flag():
    idx = pd.to_datetime(["2000-06-01", "2000-06-02", "2001-06-01", "2001-06-04"])
    d = pd.DataFrame({"A": [0.10, 0.10, 0.20, np.nan]}, index=idx)
    comp, complete = compound(d, d.index.year)
    assert comp.loc[2000, "A"] == pytest.approx(1.1 * 1.1 - 1)
    assert comp.loc[2001, "A"] == pytest.approx(0.20)  # missing day skipped
    assert bool(complete.loc[2000, "A"]) is True
    assert bool(complete.loc[2001, "A"]) is False  # had a missing day


def test_abs_diff_only_where_both_exist_and_daily_is_complete():
    comp = pd.DataFrame({"A": [0.10, 0.20, np.nan]}, index=[2000, 2001, 2002])
    ref = pd.DataFrame({"A": [0.1005, 0.30, 0.05]}, index=[2000, 2001, 2002])
    complete = pd.DataFrame({"A": [True, False, True]}, index=[2000, 2001, 2002])
    diff = abs_diff(comp, ref, complete)
    assert diff.loc[2000, "A"] == pytest.approx(0.0005)
    assert np.isnan(diff.loc[2001, "A"])  # daily data incomplete
    assert np.isnan(diff.loc[2002, "A"])  # no computed value


def test_summarize_reports_share_within_tolerance():
    diff = pd.DataFrame({"A": [0.0005, 0.0009, 0.0020, np.nan]})
    s = summarize(diff, tol=0.001)
    assert s["n"] == 3
    assert s["within_tol_pct"] == pytest.approx(200 / 3)
    assert s["max_pp"] == pytest.approx(0.20)
