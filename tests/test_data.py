"""Unit tests for the data loader, on tiny hand-made files in French's format."""

import zipfile

import numpy as np
import pandas as pd
import pytest

from src.data import (
    active_mask,
    cut_sample,
    first_evaluation_date,
    load_industry,
    load_pput,
    market_return,
    parse_french_tables,
    total_return_index,
)

INDUSTRY_TEXT = """This file was created using the 202608 CRSP database.

Missing data are indicated by -99.99 or -999.


  Average Value Weighted Returns -- Daily
,A,B
19260701,   1.00, -99.99
19260702,  -2.00,   3.00
19260706,    -999,   0.50


  Average Equal Weighted Returns -- Daily
,A,B
19260701,   0.50,   0.25
19260702,   0.00,   0.00

"""

FF3_TEXT = """The Tbill return is the simple daily rate.

,Mkt-RF,SMB,HML,RF
19260701,    0.09,   -0.23,   -0.28,    0.01
19260702,    0.44,   -0.35,   -0.08,    0.01

Copyright 2026 Eugene F. Fama and Kenneth R. French
"""


def test_splits_two_titled_tables():
    tables = parse_french_tables(INDUSTRY_TEXT)
    assert list(tables) == [
        "Average Value Weighted Returns -- Daily",
        "Average Equal Weighted Returns -- Daily",
    ]
    assert tables["Average Value Weighted Returns -- Daily"].shape == (3, 2)
    assert tables["Average Equal Weighted Returns -- Daily"].shape == (2, 2)


def test_percent_to_decimal_and_missing_codes_to_nan():
    vw = parse_french_tables(INDUSTRY_TEXT)["Average Value Weighted Returns -- Daily"]
    assert vw.loc["1926-07-01", "A"] == pytest.approx(0.01)
    assert vw.loc["1926-07-02", "A"] == pytest.approx(-0.02)
    assert np.isnan(vw.loc["1926-07-01", "B"])  # -99.99
    assert np.isnan(vw.loc["1926-07-06", "A"])  # -999


def test_untitled_table_gets_generic_key_and_footer_is_ignored():
    tables = parse_french_tables(FF3_TEXT)
    assert list(tables) == ["table1"]
    ff = tables["table1"]
    assert list(ff.columns) == ["Mkt-RF", "SMB", "HML", "RF"]
    assert len(ff) == 2  # the copyright line is not a row


def test_dates_are_parsed_as_dates():
    ff = parse_french_tables(FF3_TEXT)["table1"]
    assert ff.index[0] == pd.Timestamp("1926-07-01")
    assert ff.index.name == "date"


def test_market_return_is_mkt_rf_plus_rf():
    ff = parse_french_tables(FF3_TEXT)["table1"]
    mkt = market_return(ff)
    assert mkt.iloc[0] == pytest.approx(0.0009 + 0.0001)


def test_cut_sample_keeps_end_date_and_drops_later():
    idx = pd.to_datetime(["2025-12-30", "2025-12-31", "2026-01-02"])
    df = pd.DataFrame({"x": [1, 2, 3]}, index=idx)
    out = cut_sample(df, "2025-12-31")
    assert list(out.x) == [1, 2]


def test_active_mask_needs_exactly_273_valid_days_before():
    n = 300
    ret = pd.DataFrame(
        {"A": np.full(n, 0.001)}, index=pd.bdate_range("2000-01-03", periods=n)
    )
    act = active_mask(ret)["A"].to_numpy()
    # Day index k has k valid days before it, so it is active from k = 273.
    assert not act[272]
    assert act[273]
    assert act.sum() == n - 273


def test_missing_day_is_inactive_and_does_not_reset_history():
    n = 300
    r = np.full(n, 0.001)
    r[100] = np.nan
    ret = pd.DataFrame({"A": r}, index=pd.bdate_range("2000-01-03", periods=n))
    act = active_mask(ret)["A"].to_numpy()
    assert not act[100]  # missing day: inactive
    assert not act[273]  # only 272 valid days before it, because of the gap
    assert act[274]
    assert act[101 + 173]  # history was not reset by the gap: 274 - 1 valid


def test_total_return_index_compounds_and_skips_missing_days():
    ret = pd.Series(
        [0.10, np.nan, -0.10], index=pd.bdate_range("2000-01-03", periods=3)
    )
    tri = total_return_index(ret)
    assert tri.iloc[0] == pytest.approx(1.10)
    assert tri.iloc[1] == pytest.approx(1.10)  # unchanged on the missing day
    assert tri.iloc[2] == pytest.approx(1.10 * 0.90)


def test_first_evaluation_date_is_first_day_any_industry_is_active():
    idx = pd.bdate_range("2000-01-03", periods=400)
    a = np.full(400, 0.001)
    b = np.full(400, 0.001)
    b[:50] = np.nan  # B starts 50 days late
    ret = pd.DataFrame({"A": a, "B": b}, index=idx)
    assert first_evaluation_date(ret) == idx[273]  # A gets there first


def test_reads_straight_from_zip_and_picks_the_right_table(tmp_path):
    z = tmp_path / "49_Industry_Portfolios_daily_CSV.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("49_Industry_Portfolios_Daily.csv", INDUSTRY_TEXT)
    ew = load_industry(z, "ew", end="2025-12-31")
    vw = load_industry(z, "vw", end="2025-12-31")
    assert ew.loc["1926-07-01", "A"] == pytest.approx(0.005)
    assert vw.loc["1926-07-01", "A"] == pytest.approx(0.01)


def test_load_pput_parses_mmddyyyy_and_cuts_the_sample(tmp_path):
    p = tmp_path / "PPUT_History.csv"
    p.write_text("DATE,PPUT\n06/30/1986,100.0\n12/31/2025,2000.0\n01/02/2026,2001.0\n")
    s = load_pput(p, end="2025-12-31")
    assert list(s.index) == [pd.Timestamp("1986-06-30"), pd.Timestamp("2025-12-31")]
    assert s.iloc[0] == 100.0
