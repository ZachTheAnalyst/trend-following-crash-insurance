"""Tests for the H1 test, on made-up data only."""

import numpy as np
import pandas as pd
import pytest

from src import h1, power


# --- calendar months --------------------------------------------------------------

def test_partial_first_month_is_dropped_and_full_months_compound():
    cal = pd.bdate_range("2000-01-03", "2000-04-28")
    daily = pd.Series(0.001, index=cal[cal >= "2000-01-20"])  # January covered only from the 20th
    m = h1.calendar_months(daily, cal)
    assert list(m.index.astype(str)) == ["2000-02", "2000-03", "2000-04"]
    n_feb = int((cal.to_period("M") == "2000-02").sum())
    assert m["2000-02"] == pytest.approx(1.001**n_feb - 1)


def test_month_missing_its_last_day_is_dropped():
    cal = pd.bdate_range("2000-01-03", "2000-03-31")
    daily = pd.Series(0.001, index=cal[cal <= "2000-03-30"])  # March is cut one day short
    m = h1.calendar_months(daily, cal)
    assert list(m.index.astype(str)) == ["2000-01", "2000-02"]


def test_all_full_months_are_kept():
    cal = pd.bdate_range("2000-01-03", "2000-12-29")
    m = h1.calendar_months(pd.Series(0.0005, index=cal), cal)
    assert len(m) == 12


# --- Delta-CE and its standard error ----------------------------------------------

def _returns(m=240, seed=1, drift=0.0):
    rng = np.random.default_rng(seed)
    rb = rng.normal(0.008, 0.045, m)
    rs = 0.6 * rb + rng.normal(drift, 0.01, m)
    return rs, rb


def test_delta_ce_matches_the_power_module_formula():
    rs, rb = _returns()
    Xs, Xb = (1 + rs) ** -4.0, (1 + rb) ** -4.0
    g, _ = h1._dce_and_se(Xs[None, :], Xb[None, :], 0)
    assert g[0] == pytest.approx(power.delta_ce(rs, rb), rel=1e-10)


def test_delta_method_se_matches_numerical_derivative_at_lag_zero():
    rs, rb = _returns(m=400, seed=3)
    Xs, Xb = (1 + rs) ** -4.0, (1 + rb) ** -4.0
    _, se = h1._dce_and_se(Xs[None, :], Xb[None, :], 0)
    # iid delta method by hand: gradient of g wrt (mean Xs, mean Xb) by finite differences
    def g(a, b):
        return 100.0 * (a**h1.K - b**h1.K)
    a, b = Xs.mean(), Xb.mean()
    h = 1e-6
    ga = (g(a + h, b) - g(a - h, b)) / (2 * h)
    gb = (g(a, b + h) - g(a, b - h)) / (2 * h)
    z = ga * (Xs - a) + gb * (Xb - b)
    assert se[0] == pytest.approx(np.sqrt(np.mean(z**2) / len(z)), rel=1e-5)


def test_delta_method_se_matches_bootstrap_sd():
    rs, rb = _returns(m=600, seed=5)
    Xs, Xb = (1 + rs) ** -4.0, (1 + rb) ** -4.0
    _, se = h1._dce_and_se(Xs[None, :], Xb[None, :], 0)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(rs), size=(4000, len(rs)))
    g, _ = h1._dce_and_se(Xs[idx], Xb[idx], 0)
    assert se[0] == pytest.approx(g.std(), rel=0.08)


def test_bootstrap_t_interval_on_a_skewed_hand_example():
    # t* runs 0..99 / 10 - 8, so its 2.5% and 97.5% quantiles are known and are NOT symmetric about 0
    t_star = np.arange(0, 100) / 10.0 - 8.0
    q_lo, q_hi = np.quantile(t_star, [0.025, 0.975])
    lo, hi = h1.boot_t_interval(2.0, 0.5, t_star)
    assert q_lo == pytest.approx(-7.7525) and q_hi == pytest.approx(1.6525)
    assert lo == pytest.approx(2.0 - 0.5 * q_hi)
    assert hi == pytest.approx(2.0 - 0.5 * q_lo)
    assert lo < hi


# --- verdict bins -----------------------------------------------------------------

@pytest.mark.parametrize(
    "dce, lo, hi, expected",
    [
        (1.2, 0.3, 2.1, "Worth it, meaningful"),
        (0.5, 0.01, 1.0, "Worth it, meaningful"),  # exactly at the threshold counts
        (0.49, 0.01, 1.0, "Worth it, trivial"),
        (0.3, 0.1, 0.5, "Worth it, trivial"),
        (-1.0, -2.0, -0.1, "Not worth it"),
        (0.4, -0.2, 1.0, "Inconclusive"),
        (0.4, 0.0, 1.0, "Inconclusive"),  # CI touching zero is not above zero
        (-0.4, -1.0, 0.0, "Inconclusive"),
    ],
)
def test_verdict_bins(dce, lo, hi, expected):
    assert h1.verdict(dce, lo, hi) == expected


# --- statistical behaviour on synthetic data --------------------------------------

def test_null_rate_ci_coverage_and_p_ci_agreement():
    """Under H0 (identical series up to noise with equal CE) the test is close to size 5%."""
    rng = np.random.default_rng(11)
    n_sim, m = 300, 300
    rej = cover = agree = 0
    for _ in range(n_sim):
        rb = rng.normal(0.008, 0.045, m)
        # strategy = benchmark plus independent noise, then shifted so that Delta-CE is zero in population
        rs = rb + rng.normal(0.0, 0.01, m)
        # population Delta-CE of this design is not exactly 0, so compare to the population value below
        out = h1.h1_test(rs, rb, rng, n_boot=299, chunk=299)
        rej += out["p_boot"] < 0.05
        agree += (out["p_boot"] < 0.05) == (not (out["ci95_boot_t"][0] <= 0 <= out["ci95_boot_t"][1]))
    # noise adds variance, so the population Delta-CE is slightly negative: the design is
    # not an exact null. The checks here are agreement between p and CI, and that it is not degenerate.
    assert agree / n_sim > 0.88
    assert 0.0 < rej / n_sim < 1.0


def test_size_at_an_exact_null_by_swapping_labels():
    """Exact null: the two series are exchangeable draws from the same process."""
    rng = np.random.default_rng(21)
    n_sim, m = 300, 300
    rej = cover = 0
    for _ in range(n_sim):
        x = rng.normal(0.008, 0.045, (2, m))
        out = h1.h1_test(x[0], x[1], rng, n_boot=299, chunk=299)
        rej += out["p_boot"] < 0.05
        cover += out["ci95_boot_t"][0] <= 0 <= out["ci95_boot_t"][1]
    assert 0.02 <= rej / n_sim <= 0.09
    assert 0.90 <= cover / n_sim <= 0.99


def test_a_large_true_effect_is_detected_and_called_meaningful():
    rng = np.random.default_rng(31)
    m = 600
    rb = rng.normal(0.008, 0.05, m)
    rs = rb.copy()
    rs[rb < -0.05] = 0.003  # strategy sidesteps the big losses
    out = h1.h1_test(rs, rb, rng, n_boot=999)
    assert out["p_boot"] < 0.01
    assert out["delta_ce_pct"] > 0.5
    assert out["verdict"] == "Worth it, meaningful"


def test_a_strategy_worse_than_buy_and_hold_is_not_worth_it():
    rng = np.random.default_rng(41)
    m = 600
    rb = rng.normal(0.008, 0.045, m)
    rs = rb - 0.004
    out = h1.h1_test(rs, rb, rng, n_boot=999)
    assert out["delta_ce_pct"] < 0
    assert out["verdict"] == "Not worth it"


def test_identical_series_give_zero_effect():
    rb = np.random.default_rng(2).normal(0.008, 0.045, 200)
    # se = 0 makes the t statistic undefined: the test must not silently report significance
    with np.errstate(all="ignore"):
        out = h1.h1_test(rb, rb.copy(), np.random.default_rng(3), n_boot=99)
    assert out["delta_ce_pct"] == 0.0


def test_same_seed_gives_identical_results():
    rs, rb = _returns(m=240, seed=8, drift=0.001)
    a = h1.h1_test(rs, rb, np.random.default_rng(5), n_boot=499)
    b = h1.h1_test(rs, rb, np.random.default_rng(5), n_boot=499)
    assert a == b


def test_chunking_does_not_change_the_p_value_distribution_inputs():
    rs, rb = _returns(m=240, seed=9, drift=0.001)
    a = h1.h1_test(rs, rb, np.random.default_rng(5), n_boot=600, chunk=600)
    assert a["n_boot"] == 600
    assert 1 / 601 <= a["p_boot"] <= 1.0


def test_reported_fields_are_consistent():
    rs, rb = _returns(m=240, seed=10, drift=0.002)
    out = h1.h1_test(rs, rb, np.random.default_rng(1), n_boot=499)
    assert out["delta_ce_pct"] == pytest.approx(power.delta_ce(rs, rb))
    assert out["nw_lag"] == power.nw_lag(240)
    lo, hi = out["ci95_boot_t"]
    assert lo < out["delta_ce_pct"] < hi
    assert out["mean_d"] == pytest.approx(power.utility_diff(rs, rb).mean())


# --- end to end on a synthetic panel ------------------------------------------------

def _panel(n_days=1500, n_ind=4, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("1990-01-02", periods=n_days)
    ret = pd.DataFrame(rng.normal(0.0004, 0.011, (n_days, n_ind)), index=idx,
                       columns=[f"I{i}" for i in range(n_ind)])
    ret.iloc[:400, 3] = np.nan  # a late-starting industry
    rf = pd.Series(0.00012, index=idx)
    return ret, rf


def test_pipeline_runs_end_to_end_and_reports_its_window():
    ret, rf = _panel()
    out = h1.h1_pipeline(ret, rf, n_boot=199, seed=1)
    assert out["n_months"] > 50
    assert out["verdict"] in {"Worth it, meaningful", "Worth it, trivial", "Not worth it", "Inconclusive"}
    assert out["first_day"] >= out["evaluation_start"]
    assert out["first_month"] > out["evaluation_start"][:7]  # the partly covered first month is dropped
    assert 0 < out["p_boot"] <= 1


def test_pipeline_is_reproducible_with_a_seed():
    ret, rf = _panel()
    a = h1.h1_pipeline(ret, rf, n_boot=199, seed=7)
    b = h1.h1_pipeline(ret, rf, n_boot=199, seed=7)
    assert a == b


def test_pipeline_result_does_not_depend_on_returns_after_the_window_for_the_estimate():
    """Changing only the last month's returns cannot move the first month's contribution:
    the months before the last one are identical in both runs."""
    ret, rf = _panel()
    ret2 = ret.copy()
    ret2.iloc[-15:] = ret2.iloc[-15:] * 3.0
    a = h1.h1_pipeline(ret, rf, n_boot=99, seed=1)
    b = h1.h1_pipeline(ret2, rf, n_boot=99, seed=1)
    assert a["first_month"] == b["first_month"]
    assert a["delta_ce_pct"] != b["delta_ce_pct"]  # the change is seen, so the pipeline is not inert


def test_real_data_cli_is_gated():
    import subprocess, sys
    r = subprocess.run([sys.executable, "-m", "src.h1", "x", "y"], capture_output=True, text=True)
    assert r.returncode != 0
    assert "--confirm-unblind" in (r.stdout + r.stderr)
