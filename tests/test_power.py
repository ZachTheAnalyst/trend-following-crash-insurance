"""Unit tests for the power-analysis building blocks. Synthetic data only."""

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from arch.bootstrap import StationaryBootstrap

from src import power
from src.engine import backtest


# --- monthly, CE, utility difference ---------------------------------------------------

def test_monthly_compounds_21_day_blocks_and_drops_the_partial_month():
    x = np.full(50, 0.01)  # 2 full months + 8 leftover days
    m = power.monthly(x)
    assert len(m) == 2
    assert m == pytest.approx(np.full(2, 1.01**21 - 1))


def test_ce_of_a_constant_return_is_that_return():
    assert power.ce(np.full(10, 0.02)) == pytest.approx(0.02)


def test_ce_hand_value_and_jensen():
    R = np.array([0.10, -0.10])
    expected = (0.5 * (1.1**-4 + 0.9**-4)) ** (-1 / 4) - 1  # gamma = 5
    assert power.ce(R) == pytest.approx(expected)
    assert power.ce(R) < R.mean()  # a risk-averse investor accepts less than the mean
    assert power.ce(R) < 0


def test_delta_ce_is_zero_for_identical_series_and_annualized_in_percent():
    R = np.array([0.01, -0.02, 0.03])
    assert power.delta_ce(R, R) == pytest.approx(0.0)
    hi, lo = np.full(6, 0.01), np.full(6, 0.005)
    assert power.delta_ce(hi, lo) == pytest.approx(100 * (1.01**12 - 1.005**12))


def test_utility_difference_is_positive_exactly_when_the_strategy_month_is_better():
    d = power.utility_diff(np.array([0.05, -0.05]), np.array([0.01, 0.01]))
    assert d[0] > 0 and d[1] < 0
    assert power.utility_diff(np.array([0.02]), np.array([0.02]))[0] == 0.0
    # mean(d) > 0 <=> Delta-CE > 0
    rng = np.random.default_rng(0)
    Rs, Rb = rng.normal(0.01, 0.03, 200), rng.normal(0.008, 0.05, 200)
    assert (power.utility_diff(Rs, Rb).mean() > 0) == (power.delta_ce(Rs, Rb) > 0)


# --- Newey-West ---------------------------------------------------------------------------

@pytest.mark.parametrize("m,lag", [(100, 4), (396, 5), (1200, 6)])
def test_newey_west_lag_formula(m, lag):
    assert power.nw_lag(m) == lag  # floor(4 (M/100)^(2/9))


def test_nw_t_matches_statsmodels_hac():
    rng = np.random.default_rng(3)
    e = rng.normal(size=600)
    d = 0.05 + 0.5 * np.r_[0.0, e[:-1]] + e  # autocorrelated
    for lag in (0, 3, 6):
        mean, se, t = power.nw_t(d, lag)
        fit = sm.OLS(d, np.ones(len(d))).fit(cov_type="HAC", cov_kwds={"maxlags": lag})
        assert mean == pytest.approx(fit.params[0])
        assert se == pytest.approx(fit.bse[0])
        assert t == pytest.approx(fit.tvalues[0])


# --- stationary bootstrap ---------------------------------------------------------------------

def test_indices_are_in_range_and_reproducible():
    a = power.stationary_indices(100, 5000, 20.0, np.random.default_rng(1))
    b = power.stationary_indices(100, 5000, 20.0, np.random.default_rng(1))
    assert a.min() >= 0 and a.max() < 100
    assert (a == b).all()


@pytest.mark.parametrize("b", [5.0, 20.0, 100.0])
def test_continuation_rate_matches_the_block_length_and_arch(b):
    n = 1000
    mine = power.stationary_indices(n, 200_000, b, np.random.default_rng(2))
    cont_mine = np.mean(mine[1:] == (mine[:-1] + 1) % n)
    assert cont_mine == pytest.approx(1 - 1 / b, abs=0.004)
    arch_idx = np.asarray(StationaryBootstrap(b, np.arange(n), seed=2).update_indices())
    # arch draws only n = 1000 indices, so its rate is noisier
    cont_arch = np.mean(arch_idx[1:] == (arch_idx[:-1] + 1) % n)
    assert cont_arch == pytest.approx(1 - 1 / b, abs=0.03)
    two_d = power.stationary_indices_2d(n, 2000, b, np.random.default_rng(4), 100)
    assert two_d.shape == (100, 2000)
    assert np.mean(two_d[:, 1:] == (two_d[:, :-1] + 1) % n) == pytest.approx(1 - 1 / b, abs=0.004)


def test_simulate_path_keeps_each_day_s_return_and_tbill_rate_together():
    r = np.arange(500.0)
    rf = 10.0 * r
    rs, fs = power.simulate_path(r, rf, 3000, 30.0, np.random.default_rng(5))
    assert len(rs) == 3000
    assert (fs == 10.0 * rs).all()


# --- the 50/200 rule on a path ---------------------------------------------------------------------

def test_segmented_signal_equals_the_whole_path_signal():
    rng = np.random.default_rng(6)
    r = rng.normal(0.0005, 0.012, 6000)
    whole = power.signal_50_200(r, seg=10**9)
    parts = power.signal_50_200(r, seg=1200, overlap=250)
    pd.testing.assert_series_equal(whole, parts)


def test_signal_survives_a_path_whose_price_index_would_overflow():
    rng = np.random.default_rng(7)
    r = rng.normal(0.005, 0.01, 300_000)  # +0.5% a day: e^1500, far beyond float range
    sig = power.signal_50_200(r)
    assert np.log1p(r).sum() > 709  # the index would exceed the largest float
    assert sig.isna().sum() == 199
    assert set(sig.dropna().unique()) <= {0.0, 1.0}


def test_strategy_on_a_steadily_rising_path_equals_buy_and_hold_after_warmup():
    n = 700
    r = np.full(n, 0.002)
    rf = np.full(n, 0.0001)
    net, bh, rf_out = power.strategy_on_path(r, rf, cost=0.01)
    assert len(net) == n - 200  # the evaluation window starts after 200 days
    assert (net == bh).all()  # always long, no entry trade
    assert (rf_out == 0.0001).all()


def test_strategy_matches_engine_on_hand_built_signal_timing():
    # Falling then rising path: long -> cash -> long. Check the day of each switch
    # by recomputing with the engine directly.
    rng = np.random.default_rng(8)
    r = rng.normal(0.0, 0.01, 900)
    rf = np.full(900, 0.0001)
    net, bh, _ = power.strategy_on_path(r, rf, cost=0.001)
    sig = power.signal_50_200(r)
    k = int(sig.first_valid_index())
    ref = backtest(
        pd.Series(r[k:]), pd.Series(rf[k:]), sig.iloc[k:].reset_index(drop=True), 0.001,
        initial_weight=float(sig.iloc[k]),
    ).net_ret.to_numpy()[1:]
    assert net == pytest.approx(ref, abs=0, rel=0)


# --- calibration -----------------------------------------------------------------------------------

def test_calibrated_shift_hits_the_target_delta_ce():
    rng = np.random.default_rng(9)
    r = rng.normal(0.0004, 0.01, 20_000)
    rf = np.full(20_000, 0.0001)
    net, bh, _ = power.strategy_on_path(r, rf)
    prev = -np.inf
    for target in (0.25, 0.5, 1.0, 2.0):
        shift = power.calibrate_shift(net, bh, target)
        got = power.delta_ce(power.monthly(net + shift), power.monthly(bh))
        assert got == pytest.approx(target, abs=1e-6)
        assert shift > prev  # a bigger target needs a bigger shift
        prev = shift


def test_calibrate_cell_uses_a_path_100_times_the_period_and_returns_one_shift_per_target():
    rng = np.random.default_rng(10)
    r = rng.normal(0.0004, 0.01, 1000)
    rf = np.full(1000, 0.0001)
    shifts, baseline, cal_se = power.calibrate_cell(r, rf, 20.0, (0.5, 1.0), np.random.default_rng(11), factor=5)
    assert set(shifts) == {0.5, 1.0}
    assert shifts[1.0] > shifts[0.5]
    assert np.isfinite(baseline)
    assert set(cal_se) == {0.5, 1.0} and all(v > 0 for v in cal_se.values())


# --- the bootstrap test and the whole power machinery ---------------------------------------------------

def test_studentized_bootstrap_has_about_the_right_size_and_detects_a_real_mean():
    rng = np.random.default_rng(12)
    null_rej = sum(
        power.studentized_bootstrap_p(rng.normal(0, 1, 200), rng, n_boot=299) < 0.05
        for _ in range(200)
    )
    assert 0.01 <= null_rej / 200 <= 0.11  # nominal 5%
    p_alt = power.studentized_bootstrap_p(rng.normal(0.4, 1, 200), rng, n_boot=499)
    assert p_alt < 0.01


def test_mde_is_the_smallest_delta_with_at_least_80_percent_power():
    assert power.mde({0.25: 0.1, 0.5: 0.79, 1.0: 0.8, 1.5: 0.95}) == 1.0
    assert power.mde({0.25: 0.1, 0.5: 0.2}) is None


def test_end_to_end_size_near_five_percent_at_zero_effect_and_power_rises_with_delta():
    """Iid returns, so the 50/200 rule has no edge. Calibrate the shift for
    Delta = 0, then for a large Delta, and run the cell."""
    rng = np.random.default_rng(13)
    n = 6000
    r = rng.normal(0.0004, 0.01, n)
    rf = np.full(n, 0.0001)
    shifts, baseline, cal_se = power.calibrate_cell(r, rf, 10.0, (0.0, 8.0), np.random.default_rng(14), factor=100)
    ss = np.random.SeedSequence(15)
    lo, hi = ss.spawn(2)
    at0 = power.power_cell(r, rf, 10.0, shifts[0.0], lo, n_paths=200, n_spot=10, n_boot=299)
    at8 = power.power_cell(r, rf, 10.0, shifts[8.0], hi, n_paths=200, n_spot=10, n_boot=299)
    assert at0["rej_nw"] <= 0.15  # nominal 5%, Newey-West is a little liberal on short samples
    assert at8["rej_nw"] >= 0.80
    assert at8["rej_nw"] > at0["rej_nw"] + 0.5
    assert at8["mean_est_dce"] == pytest.approx(8.0, abs=0.6)
    assert at0["mean_est_dce"] == pytest.approx(0.0, abs=0.6)
    assert 0.0 < cal_se[8.0] < 0.5  # the calibration error is small on a 100x path


def test_position_held_on_day_t_ignores_returns_from_day_t_onward():
    """The plan's look-ahead test: scramble returns from day t onward (day t
    included); the position held on day t and before must not change."""
    rng = np.random.default_rng(16)
    n = 1500
    r = rng.normal(0.0003, 0.012, n)
    rf = np.full(n, 0.0001)
    base = power.weights_on_path(r, rf)
    changed = 0
    for t0 in range(400, 1400, 37):
        r2 = r.copy()
        r2[t0:] = rng.normal(0.0, 0.05, n - t0)
        r2[t0] = 0.5 if t0 % 2 else -0.5  # a big shock on day t0 itself
        alt = power.weights_on_path(r2, rf)
        j = t0 - 200  # evaluation index of path day t0 (the first evaluated day is path day 200)
        assert (alt[: j + 1] == base[: j + 1]).all()
        changed += (alt != base).any()
    assert changed > 0  # the scrambling does change positions after day t


# --- the run script ------------------------------------------------------------------------------

def _tiny_inputs():
    rng = np.random.default_rng(17)
    idx = pd.bdate_range("1980-01-01", "2004-12-31")
    return pd.DataFrame({"bh": rng.normal(0.0004, 0.01, len(idx)), "rf": 0.0001}, index=idx)


def test_run_writes_one_row_per_cell_the_mde_table_and_resumes(tmp_path, monkeypatch):
    from src import power_run

    inputs = _tiny_inputs()
    cells, table = power_run.run(
        inputs, tmp_path, n_paths=4, n_spot=2, n_boot=99, factor=3, workers=1, log=lambda *_: None
    )
    assert len(cells) == 3 * 3 * 5  # periods x block settings x Deltas
    assert set(cells["period"]) == {"full", "pre1993", "post1992"}
    assert set(cells["block_setting"].astype(str)) == {"pw", "252", "1260"}
    assert (tmp_path / "power_meta.json").exists() and (tmp_path / "power_mde.csv").exists()
    assert len(table) == 9
    # pre + post evaluation days add up to the full period
    n = cells.groupby("period")["n_eval_days"].first()
    assert n["pre1993"] + n["post1992"] == n["full"]

    # a second run finds everything done and computes nothing
    def boom(*a, **k):
        raise AssertionError("resume should not recompute")

    monkeypatch.setattr(power_run, "_task", boom)
    cells2, _ = power_run.run(
        inputs, tmp_path, n_paths=4, n_spot=2, n_boot=99, factor=3, workers=1, log=lambda *_: None
    )
    assert len(cells2) == len(cells)


def test_period_slices_split_at_1993_with_no_overlap_or_gap():
    inputs = _tiny_inputs()
    pre, post = power.period_slice(inputs, "pre1993"), power.period_slice(inputs, "post1992")
    assert pre.index[-1] < pd.Timestamp("1993-01-01") <= post.index[0]
    assert len(pre) + len(post) == len(inputs)
