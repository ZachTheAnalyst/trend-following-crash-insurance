"""Power analysis for the primary test H1 (pre-registration, "Economic
significance and power").

What the plan says, step by step, and where it lives here:

1. Build the equal-weighted industry buy-and-hold index and resample its daily
   returns, with the matching T-bill rates, by stationary bootstrap:
   `bh_inputs`, `stationary_indices`, `simulate_path`.
2. Run the 50/200 rule on each synthetic path: `strategy_on_path`. The rule is
   NEVER run on the real index, so no real strategy return is computed.
3. Calibrate a constant shift on one path 100 times the sample length so the
   population Delta-CE equals the target: `calibrate_shift`.
4. Run the H1 test on 500 independent paths per Delta with the Newey-West
   version (`nw_t`), and spot-check 50 with the full studentized stationary
   block bootstrap (`studentized_bootstrap_p`): `power_cell`.
5. MDE = smallest Delta rejected in at least 80% of runs at alpha = 0.05: `mde`.
6. Separately for the full sample, pre-1993 and post-1992: `PERIODS`.

Choices where the plan is silent are logged in the README (Deviations,
"Power-analysis clarifications", and Implementation notes):

* The effect size is a fixed additive daily return shift, found by root-finding.
* Pre-1993 and post-1992 paths are resampled only from that period's own
  returns, each at that period's length.
* One calibration per block setting x period x Delta (45), each on one long path.
* Block length: `arch.bootstrap.optimal_block_length`, "stationary" column
  (Patton-Politis-White correction), on the period's daily returns; fixed
  settings 252 and 1,260 days.
* A month is 21 trading days (the plan's definition), so synthetic paths need
  no calendar. A leftover partial month at the end of a path is dropped.
* A synthetic path is 200 days longer than the evaluation window, because the
  50/200 rule needs 200 days before its first signal. The evaluation window has
  exactly the length of the real period.
* Two-sided test at alpha = 0.05 using the normal critical value 1.96, and the
  Newey-West lag floor(4 (M/100)^(2/9)) with M the number of months.
* The stationary bootstrap is implemented here (vectorized) because the
  bootstrap test needs 10,000 draws per run and the long calibration path is
  longer than the data. `tests/test_power.py` checks it against arch's
  StationaryBootstrap.
"""

import numpy as np
import pandas as pd
from arch.bootstrap import optimal_block_length
from scipy.optimize import brentq

from src import data, rules
from src.engine import backtest

GAMMA = 5.0
MONTH = 21  # trading days
COST = 0.0010  # 10 bp one-way
SHORT, LONG = 50, 200
DELTAS = (0.25, 0.5, 1.0, 1.5, 2.0)  # percent per year
N_PATHS = 500
N_SPOT = 50
N_BOOT = 10_000
ALPHA = 0.05
Z_CRIT = 1.959963984540054
LONG_FACTOR = 100
POWER_TARGET = 0.80
SPLIT = pd.Timestamp("1993-01-01")
PERIODS = ("full", "pre1993", "post1992")
BLOCK_SETTINGS = ("pw", 252, 1260)
SEED = 20260929


# --- inputs from the real data: buy-and-hold only ---------------------------------

def bh_inputs(raw_dir):
    """Equal-weighted industry buy-and-hold daily return and T-bill rate.

    Equal weight across the industries active that day (273-day rule), value-
    weighted within each industry, rebalanced daily. Starts on the evaluation
    start date. Only buy-and-hold returns are built here: no rule is run.
    Returns a DataFrame with columns bh and rf, indexed by date.
    """
    from pathlib import Path

    raw = Path(raw_dir)
    ind = data.load_industry(raw / "49_Industry_Portfolios_daily_CSV.zip", "vw")
    ff3 = data.load_ff3(raw / "F-F_Research_Data_Factors_daily_CSV.zip")
    active = data.active_mask(ind)
    bh = ind.where(active).mean(axis=1, skipna=True)
    start = rules.evaluation_start(ind.index, data.first_evaluation_date(ind))
    out = pd.DataFrame({"bh": bh, "rf": ff3["RF"]}).loc[start:]
    if out.isna().any().any():
        raise ValueError("missing values in buy-and-hold or T-bill series")
    return out


def period_slice(df, period):
    """Rows of `df` in "full", "pre1993" or "post1992"."""
    if period == "full":
        return df
    if period == "pre1993":
        return df.loc[: SPLIT - pd.Timedelta(days=1)]
    if period == "post1992":
        return df.loc[SPLIT:]
    raise ValueError(period)


def pw_block_length(r):
    """Politis-White block length for the stationary bootstrap (Patton correction)."""
    b = float(optimal_block_length(np.asarray(r, dtype=float))["stationary"].iloc[0])
    return max(b, 1.0)


# --- stationary bootstrap -----------------------------------------------------------

def stationary_indices(n, length, mean_block, rng):
    """Indices of one stationary-bootstrap sample (Politis-Romano 1994).

    Each step continues the current block (next index, wrapping around) with
    probability 1 - 1/mean_block and otherwise jumps to a uniform random index.
    """
    restart = rng.random(length) < 1.0 / mean_block
    restart[0] = True
    pos = np.arange(length)
    last = np.maximum.accumulate(np.where(restart, pos, 0))
    start = rng.integers(0, n, size=length)
    return (start[last] + pos - last) % n


def stationary_indices_2d(n, length, mean_block, rng, draws):
    """`draws` stationary-bootstrap samples at once: an array (draws, length)."""
    restart = rng.random((draws, length)) < 1.0 / mean_block
    restart[:, 0] = True
    pos = np.arange(length)
    last = np.maximum.accumulate(np.where(restart, pos, 0), axis=1)
    start = rng.integers(0, n, size=(draws, length))
    base = np.take_along_axis(start, last, axis=1)
    return (base + pos - last) % n


def simulate_path(r, rf, length, mean_block, rng):
    """One synthetic path: returns and T-bill rates resampled with the same indices."""
    idx = stationary_indices(len(r), length, mean_block, rng)
    return r[idx], rf[idx]


# --- the 50/200 rule on a synthetic path ---------------------------------------------

def signal_50_200(r, seg=60_000, overlap=250):
    """Family 1 flagship signal (50/200 SMA, no band) from daily returns.

    The rule only compares averages of the price index, so it does not depend
    on the index's scale. A path 100 times the sample would overflow a float
    index, so long paths are processed in overlapping segments, each rebased to
    1.0. `overlap` is above the 200-day window, so every kept signal sees a
    full window inside its segment.
    """
    n = len(r)
    if n <= seg:
        level = data.total_return_index(pd.Series(r))
        return rules.family1_sma(level, SHORT, LONG, 0.0)
    out = np.full(n, np.nan)
    start = 0
    while start < n:
        stop = min(start + seg, n)
        level = data.total_return_index(pd.Series(r[start:stop]))
        sig = rules.family1_sma(level, SHORT, LONG, 0.0).to_numpy()
        keep = 0 if start == 0 else overlap
        out[start + keep : stop] = sig[keep:]
        if stop == n:
            break
        start = stop - overlap
    return pd.Series(out)


def _run_engine(r, rf, cost):
    """Engine result on the evaluation window of a path, and the first-signal day k."""
    sig = signal_50_200(r)
    k = int(sig.first_valid_index())  # 199: first day the 200-day average exists
    s = sig.iloc[k:]
    rr = pd.Series(r[k:], index=s.index)
    ff = pd.Series(rf[k:], index=s.index)
    return backtest(rr, ff, s, cost, initial_weight=float(s.iloc[0])), k


def strategy_on_path(r, rf, cost=COST):
    """Net daily strategy return, buy-and-hold return and T-bill rate on the
    evaluation window of a path (everything after the first signal day).
    """
    res, k = _run_engine(r, rf, cost)
    # Day k only carries the initial position; the evaluation starts the day after.
    return res.net_ret.to_numpy()[1:], r[k + 1 :], rf[k + 1 :]


def weights_on_path(r, rf, cost=COST):
    """Stock weight held on each evaluation day of a path (for the look-ahead test)."""
    res, _ = _run_engine(r, rf, cost)
    return res.weight.to_numpy()[1:]


# --- certainty equivalent, utility difference, tests ---------------------------------

def monthly(x):
    """Compound daily returns into 21-day months. A partial last month is dropped."""
    m = len(x) // MONTH
    return np.prod(1.0 + np.asarray(x)[: m * MONTH].reshape(m, MONTH), axis=1) - 1.0


def ce(R, gamma=GAMMA):
    """CRRA certainty equivalent of monthly returns R."""
    return np.mean((1.0 + R) ** (1.0 - gamma)) ** (1.0 / (1.0 - gamma)) - 1.0


def delta_ce(R_s, R_b, gamma=GAMMA):
    """Annualized strategy-minus-buy-and-hold CE, in percent per year."""
    return 100.0 * ((1.0 + ce(R_s, gamma)) ** 12 - (1.0 + ce(R_b, gamma)) ** 12)


def utility_diff(R_s, R_b, gamma=GAMMA):
    """Monthly utility difference d_m; H0: E[d_m] = 0 (the plan's H1)."""
    return ((1.0 + R_s) ** (1.0 - gamma) - (1.0 + R_b) ** (1.0 - gamma)) / (1.0 - gamma)


def nw_lag(m):
    """Newey-West lag floor(4 (M/100)^(2/9))."""
    return int(np.floor(4.0 * (m / 100.0) ** (2.0 / 9.0)))


def _nw_var_rows(X, lag):
    """Newey-West (Bartlett) variance of the mean, for each row of demeaned X."""
    m = X.shape[1]
    v = np.einsum("ij,ij->i", X, X) / m
    for j in range(1, lag + 1):
        g = np.einsum("ij,ij->i", X[:, j:], X[:, :-j]) / m
        v = v + 2.0 * (1.0 - j / (lag + 1.0)) * g
    return v / m


def nw_t(d, lag=None):
    """Mean of d, its Newey-West standard error, and the t statistic."""
    d = np.asarray(d, dtype=float)
    lag = nw_lag(len(d)) if lag is None else lag
    mean = d.mean()
    se = float(np.sqrt(_nw_var_rows((d - mean)[None, :], lag)[0]))
    return mean, se, mean / se


def studentized_bootstrap_p(d, rng, n_boot=N_BOOT, chunk=1000):
    """Two-sided p-value for E[d] = 0: studentized stationary block bootstrap.

    Block length by the Politis-White rule on d. Each draw's t statistic is
    (mean* - mean) / se*, with se* the Newey-West error of the draw.
    """
    d = np.asarray(d, dtype=float)
    m = len(d)
    lag = nw_lag(m)
    mean, se, t_obs = nw_t(d, lag)
    b = pw_block_length(d)
    count = 0
    done = 0
    while done < n_boot:
        k = min(chunk, n_boot - done)
        idx = stationary_indices_2d(m, m, b, rng, k)
        ds = d[idx]
        mstar = ds.mean(axis=1)
        sestar = np.sqrt(_nw_var_rows(ds - mstar[:, None], lag))
        tstar = (mstar - mean) / sestar
        count += int(np.sum(np.abs(tstar) >= abs(t_obs)))
        done += k
    return (1.0 + count) / (n_boot + 1.0)


# --- calibration and power ------------------------------------------------------------

def calibrate_shift(net_s, r_bh, target_pct):
    """Daily additive shift so the population Delta-CE equals `target_pct`.

    `net_s` and `r_bh` are daily strategy and buy-and-hold returns on one long
    simulated path. CE is not linear, so the shift is found by root-finding.
    """
    R_b = monthly(r_bh)

    def f(shift):
        return delta_ce(monthly(net_s + shift), R_b) - target_pct

    lo, hi = -5e-4, 5e-4
    while f(lo) > 0:
        lo *= 2
    while f(hi) < 0:
        hi *= 2
    return brentq(f, lo, hi, xtol=1e-13, rtol=1e-12)


def simulate_strategy(r, rf, length, mean_block, rng):
    """Synthetic path -> (net strategy, buy-and-hold) daily returns on its evaluation window."""
    rs, fs = simulate_path(r, rf, length + LONG, mean_block, rng)
    net, bh, _ = strategy_on_path(rs, fs)
    return net, bh


def calibrate_cell(r, rf, mean_block, targets, rng, factor=LONG_FACTOR):
    """Calibrate the shift for each target Delta on ONE path `factor` times as long
    as the period.

    Returns ({target: shift}, baseline Delta-CE at shift 0, {target: calibration
    standard error in percent per year}). The path is finite, so the population
    Delta-CE the shift achieves is only known up to sampling error; the standard
    error is the Newey-West error of mean d_m on the long path divided by the
    slope of mean d_m in Delta.
    """
    net, bh = simulate_strategy(r, rf, factor * len(r), mean_block, rng)
    shifts = {t: calibrate_shift(net, bh, t) for t in targets}
    R_b = monthly(bh)
    d = {t: utility_diff(monthly(net + shifts[t]), R_b) for t in targets}
    baseline = delta_ce(monthly(net), R_b)
    cal_se = {}
    if len(targets) >= 2:
        lo, hi = min(targets), max(targets)
        slope = (d[hi].mean() - d[lo].mean()) / (hi - lo)
        cal_se = {t: nw_t(d[t])[1] / slope for t in targets}
    return shifts, baseline, cal_se


def path_test(net, bh, shift):
    """H1 on one simulated path: (estimated Delta-CE in %, NW t, monthly d_m)."""
    R_s, R_b = monthly(net + shift), monthly(bh)
    d = utility_diff(R_s, R_b)
    _, _, t = nw_t(d)
    return delta_ce(R_s, R_b), t, d


def check_calibration(r, rf, mean_block, shifts, rng, factor=LONG_FACTOR):
    """Delta-CE the calibrated shifts actually give on an INDEPENDENT long path.

    On the calibration path itself the shift hits its target by construction, so
    an independent path (fresh draws, same length) is what shows the calibration
    error in numbers. Returns {target: achieved Delta-CE in percent per year}.
    """
    net, bh = simulate_strategy(r, rf, factor * len(r), mean_block, rng)
    R_b = monthly(bh)
    return {t: delta_ce(monthly(net + shift), R_b) for t, shift in shifts.items()}


def power_cell(r, rf, mean_block, shift, seed_seq, n_paths=N_PATHS, n_spot=N_SPOT, n_boot=N_BOOT):
    """Rejection rates for one (period, block setting, Delta) cell.

    r, rf   the period's real buy-and-hold returns and T-bill rates (arrays)
    shift   the calibrated daily shift for this cell
    Returns the share of paths rejected by the Newey-West test (both tails, and
    each tail alone), the mean and SD of the t statistic, the share of the first
    `n_spot` paths rejected by the full bootstrap test, and the mean and SD of the
    estimated Delta-CE across paths.
    """
    rng = np.random.default_rng(seed_seq)
    rej_boot = 0
    est, ts = [], []
    for i in range(n_paths):
        net, bh = simulate_strategy(r, rf, len(r), mean_block, rng)
        dce, t, d = path_test(net, bh, shift)
        est.append(dce)
        ts.append(t)
        if i < n_spot:
            rej_boot += studentized_bootstrap_p(d, rng, n_boot) < ALPHA
    ts = np.asarray(ts)
    return {
        "rej_nw": float(np.mean(np.abs(ts) > Z_CRIT)),
        "rej_pos": float(np.mean(ts > Z_CRIT)),
        "rej_neg": float(np.mean(ts < -Z_CRIT)),
        "mean_t": float(ts.mean()),
        "sd_t": float(ts.std(ddof=1)),
        "rej_boot": rej_boot / n_spot if n_spot else float("nan"),
        "mean_est_dce": float(np.mean(est)),
        "sd_est_dce": float(np.std(est, ddof=1)),
    }


def mde(rates_by_delta, level=POWER_TARGET):
    """Smallest Delta whose rejection rate is at least `level`, or None."""
    for delta in sorted(rates_by_delta):
        if rates_by_delta[delta] >= level:
            return delta
    return None
