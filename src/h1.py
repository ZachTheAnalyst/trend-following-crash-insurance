"""Primary test H1 (pre-registration, "Tests and inference").

H1: the CRRA certainty equivalent (gamma = 5) of the 50/200 strategy equals
buy-and-hold's. Equal-weighted industry portfolio, full sample, 10 bp cost.

    Delta-CE = (1 + CE_S)^12 - (1 + CE_BH)^12, in percent per year, on monthly
    net returns compounded from daily returns.
    H1 is a test on the mean of the monthly utility differences d_m.

Test: studentized stationary block bootstrap, 10,000 draws, block length from the
Politis-White rule (Patton-Politis-White correction). The same draws give the 95%
CI for Delta-CE. A Newey-West t-test on d_m is reported as a check.

Implementation choices where the plan is silent (documented in the README, and
fixed there before this code is run on real data):

* Months are calendar months. A month that the evaluation window covers only
  partly (in practice the first one) is dropped from H1.
* The bootstrap resamples months in pairs (strategy month, buy-and-hold month),
  so each draw keeps both series matched.
* p-value: two-sided, (1 + number of draws with |t*| >= |t|) / (draws + 1), where
  t = mean(d) / se and t* is the same statistic on the draw, centered at mean(d).
* CI: bootstrap-t for Delta-CE. Each draw's Delta-CE has a delta-method Newey-West
  standard error, t* = (Delta-CE* - Delta-CE) / se*, and the CI is
  [D - q(0.975) se, D - q(0.025) se]. The percentile CI is reported as a check.
  The verdict bin uses the bootstrap-t CI.
* Newey-West lag floor(4 (M/100)^(2/9)) with M months, Bartlett kernel; the
  check p-value uses the normal distribution.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from src import data, portfolio, power, rules

GAMMA = 5.0
K = 12.0 / (1.0 - GAMMA)  # Delta-CE = 100 (m_S^K - m_BH^K), m = mean (1 + R)^(1-gamma)
N_BOOT = 10_000
MEANINGFUL = 0.5  # percent per year, the plan's threshold


def calendar_months(daily, calendar_index):
    """Compound daily returns into calendar months; drop months only partly covered.

    A month is kept when the series has both the first and the last trading day
    of that month in `calendar_index` (the full trading calendar).
    """
    per = daily.index.to_period("M")
    monthly = (1.0 + daily).groupby(per).prod() - 1.0
    cal_days = pd.Series(calendar_index, index=calendar_index)
    cal_per = calendar_index.to_period("M")
    first, last = cal_days.groupby(cal_per).min(), cal_days.groupby(cal_per).max()
    have = daily.index.to_series()
    ok = (have.groupby(per).min() == first.reindex(monthly.index)) & (
        have.groupby(per).max() == last.reindex(monthly.index)
    )
    return monthly[ok]


def _dce_and_se(Xs, Xb, lag):
    """Delta-CE (percent per year) and its delta-method Newey-West standard error.

    Xs, Xb: arrays (rows, months) of (1 + R)^(1 - gamma) for strategy and benchmark.
    """
    ms, mb = Xs.mean(axis=1), Xb.mean(axis=1)
    g = 100.0 * (ms**K - mb**K)
    dgs = 100.0 * K * ms ** (K - 1.0)
    dgb = -100.0 * K * mb ** (K - 1.0)
    z = dgs[:, None] * (Xs - ms[:, None]) + dgb[:, None] * (Xb - mb[:, None])
    return g, np.sqrt(power._nw_var_rows(z, lag))


def verdict(dce, ci_lo, ci_hi):
    """The plan's four verdict bins, from Delta-CE and its 95% CI."""
    if ci_lo > 0:
        return "Worth it, meaningful" if dce >= MEANINGFUL else "Worth it, trivial"
    if ci_hi < 0:
        return "Not worth it"
    return "Inconclusive"


def boot_t_interval(dce, se, t_star, level=0.95):
    """Bootstrap-t interval: [D - q(1 - a/2) se, D - q(a/2) se] from the draws' t* values."""
    a = 1.0 - level
    q_lo, q_hi = np.quantile(t_star, [a / 2.0, 1.0 - a / 2.0])
    return dce - q_hi * se, dce - q_lo * se


def h1_test(R_s, R_b, rng, n_boot=N_BOOT, chunk=1000):
    """H1 on monthly strategy and benchmark returns (arrays of equal length)."""
    R_s, R_b = np.asarray(R_s, dtype=float), np.asarray(R_b, dtype=float)
    m = len(R_s)
    lag = power.nw_lag(m)
    Xs, Xb = (1.0 + R_s) ** (1.0 - GAMMA), (1.0 + R_b) ** (1.0 - GAMMA)
    d = (Xs - Xb) / (1.0 - GAMMA)
    mean_d, se_d, t_nw = power.nw_t(d, lag)
    dce, se_dce = (v[0] for v in _dce_and_se(Xs[None, :], Xb[None, :], lag))
    block = power.pw_block_length(d)

    count = 0
    g_all, t_all = [], []
    done = 0
    while done < n_boot:
        k = min(chunk, n_boot - done)
        idx = power.stationary_indices_2d(m, m, block, rng, k)
        xs, xb = Xs[idx], Xb[idx]
        ds = (xs - xb) / (1.0 - GAMMA)
        mstar = ds.mean(axis=1)
        sestar = np.sqrt(power._nw_var_rows(ds - mstar[:, None], lag))
        count += int(np.sum(np.abs((mstar - mean_d) / sestar) >= abs(t_nw)))
        g, se = _dce_and_se(xs, xb, lag)
        g_all.append(g)
        t_all.append((g - dce) / se)
        done += k
    g_all, t_all = np.concatenate(g_all), np.concatenate(t_all)
    ci = boot_t_interval(dce, se_dce, t_all)
    pct = tuple(np.quantile(g_all, [0.025, 0.975]))
    return {
        "n_months": m,
        "delta_ce_pct": float(dce),
        "delta_ce_se_pct": float(se_dce),
        "ce_strategy_monthly": float((Xs.mean()) ** (1.0 / (1.0 - GAMMA)) - 1.0),
        "ce_buy_hold_monthly": float((Xb.mean()) ** (1.0 / (1.0 - GAMMA)) - 1.0),
        "mean_d": float(mean_d),
        "nw_lag": lag,
        "nw_t": float(t_nw),
        "nw_p": float(2.0 * norm.sf(abs(t_nw))),
        "block_length_months": float(block),
        "n_boot": n_boot,
        "p_boot": (1.0 + count) / (n_boot + 1.0),
        "ci95_boot_t": [float(ci[0]), float(ci[1])],
        "ci95_percentile": [float(pct[0]), float(pct[1])],
        "verdict": verdict(dce, ci[0], ci[1]),
    }


def h1_pipeline(ret, rf, n_boot=N_BOOT, seed=None, first_273=None):
    """Industry returns and T-bill rate -> H1. The full chain the plan describes.

    ret  daily industry returns (date x industry), NaN where missing
    rf   daily T-bill rate on the same dates
    """
    active = data.active_mask(ret)
    first_273 = data.first_evaluation_date(ret) if first_273 is None else first_273
    start = rules.evaluation_start(ret.index, first_273)
    port = portfolio.strategy_portfolio(ret, rf, portfolio.signals_50_200(ret), active)
    port = portfolio.evaluation_frame(port, start)
    R_s = calendar_months(port["net"], ret.index)
    R_b = calendar_months(port["bh"], ret.index)
    assert R_s.index.equals(R_b.index)
    rng = np.random.default_rng(np.random.SeedSequence(power.SEED if seed is None else seed))
    out = h1_test(R_s.to_numpy(), R_b.to_numpy(), rng, n_boot)
    out.update(
        evaluation_start=str(start.date()),
        first_month=str(R_s.index[0]),
        last_month=str(R_s.index[-1]),
        first_day=str(port.index[0].date()),
        last_day=str(port.index[-1].date()),
    )
    return out


def run(raw_dir, out_dir, n_boot=N_BOOT):
    """Run H1 on the real data. Called only after the plan and these choices are locked."""
    raw = Path(raw_dir)
    ret = data.load_industry(raw / "49_Industry_Portfolios_daily_CSV.zip", "vw")
    ff3 = data.load_ff3(raw / "F-F_Research_Data_Factors_daily_CSV.zip")
    rf = ff3["RF"].reindex(ret.index)
    if rf.isna().any():
        raise ValueError("T-bill rate missing on some industry dates")
    out = h1_pipeline(ret, rf, n_boot=n_boot)
    out["seed"] = power.SEED
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "h1_primary.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[3] != "--confirm-unblind":
        sys.exit(
            "usage: python -m src.h1 <raw data folder> <output folder> --confirm-unblind\n"
            "This computes the first real strategy result. Run it only after the plan and the\n"
            "H1 implementation choices in the README are pushed and you have decided to look."
        )
    print(json.dumps(run(sys.argv[1], sys.argv[2]), indent=2))
