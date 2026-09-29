# Is Trend-Following Worth Its Cost as Crash Insurance?

**Evidence from a Century of Industry Returns**

Zach Arnold · University of Missouri–St. Louis

## Pre-registration
The full analysis plan was pre-registered on OSF on September 28, 2026, before any real-data results were computed:
**https://osf.io/x3dj9**

The same plan is in this repo: [`Arnold_PreRegistration_2026-09-28.pdf`](Arnold_PreRegistration_2026-09-28.pdf)

## The question
Moving-average trend-following usually earns less than buy-and-hold but loses less in crashes. This study asks whether that protection is worth what it costs. It tests 90 pre-specified rules on the survivorship-free Kenneth French 49 U.S. industry portfolios (daily, 1926–2025), with trading costs, a utility-based primary test, and data-snooping correction.

## Status
- [x] Analysis plan written and pre-registered
- [x] Validation tests (all 6 done) and trend rules for families 1-5 (`src/rules.py`)
- [x] Backtest engine (`src/engine.py`)
- [x] Data download (raw files in `data/raw/2026-09-28/`)
- [x] Data loader and SQL data checks (`src/data.py`, `sql/data_checks.sql`)
- [x] Replication checks run (checks 1 and 2 not met, see Deviations; check 3 is covered by the SQL data checks)
- [ ] Power analysis
- [ ] Primary test (H1)
- [ ] Secondary tests and robustness
- [ ] Paper

## Implementation notes
Choices made while writing the code, for places the plan does not spell out. None changes a test, a hypothesis or a parameter. Logged 2026-09-28, before any strategy result was computed.

- **Sample end.** The downloaded files (CRSP 202608 vintage) run through 2026-08-31. The loader cuts everything at 2025-12-31, as the plan's sample says.
- **Missing days inside an industry's history.** Eight industries have missing days after they first appear, all between 1927 and 1945 (Rubber has one 303-day gap in 1943-44; most others are single days). The plan says missing values are treated as missing. The loader takes that literally: an industry is inactive on a missing day, and the 273-day history count is the number of valid returns before the day, so a gap does not reset it. In the total-return index a missing day leaves the index unchanged.
- **Extreme early returns.** The raw data has daily moves over 40% in thin early-period industries (39 days between 1926 and 1949, for example Paper +150% on 1932-08-11). They are kept as they are in the source. The plan has no winsorizing step, so none is applied.

Rule code for families 1-5 (`src/rules.py`), logged 2026-09-29. No strategy result on the industry data had been computed. None of these changes a grid, a parameter or a hypothesis.

- **Ties.** Every entry test is a strict ">" and every exit test a strict "<", so a tie means not long.
- **Family 4 lookback.** "L months" is L x 21 trading days, the plan's definition of a month, measured at month-end. The T-bill return is compounded over the same days.
- **Family 2.** Averages the last N month-end index values, including the current one.
- **Start of a band or channel rule.** On its first valid day it is in cash, unless its entry condition is already true.
- **Family 5.** The data has closes only, so "prior N-day high/low" is the highest/lowest close of the N days before today.

Run the data checks with `python -m src.validate_data data/raw/2026-09-28`.

## Deviations from the plan
Any change after registration is logged here with a date and reason.

### 2026-09-28: Replication check 1 not met
**Plan:** daily industry returns compounded over a calendar year match French's annual industry file to within 0.1% per year.

**Result** (value-weighted industries, CRSP 202608 files, 1927-2025, years where the daily data is complete): 75.1% of the 4,591 industry-years are within 0.1 percentage point. The median gap is 0.05 points, the 95th percentile is 0.37 points, and the largest is 9.95 points (Real Estate, 1932). The gap appears in every era, so it is not a problem of the early years only. Reproduce with `python -m src.replication data/raw/2026-09-28`.

**Not changed:** the tolerance, the data, and every test in the plan. This check feeds none of the hypotheses. No strategy result had been computed when this was logged.

**Why the loader is not the cause:** French's annual file equals his own monthly file compounded to within 0.06 points (all 4,611 industry-years within 0.1 point). The gap is between the daily and the monthly files: 98.5% of industry-months are within 0.1 point, and the misses are the small, volatile industries.

**Likely cause (not confirmed):** the daily portfolios are re-weighted every day, while the monthly portfolios keep their start-of-month weights. Compounded daily returns from a daily-rebalanced portfolio need not equal the monthly return.

**What the paper will do:** report the check as not met, with the table above, in Appendix B. No replacement criterion is adopted.

### 2026-09-29: Replication check 2 (Faber) not met, by 0.03 points
**Plan:** the 10-month SMA rule on the market, 1972-2005, reproduces Faber's timing-minus-buy-and-hold differences (-0.06 points of CAGR, -3.47 points of standard deviation) to within 0.5 percentage points each. Faber's numbers were checked against Exhibit 9 of the published 2007 paper (S&P 500 11.24% / 17.47%, timing 11.18% / 14.00%) before the run. They match the plan.

**Result** (French market return Mkt-RF + RF and French T-bill rate, daily data compounded to months, 408 months):

| | CAGR % | SD % |
|---|---|---|
| Buy-and-hold | 11.15 | 15.85 |
| 10-month SMA | 10.56 | 12.38 |
| Difference | -0.59 | -3.46 |
| Faber's difference | -0.06 | -3.47 |
| Gap | 0.53 | 0.01 |

The SD gap is inside the tolerance. The CAGR gap is 0.53 points against a limit of 0.5, so the criterion is not met. Reproduce with `python -m src.faber_check data/raw/2026-09-28`.

**Not changed:** the tolerance, the rule, the data, the window and the sample dates. The check was run once. It feeds none of the hypotheses.

**Possible causes (not tested):** Faber's series is the S&P 500, not the French market portfolio, and his cash rate is 90-day commercial paper, not the T-bill rate. Both would move the CAGR difference.

**What the paper will do:** report the check as not met, with the table above, in Appendix B. No replacement criterion is adopted.

### 2026-09-29: Evaluation start date moved from 1927-05-31 to 1927-06-30
**Plan:** evaluation starts on "the first day every rule in the catalog has a valid signal", and the plan says 273 trading days of history covers the longest rule.

**Found:** the two statements disagree on the real calendar. In 1926-27 markets traded six days a week, about 25 trading days a month, so 273 trading days is only about 10.5 calendar months. Family 2 with N = 12 needs 12 month-end prices, and its first valid signal is 1927-06-30 (trading day 298). The 273-day rule gives 1927-05-31. Every other rule of families 1-5 is valid by then. This was measured on a flat price series over the real trading dates, so no strategy return was computed.

**Decision (Zach, 2026-09-29):** follow the plan's literal sentence. Evaluation starts on the later of the two dates, 1927-06-30, so every rule has a valid signal on day one. `rules.evaluation_start` computes it and `python -m src.validate_data data/raw/2026-09-28` reports it. The 273-day rule still decides when each industry enters the portfolios. Families 6-8 need at most 273 trading days, so they do not change the date.

**Not changed:** the rules, the grids, the tolerances and every hypothesis. The 1929 requirement (check 3) still holds: 1927-06-30 is before 1929-09-01. No strategy result had been computed when this was logged.

### 2026-09-29: Power-analysis clarifications (not a change to the plan)
The plan's power analysis leaves four details open. They are fixed here, before any power-analysis code has been written or run. None touches a locked item (primary test, rule catalog, sample splits, costs, crash definition, verdict thresholds, alpha).

| Date | Change | Reason |
|---|---|---|
| 2026-09-29 | (1) The effect size Delta is applied as a fixed additive daily return shift, found with a numerical root-finder so the synthetic strategy's population Delta-CE equals the target. | The plan says "constant shift" without saying how. An additive shift adds return without changing volatility. CRRA CE is not linear, so the shift cannot be computed in closed form. |
| 2026-09-29 | (2) The pre-1993 and post-1992 MDEs use paths resampled only from that period's own buy-and-hold returns (1926-1992, 1993-2025) with the matching T-bill rates, each at that period's length. The full-sample MDE uses the whole sample. | Cutting one full-sample path in two would give both halves the same mix of history. Only buy-and-hold returns are used, so no real strategy return is computed. |
| 2026-09-29 | (3) The shift is calibrated separately for every block setting x period x Delta (3 x 3 x 5 = 45 calibrations), each on one path 100 times the length of that period. | Block length changes how trendy the paths are and the period changes their volatility, so both move the strategy's baseline Delta-CE. |
| 2026-09-29 | (4) Block length comes from `arch.bootstrap.optimal_block_length` (the "stationary" column, which includes the Patton-Politis-White correction), computed on the daily buy-and-hold returns. Returns and T-bill rates are resampled together so each day's pair stays matched. The fixed settings are 252 and 1,260 days. | The plan names the method (Politis-White with the 2009 correction) but not an implementation. |

The 50/200 rule needs 200 days of warm-up on each synthetic path, so the evaluation window of each path starts after the warm-up.

## License
Code: MIT. Analysis plan: CC-BY 4.0.
