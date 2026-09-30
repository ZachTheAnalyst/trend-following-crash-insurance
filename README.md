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
- [x] Power analysis (run 2026-09-29: no cell reaches 80% power by 2.0% per year, see Power analysis results)
- [x] Portfolio layer and H1 code written and tested on made-up data (`src/portfolio.py`, `src/h1.py`); NOT yet run on real data
- [ ] Primary test (H1) run on real data
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

Power-analysis code (`src/power.py`, `src/power_run.py`), logged 2026-09-29, before any real-data power run. It builds the plan's steps and adds no test, hypothesis or parameter. Only the equal-weighted industry buy-and-hold index and the T-bill rate are read from the real data; the 50/200 rule runs on synthetic paths only.

- **Months.** A month is 21 trading days (the plan's definition), so synthetic paths need no calendar. A partial last month is dropped.
- **Path length.** A synthetic path is 200 days longer than the period, because the 50/200 rule needs 200 days before its first signal. The evaluation window has exactly the period's length.
- **Calibration path.** One path 100 times the period's length (plus the 200 days), reused for all five Deltas, giving 45 calibrations. Such a path would overflow a float price index, so the signal is computed in overlapping segments, each rebased to 1.0. The rule only compares averages of the index, so rebasing changes nothing, and a test checks that segmented and whole-path signals are identical.
- **Calibration error.** The long path is finite, so the population Delta-CE the shift reaches carries sampling error. It is reported per cell as `calibration_se_pct`.
- **Test.** Two-sided, alpha = 0.05, normal critical value 1.96, Newey-West lag floor(4 (M/100)^(2/9)) with M months. The bootstrap version is a studentized stationary block bootstrap with 10,000 draws on the monthly utility differences, block length from the Politis-White rule on those differences, two-sided p = (1 + number of draws with |t*| >= |t|) / (draws + 1). The spot-check uses the first 50 paths of each cell.
- **Bootstrap code.** The stationary bootstrap is written in `src/power.py` (vectorized) because the bootstrap test needs 10,000 draws per run and the calibration path is longer than the data. `arch` supplies the block lengths, and a test checks the bootstrap against arch's `StationaryBootstrap`.
- **Seeds.** Every cell has a fixed seed derived from 20260929, so the run reproduces exactly. The run resumes if interrupted.

Portfolio and H1 code (`src/portfolio.py`, `src/h1.py`), logged 2026-09-29, before any real strategy result was computed. Tested on made-up data only. It adds no test, hypothesis or parameter; H1 stays as registered (CRRA gamma = 5, 50/200, equal-weighted industries, 10 bp, full sample).

- **First active day.** An industry's first active day is not an entry trade. The strategy holds the weight its signal gives, as the engine does with `initial_weight`. Buy-and-hold and the strategy start together.
- **After a missing day.** The industry is not traded on the missing day. The next active day's weight change is measured against its last active weight.
- **Signal.** The 50/200 signal is family 1 (no band) on each industry's total-return index, in which a missing day leaves the index unchanged.
- **Months.** H1 uses calendar months of compounded daily net returns. A month the window covers only partly is dropped (in practice June 1927, since the window starts 1927-06-30). Strategy and buy-and-hold use the same months. (The power analysis used 21-day months because its paths have no calendar; the plan's 21-day month is for the engine and the rules.)
- **Bootstrap.** Months are resampled in pairs (strategy month, buy-and-hold month) so each draw keeps them matched. Stationary bootstrap, 10,000 draws, Politis-White block length on the monthly utility differences.
- **p-value.** Two-sided, (1 + number of draws with |t*| >= |t|) / (draws + 1), where t = mean(d) / NW standard error and t* is the same statistic on a draw, centered at mean(d).
- **Confidence interval for Delta-CE.** Bootstrap-t. Each draw's Delta-CE gets a delta-method Newey-West standard error, t* = (Delta-CE* - Delta-CE) / se*, and the interval is [D - q(0.975) se, D - q(0.025) se]. The percentile interval is reported next to it as a check. The verdict uses the bootstrap-t interval.
- **Verdict bins.** Worth it, meaningful: CI lower bound > 0 and Delta-CE >= 0.5. Worth it, trivial: CI lower bound > 0 and Delta-CE < 0.5. Not worth it: CI upper bound < 0. Otherwise inconclusive.
- **Newey-West check.** Same lag rule as the power analysis, Bartlett kernel, normal p-value. Reported only as a check.
- **Seed.** 20260929. `python -m src.h1` refuses to run without `--confirm-unblind`.

Run the power analysis with `python -m src.power_run data/raw/2026-09-28 results/power`.

Run the data checks with `python -m src.validate_data data/raw/2026-09-28`.

## Power analysis results
Run 2026-09-29 with `python -m src.power_run data/raw/2026-09-28 results/power`, from commit 62e7067, on synthetic paths resampled from the equal-weighted industry buy-and-hold index. No real strategy return was computed. Raw output: `results/power/power_cells.csv` (one row per cell), `power_mde.csv`, `power_meta.json`.

Share of 500 paths in which the Newey-West version of H1 rejects at alpha = 0.05, by true Delta (percent per year of certainty-equivalent gain):

| Period | Block length | Eval. days | 0.25 | 0.5 | 1.0 | 1.5 | 2.0 | SD of estimated Delta-CE (pts/yr) | MDE |
|---|---|---|---|---|---|---|---|---|---|
| Full | Politis-White (8.1 d) | 25,853 | 6.2% | 6.2% | 7.6% | 14.2% | 26.4% | 1.3 | > 2.0 |
| Full | 252 d | 25,853 | 7.4% | 6.8% | 7.2% | 9.2% | 17.8% | 1.8 | > 2.0 |
| Full | 1260 d | 25,853 | 3.6% | 2.8% | 2.6% | 3.2% | 8.6% | 1.6 | > 2.0 |
| Pre-1993 | Politis-White (26.9 d) | 17,546 | 6.8% | 5.4% | 6.8% | 7.2% | 19.4% | 1.9 | > 2.0 |
| Pre-1993 | 252 d | 17,546 | 9.8% | 7.8% | 5.6% | 9.4% | 11.0% | 2.3 | > 2.0 |
| Pre-1993 | 1260 d | 17,546 | 5.0% | 4.2% | 3.6% | 1.6% | 2.8% | 1.9 | > 2.0 |
| Post-1992 | Politis-White (3.6 d) | 8,307 | 5.6% | 5.0% | 5.2% | 7.8% | 12.2% | 2.0 | > 2.0 |
| Post-1992 | 252 d | 8,307 | 8.8% | 9.4% | 5.8% | 3.8% | 2.8% | 3.1 | > 2.0 |
| Post-1992 | 1260 d | 8,307 | 6.2% | 6.0% | 2.2% | 1.0% | 0.2% | 2.8 | > 2.0 |

**Result:** the minimum detectable effect (80% power) is above 2.0% per year in all nine cells. The plan's grid stops at 2.0, so the plan's answer is "MDE > 2.0%", far above the 0.5% "meaningful" threshold. Under the plan's reading rule, an inconclusive H1 with MDE above 0.5% is read as "the test is too weak to decide", not as evidence of no effect.

**Why:** the estimated Delta-CE varies by 1.3 to 3.1 points a year from one synthetic path to the next, so a 2-point effect is about one standard error. The rejection rates below 5% in the long-block rows come from the Newey-West test being conservative when a few crash months dominate the utility differences (the t statistic's spread is about 0.75, not 1). Some rows fall as Delta rises because at small Delta the few rejections come from the wrong-sign tail, caused by calibration error (`calibration_se_pct`: 0.14 to 0.35 points a year).

**Spot-check:** the full studentized bootstrap on the first 50 paths of each cell (`rej_boot`) agrees with the Newey-West rates within the noise of 50 runs.

**Exploratory, superseded:** a first look with larger effects (300 paths, unsaved) suggested 80% power arrives near 3.5-4% a year for the full sample and near 8% a year for post-1992. Those numbers are not to be cited. They are replaced by the supplementary run logged under Deviations (2026-09-29), which saves everything.

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

### 2026-09-29: Supplementary power run (addition, not a change)
**Plan:** the power analysis uses Delta = 0.25, 0.5, 1.0, 1.5, 2.0. **Registered result:** no cell reaches 80% power by 2.0, so the plan's answer is "MDE > 2.0%", see Power analysis results. That result stays as it is and remains the answer to the plan's power analysis.

**Added (Zach's decision, 2026-09-29).** The grid and the rules below are locked now, before the supplementary run, and will not be adjusted after seeing its results. The code is `src/power_supp.py`.

1. **Extra effect sizes.** Delta = 0 and Delta = 3, 4, 6, 8, 12 (percent per year) in all nine cells, 500 paths each. Delta = 0 shows how often the test rejects when there is no effect (the size check). If no Delta up to 12 reaches 80% in a cell, its MDE is reported as "> 12"; the grid is not extended.
2. **More paths where the rate dipped.** Rule, fixed now: a cell is flagged if its registered rejection rate at Delta = 2.0 was below its rate at Delta = 0.25. Applying it to `results/power/power_cells.csv` flags three cells: post-1992 with 252-day blocks, post-1992 with 1,260-day blocks, and pre-1993 with 1,260-day blocks. In those cells every Delta (0, 0.25, 0.5, 1.0, 1.5, 2.0, 3, 4, 6, 8, 12) runs with 2,000 paths. If the dip disappears at 2,000 paths, it was noise. The MDE for a flagged cell uses its 2,000-path rows; for other cells it uses the registered 500-path rows plus the supplementary rows.
3. **Calibration check with numbers.** For every target, the Delta-CE the calibrated shift gives on an independent long path (fresh draws, same length) is saved next to the target, together with the calibration standard error. On the calibration path itself the shift hits the target by construction, so an independent path is what shows the error.
4. **Extra outputs.** Rejections in each tail, and the mean and SD of the t statistic. They show whether the test is conservative (SD below 1) or the rejections come from the wrong-sign tail.
5. **The unsaved exploratory runs are replaced** by this run, so the paper never cites a number that was not saved.
6. **Seeds** are fresh, so nothing repeats the registered run's draws. Spot-checks (50 paths per cell, 10,000 bootstrap draws) run as before.

**Not changed:** H1, its metric (CRRA Delta-CE, gamma = 5), the verdict thresholds, the rule catalog, the splits, costs, alpha, and the registered power result. H1 is not replaced by a Sharpe test or a lower gamma because it looks weak; those are already secondary tests. The supplementary run finishes before H1 runs on real data.

**How it will be reported:** in Appendix A next to the registered result, labeled supplementary. Run with `python -m src.power_supp data/raw/2026-09-28 results/power results/power_supp`.

## License
Code: MIT. Analysis plan: CC-BY 4.0.
