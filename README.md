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
- [x] Validation tests (5 of 6 done; the flat-price test waits for the SMA rule code)
- [x] Backtest engine (`src/engine.py`)
- [x] Data download (raw files in `data/raw/2026-09-28/`)
- [x] Data loader and SQL data checks (`src/data.py`, `sql/data_checks.sql`)
- [ ] Replication checks (check 1 not met, see Deviations; the Faber check is still to do)
- [ ] Power analysis
- [ ] Primary test (H1)
- [ ] Secondary tests and robustness
- [ ] Paper

## Implementation notes
Choices made while writing the code, for places the plan does not spell out. None changes a test, a hypothesis or a parameter. Logged 2026-09-28, before any strategy result was computed.

- **Sample end.** The downloaded files (CRSP 202608 vintage) run through 2026-08-31. The loader cuts everything at 2025-12-31, as the plan's sample says.
- **Missing days inside an industry's history.** Eight industries have missing days after they first appear, all between 1927 and 1945 (Rubber has one 303-day gap in 1943-44; most others are single days). The plan says missing values are treated as missing. The loader takes that literally: an industry is inactive on a missing day, and the 273-day history count is the number of valid returns before the day, so a gap does not reset it. In the total-return index a missing day leaves the index unchanged.
- **Extreme early returns.** The raw data has daily moves over 40% in thin early-period industries (39 days between 1926 and 1949, for example Paper +150% on 1932-08-11). They are kept as they are in the source. The plan has no winsorizing step, so none is applied.

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

## License
Code: MIT. Analysis plan: CC-BY 4.0.
