-- Data validation checks, run in DuckDB by `python -m src.validate_data`.
--
-- Tables (built by src/data.py from the raw files, sample cut at 2025-12-31,
-- returns in decimal, missing codes already turned into NULL):
--   ind_vw(date, industry, ret)   49 industries, value-weighted, daily
--   ind_ew(date, industry, ret)   49 industries, equal-weighted, daily
--   ff3(date, mkt_rf, smb, hml, rf)
--   ff5(date, mkt_rf, smb, hml, rmw, cma, rf)
--   mom(date, mom)
--   pput(date, pput)              Cboe PPUT index level
--
-- Each block is: "-- check: <name>", "-- expect: zero_rows | info", then a
-- query. zero_rows means the query must return nothing (any row is a failure).
-- info just prints what the query returns.

-- check: no_duplicate_dates_per_industry
-- expect: zero_rows
SELECT industry, date, COUNT(*) AS n
FROM ind_ew
GROUP BY industry, date
HAVING COUNT(*) > 1;

-- check: vw_and_ew_cover_the_same_industry_days
-- expect: zero_rows
SELECT COALESCE(v.date, e.date) AS date, COALESCE(v.industry, e.industry) AS industry
FROM ind_vw v
FULL OUTER JOIN ind_ew e ON v.date = e.date AND v.industry = e.industry
WHERE v.date IS NULL OR e.date IS NULL;

-- check: industry_dates_match_ff3_dates
-- expect: zero_rows
SELECT COALESCE(a.date, b.date) AS date
FROM (SELECT DISTINCT date FROM ind_ew) a
FULL OUTER JOIN ff3 b ON a.date = b.date
WHERE a.date IS NULL OR b.date IS NULL;

-- check: sample_ends_no_later_than_2025_12_31
-- expect: zero_rows
SELECT 'ind_ew' AS tbl, MAX(date) AS last_date FROM ind_ew HAVING MAX(date) > DATE '2025-12-31'
UNION ALL SELECT 'ff3', MAX(date) FROM ff3 HAVING MAX(date) > DATE '2025-12-31'
UNION ALL SELECT 'pput', MAX(date) FROM pput HAVING MAX(date) > DATE '2025-12-31';

-- check: no_return_of_minus_100_percent_or_worse
-- expect: zero_rows
SELECT 'ind_vw' AS tbl, date, industry, ret FROM ind_vw WHERE ret <= -1
UNION ALL
SELECT 'ind_ew', date, industry, ret FROM ind_ew WHERE ret <= -1;

-- check: ff3_has_no_missing_values
-- expect: zero_rows
SELECT * FROM ff3
WHERE mkt_rf IS NULL OR smb IS NULL OR hml IS NULL OR rf IS NULL;

-- check: risk_free_rate_is_never_negative
-- expect: zero_rows
SELECT date, rf FROM ff3 WHERE rf < 0;

-- check: pput_levels_are_positive_and_dates_unique
-- expect: zero_rows
SELECT date, pput FROM pput WHERE pput IS NULL OR pput <= 0
UNION ALL
SELECT date, COUNT(*) FROM pput GROUP BY date HAVING COUNT(*) > 1;

-- check: first_evaluation_date_is_before_1929_09_01
-- expect: zero_rows
-- The 273-day rule from the plan: an industry is active on a day when its
-- return is valid and it has at least 273 valid returns before that day.
WITH v AS (
    SELECT industry, date, ret,
           COUNT(ret) OVER (
               PARTITION BY industry ORDER BY date
               ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
           ) AS prior_valid
    FROM ind_ew
),
first_eval AS (
    SELECT MIN(date) AS first_date FROM v WHERE ret IS NOT NULL AND prior_valid >= 273
)
SELECT first_date FROM first_eval WHERE first_date >= DATE '1929-09-01';

-- check: sample_window_by_table
-- expect: info
SELECT 'ind_ew' AS tbl, MIN(date) AS first_date, MAX(date) AS last_date, COUNT(DISTINCT date) AS days FROM ind_ew
UNION ALL SELECT 'ind_vw', MIN(date), MAX(date), COUNT(DISTINCT date) FROM ind_vw
UNION ALL SELECT 'ff3', MIN(date), MAX(date), COUNT(*) FROM ff3
UNION ALL SELECT 'ff5', MIN(date), MAX(date), COUNT(*) FROM ff5
UNION ALL SELECT 'mom', MIN(date), MAX(date), COUNT(*) FROM mom
UNION ALL SELECT 'pput', MIN(date), MAX(date), COUNT(*) FROM pput;

-- check: first_evaluation_date
-- expect: info
WITH v AS (
    SELECT industry, date, ret,
           COUNT(ret) OVER (
               PARTITION BY industry ORDER BY date
               ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
           ) AS prior_valid
    FROM ind_ew
)
SELECT MIN(date) AS first_evaluation_date FROM v WHERE ret IS NOT NULL AND prior_valid >= 273;

-- check: missing_values_by_industry
-- expect: info
-- gaps_after_start = missing days after the industry first appears (mid-sample holes).
WITH s AS (
    SELECT industry,
           MIN(date) FILTER (WHERE ret IS NOT NULL) AS first_valid,
           COUNT(*) FILTER (WHERE ret IS NULL) AS missing_days
    FROM ind_ew
    GROUP BY industry
),
g AS (
    SELECT e.industry, COUNT(*) AS gaps_after_start
    FROM ind_ew e JOIN s ON e.industry = s.industry
    WHERE e.ret IS NULL AND e.date > s.first_valid
    GROUP BY e.industry
)
SELECT s.industry, s.first_valid, s.missing_days, COALESCE(g.gaps_after_start, 0) AS gaps_after_start
FROM s LEFT JOIN g ON s.industry = g.industry
WHERE s.missing_days > 0
ORDER BY gaps_after_start DESC, s.missing_days DESC;

-- check: extreme_daily_returns_over_40_percent_by_decade
-- expect: info
SELECT (EXTRACT(year FROM date) // 10 * 10)::INT AS decade, COUNT(*) AS days_over_40pct
FROM ind_ew
WHERE ABS(ret) > 0.40
GROUP BY decade
ORDER BY decade;

-- check: ten_biggest_daily_moves_equal_weighted
-- expect: info
SELECT date, industry, ROUND(ret * 100, 2) AS ret_pct
FROM ind_ew
WHERE ret IS NOT NULL
ORDER BY ABS(ret) DESC
LIMIT 10;

-- check: active_industries_at_year_end
-- expect: info
WITH v AS (
    SELECT industry, date, ret,
           COUNT(ret) OVER (
               PARTITION BY industry ORDER BY date
               ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
           ) AS prior_valid
    FROM ind_ew
),
a AS (
    SELECT date, COUNT(*) FILTER (WHERE ret IS NOT NULL AND prior_valid >= 273) AS active_industries
    FROM v GROUP BY date
)
SELECT EXTRACT(year FROM date)::INT AS year, MAX_BY(active_industries, date) AS active_on_last_day
FROM a
WHERE EXTRACT(year FROM date) IN (1927, 1929, 1935, 1950, 1975, 2000, 2025)
GROUP BY year
ORDER BY year;
