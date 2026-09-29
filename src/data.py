"""Load and clean the raw data files (Kenneth French Data Library, Cboe PPUT).

Rules come from the pre-registration ("Data" and "Handling rules"):

* French's missing-value codes (-99.99 and -999) are treated as missing.
* Returns in the files are in percent; everything here is decimal (1% = 0.01).
* The main sample ends 2025-12-31. The files hold newer data, which is cut off.
* An industry needs 273 trading days of history before it enters the strategy
  or the buy-and-hold portfolio.
* Moving averages run on a total-return index built from the daily returns.
* Raw files are never edited: they are read straight from the zip.

Implementation choices where the plan is silent (documented in the README):

* A missing day is not a return of zero. The industry is inactive that day.
* "273 days of history" counts valid (non-missing) daily returns before day t.
  A gap does not reset the count.
* In the total-return index a missing day leaves the index unchanged.
"""

import io
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

MISSING_CODES = (-99.99, -999.0)
SAMPLE_END = "2025-12-31"
MIN_HISTORY = 273  # trading days

# A data row starts with a 4-, 6- or 8-digit date followed by a comma.
_DATA_ROW = re.compile(r"^\s*\d{4}(?:\d{2}(?:\d{2})?)?\s*,")


def read_text(path, member=None):
    """Return a file's text, reading straight from a zip if it is one."""
    path = Path(path)
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            name = member or zf.namelist()[0]
            return zf.read(name).decode("latin-1")
    return path.read_text(encoding="latin-1")


def parse_french_tables(text):
    """Split a French-format CSV into its tables.

    Each table is a header line that starts with a comma (",Agric,Food,...")
    followed by data rows. The table's key is the title line right above the
    header, or "table1", "table2", ... when there is none. Values are returned
    as decimal returns with missing codes turned into NaN, indexed by date.
    """
    lines = text.splitlines()
    tables = {}
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith(",") and not _DATA_ROW.match(line):
            names = [h.strip() for h in line.split(",")[1:]]
            title = lines[i - 1].strip() if i > 0 else ""
            key = title if title else f"table{len(tables) + 1}"
            rows = []
            j = i + 1
            while j < len(lines) and _DATA_ROW.match(lines[j]):
                rows.append(lines[j])
                j += 1
            df = pd.read_csv(
                io.StringIO("\n".join(rows)),
                header=None,
                names=["date"] + names,
                index_col=0,
                dtype={"date": str},
            )
            df.index = _parse_dates(df.index)
            df = df.astype(float).where(~df.isin(MISSING_CODES)) / 100.0
            tables[key] = df
            i = j
        else:
            i += 1
    return tables


def _parse_dates(idx):
    """8 digits = day (YYYYMMDD), 6 = month (YYYYMM), 4 = year (YYYY)."""
    s = pd.Index([str(x).strip() for x in idx])
    n = {len(x) for x in s}
    if n == {8}:
        return pd.DatetimeIndex(pd.to_datetime(s, format="%Y%m%d"), name="date")
    if n == {6}:
        return pd.DatetimeIndex(pd.to_datetime(s, format="%Y%m"), name="date")
    if n == {4}:
        return pd.DatetimeIndex(pd.to_datetime(s, format="%Y"), name="date")
    raise ValueError(f"mixed or unsupported date formats in one table: {sorted(n)}")


def cut_sample(df, end=SAMPLE_END):
    """Keep dates up to and including `end`."""
    return df.loc[:end]


def _find_table(tables, contains):
    hits = [k for k in tables if contains.lower() in k.lower()]
    if len(hits) != 1:
        raise KeyError(f"expected one table containing {contains!r}, found {hits}")
    return tables[hits[0]]


def load_industry(path, weighting="ew", end=SAMPLE_END):
    """Daily returns of the 49 industries. weighting is "vw" or "ew"."""
    word = {"vw": "Value Weighted", "ew": "Equal Weighted"}[weighting]
    tables = parse_french_tables(read_text(path))
    return cut_sample(_find_table(tables, word), end)


def _single_table(path, end):
    tables = parse_french_tables(read_text(path))
    if len(tables) != 1:
        raise ValueError(f"{path}: expected 1 table, found {list(tables)}")
    return cut_sample(next(iter(tables.values())), end)


def load_ff3(path, end=SAMPLE_END):
    """Fama-French 3 factors + RF, daily. Columns: Mkt-RF, SMB, HML, RF."""
    return _single_table(path, end)


def load_ff5(path, end=SAMPLE_END):
    """Fama-French 5 factors + RF, daily (from 1963)."""
    return _single_table(path, end)


def load_momentum(path, end=SAMPLE_END):
    """Momentum factor (UMD), daily. Column: Mom."""
    return _single_table(path, end)


def market_return(ff3):
    """Total market return = Mkt-RF + RF (used to define crash cycles)."""
    return ff3["Mkt-RF"] + ff3["RF"]


def load_pput(path, end=SAMPLE_END):
    """Cboe PPUT index level, daily. Returns a Series named "PPUT"."""
    df = pd.read_csv(path)
    df["DATE"] = pd.to_datetime(df["DATE"], format="%m/%d/%Y")
    s = df.set_index("DATE")["PPUT"].astype(float)
    s.index.name = "date"
    return s.loc[:end]


def active_mask(ret, min_history=MIN_HISTORY):
    """True where an industry may be held on that day.

    Active on day t means: the return on day t is valid, and there are at
    least `min_history` valid returns strictly before day t.
    """
    valid = ret.notna()
    prior_valid = valid.cumsum().shift(1, fill_value=0)
    return valid & (prior_valid >= min_history)


def total_return_index(ret):
    """Total-return index starting at 1.0. A missing day leaves it unchanged."""
    return (1.0 + ret.fillna(0.0)).cumprod()


def first_evaluation_date(ret, min_history=MIN_HISTORY):
    """First day at least one industry is active."""
    active = active_mask(ret, min_history)
    days = active.any(axis=1)
    return days.index[np.argmax(days.to_numpy())]
