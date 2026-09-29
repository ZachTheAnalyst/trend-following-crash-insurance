"""Run the SQL data checks on the raw files.

    python -m src.validate_data data/raw/2026-09-28

Loads the raw files with src/data.py, puts them in an in-memory DuckDB, runs
sql/data_checks.sql and prints a report. Exits with status 1 if any check
with "expect: zero_rows" returns rows.
"""

import re
import sys
from pathlib import Path

import duckdb
import pandas as pd

from src import data

SQL_FILE = Path(__file__).resolve().parent.parent / "sql" / "data_checks.sql"

FILES = {
    "industry": "49_Industry_Portfolios_daily_CSV.zip",
    "ff3": "F-F_Research_Data_Factors_daily_CSV.zip",
    "ff5": "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",
    "mom": "F-F_Momentum_Factor_daily_CSV.zip",
    "pput": "PPUT_History.csv",
}


def _long(df):
    """Wide (date x industry) to long (date, industry, ret)."""
    out = df.rename_axis("date").reset_index().melt(
        id_vars="date", var_name="industry", value_name="ret"
    )
    return out.sort_values(["industry", "date"]).reset_index(drop=True)


def _flat(df, names):
    out = df.rename_axis("date").reset_index()
    out.columns = ["date"] + names
    return out


def build_db(raw_dir):
    """Load every raw file into an in-memory DuckDB and return the connection."""
    raw = Path(raw_dir)
    path = {k: raw / v for k, v in FILES.items()}
    con = duckdb.connect()
    con.register("ind_vw", _long(data.load_industry(path["industry"], "vw")))
    con.register("ind_ew", _long(data.load_industry(path["industry"], "ew")))
    con.register("ff3", _flat(data.load_ff3(path["ff3"]), ["mkt_rf", "smb", "hml", "rf"]))
    con.register(
        "ff5",
        _flat(data.load_ff5(path["ff5"]), ["mkt_rf", "smb", "hml", "rmw", "cma", "rf"]),
    )
    con.register("mom", _flat(data.load_momentum(path["mom"]), ["mom"]))
    con.register("pput", data.load_pput(path["pput"]).rename("pput").reset_index())
    return con


def parse_checks(sql_text):
    """Split the SQL file into (name, expect, query) blocks."""
    blocks = re.split(r"(?m)^-- check: ", sql_text)[1:]
    checks = []
    for block in blocks:
        name, _, rest = block.partition("\n")
        m = re.match(r"-- expect: (zero_rows|info)", rest)
        if not m:
            raise ValueError(f"check {name!r} has no '-- expect:' line")
        query = "\n".join(
            line for line in rest.splitlines()[1:] if not line.strip().startswith("--")
        ).strip()
        checks.append((name.strip(), m.group(1), query))
    return checks


def run(raw_dir):
    con = build_db(raw_dir)
    failures = 0
    for name, expect, query in parse_checks(SQL_FILE.read_text()):
        df = con.execute(query).df()
        if expect == "zero_rows":
            ok = len(df) == 0
            failures += 0 if ok else 1
            print(f"[{'PASS' if ok else 'FAIL'}] {name}")
            if not ok:
                print(df.head(20).to_string(index=False))
        else:
            print(f"[INFO] {name}")
            with pd.option_context("display.width", 120, "display.max_rows", 60):
                print(df.to_string(index=False))
        print()
    print(f"{failures} check(s) failed." if failures else "All checks passed.")
    return failures


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python -m src.validate_data <raw data folder>")
    sys.exit(1 if run(sys.argv[1]) else 0)
