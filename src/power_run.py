"""Run the power analysis (plan: "Economic significance and power").

    python -m src.power_run data/raw/2026-09-28 results/power

Only the equal-weighted industry BUY-AND-HOLD index and the T-bill rate are
built from the real data. The 50/200 rule runs on synthetic paths only, so no
real strategy return is computed.

For each period (full, pre-1993, post-1992) and block setting (Politis-White,
252, 1,260 days) the shift is calibrated for every Delta on one path 100 times
the period, then 500 independent paths per Delta run the Newey-West version of
H1 and the first 50 also run the full studentized bootstrap.

Writes, in the output folder:
    power_cells.csv   one row per period x block setting x Delta
    power_mde.csv     minimum detectable effect per period x block setting
    power_meta.json   seed, settings, sample sizes and package versions
The run resumes: cells already in power_cells.csv are skipped.
"""

import json
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from src import power

CELL_COLUMNS = [
    "period", "block_setting", "mean_block_days", "delta_pct", "shift_per_day",
    "baseline_dce_pct", "calibration_se_pct", "n_eval_days", "n_paths", "n_spot",
    "rej_nw", "rej_boot", "mean_est_dce_pct", "sd_est_dce_pct",
]


def _task(args):
    """Calibrate one (period, block setting) and run all its Delta cells."""
    (p_i, period, b_i, setting, r, rf, mean_block, n_paths, n_spot, n_boot, factor) = args
    targets = power.DELTAS
    shifts, baseline, cal_se = power.calibrate_cell(
        r, rf, mean_block, targets,
        np.random.default_rng(np.random.SeedSequence([power.SEED, p_i, b_i, 99])),
        factor=factor,
    )
    rows = []
    for d_i, delta in enumerate(targets):
        out = power.power_cell(
            r, rf, mean_block, shifts[delta],
            np.random.SeedSequence([power.SEED, p_i, b_i, d_i]),
            n_paths=n_paths, n_spot=n_spot, n_boot=n_boot,
        )
        rows.append({
            "period": period, "block_setting": setting, "mean_block_days": mean_block,
            "delta_pct": delta, "shift_per_day": shifts[delta], "baseline_dce_pct": baseline,
            "calibration_se_pct": cal_se.get(delta, float("nan")), "n_eval_days": len(r),
            "n_paths": n_paths, "n_spot": n_spot, "rej_nw": out["rej_nw"],
            "rej_boot": out["rej_boot"], "mean_est_dce_pct": out["mean_est_dce"],
            "sd_est_dce_pct": out["sd_est_dce"],
        })
    return rows


def mde_table(cells):
    """Minimum detectable effect per (period, block setting), from the cell table."""
    out = []
    for (period, setting), g in cells.groupby(["period", "block_setting"], sort=False):
        nw = dict(zip(g["delta_pct"], g["rej_nw"]))
        boot = dict(zip(g["delta_pct"], g["rej_boot"]))
        out.append({
            "period": period, "block_setting": setting,
            "mean_block_days": g["mean_block_days"].iloc[0],
            "mde_nw_pct": power.mde(nw), "mde_boot_spotcheck_pct": power.mde(boot),
        })
    return pd.DataFrame(out)


def run(inputs, out_dir, n_paths=power.N_PATHS, n_spot=power.N_SPOT, n_boot=power.N_BOOT,
        factor=power.LONG_FACTOR, workers=2, log=print):
    """`inputs` is a DataFrame with daily columns bh and rf, indexed by date."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cells_path = out / "power_cells.csv"
    done = pd.read_csv(cells_path) if cells_path.exists() else pd.DataFrame(columns=CELL_COLUMNS)
    finished = set(zip(done["period"], done["block_setting"].astype(str)))

    tasks, sizes = [], {}
    for p_i, period in enumerate(power.PERIODS):
        d = power.period_slice(inputs, period)
        r, rf = d["bh"].to_numpy(), d["rf"].to_numpy()
        sizes[period] = {"days": len(r), "first": str(d.index[0].date()), "last": str(d.index[-1].date()),
                         "pw_block_days": power.pw_block_length(r)}
        for b_i, setting in enumerate(power.BLOCK_SETTINGS):
            if (period, str(setting)) in finished:
                continue
            mean_block = sizes[period]["pw_block_days"] if setting == "pw" else float(setting)
            tasks.append((p_i, period, b_i, str(setting), r, rf, mean_block, n_paths, n_spot, n_boot, factor))

    meta = {
        "seed": power.SEED, "deltas_pct": list(power.DELTAS), "n_paths": n_paths, "n_spot": n_spot,
        "n_boot": n_boot, "long_path_factor": factor, "alpha": power.ALPHA, "gamma": power.GAMMA,
        "cost": power.COST, "periods": sizes, "python": platform.python_version(),
        "numpy": np.__version__, "pandas": pd.__version__,
    }
    try:
        import arch
        meta["arch"] = arch.__version__
    except ImportError:  # pragma: no cover
        pass
    (out / "power_meta.json").write_text(json.dumps(meta, indent=2))

    log(f"{len(tasks)} of {len(power.PERIODS) * len(power.BLOCK_SETTINGS)} (period, block) tasks to run")
    start = time.time()
    frames = [done]
    if tasks:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for rows in pool.map(_task, tasks):
                new = pd.DataFrame(rows, columns=CELL_COLUMNS)
                frames.append(new)
                pd.concat(frames, ignore_index=True).to_csv(cells_path, index=False)
                log(f"  done {rows[0]['period']:9s} block {rows[0]['block_setting']:>5s}  "
                    f"({time.time() - start:5.0f}s)")
    cells = pd.concat(frames, ignore_index=True)
    table = mde_table(cells)
    table.to_csv(out / "power_mde.csv", index=False)
    return cells, table


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: python -m src.power_run <raw data folder> <output folder>")
    inputs = power.bh_inputs(sys.argv[1])
    cells, table = run(inputs, sys.argv[2])
    with pd.option_context("display.width", 140, "display.max_columns", 30):
        print()
        print(cells.round(4).to_string(index=False))
        print()
        print(table.to_string(index=False))
