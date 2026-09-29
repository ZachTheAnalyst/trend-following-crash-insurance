"""Supplementary power run (README, Deviations, 2026-09-29: "Supplementary power
run (addition, not a change)"). The grid and the rules below were locked in the
README before this script was run.

    python -m src.power_supp data/raw/2026-09-28 results/power results/power_supp

Only the buy-and-hold index and the T-bill rate are read from the real data.
The registered run (results/power) stays as it is and remains the plan's answer.

What this adds to the registered run:

* Delta = 0 and Delta = 3, 4, 6, 8, 12 (percent per year) in all 9 cells,
  500 paths each. The grid is fixed and is not extended after seeing results.
* In the three cells where the registered run's rejection rate at Delta = 2.0 was
  below its rate at Delta = 0.25 (post-1992 with 252-day blocks, post-1992 with
  1,260-day blocks, pre-1993 with 1,260-day blocks), EVERY Delta (0, 0.25, 0.5,
  1.0, 1.5, 2.0, 3, 4, 6, 8, 12) is run with 2,000 paths, to see whether the dip
  is noise.
* Extra outputs: rejections in each tail, mean and SD of the t statistic, and
  the Delta-CE each calibrated shift gives on an independent long path.
* Fresh seeds, so nothing repeats the registered run's draws.
"""

import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from src import power

SUPP_DELTAS = (0.0, 3.0, 4.0, 6.0, 8.0, 12.0)
FLAGGED_DELTAS = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0)
FLAGGED_CELLS = (("post1992", "252"), ("post1992", "1260"), ("pre1993", "1260"))
N_PATHS_FLAGGED = 2000
RUN_ID = 1  # part of every seed, so the draws differ from the registered run

COLUMNS = [
    "period", "block_setting", "mean_block_days", "delta_pct", "shift_per_day",
    "baseline_dce_pct", "calibration_se_pct", "independent_path_dce_pct", "n_eval_days",
    "n_paths", "n_spot", "rej_nw", "rej_pos", "rej_neg", "rej_boot", "mean_t", "sd_t",
    "mean_est_dce_pct", "sd_est_dce_pct", "run_type",
]


def flagged_from_registered(cells):
    """The cells where the registered rejection rate at 2.0 was below the rate at 0.25."""
    out = []
    c = cells.assign(block_setting=cells["block_setting"].astype(str))
    for (period, setting), g in c.groupby(["period", "block_setting"], sort=False):
        g = g.set_index("delta_pct")["rej_nw"]
        if g[2.0] < g[0.25]:
            out.append((period, setting))
    return tuple(out)


def _task(args):
    (p_i, period, b_i, setting, r, rf, mean_block, targets, n_paths, n_spot, n_boot, factor, run_type) = args
    shifts, baseline, cal_se = power.calibrate_cell(
        r, rf, mean_block, targets,
        np.random.default_rng(np.random.SeedSequence([power.SEED, RUN_ID, p_i, b_i, 99])),
        factor=factor,
    )
    achieved = power.check_calibration(
        r, rf, mean_block, shifts,
        np.random.default_rng(np.random.SeedSequence([power.SEED, RUN_ID, p_i, b_i, 98])),
        factor=factor,
    )
    rows = []
    for d_i, delta in enumerate(targets):
        out = power.power_cell(
            r, rf, mean_block, shifts[delta],
            np.random.SeedSequence([power.SEED, RUN_ID, p_i, b_i, d_i]),
            n_paths=n_paths, n_spot=n_spot, n_boot=n_boot,
        )
        rows.append({
            "period": period, "block_setting": setting, "mean_block_days": mean_block,
            "delta_pct": delta, "shift_per_day": shifts[delta], "baseline_dce_pct": baseline,
            "calibration_se_pct": cal_se.get(delta, float("nan")),
            "independent_path_dce_pct": achieved[delta], "n_eval_days": len(r),
            "n_paths": n_paths, "n_spot": n_spot, "rej_nw": out["rej_nw"],
            "rej_pos": out["rej_pos"], "rej_neg": out["rej_neg"], "rej_boot": out["rej_boot"],
            "mean_t": out["mean_t"], "sd_t": out["sd_t"],
            "mean_est_dce_pct": out["mean_est_dce"], "sd_est_dce_pct": out["sd_est_dce"],
            "run_type": run_type,
        })
    return rows


def combined_mde(registered, supp, flagged=FLAGGED_CELLS):
    """MDE per cell over the locked grids.

    Flagged cells use their 2,000-path rows over the whole grid. Other cells use
    the registered 500-path rows (0.25 to 2.0) plus the supplementary rows (3 to 12).
    Delta = 0 is never a candidate (it is the size check). Returns None where no
    Delta up to 12 reaches 80%.
    """
    reg = registered.assign(block_setting=registered["block_setting"].astype(str))
    sup = supp.assign(block_setting=supp["block_setting"].astype(str))
    out = []
    for (period, setting), g in sup.groupby(["period", "block_setting"], sort=False):
        if (period, setting) in flagged:
            rates = g[g["delta_pct"] > 0]
        else:
            r0 = reg[(reg["period"] == period) & (reg["block_setting"] == setting)]
            rates = pd.concat([r0[["delta_pct", "rej_nw"]], g[g["delta_pct"] > 0][["delta_pct", "rej_nw"]]])
        out.append({
            "period": period, "block_setting": setting,
            "mde_pct": power.mde(dict(zip(rates["delta_pct"], rates["rej_nw"]))),
            "paths_used": N_PATHS_FLAGGED if (period, setting) in flagged else power.N_PATHS,
        })
    return pd.DataFrame(out)


def run(inputs, registered_cells, out_dir, n_paths=power.N_PATHS, n_paths_flagged=N_PATHS_FLAGGED,
        n_spot=power.N_SPOT, n_boot=power.N_BOOT, factor=power.LONG_FACTOR, workers=2, log=print):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "power_supp_cells.csv"
    done = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=COLUMNS)
    finished = set(zip(done["period"], done["block_setting"].astype(str)))
    flagged = flagged_from_registered(registered_cells)

    tasks = []
    for p_i, period in enumerate(power.PERIODS):
        d = power.period_slice(inputs, period)
        r, rf = d["bh"].to_numpy(), d["rf"].to_numpy()
        pw = power.pw_block_length(r)
        for b_i, setting in enumerate(power.BLOCK_SETTINGS):
            if (period, str(setting)) in finished:
                continue
            mb = pw if setting == "pw" else float(setting)
            is_flagged = (period, str(setting)) in flagged
            tasks.append((
                p_i, period, b_i, str(setting), r, rf, mb,
                FLAGGED_DELTAS if is_flagged else SUPP_DELTAS,
                n_paths_flagged if is_flagged else n_paths, n_spot, n_boot, factor,
                "registered grid rerun + supplementary" if is_flagged else "supplementary",
            ))
    log(f"flagged cells: {list(flagged)}")
    log(f"{len(tasks)} tasks to run")
    start = time.time()
    frames = [done]
    if tasks:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for rows in pool.map(_task, tasks):
                frames.append(pd.DataFrame(rows, columns=COLUMNS))
                pd.concat(frames, ignore_index=True).to_csv(path, index=False)
                log(f"  done {rows[0]['period']:9s} block {rows[0]['block_setting']:>5s}  "
                    f"{rows[0]['n_paths']} paths ({time.time() - start:5.0f}s)")
    cells = pd.concat(frames, ignore_index=True)
    table = combined_mde(registered_cells, cells, flagged)
    table.to_csv(out / "power_supp_mde.csv", index=False)
    return cells, table


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit("usage: python -m src.power_supp <raw data folder> <registered results folder> <output folder>")
    registered = pd.read_csv(Path(sys.argv[2]) / "power_cells.csv")
    inputs = power.bh_inputs(sys.argv[1])
    cells, table = run(inputs, registered, sys.argv[3])
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print()
        print(cells.round(4).to_string(index=False))
        print()
        print(table.to_string(index=False))
