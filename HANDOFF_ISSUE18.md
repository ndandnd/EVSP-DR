# EVSP Demand-Response Handoff

> Historical handoff: paths and random-instance launchers below describe the
> earlier issue18 campaign. For a clean clone and the maintained Goal-1 Unicorn
> workflow, use [`UNICORN_RUNBOOK.md`](UNICORN_RUNBOOK.md).

This repository contains a column-generation EVSP workflow run locally on macOS
and remotely on the Cornell Unicorn Slurm cluster. This handoff captures the
current state after adding greedy initialization and comparing GREEDY,
NO_CHEAT, and CHEAT price-scenario experiments.

## Git state

- The live working branch at handoff was `issue20`; this handoff is being
  published as a clean, source-only commit on `issue18`.
- The clean `issue18` update intentionally omits older oversized result and
  column-pool artifacts that GitHub refuses to accept. It contains the current
  code, scripts, and this handoff only.
- Do not add generated `results/`, `grbnodes*`, `__pycache__/`, PDFs, copied
  result CSVs, or large raw input CSVs unless explicitly requested.

## Core model and initialization modes

The active experiment driver is `src/run_ex_unicorn.py`.

- `--cheat`: seed routes reconstructed from the historical `VehicleTask`
  solution.
- default / no flag: `NO_CHEAT`; begin with an empty `R_truck`.
- `--greedy`: use the new greedy feasible route cover in `src/greedy_init.py`.
  `--cheat` and `--greedy` are intentionally mutually exclusive.

`src/pricing_dp_og.py` is the active DP pricer. The source state now starts at
full SOC (`soc=float(G)`) at the depot, with no artificial initial charging
cost. `src/pricing_dp.py` is legacy/not normally used, but its source label was
also changed to full SOC for consistency.

The greedy initializer uses the same trip/depot/station graph conventions as
the DP pricer. It builds routes until every trip is covered, respects time,
deadhead-energy, charging-rate, battery-capacity, recharge-count, and depot
horizon checks, then returns routes in the `R_truck` format expected by the
master problem.

For a fast local inspection without column generation or Gurobi, use:

```bash
cd /Users/nadan/Documents/projects/demandresponse
python3 src/run_greedy_init_only.py --csv Inst_10B_RND001.csv --G 300
```

It writes a route-level JSON plus a summary CSV under
`src/results/greedy_init_debug/`. Exit `less`/pager output with `q`.

## Pricing scenario inputs

`src/make_spatiotemporal_price_scenarios.py` converts temporal input files
`hourly_prices_single_peak_08.csv`, `_12.csv`, and `_18.csv` to the model's
spatiotemporal schema by replicating each hourly price across all stations:

```bash
python3 src/make_spatiotemporal_price_scenarios.py
```

The generated files are:

- `data/spatiotemporal_single_peak_08.csv`
- `data/spatiotemporal_single_peak_12.csv`
- `data/spatiotemporal_single_peak_18.csv`

## Cluster notes

Cluster login host:

```bash
nc437@unicorn-login-01.coecis.cornell.edu
```

In non-interactive `ssh '...'` commands, Slurm may not be on `PATH`. Use the
absolute executable `/usr/local/slurm/current/bin/sbatch` or export:

```bash
export PATH=/usr/local/slurm/current/bin:$PATH
```

The standard job environment is:

```bash
export GRB_LICENSE_FILE=/share/apps/software/gurobi/gurobi.lic
unset LM_LICENSE_FILE
source /share/apps/software/anaconda3/etc/profile.d/conda.sh
conda activate /home/nc437/evsp_env
```

For preemption-safe column generation, use `--requeue`, locate the most recent
`ckpt_latest_*.json`, export `RESUME_CKPT`, and let
`run_ex_unicorn.py` continue its active-time accounting. Do not add a final MIP
inside preemptible column-generation jobs; run it separately on the Scaglione
partition after a snapshot exists.

## Existing scenario batches and naming tags

All use fleet sizes 10B/15B, RND001-004, and peak08/peak12/peak18 unless
otherwise noted.

- Earlier CHEAT/NO_CHEAT batch:
  `stag999998_imp-2.0_peakXX`
- Full-SOC CHEAT/NO_CHEAT batch:
  `stag999997_imp-2.0_peakXX_fullsoc`
- GREEDY batch:
  `GREEDY_stag999996_imp-2.0_peakXX_greedy`

GREEDY was run for 12 hours of active CG. It initially saved only 12h
snapshots; the 3h snapshots were safely reconstructed later from pricing stats
using `src/reconstruct_milestone_snapshots.py`. The reconstruction uses the
pricing CSV's cumulative active time and takes the exact route-prefix from the
final checkpoint. It wrote 24 GREEDY 3h snapshots successfully.

## Submit scripts

Full-SOC pipeline scripts are retained as working examples:

- `src/submit_12h_price_scenarios.sub`: 48 CHEAT/NO_CHEAT 12h CG tasks with
  `--skip_final_mip`.
- `src/submit_mip_40m_fullsoc.sub`: dependent 40-minute Scaglione MIP array.
- `src/submit_fullsoc_pipeline.sh`: submits both arrays; uses `afterany`, so
  the MIP script must skip task instances with no 12h snapshot.
- `scripts/collect_fullsoc_results_and_plot.sh`: macOS-side poll/rsync/Gantt
  helper. It filters `stag999997` folders and then runs the local Gantt
  plotter.

For MIPs from any particular snapshot list, `src/submit_mip_only.sub` accepts
the snapshot path as argument 1 and a time limit in hours as argument 2:

```bash
sbatch -J "my_mip" submit_mip_only.sub /absolute/path/to/routes_12h_snapshot_....json 1
```

`run_final_mip.py` saves `final_mip_*.sol`, `final_mip_summary_*.json`, and
the Gurobi log into the run directory. When choosing a solution for a Gantt
plot, prefer the summary whose `ckpt_source` matches the desired snapshot so a
3h solution is not mixed with a 12h solution.

## Local results and plotting

Current local result roots:

- GREEDY: `/Users/nadan/Downloads/evsp_final_results_greedy_stag999996`
- Earlier CHEAT/NO_CHEAT price scenarios:
  `/Users/nadan/Downloads/evsp_final_results`
- Full-SOC collected results, when present:
  `/Users/nadan/Downloads/evsp_final_results_stag999997_fullsoc`

`src/plot_charging_gantt_from_solutions.py` supports the GREEDY naming scheme
and `--snapshot-kind` (`auto`, `3h`, `10h`, `12h`, `24h`). It uses
`final_mip_summary_*.json` metadata to pair a `.sol` file with its source
snapshot before drawing charging schedules.

`src/plot_greedy_vs_nocheat_master.py` discovers pricing CSVs by parsing their
run-directory names, not by assuming a specific timestamp. It produces:

- per-instance log-scale curves,
- overall GREEDY-vs-NO_CHEAT distribution-band plots,
- overall all-three (GREEDY/NO_CHEAT/CHEAT) plots,
- fleet-separated 10B and 15B versions of both comparisons,
- 8-10 active-hour zooms, and
- CSV manifests of matched runs.

Run it locally with:

```bash
cd /Users/nadan/Documents/projects/demandresponse
MPLCONFIGDIR=/private/tmp/mplconfig python3 src/plot_greedy_vs_nocheat_master.py
```

Plots are written to:

`output/greedy_vs_nocheat_master/`

The current script found 24 matched pricing CSVs for each of GREEDY,
NO_CHEAT, and CHEAT. Each fleet-specific plot aggregates 12 runs per mode
(four random instances times three peak-price scenarios). In the plots, faint
lines are individual runs, the dark band is the 25th-75th percentile, the
light band is the 10th-90th percentile, and the thick line is the pointwise
median. These bands show empirical run-to-run spread; they are not formal
statistical confidence intervals.

## Known caveats

- GREEDY, NO_CHEAT, and CHEAT comparisons are useful operationally but not a
  perfectly controlled experiment: tags, initialization rules, and some code
  changes differ across batches.
- The full-SOC collector previously hit an intermittent `rsync` broken pipe;
  rerunning the same filtered rsync is safe and resumes transferred files.
- Aggregated plots trim to the common active-time horizon to avoid summarizing
  only a shrinking subset of runs at the far right.
- The repository presently has many user-owned untracked data and generated
  artifacts. Preserve them; stage only clearly intentional source/scripts.

## Good next tasks

1. Inspect final MIP summaries by fleet and initialization to compare vehicle
   count, objective, bound, and gap at 3h/12h.
2. Create Gantt plots from correctly matched `final_mip_summary_*.json` and
   snapshots for the GREEDY 3h and 12h MIPs.
3. Run a truly apples-to-apples batch where GREEDY, NO_CHEAT, and CHEAT share
   the same DP/pricing code, full-SOC convention, price files, active-time
   budget, and MIP budget.
