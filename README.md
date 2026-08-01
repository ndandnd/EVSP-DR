# Joint Truck & Battery Routing under Solar & V2G EVSP

**Authors:** Nathan Cho, nc437@cornell.edu; Andrea Lodi; Anna Scaglione; Nada Ae


This repository contains the Python implementation and data needed to reproduce all results in:
> **Electric Vehicle Scheduling and Vehicle-to-Grid Integration in Microgrids**



We solve a two-stage column-generation model for joint truck and battery routing under solar and vehicle-to-grid planning.

## Current Workflow

The maintained experiment path is intentionally small:

- `src/run_ex_unicorn.py`: column-generation driver.
- `src/pricing_dp_og.py`: dynamic-programming pricing problem.
- `src/greedy_init.py`: feasible greedy route-cover initialization.
- `src/run_final_mip.py`: final set-covering MIP over a saved column pool.
- `src/reconstruct_milestone_snapshots.py`: reconstruct 3h/10h/24h route pools from checkpoints and pricing statistics.
- `src/plot_charging_gantt_from_solutions.py` and `src/plot_greedy_vs_nocheat_master.py`: result visualization.

Canonical static inputs live in `data/`. Experiment instances, checkpoints, solver logs, route pools, MIP solutions, and plots are generated artifacts and are deliberately ignored by Git. Keep research results in a dated results directory or external storage rather than committing them to the source repository.

For the current reproducible Unicorn checkpoint, exact 3h/6h/unlimited launch
commands, result audits, and next-step decision rules, see
[`UNICORN_RUNBOOK.md`](UNICORN_RUNBOOK.md). `HANDOFF_ISSUE18.md` is retained as
historical context.
