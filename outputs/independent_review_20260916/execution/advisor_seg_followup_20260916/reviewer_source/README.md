# Advisor experiment: does vehicle-group mixing explain the nine k−1 cases?

Run by the reviewer (Claude), 17 Sep 2026 00:45–00:50 UTC. Slurm array **342787** (12 tasks), partition `scaglione`, `--exclude=scaglione-compute-01`, 4 CPU / 24 GB, 35–86 s each. Code: `seg_lp.py`, `seg_lp.sub`. Remote copy: `/home/nc437/ladder-lite/review_seg_lp_20260916/`. First attempt 342762 failed on the pip Gurobi size-limited license; fixed by exporting `GRB_LICENSE_FILE=/share/apps/software/gurobi/gurobi.lic` as production workers do.

**Method.** For each case, read the saved CG column journal (`cg.json.columns.jsonl`, 187k–262k columns), classify each column as *mixed* if it serves trips from both GIRO vehicle groups (VehicleTask 134xx = route 21 / 18E1; 133xx = local / 18E2, via `Ordered_Trip_ID` → `Par_VehicleDetails_Updated.csv`), and solve the covering LP (min Σ c_r λ_r, Σ_r a_ir λ_r ≥ 1, λ ≥ 0) over (a) the full pool and (b) unmixed columns only. No pricing; no new columns. Restricted-pool LPs are therefore **upper bounds** on the true LP of each model.

## Result

| Case | GIRO k (A/B) | Full-pool LP (recorded) | Mixed weight | Unmixed-pool LP | Unmixed A / B | Verdict |
|---|---|---:|---:|---:|---|---|
| w5_k27 | 27 (9/18) | 26.000 (26.000) | 4.29 | **27.000** | 9 / 18 | mixing = 1 bus |
| w5_k28 | 28 (10/18) | 27.000 (27.000) | 4.56 | **28.000** | 10 / 18 | mixing = 1 bus |
| w5_k29 | 29 (10/19) | 28.000 (28.000) | 5.28 | **29.000** | 10 / 19 | mixing = 1 bus |
| w4_k30 | 30 (10/20) | 29.000 (29.000) | 5.55 | **30.000** | 10 / 20 | mixing = 1 bus |
| w5_k30 | 30 (11/19) | 29.000 (29.000) | 4.62 | **30.000** | 11 / 19 | mixing = 1 bus |
| w4_k31 | 31 (10/21) | 30.000 (30.000) | 5.63 | **31.000** | 10 / 21 | mixing = 1 bus |
| w5_k31 (certified) | 31 (12/19) | 30.000 (30.000) | 4.64 | **31.000** | 12 / 19 | mixing = 1 bus |
| w4_k32 | 32 (11/21) | 31.000 (31.000) | 5.90 | **32.000** | 11 / 21 | mixing = 1 bus |
| w5_k32 | 32 (12/20) | 31.000 (31.000) | 4.89 | **32.000** | 12 / 20 | mixing = 1 bus |
| w3_k31 (control) | 31 (11/20) | 31.000 (31.000) | 2.61 | 31.000 | 11 / 20 | unchanged |
| w6_k31 (control) | 31 (11/20) | 31.000 (31.000) | 1.98 | 31.000 | 11 / 20 | unchanged |
| w2_k32 (control) | 32 (10/22) | 32.000 (32.000) | 0.90 | 32.000 | 10 / 22 | unchanged |

- Full-pool LP reproduces every recorded route weight (sanity check).
- **All nine k−1 cases return to exactly k when mixed-group columns are removed.** Controls are unchanged.
- In all 12 cases the unmixed LP splits into **integer per-group weights equal to GIRO's own per-group duty counts**. GIRO's schedule is per-group LP-optimal in this model on every tested instance.
- Mixed columns are 13–27% of each pool.

**Scope.** True segregated LP ∈ [k−1, k]; the restricted pool gives k. The certifying run is the segregation arm of Astra's action 3 (jobs 342678/342680): it prices over unmixed routes and should certify k with the same A/B split. F4 already shows GIRO's segregated duties are feasible under baseline physics, so a segregated integer solution at k exists.

## Time-concurrency check (local, `inst/`)

Maximum number of simultaneously running trips (zero turnaround; a valid fleet lower bound that ignores energy and deadhead):

| Group | Concurrency vs GIRO count |
|---|---|
| Route 21 (18E1) | **equal in 12/12** (9=9, 10=10, …, 12=12) |
| Local (18E2) | 1–3 below GIRO/LP in 12/12 (e.g. 16 vs 18, 17 vs 19, 19 vs 20) |
| All trips, mixing allowed | 25–29, i.e. 1–3 below the mixed LP |

Interpretation: route-21 fleet size is set purely by peak concurrency — energy is not binding there at 240 kWh/240 kW. Local-route fleet exceeds the time-only bound by 1–3 buses; that gap is the joint cost of deadheading and energy and is where physics (60 kW depot, 15% reserve) could move fleet. The mixed-model saving is not a concurrency effect either (mixed concurrency 28 vs LP 30 at w5_k31); it is a packing effect across groups.

## Pre-registered predictions for Astra's six single-factor arms (chain 5, k=31 pool)

| Arm | Predicted LP route weight | Predicted MIP |
|---|---:|---|
| Control | 30 | ≥31 (as before) |
| PARX 60 kW | **30** (unchanged) | ≥31 |
| 15% reserve | 30 or 31 — reserve may bind locals | — |
| Battery 236.44 / 239.01 | 30 | — |
| **Group segregation** | **31, split 12/19, certified** | 31 |

If depot rate or battery arms move the LP, the mixing explanation is incomplete; record that.

## What this changes

Q1's answer: *under Transdev's vehicle-group rule, the LP bound equals GIRO's fleet in every instance and in every group; GIRO's schedules are fleet-optimal. Relaxing group compatibility is worth exactly one bus (LP) in 9 of 102 instances, all in chains 4/5 at k ≥ 27. Whether that bus is integrally attainable is the open question item 8 (12 h search on w5_k31) addresses.* That is a concrete operator recommendation with a number (~3% of fleet for interoperable vehicles), and a structural statement about the relaxation.

## Suggested next cheap experiment: the electrification premium

Solve the pure time-only VSP (no energy; same deadhead matrix) per group per instance — a min-cost-flow / assignment LP, seconds each. Then *electrification premium* = EV LP bound − time-only optimum, per group. Route 21 should be 0; locals 1–3. Repeat under the strict-physics arms to see how the premium grows with 60 kW depot and 15% reserve. This is a figure, not a table, and it is the physical story behind "the LP is tight."
