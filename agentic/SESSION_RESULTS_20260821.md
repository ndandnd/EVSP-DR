# Agent results, 2026-08-21 — verbatim, pending curation

Reported by the six Cursor agents. **Not yet folded into `STATUS_20260821.md`** —
that is curator deliverable 5. Recorded here so the numbers survive independently
of any chat session.

## Agent B — adversarial correctness (`cursor/branch-and-price-experiment-2969-1451`)

Seed `20260821`.

| check | agreement |
|---|---|
| arbitrary-dual pricing vs exhaustive path enumeration | **7,680 / 7,680** |
| LP bounds, all methods | 240 / 240 |
| brute force vs branch-and-price vs arc-flow integer fleet | 240 / 240 |
| CG final-pool integer fleet | **216 / 240** |

All 24 disagreements are CG pool-composition failures: the certified LP is
correct, the pool integer optimum needs one extra bus. Each has an irreducible
reproducer; **smallest is 5 trips, 2 stations.** 32 dual samples per network
covering negative components, near and exact ties, SOC steps 0.5/1/2, one and two
stations, varied reachability, delayed charging, flat and time-varying tariffs.
**Maximum pricing error 3.50e-10.**

This establishes the pricing DP as an exact oracle **over its domain**, not merely
along the CG trajectory.

## Agent C — public benchmark and reproducible family (`cursor/arcflow-oracle-2969-3a99`)

2017 Leuven data is not public; selected the **Utrecht qlink 8 successor**
benchmark. Converted result: **LP fleet 10.2273, proven fleet 11 vs published
10 — +1 bus / +10%.** Objectives are documented as not comparable.

Public synthetic family, seed `20260821`: fine grid fleet 2, coarse grid fleet 3,
pair-limited pool 3.

- `analysis/public_benchmark_evsp_20260821/README.md`
- `analysis/public_synthetic_evsp_20260821/README.md`

Also added `--g-kwh`, `--charge-kw`, `--min-soc-frac` with commensurate-grid
validation. Legacy regression held: LP **2.1875**, integer **3**, identical
payload hashes. At 240/240 with 10/10: LP **2.4**, proven integer fleet **3**.

## Agent D — full-schedule fixed-duty charging (`cursor/fixed-duty-charging-2969-b22e`)

All percentages on one fixed **30-duty intersection** feasible across every
tariff, terminal policy and arm.

| tariff / terminal | uncapped timing | capped timing |
|---|---:|---:|
| flat / ≥reserve | **0.000%** | **0.000%** |
| flat / ≥initial | **0.000%** | **0.000%** |
| peak12 / ≥reserve | 2.208% | 2.015% |
| **two-peak / ≥reserve** | **4.720%** | **4.755%** |
| peak12 / ≥initial | 10.030% | 9.794% |
| two-peak / ≥initial | 30.565% | 31.487% |

The flat rows at exactly 0.000% are the null control.

Infeasibility, reported separately: all-40 ≥reserve — 0 uncapped, 1 capped;
all-40 ≥initial — 7 uncapped, 10 capped; prior k5 observation 3/15
(`13302`, `13410`, `13304`).

**Unconstrained charger demand** on a shared service-day timeline —
two-peak/≥reserve optimized: 18–20 simultaneous buses fleet-wide, 6 at one
station-time, **4.32–4.80 MW**. Two-peak/≥initial: **30 buses at `PARX_1`,
7.2 MW.** These are demand measurements, **not** capacity-feasible schedules,
and license **no peak-shaving claim**.

Earlier matched-set correction (15/12 duties) that closed `B0030`:

| configuration | duties | uncapped | capped | cap cost | prefix bound | achieved/bound |
|---|---:|---:|---:|---:|---:|---:|
| peak12 / ≥reserve | 15 | 7.795 (1.509%) | 6.428 (1.212%) | 13.782 | 32.539 (6.299%) | 24.0% |
| peak12 / ≥initial | 12 | 40.412 (7.518%) | 40.412 (7.395%) | 8.922 | 77.435 (14.406%) | 52.2% |
| two-peak / ≥reserve | 15 | 27.849 (5.328%) | 28.834 (5.447%) | 6.648 | 40.225 (7.695%) | **69.2%** |
| two-peak / ≥initial | 12 | 181.691 (26.447%) | 193.489 (27.427%) | 18.483 | 211.646 (30.807%) | 85.8% |

Main report: `analysis/fixed_duty_continuous_20260820/K40_FIXED_DUTY_RESULTS.md`

## Agent E — event-based pricer v2 (`cursor/event-based-pricer-2969`, `2e6cb9b`)

- `k02_s3`: LP **2.0**, 557 s, **671 MiB**, 15,882 / 15,882 routes replayed
- `k05_s2`: LP **5.0**, 1,013 s, **1,069 MiB**, 16,253 / 16,253 replayed
- **96.01% memory reduction** versus explicit arcs, via packed lazy arcs
- Uniform-mode scientific identity preserved; timing and RSS excluded
- Report: `analysis/event_based_pricer_v2_20260821/REPORT.md`

Both performance gates passed. Supersedes the earlier v1 result where G2 was
incomplete, duty 13411 was representable only at 2.5/10, 1/10 and 1/5, G5 failed
for 1,671 of 1,710 columns, and k5 reached 22.2M arcs / 26.8 GiB. The
`sink_predecessor_route_batch` is an **enrichment heuristic** affecting
convergence and pool quality, not the reduced-cost certificate.

## Agent B — MIP restart audit

`--progress-dir` is **observational only.** It restores neither incumbent nor
Gurobi tree, bound or node state. Interrupted run: 20 nodes, incumbent 35, bound
6. Relaunch: 19 nodes, incumbent 35, bound 6 — rebuilt the same greedy MIP start
and repeated from root. Reusing a prior progress directory is explicitly
rejected. **Therefore large pool MIPs still need the protected partition.**

Report: `analysis/mip_progress_audit_20260821/REPORT.md`

## Agent F — records curation

§9 of `STATUS_20260821.md` audited into a self-guaranteeing reading list; the
external handoff correctly tagged. Commit `ba62371`, PR #38. **Still open:**
ID registry, old→canonical mapping, `RESULTS_LOG.csv`, the R14/R16/R3/R11/R12
corrections, and validation of Agent A's 18 duty-union instances.

## Agent A — instances

All 18 duty-union CSVs committed at ladder commit `72c7bf4`, marked
**producer-only, pending curator validation.** SyntheticRandom kept separate.
Authoritative ledgers restored unchanged; findings moved to
`records/inbox/cursor-ladder-lite-20260819-2969.md` as `LOCAL-*`.
