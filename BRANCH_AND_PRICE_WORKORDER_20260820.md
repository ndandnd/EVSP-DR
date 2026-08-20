# Work order: exact branch-and-price experiment (self-contained briefing)

**For a new agent with no prior context. Read all of §1–§4 before writing code.**

Date: 2026-08-20. Author: Claude (research manager for this project).
Operator: Nathan (`nc437`) — **he owns the cluster; you never submit cluster
jobs.** You work locally and report executed output.

---

## 1. Project context

### 1.1 What the project is

EVSP-DR is an **electric vehicle scheduling problem (E-VSP)** solver for a real
bus network in Partille, Sweden. A commercial tool (GIRO) produced an industrial
schedule for this network **without** modelling time- or location-varying
electricity prices. The project has two goals:

- **Goal 1 (current focus): algorithmic trust.** Can our column-generation
  pipeline reproduce or improve on GIRO's fleet quality? We are not claiming
  novelty here — GIRO will not share code, so we must convince ourselves our own
  code is correct before we use it for anything.
- **Goal 2 (the actual contribution, not started): demand response.** Given
  time-varying tariffs, how much value comes from re-timing charging versus
  re-routing vehicles?

### 1.2 The optimization model

Set partitioning over vehicle **routes** (a route = one bus's whole day: a
sequence of trips, deadhead legs, and charging stops):

- minimize `100000 · Σ_r x_r  +  charging_cost(x)`
- subject to `Σ_{r ∋ i} x_r = 1` for every trip `i` (partition, equality)
- Bus fixed cost is exactly **100,000**. Charging cost at the LP optimum is
  **$86–$2,469**. That ratio matters — see §3.4.

**Pricing subproblem:** shortest path over an **SOC-by-time expanded directed
acyclic graph**. Explicit SOC states (`--soc-step` kWh), explicit time blocks
(`--block-min` minutes), station/wait/charge choices, delayed-start charging.
No heuristic label cap, no dominance pruning — it is exact for the named
discretized route space. Implemented in `src/exact_pricer_expanded.py`.

**Physics currently in use for the baselines you must reproduce:** 300 kWh
battery, 300 kW charging, reserve 0, flat historical tariff. (The project is
migrating to 240 kWh / 240 kW, but **the baselines in §3 are at 300/300** — use
those.)

### 1.3 Instances

Each "cell" is the union of *k* GIRO duties. GIRO used one bus per duty, so the
**target fleet equals k**. Naming: `k05_s2_c1` = 5 duties, selection replicate 2,
CG replicate 1.

Instance CSVs live at `data/scale_ladder/instances/` and
`data/tariff_response/frozen_instances/`, and are staged for runs into
`data/scale_ladder_inputs/k{NN}_s{N}_c1_Practice_Custom_DutyUnion_k{NN}_r{N}.csv`.
The `--csv` argument is **relative to `data/`**.

---

## 2. Repository

- GitHub: `https://github.com/ndandnd/EVSP-DR`
- **Base your branch on `origin/records/ladder-lite-20260819-2969`.** It contains
  the status document, the records CSVs, and the result data you need.
- **Create `cursor/branch-and-price-experiment-2969`.** This is an isolated
  experiment and may be deleted wholesale; that is an accepted outcome.
- **Do not touch** `peel-and-price`, `main`, or any other `cursor/*` branch.
  Another agent is actively working on `cursor/ladder-lite-20260819-2969`.

Files worth reading before you start:

| path | why |
|---|---|
| `STATUS_20260820.md` | current state, 9 findings, what remains |
| `analysis/scale_ladder/ll_20260820c/ladder_summary.csv` | the certified LP values you must reproduce |
| `analysis/scale_ladder/ll_20260820c/resolution_matrix.csv` | LP by instance × grid resolution |
| `analysis/scale_ladder/ll_20260820c/RESOLUTION_ANSWER_20260820.md` | why time discretization binds, not SOC |
| `records/DECISION_LOG.csv`, `records/BUG_LOG.csv` | append-only project record; add rows for anything you find |
| `src/exact_pricer_expanded.py` | the exact pricer (do not modify — see §4) |
| `src/master_lp_scipy.py` | the restricted master LP (reuse it) |
| `src/run_exact_pool_mip.py` | current one-shot pool MIP, for comparison |

---

## 3. What we already know — you must not contradict these without evidence

### 3.1 The pricer is exact and certified

On 10 primary-grid cells and 28 of 30 finer-grid cells, column generation
terminates with `min_rc = 0`: a **reduced-cost certificate** that no improving
route exists in the named route space. This is proof, not a claim. Your root
node must agree with it (gate G1).

### 3.2 The model is validated against the industrial solution

The `MIP_KNOWN` arm injects GIRO's validated partition into the column pool.
Result: **17 of 17 cells return exactly the target fleet, `fleet_proven = true`,
`bound = target`, gap 0.0**, for k2, k3, k5, k8, k13 and k20. So our cost model,
physics, and feasibility conditions do **not** wrongly reject the real schedule.

**This is a positive control, NOT algorithmic recovery.** RAW and KNOWN arms must
never share a row or a sentence. Conflating them is the single most common error
in this project's history.

### 3.3 The dominant failure is integral pool composition, not pricing

The `MIP_RAW` arm solves a one-shot MIP over the final CG pool. At k2, from a
**pricing-certified** 2,272-column pool with a certified LP of 2.1818:

| RAW cell | integer result | target | status |
|---|---:|---:|---|
| k02 s1 / s2 / s3 | **4 / 4 / 7** | 2 | OPTIMAL, proven for that pool |
| k03 s1 / s2 / s3 | **5 / 10 / 4** | 3 | OPTIMAL, proven for that pool |
| k05 s1 / s2 | **11 / 6** | 5 | OPTIMAL, proven for that pool |
| k08 s1 / s2 / s3 | 58 / 29 / 27 | 8 | TIME_LIMIT |
| k13 s1 / s2 / s3 | 100 / 61 / 145 | 13 | TIME_LIMIT |

So at k2 the failure decomposes as: pricing censoring **0 buses**, route-space
discretization **~1 bus** (2 → 2.18 LP), **pool integral composition ~2 buses**
(2.18 → 4). The pool provably contains no 2-bus and no 3-bus partition.

**This is why branch-and-price is being tried.** Converging the LP does not
populate the pool with the complementary columns an integer solution needs.

### 3.4 `route_weight` semantics — get this right in all output

`route_weight = Σ_r x_r`, a fractional bus count from the LP relaxation. It is
**"combined-cost-master route weight"**, not a fleet LP lower bound, because the
master minimizes bus cost *plus* charging. However: bus cost is 100,000 and
charging at the optimum is ≤ $2,469, so for the combined-cost optimum `w` and any
minimum-fleet solution, `w − W* ≤ C/100000 ≈ 0.03` buses. So a **certified**
route weight brackets the fleet LP bound within ~0.03. An **uncertified** route
weight is only an upper bound on the LP value and gives **no** fleet bound.

Also: a certified bound is a bound **for the discretized model only**. The grid
both removes routes and overstates costs (SOC flooring is conservative). "LP
2.1818 proves 3 buses are needed" is false. "On the 15/10 grid 3 buses are
needed, and GIRO's real 2-bus schedule shows the grid discards a bus" is true.

### 3.5 Time discretization binds; SOC saturates

All three k2 cells reach route weight **exactly 2.0000, certified, at
1 kWh / 5 min**. Controlled comparisons at fixed 1.0 kWh SOC changing only
`block_min` 10 → 5: `k02_s2` 2.1538 → 2.0000, `k02_s3` 2.1400 → 2.0000. SOC
refinement alone plateaus (`k02_s2`: 2.1875 / 2.1667 / 2.1538 / 2.1538 across
15 / 5 / 2.5 / 1.0 kWh). Do not attribute the gap to SOC granularity.

---

## 4. The work order

### 4.1 Why this experiment has a known answer

Our data predicts the outcome, which makes this a falsifiable test rather than
an open-ended build:

- At k2 on the **primary grid**, certified LP = 2.1818, so the discretized
  integer optimum is **≥ 3**. The pool MIP says 4. B&P must prove 3 or 4.
- At k2 on **1 kWh / 5 min**, certified LP = **2.0000**. B&P should **prove 2** —
  which would be end-to-end RAW recovery of the industrial fleet with a proof,
  the central Goal-1 result.

### 4.2 Constraints

- **New module(s) only**, e.g. `src/branch_and_price.py`. Do **not** modify
  `src/exact_pricer_expanded.py` except, if unavoidable, to expose an importable
  pricing entry point. If that needs more than ~30 changed lines, write an
  adapter in your new module instead and say so explicitly.
- Budget **1200 lines** excluding tests.
- Scope: **k2, k3, k5 only.** Do not attempt k8 or above.
- **No cluster submission.** Local runs only. Nathan operates the cluster.
- Reuse `src/master_lp_scipy.py` for the restricted master.

### 4.3 Design — follow this; do not substitute another branching scheme

1. **Master.** Set partitioning as in §1.2, equality rows, same objective.
2. **Branching: Ryan–Foster.** From the fractional LP compute
   `α_ij = Σ_{r : i,j ∈ r} x_r` over trip pairs. If every `α_ij ∈ {0,1}` the
   solution is integral — assert that and use it as a test (gate G4). Otherwise
   branch on the pair with `α_ij` nearest 0.5 into `together(i,j)` /
   `apart(i,j)`.
3. **Pricing under branching — the correctness crux.** Our pricer is an exact
   shortest path over a time-expanded DAG, which makes both constraints
   decompose exactly:
   - `apart(i,j)`: any route avoiding *both together* must omit `i` or omit `j`.
     Solve twice — once with node `i` deleted, once with `j` deleted — take the
     better. **Exact.**
   - `together(i,j)`: a route contains both or neither. Solve twice — once with
     both forced in, once with both deleted — take the better. Forcing is easy
     because the DAG is time-ordered, so concatenate shortest paths
     `source→i`, `i→j`, `j→sink`. **Exact.**
   - With `k` active constraints on the root-to-node path this gives `2^k`
     subproblems. Enumerate them exactly. **Cap `k` at 8 by default**
     (configurable). Beyond the cap, stop expanding and record the node as an
     open frontier node. **Never switch to inexact pricing** — that silently
     invalidates the lower bound, which would make the whole experiment
     worthless.
4. **Search.** Depth-first with diving to find integer solutions early. Maintain
   a global lower bound as the minimum LP bound over open frontier nodes.
   Configurable node limit and wall limit.
5. **Output JSON.** `best_integer_fleet`, `best_integer_cost`,
   `global_lower_bound`, `gap`, `proven_optimal` (true **only** if the tree
   closed with no frontier nodes and no depth-capped nodes), `nodes_explored`,
   `nodes_depth_capped`, `pricing_solves`, `wall_s`, plus the provenance fields
   the existing pricer writes (instance sha256, grid, physics, commit).

### 4.4 Validation gates — build these first, in order, and stop at the first failure

- **G1 — root LP equality.** The root node's LP bound must match our certified CG
  values to 1e-6 on the primary 15 kWh / 10 min grid at 300 kWh / 300 kW:

  | cell | certified LP |
  |---|---:|
  | k02_s1 | 2.1818 |
  | k02_s2 | 2.1875 |
  | k02_s3 | 2.2747 |
  | k03_s1 | 3.1818 |
  | k03_s2 | 3.4047 |
  | k03_s3 | 3.0000 |
  | k05_s1 | 5.3237 |
  | k05_s2 | 5.0000 |
  | k05_s3 | 5.0000 |

  Full-precision values are in `ladder_summary.csv`. The baseline command was:

  ```
  python -u src/exact_pricer_expanded.py --csv <rel.csv> \
    --prices_csv hourly_prices_flat.csv --g-kwh 300 --charge-kw 300 \
    --min-soc-frac 0 --soc-step 15.0 --block-min 10 \
    --master-sense partition --initial-pool singletons \
    --checkpoint-every 25 --out <out.json>
  ```

  **If G1 fails, your master or pricer adapter is wrong and nothing downstream
  means anything. Report and stop.**
- **G2 — bound monotonicity.** Every child node's LP bound ≥ its parent's.
  Assert at runtime, not only in tests.
- **G3 — pricing exactness under branching.** On a deliberately tiny grid,
  brute-force enumerate all feasible routes and verify constrained pricing
  returns the true minimum reduced cost under `apart` and under `together`.
  **This is the gate that catches the subtle errors.** Do not skip it.
- **G4 — integrality certificate.** When all `α_ij ∈ {0,1}`, verify the solution
  is genuinely integral.
- **G5 — upper-bound sanity.** Any integer solution must be ≥ the certified root
  LP and must cover every trip exactly once. Reuse the existing overcoverage
  audit in `run_exact_pool_mip.py`.

### 4.5 Deliverable

One table, k2/k3/k5 × {15 kWh/10 min, 1 kWh/5 min}:

`instance | grid | root LP | best integer | global LB | gap | proven? | nodes | depth-capped | wall s`

Then one paragraph each on:

1. **On the primary grid at k2, is the proven integer optimum 3 or 4?** If 3, our
   pool MIP was missing columns and pool composition is fixable. If 4, then 4 is
   genuinely the grid optimum and the discretization costs two vehicles, not one.
2. **At 1 kWh / 5 min, does B&P prove 2 vehicles for all three k2 instances?**
   A yes is end-to-end RAW recovery of the industrial fleet, with a proof.

---

## 5. Reporting conventions for this project

1. **Report executed output, never readiness.** "Packaged", "pushed", "tests
   pass", "ready" are not results. The reportable states are: code written →
   gates passing with pasted output → table produced.
2. **If a gate fails**, say so plainly and stop. A failed G1 reported honestly is more, say so plainly and stop. A failed G1 reported honestly is more
3. **RAW and KNOWN never share a row.** KNOWN is a plumbing positive control.
4. **`route_weight` is labelled "combined-cost-master route weight"** unless
   accompanied by the §3.4 bracket.
5. **Append to `records/BUG_LOG.csv` and `records/DECISION_LOG.csv`** for
   anything you find or decide. Append-only — never edit a row in place; a
   correction is a new row.
6. Every claim of exactness must name what it is exact *for*: the discretized
   route space at a stated grid, not the real-world problem.
