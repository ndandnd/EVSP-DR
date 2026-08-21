> **TASK STATE 2026-08-21: complete.** Confirmed the primary-grid k2 optimum is 3/3/3, found arc-flow LP equals the set-partitioning LP on all nine cells (`D0026`), and audited all 36 per-route constraints (`D0030`). Do not start this brief fresh.

# Work order: independent arc-flow oracle (self-contained briefing)

**For a new agent with no prior context. Read §1–§3 before writing code.**

Date 2026-08-20. Operator: Nathan (`nc437`). **He owns the cluster; you never
submit cluster jobs.** You work locally and report executed output.

---

## 1. Project context

EVSP-DR solves an **electric vehicle scheduling problem (E-VSP)** for a real bus
network in Partille, Sweden. A commercial tool (GIRO) produced an industrial
schedule without modelling electricity prices. We are currently in **Goal 1:
convincing ourselves our own code is correct**, before using it for anything.

### 1.1 The model

Set partitioning over vehicle **routes** (a route = one bus's whole day: trips,
deadhead legs, charging stops):

- minimize `100000 · Σ_r x_r + charging_cost(x)`
- subject to `Σ_{r ∋ i} x_r = 1` for every trip `i`
- Bus fixed cost is exactly **100,000**; charging at the optimum is **$86–$333**
  on small instances.

It is solved by **column generation**. The pricing subproblem is an exact
shortest path over an **SOC-by-time expanded DAG**: explicit state of charge
(`--soc-step` kWh), explicit time blocks (`--block-min` minutes), station /
wait / charge / delayed-start-charge choices. Implemented in
`src/exact_pricer_expanded.py`. The restricted master is
`src/master_lp_scipy.py` (scipy → HiGHS). Physics for the baselines you must
match: **300 kWh battery, 300 kW charging, reserve 0, flat historical tariff.**

### 1.2 Instances

A "cell" is the union of *k* GIRO duties. GIRO used one bus per duty, so the
**target fleet equals k**. `k05_s2_c1` = 5 duties, selection replicate 2.
Instance CSVs are under `data/scale_ladder/instances/` and
`data/tariff_response/frozen_instances/`; staged copies used by runs live at
`data/scale_ladder_inputs/`. The `--csv` argument is **relative to `data/`**.

---

## 2. Repository

- GitHub `https://github.com/ndandnd/EVSP-DR`
- **Base your branch on `origin/records/ladder-lite-20260819-2969`.**
- **Create `cursor/arcflow-oracle-2969`.**
- **Do not touch** `peel-and-price`, `main`, or any other `cursor/*` branch.
  Three other agents are active on their own branches.

Read first: `STATUS_20260820.md`;
`analysis/scale_ladder/ll_20260820c/ladder_summary.csv`;
`analysis/scale_ladder/ll_20260820c/resolution_matrix.csv`;
`records/DECISION_LOG.csv`; `src/exact_pricer_expanded.py` (for how the SOC×time
network is built — **reuse that construction, do not invent a second one**).

---

## 3. Why this experiment exists — the question only you can answer

Our column generation is **exact and certified**: on 10 primary-grid cells the
pricer terminates with `min_rc = 0`, a reduced-cost certificate that no
improving route exists in the discretized route space. At k2 the certified LP is
**2.1818** against an industrial target of **2**.

But when we take that same certified pool and solve a one-shot MIP over it, we
get **4 buses, proven optimal for that pool**. So the pool provably contains no
2-bus and no 3-bus partition.

**We do not know whether 4 is the true integer optimum of the discretized model,
or whether the pool is simply missing columns.** Everything we are currently
doing about it — column diversification, pool unions, an experimental
branch-and-price — assumes the true optimum is 3 and the pool is deficient. That
assumption has never been tested.

**Your arc-flow oracle tests it.** By formulating the same discretized problem
directly as an arc-flow MIP over the SOC×time network — with no column
generation, no route enumeration, no pool — you compute the true integer optimum
of the model. Then:

- If the arc-flow integer optimum is **4**, our pool MIP was already optimal,
  there is **no pool-composition failure at k2**, and the whole 2→4 gap is
  discretization. Three other workstreams would be attacking a non-problem.
- If it is **3**, pool composition is real and those workstreams are correct.

Either answer is decisive. This is the highest-value open question in the
project.

### 3.1 A bound relationship you must not mistake for a bug

```
arcflow_LP  ≤  setpartitioning_LP  ≤  integer_optimum
```

The arc-flow LP relaxation is **weaker** than the set-partitioning LP — that is
the entire reason column generation is used for this problem class. So your LP
bound will come out **below 2.1818 at k02_s1, and that is correct.** Do not file
a bug, do not "fix" it, and do not conclude the master is wrong.

The comparisons that must agree are the **integer optimum**, the route
incidence, and the charging schedule.

---

## 4. The work order

### 4.1 Constraints

- New module(s) only, e.g. `src/arcflow_oracle.py`. **Do not modify**
  `src/exact_pricer_expanded.py`. Import its network construction; if that needs
  more than ~30 changed lines to expose, write an adapter in your module and say
  so explicitly.
- Budget **900 lines** excluding tests.
- Scope **k2, k3, k5 only**, on grids `15 kWh/10 min` and `1 kWh/5 min`.
  **Do not attempt k8 or above** — the network size makes it pointless.
- **No cluster submission.** Local only.
- MIP solver: `scipy.optimize.milp` (HiGHS) is available and license-free and is
  the default. Gurobi is available only on the cluster, so do not depend on it;
  if you support it, keep it strictly optional behind a flag.

### 4.2 Formulation

Build a flow model on the same SOC×time DAG the pricer builds:

1. **Arc variables** `f_a ≥ 0`, integer, one per network arc (deadhead, wait,
   charge, trip-service arcs).
2. **Flow conservation** at every SOC×time node.
3. **Trip coverage:** for every trip `i`, the total flow on arcs that service
   trip `i` equals exactly **1**.
4. **Fleet count:** flow out of the source (equivalently into the sink) equals
   the number of vehicles; that is the primary objective term at 100,000 each.
5. **Charging cost** on charge arcs, priced identically to the pricer's cost
   model — same tariff lookup, same SOC-flooring convention. If you cannot make
   them identical, say so and state the difference precisely rather than
   papering over it.
6. Objective: `100000 · vehicles + charging_cost`, matching §1.1 exactly.

Then also solve a **fleet-only** variant (`minimize vehicles`, constant charging)
so it can be compared against our three-phase lexicographic results.

### 4.3 Validation gates — build in order, stop at the first failure

- **G1 — network identity.** Your network must have the same node and arc counts
  as the pricer's for the same instance, grid, and physics. Print both and assert
  equality. If you cannot reuse the pricer's construction, this gate is how you
  prove your reimplementation matches.
- **G2 — route feasibility round trip.** Take a known-feasible route from an
  existing CG column journal (`<out>.json.columns.jsonl` in the ladder results),
  map it onto your arcs, and verify it is feasible in your model at the same
  cost. Any known-good route that your model rejects is a modelling error in
  your code.
- **G3 — LP ordering.** Assert `arcflow_LP ≤ setpartitioning_LP` for every cell
  (§3.1). A violation means your relaxation is wrong, not that the master is.
- **G4 — integer upper bound.** Your integer solution must cover every trip
  exactly once; reuse the overcoverage audit in `src/run_exact_pool_mip.py`
  (it does **not** import gurobipy at module scope, so it is importable without
  a licence).
- **G5 — agreement or disagreement, explicitly.** For each cell, print the
  arc-flow integer optimum beside our pool-MIP result:
  `k02_s1 = 4, k02_s2 = 4, k02_s3 = 7, k03_s1 = 5, k03_s2 = 10, k03_s3 = 4,
  k05_s1 = 11, k05_s2 = 6` (primary grid, from RAW pools). Arc-flow must be
  **≤** each of these, since the pool is a restriction of the model. **An
  arc-flow optimum strictly greater than a pool-MIP result is a bug in your
  code** — flag it loudly rather than reporting it as a finding.

### 4.4 Deliverable

One table, k2/k3/k5 × {15 kWh/10 min, 1 kWh/5 min}:

`instance | grid | arcflow_LP | setpart_LP (given) | arcflow_integer | pool_MIP (given) | GIRO target | solve_s | status`

Then answer, in one paragraph each:

1. **At k2 on the primary grid, is the true discretized integer optimum 3 or 4?**
   State it plainly. This single number redirects three other workstreams.
2. **Where arc-flow and the pool MIP disagree, by how much?** That difference is
   the exact size of the pool-composition failure, measured rather than inferred.

If you cannot pass G1 or G2, report that and stop. A correct answer to "our two
network constructions disagree" is more valuable than an integer optimum from a
network we cannot trust.

---

## 5. Reporting conventions

1. **Report executed output, never readiness.** "Implemented", "tests pass",
   "ready" are not results.
2. **If a gate fails, say so plainly and stop.**
3. Every exactness claim must name what it is exact **for**: the discretized
   model at a stated grid, never the real-world problem. The discretization both
   removes routes and overstates costs, so a bound on the discretized model is
   not a bound on reality.
4. **Append to `records/BUG_LOG.csv` and `records/DECISION_LOG.csv`** for
   anything you find or decide. Append-only; a correction is a new row.
