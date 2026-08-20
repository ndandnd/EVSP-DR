# Work order: continuous / event-based fixed-duty charging optimizer

**For a new agent with no prior context. Read §1–§3 before writing code.**

Date 2026-08-20. Operator: Nathan (`nc437`). **He owns the cluster; you never
submit cluster jobs.** You work locally and report executed output.

**This is the task that leads to the project's publishable result.** Everything
else in flight is Goal-1 self-validation. This is Goal 2.

---

## 1. Project context

EVSP-DR solves an **electric vehicle scheduling problem (E-VSP)** for a real bus
network in Partille, Sweden. A commercial tool (GIRO) produced the industrial
schedule **without modelling time- or location-varying electricity prices.**

- **Goal 1 (largely done, other agents):** convince ourselves our
  column-generation code is correct.
- **Goal 2 (yours):** given time-varying tariffs, how much value comes from
  **re-timing charging** versus **re-routing vehicles**?

That decomposition is the contribution:

| tier | what varies | what it measures |
|---|---|---|
| **Observed** | nothing — GIRO's recorded schedule, cross-priced under each tariff | the baseline cost of a price-blind schedule |
| **Fixed sequence** | charging only; GIRO's trip sequences held fixed | **value of re-timing charging alone** |
| **Joint** | routes and charging together | **incremental value of re-routing** |

**Your job is the middle tier**, which is also the one that works even if
route-level recovery stays hard. That property is why it is the priority.

---

## 2. Repository and what already exists

- GitHub `https://github.com/ndandnd/EVSP-DR`
- **Base your branch on `origin/records/ladder-lite-20260819-2969`.**
- **Create `cursor/fixed-duty-charging-2969`.**
- **Do not touch** `peel-and-price`, `main`, or any other `cursor/*` branch.
  Three other agents are active on their own branches.

**Do not start from scratch.** These exist and you must build on them:

| path | what it is |
|---|---|
| `src/fixed_duty_expanded_optimizer.py` | existing fixed-duty optimizer on the discrete SOC×time grid |
| `src/audit_fixed_duty_grid_transitions.py` | grid-transition auditor; explains why some duties are unrepresentable |
| `src/expanded_path_realization.py` | maps grid paths to physical charging schedules |
| `src/tariff_response_core.py`, `src/prepare_tariff_fixed_duty_seed.py` | tariff-response scaffolding |
| `src/rerealize_routes.py` | re-realizes routes under a model |
| `data/hourly_prices_flat.csv` | the flat historical tariff |

Read `STATUS_20260820.md` and `records/DECISION_LOG.csv` first. Then read the
existing fixed-duty optimizer carefully and report what it already does before
proposing changes.

---

## 3. Frozen physics — do not renegotiate these

Decided by the operator on 2026-08-20. Use exactly these:

| parameter | value | note |
|---|---|---|
| battery | **240 kWh** | source data supports ~239–240, not the historical 300 |
| charger power | **240 kW** | 1C; 0→100% in one hour |
| reserve SOC | **0** | revisit later |
| charge-start cost | **$5** | a tie-break that also discourages implausibly many charge events |
| charger capacity | **unlimited** | deferred; see the honesty requirement below |
| deadhead | **zone-centroid** | coarse; trips sharing a zone may appear to teleport |
| all buses start | **full** | |

Two things to state explicitly in every output, never hide:

1. Real charging **tapers above ~80% SOC** (constant-current then
   constant-voltage). Constant power to 100% is mildly optimistic. This is the
   standard linear idealization in the E-VSP literature; say so.
2. **Unlimited chargers means no peak-shaving claim is licensed yet.** You may
   report energy cost and load profiles; you may **not** report peak-shaving
   value. Emit `charger_concurrency_max` per solution so the eventual capacity
   constraint has a measured starting point.

Note `240 kW × 10 min = 40 kWh` and `× 5 min = 20 kWh`, so SOC steps of 10 or
2.5 kWh are commensurate with the time lattice. The old primary grid
(300 kW, 15 kWh steps, 10 min) floored 50 kWh down to 45 — **discarding 10% of
every charge**, which distorts exactly the cost you are trying to measure.

---

## 4. Why continuous / event-based, not a finer grid

Measured facts from the Goal-1 work that motivate your design:

- Of 22 selected GIRO duty partitions, **0** are representable on the primary
  15 kWh / 10 min grid; only 9 of 40 duties fit.
- Duty **13411** is continuously feasible but unrepresentable on **all five**
  tested grids, down to 1 kWh / 5 min. Its failures are always *transitions*
  (106→119, 119→132, 158→167), never SOC states.
- Refining SOC alone **plateaus**: `k02_s2` route weight went 2.1875 → 2.1667 →
  2.1538 → 2.1538 across 15 / 5 / 2.5 / 1.0 kWh. Halving the **time block**
  10 → 5 took it straight to 2.0000. **Time discretization binds; SOC
  saturates.** See `analysis/scale_ladder/ll_20260820c/RESOLUTION_ANSWER_20260820.md`.

So indefinitely refining the grid is the wrong response. A **continuous-time or
event-based** model for a *fixed* duty sequence is tractable — the sequence is
given, so only the charging decisions are free — and it sidesteps
representability entirely.

---

## 5. The work order

### 5.1 Constraints

- New module(s) preferred, e.g. `src/fixed_duty_continuous_optimizer.py`.
  Extend `src/fixed_duty_expanded_optimizer.py` only additively; keep its
  existing grid path **bit-identical** when your new mode is not requested.
- Budget **1000 lines** excluding tests.
- **No cluster submission.** Local only. Validate on individual duties and small
  unions (k2/k3/k5) plus duty `13411` specifically.
- Solver: `scipy.optimize.linprog` / `milp` (HiGHS) — license-free and already
  used by `src/master_lp_scipy.py`. Gurobi is cluster-only; do not depend on it.

### 5.2 What to build

For a **fixed ordered trip sequence** (one duty), decide charging to minimize
energy cost, with:

1. **Continuous or event-based time** — charging start times and durations are
   continuous (or on a fine event set derived from actual trip/arrival times),
   not lattice points.
2. **Delayed-start charging and waiting** — a bus at a station may wait, then
   charge later. This is essential: charge-on-arrival makes load-shifting
   impossible, and it was the structural blocker that prevented this project
   from having a demand-response story at all.
3. **Time-varying tariffs** — piecewise-constant hourly prices; the optimum will
   place charging in cheap windows subject to SOC feasibility.
4. **Terminal SOC policy** — implement as an explicit, switchable option
   (`free`, `>= reserve`, `>= start`, or priced terminal energy). Do **not** bake
   one in; the choice is unresolved and materially affects reported savings.
5. **$5 per charge start**, as a fixed charge per charging event.
6. **Exact physical replay** — output a schedule with concrete start times,
   durations, delivered kWh, and per-event cost, and validate that replaying it
   never violates capacity, never goes below reserve, and reproduces the
   reported cost. Reuse `src/expanded_path_realization.py` conventions.

For piecewise-constant prices with continuous charging this is a **linear
program** (or a small MIP once per-event fixed costs are included). Do not reach
for anything more elaborate.

### 5.3 Validation gates — build in order, stop at the first failure

- **G1 — lattice reproduction.** Constrain your continuous model to the grid's
  lattice (start times on 10-minute blocks, energy in 15 kWh steps, 300 kWh /
  300 kW) and it must reproduce `fixed_duty_expanded_optimizer.py`'s cost for
  the same duty **exactly**. This is the gate that proves the two models are the
  same model. Without it nothing downstream is trustworthy.
- **G2 — relaxation ordering.** Unconstrained, your continuous cost must be
  **≤** the grid cost for the same duty and tariff, because continuous relaxes
  the lattice. A continuous cost *above* the grid cost is a bug in your code.
- **G3 — duty 13411.** It is continuously feasible but unrepresentable on all
  five tested grids. Your model **must** find a feasible charging plan for it.
  If it cannot, either your model is wrong or the "continuously feasible" claim
  in the record is wrong — investigate and say which. This is the single
  sharpest test available.
- **G4 — physical replay.** Every emitted schedule replays without violating
  capacity or reserve, and the replayed cost equals the reported cost to 1e-6.
- **G5 — flat-tariff invariance.** Under the flat historical tariff, delayed
  charging must give **no** advantage over charge-on-arrival except through the
  $5 start term. If it does, your tariff handling is wrong.

### 5.4 Deliverable

For every duty in the k2/k3/k5 cells plus duty 13411, under the flat tariff and
at least one synthetic time-varying tariff:

`duty | tariff | grid_cost | continuous_cost | saving_% | charge_events | delayed_starts | peak_kW | charger_concurrency_max | terminal_SOC_policy | replay_ok`

Then answer in one paragraph each:

1. **How much does re-timing charging alone save on GIRO's own duty sequences,
   under a time-varying tariff?** This is the project's first real Goal-2 number.
2. **Does the continuous model admit duty 13411, and what does that imply about
   the 15 kWh / 10 min grid?**

State clearly that these are **fixed-sequence** savings — the value of re-timing
only. Re-routing value requires the joint tier and is not yours.

---

## 6. Reporting conventions

1. **Report executed output, never readiness.**
2. **If a gate fails, say so plainly and stop.**
3. **No peak-shaving claims** while chargers are unlimited (§3).
4. Name what every number is conditional on: tariff, terminal-SOC policy,
   physics, and whether it is grid or continuous.
5. **Append to `records/BUG_LOG.csv` and `records/DECISION_LOG.csv`**;
   append-only, a correction is a new row.
6. Synthetic tariffs are **software demonstrations, not research results**, until
   a real tariff is frozen. Label them as such.
