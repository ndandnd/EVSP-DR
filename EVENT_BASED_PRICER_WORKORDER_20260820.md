# Work order: event-based time discretization in the pricer

**For a new agent. Read §1 (required reading) before writing any code.**

Date 2026-08-20. Operator: Nathan (`nc437`). **He owns the cluster; you never
submit cluster jobs.** Work locally, report executed output.

Branch: base on `origin/records/ladder-lite-20260819-2969`, create
**`cursor/event-based-pricer-2969`**. Do **not** touch `peel-and-price`, `main`,
or any other `cursor/*` branch — four other agents are active.

---

## 1. Required reading, in order

1. `STATUS_20260820.md` — project state, the nine findings, terminology.
2. `analysis/scale_ladder/ll_20260820c/RESOLUTION_ANSWER_20260820.md` — **the
   direct motivation for your task.**
3. `records/DECISION_LOG.csv` — entries `D0019`, `D0023`–`D0030`.
4. `src/exact_pricer_expanded.py` — the pricer you are changing.
5. `src/fixed_duty_expanded_optimizer.py` and
   `src/fixed_duty_continuous_optimizer.py` — existing continuous/event
   machinery. **Reuse it. Do not write a third time model.**

Do not proceed until you can state, in your own words, why time discretization
binds and SOC saturates.

---

## 2. The problem you are solving

The pricer discretizes time into uniform blocks (`--block-min`) and state of
charge into uniform steps (`--soc-step`). Measured consequences:

- Of 22 selected industrial duty partitions, **0** are representable on the
  15 kWh / 10 min grid; only 9 of 40 duties fit.
- Duty **13411** is continuously feasible but unrepresentable on **all five**
  tested grids down to 1 kWh / 5 min. Its failures are always *transitions*
  (`106→119`, `119→132`, `158→167`), never SOC states.
- Refining SOC alone **plateaus**: `k02_s2` route weight went 2.1875 → 2.1667 →
  2.1538 → 2.1538 across 15 / 5 / 2.5 / 1.0 kWh. Halving the **time block**
  10 → 5 took it to exactly 2.0000, and branch-and-price then **proved** a
  2-bus optimum, matching the industrial fleet.
- Two independent exact methods (branch-and-price and a direct arc-flow MIP)
  agree the 15/10 optimum is **3** against an industrial fleet of **2**. The
  coarse grid provably costs exactly one vehicle.

So the route space is the binding constraint, and the responsible axis is
**time**. Uniform refinement works but is expensive, and `--block-min` is
declared `type=int`, so the axis is only `{10, 5, 2, 1}` minutes.

**Your task: replace the uniform time lattice with the event set the instance
implies**, so industrial transitions are representable **by construction**
rather than by brute-force refinement.

---

## 3. Design

Build the time lattice from the instance, not from a parameter:

1. **Event times** = trip start times, trip end times, station arrival times
   implied by each trip→station deadhead, and the trip departure deadlines
   implied by each station→trip deadhead.
2. **Charging breakpoints** = for each station, the times at which a charge run
   could start or end such that it begins no earlier than an arrival and ends no
   later than a departure deadline. Charge durations become continuous or take
   values on this induced set, rather than whole multiples of `block_min`.
3. Keep SOC handling exactly as it is for the first iteration. SOC saturates
   below ~2.5 kWh, so it is not where the value is, and changing two axes at
   once makes failures undiagnosable.
4. Expose it as an **opt-in flag**, e.g. `--time-model {uniform,event}`, default
   `uniform`, using `argparse.SUPPRESS` so the default path is untouched.

---

## 4. Validation gates — build in order, stop at the first failure

- **G1 — default bit-identity.** With `--time-model` omitted, route hashes,
  column ordering, reduced costs, certification status, and the `iters.csv` LP
  trace must be **byte-identical** to the current code on a k2 instance.
  Operational fields (`wall_s`, `peak_rss_mb`, timings) are excluded and must be
  listed as excluded. See `tests/test_k2_default_bit_identity.py`.
- **G2 — lattice containment.** For any uniform grid, the event lattice must
  contain every uniform breakpoint that is actually reachable, so the event
  model can never be *worse*. Assert `event_LP ≤ uniform_LP` for the same
  instance and physics on all nine k2/k3/k5 cells.
- **G3 — the target that motivates the task.** On `k02_s2`, the event model's
  certified LP must reach **≤ 2.0**, and duty `13413`'s failing `14→16`
  transition must become representable. Duty `13411` must become representable
  on all five of its previously failing transitions. These are known answers;
  they are the point.
- **G4 — size.** Report `dag_nodes` and `dag_arcs` for the event lattice beside
  the uniform 1 kWh / 5 min grid on the same instance. **The event lattice
  should be smaller.** If it is larger, say so plainly — that is a finding, and
  it would undercut the whole rationale.
- **G5 — physical replay.** Every route the event pricer generates must pass
  `realize_expanded_path` and `validate_injected_route` unchanged. The event
  model may not weaken any of the 36 per-route constraints audited in
  `analysis/arcflow_oracle_20260820/REPORT.md` — read that table and confirm
  each one still holds.

---

## 5. Deliverable

One table, k2/k3/k5 (three selections each), frozen physics **240 kWh / 240 kW,
reserve 0**:

`instance | model | certified LP | certified? | dag_nodes | dag_arcs | iterations | wall_s | reaches GIRO target?`

with `model` ∈ {uniform 15/10, uniform 1/5, **event**}.

Then answer in one paragraph each:

1. **Does the event lattice reach the industrial fleet on cells where uniform
   15/10 cannot?** Name the cells.
2. **Is it smaller than uniform 1 kWh / 5 min, and by how much?** This is the
   number that decides whether the approach scales.

If G1 or G3 fails, report and stop.

---

## 6. Conventions

1. Report executed output, never readiness. "Implemented" and "tests pass" are
   not results.
2. If a gate fails, say so plainly and stop.
3. Every exactness claim must name what it is exact **for** — a stated route
   space, never the real-world problem.
4. Append to `records/BUG_LOG.csv` and `records/DECISION_LOG.csv`; append-only,
   a correction is a new row.
5. A **certified** route weight may be reported as a fleet LP lower bound for
   the discretized model (`D0019`); an **uncertified** one gives no bound in
   either direction.
