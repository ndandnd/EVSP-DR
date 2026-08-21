# agentic/ — operator handoff for an LLM assistant

**Read this first, then `../STATUS_20260821.md`.**

You are assisting Nathan (`nc437`) on EVSP-DR. He operates the Unicorn cluster
from a laptop over VPN. **You never submit cluster jobs yourself** — you give him
commands to paste, and he pastes them.

## 1. Where you are in the research

Two goals. **Goal 1 (algorithmic trust) is essentially answered at small scale
with proofs.** **Goal 2 (demand response) has real full-schedule numbers but is
not yet an empirical result**, because the tariffs are synthetic.

### Goal 1 — what is proven

| claim | evidence |
|---|---|
| The pricing DP is an **exact oracle over its domain** | 7,680/7,680 agreement with exhaustive path enumeration across random dual vectors (negative components, exact ties, varied SOC steps, stations, delayed charging, flat and time-varying tariffs); max error 3.5e-10 |
| Three independent exact methods agree | brute force vs branch-and-price vs arc-flow: **240/240** integer fleets; LP bounds 240/240 |
| The LP is solved to certified optimality | reduced-cost certificates (`min_rc = 0`); arc-flow LP equals the set-partitioning LP on all nine primary cells; B&P root LP reproduces the CG LP exactly |
| Certified route weight is the fleet LP bound | lexicographic phase-2 equals combined-cost route weight to 1e-7 on eight cells |
| The industrial fleet is recoverable | `k02_s2`/`k02_s3` reach exactly 2 buses, proven, from RAW pools at 1 kWh/5 min; `k03_s1`/`k03_s3` reach 3 |
| Discretization cost is measured | at `k02_s2` on the coarse grid: **+1 bus representation, +1 bus pool composition**, both vanishing at 1 kWh/5 min |
| Against a public benchmark | Utrecht qlink-8 successor: LP 10.2273, proven 11 vs published 10 (**+1 bus / +10%**); objectives documented as non-comparable |

### Goal 1 — the known limitation, quantified

**Column generation's finite pool does not contain the integer optimum**, even
when the LP is certified. Integer-useful routes can carry zero or positive
reduced cost at the LP optimum and never enter the pool. Measured: **24 of 240**
tiny-network cases, smallest reproducer 5 trips / 2 stations. Across ~100
real-instance pool MIPs, integer results are **non-monotone in grid resolution**.

**Consequence for wording:** Gurobi `OPTIMAL` means optimal over the columns it
received. Never write "proven optimal" without a scope. Use
`optimality_scope = finite_pool | discrete_model`.

### Goal 2 — current numbers

Fixed-duty (charging-only) timing value on a fixed 30-duty intersection:

| tariff / terminal | uncapped | capped |
|---|---:|---:|
| flat / ≥reserve | **0.000%** | **0.000%** |
| peak12 / ≥reserve | 2.208% | 2.015% |
| **two-peak / ≥reserve** | **4.720%** | **4.755%** |
| peak12 / ≥initial | 10.030% | 9.794% |
| two-peak / ≥initial | 30.565% | 31.487% |

The flat row at exactly 0.000% is the null control. Unconstrained charger demand:
18–20 buses simultaneously fleet-wide (4.32–4.80 MW) under ≥reserve; **30 buses
at `PARX_1` / 7.2 MW** under ≥initial — which is why **no peak-shaving claim is
licensed** while chargers are unlimited.

**Blocking Goal 2:** a real frozen Nord Pool SE3 price series, and a decided
terminal / periodic-horizon policy. Tariffs are synthetic until then.

## 2. Frozen model decisions

240 kWh battery, 240 kW charging (1C, 0→100% in one hour), reserve SOC 0,
zone-centroid deadhead, all buses start full, `columns_per_iter = 30`. SOC steps
must be commensurate with charge-per-block (240 kW × 10 min = 40 kWh).

**Open, not frozen:** the $5 charge-start cost (it exceeded the entire energy
bill in one configuration), terminal SOC policy, charger capacity.

## 3. The evidence table reviewers need

For each instance and representation, four numbers:

| symbol | meaning |
|---|---|
| `L_model` | certified fleet LP bound (lexicographic phase 2) |
| `I_model` | true integer optimum of that discrete model |
| `I_pool` | exact optimum over the generated finite pool |
| `I_timed` | best incumbent under the MIP time budget |

Decomposing into: **representation gap** (industrial fleet → `I_model`),
**LP integrality gap** (`L_model` → `I_model`), **pool-composition gap**
(`I_model` → `I_pool`), **MIP-search gap** (`I_pool` → `I_timed`).

**Cheap proof of `I_model` — use this before any expensive solve.** The certified
fleet LP `L` is a valid model-wide bound and fleets are integral, so
`I_model ≥ ⌈L⌉`. If a **physically validated** incumbent equals `⌈L⌉`, the
discrete-model optimum is **proven** with no arc-flow or B&P solve. Record
`model_optimality_method = sandwich`.

**Rigorous relaxed-tolerance bound.** In the fleet-only phase every route costs
1, so if the final exact pricing call returns minimum reduced cost `−δ`, then
`LB = z_RMP / (1+δ)` is a valid model-wide LP lower bound. This converts a loose
stopping tolerance into a certified gap rather than "approximately converged."
**Void if pricing timed out without a valid reduced-cost bound.**

## 4. Reading order

1. `../STATUS_20260821.md` — full state, resolved findings, open questions
2. `CLUSTER_PLAYBOOK.md` — every command you will need
3. `../CLUSTER_OPERATING_RULES_20260821.md` — partition choice, mandatory `--mem`, paste habits
4. `../records/DECISION_LOG.csv`, `../records/BUG_LOG.csv` — authoritative record
5. `../records/inbox/` — unadjudicated findings awaiting canonical IDs
6. `../analysis/` — per-experiment reports

## 5. Hard rules

1. **Report executed output, never readiness.** "Implemented", "tests pass",
   "ready" are not results.
2. **Read the artifact, not the summary.** An agent once reported a gate passing,
   corrected itself in later commits, and the chat claim reached the status doc
   unverified. Do not repeat that.
3. **Never assume a git branch name** — `git ls-remote --heads origin` first.
   Cursor appends suffixes (`-3a99`, `-1451`, `-b22e`).
4. **`cat > file <<'EOF'` then run the file**, rather than long `bash <<'BASH'`
   blocks with nested `--wrap "..."` strings — those were mangled in paste four
   times.
5. **One paste block per request.**
6. **RAW and KNOWN arms never share a row.** KNOWN injects the industrial
   partition; it is a plumbing control, not algorithmic recovery.
7. **Duty unions and synthetic instances never share a table.** Only duty unions
   carry `target_fleet = k` as ground truth.
8. **Never append to `records/*.csv`** — write to `records/inbox/<branch>.md`
   with `LOCAL-*` labels; the curator assigns canonical IDs.
