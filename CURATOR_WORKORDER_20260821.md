# Work order: integration and records curator

**For a NEW agent with no prior context.** You are deliberately fresh: five
feature-branch agents each wrote to the shared record ledgers and reused the same
IDs for unrelated findings. None of them can impartially adjudicate its own IDs.
That is your job.

Date 2026-08-21. Operator: Nathan (`nc437`). **He owns the cluster; you never
submit cluster jobs.** Work locally, report executed output.

---

## 1. Required reading, in order

1. `STATUS_20260821.md` — current project state. **Note: it contains errors you
   are tasked with fixing (§5 below).**
2. `CLUSTER_OPERATING_RULES_20260821.md` — operational habits.
3. `records/DECISION_LOG.csv`, `records/BUG_LOG.csv`, `records/README.md`.

Branch: base on `records/ladder-lite-20260819-2969@739d8dd`, create
**`cursor/records-curator-2969`**. **Do not merge any feature branch wholesale.**

---

## 2. The problem

Five agents were told to append to `records/DECISION_LOG.csv` and
`records/BUG_LOG.csv`. No ID ranges were allocated — that was a management
error, not theirs. Result: the same IDs now mean different things on different
branches.

| branch | colliding IDs |
|---|---|
| ladder-lite | `B0023`–`B0026`, `D0021` |
| fixed-duty | `B0023`, `D0021`–`D0023` |
| event-based pricer | `B0031`, `D0031`–`D0033` |
| branch-and-price / resolution | `B0031`–`B0033`, `D0031`–`D0035` |

A whole-branch merge would silently make one bug ID mean two unrelated things.

---

## 3. Deliverable 1 — stop the bleeding, first

Create **`records/ID_REGISTRY.csv`** with columns
`canonical_id, kind (B|D), source_branch, source_local_label, title, assigned_utc`.

Reserve **`B0100+` and `D0100+`** for all future allocations so nothing can
collide with existing IDs.

Then confirm to the operator that the shared ledgers are frozen and unfrozen
only through you. All other agents have been told to write findings to
`records/inbox/<branch>.md` with provisional local labels (`LOCAL-1`, …) instead.

## 4. Deliverables 2–4 — make the branch self-contained

**2. Old → canonical mapping.** One row per colliding `(branch, old_id)` →
`canonical_id`, so existing reports on feature branches stay traceable.

**3. Evidence import at exact producer SHAs.** Cherry-pick reviewed code and
copy evidence artifacts from the branch-and-price, arc-flow, fixed-duty and
event branches, recording each file's sha256 and producing commit. **Do not
import feature-branch CSV ledgers.**

Acceptance: **every file cited by `STATUS_20260821.md` §9 must exist on this
branch.** Today `analysis/arcflow_oracle_20260820/REPORT.md` and
`analysis/fixed_duty_continuous_20260820/FACTORIAL_DIAGNOSTIC.md` do not, which
means the status file sends readers to files they cannot open.

**4. Populate `records/RESULTS_LOG.csv`.** It is still header-only. Use
`scripts/ladder_lite/record_results.sh` with the normalized outputs, and add an
artifact inventory with hashes. Include raw status files, iteration traces and
MIP checkpoints, or record explicitly which are absent and why.

## 5. Deliverable 5 — correct `STATUS_20260821.md`

These are known-wrong or overbroad. Fix each:

- **R14 is wrong.** Per `analysis/event_based_pricer_20260821/REPORT.md` at
  commit `92dd6c1`: **G2 is incomplete**; duty **13411 is representable only at
  2.5/10, 1/10 and 1/5 — not 15/10 or 5/10**; **G5 fails for 1,671 of 1,710
  event columns** because the realization contract rejects irregular event
  windows; k5 construction reached **22.2M arcs and 26.8 GiB**; and batch
  generation is **not** genuine k-shortest-path enumeration. Rewrite R14 as a
  partial, unlicensed result and state plainly: **no exact event-route-space
  claim is currently licensed.** (The original R14 was written from an agent's
  chat summary rather than the committed artifact — do not repeat that.)
- **R16:** the achieved-over-bound fraction is **69.2%** (5.328 / 7.695), not
  62%. The 8.85% figure is a per-duty range endpoint and is not the aggregate
  bound in the cited artifact.
- **R3:** narrow to "the lexicographic phase-2 fleet bound equalled the
  combined-cost route weight to 1e-7 **on eight k2/k3/k5 cells**." The general
  statement is the bracket argument only, not a theorem. Elsewhere prefer citing
  the explicit lexicographic phase-2 bound.
- **R11:** "warm pools **dramatically improve uncertified** k30 endpoints" — not
  "fix k30." Those runs ended with `rc ≈ −100k`, uncertified.
- **R12:** "best **tested** choice on the matched k20 comparison" — not
  universally optimal.
- **§4 Frozen decisions:** move the **$5 charge-start cost** and **terminal ≥
  reserve** into open modelling choices. `B0023` shows the $5 term exceeded the
  entire energy bill, which contradicts treating it as settled.

## 6. Deliverable 6 — validate the new instances

Independently validate the **18 new GIRO duty-union instances** from ladder
commit `72c7bf4`: duty membership, `target_fleet = k`, weekday-variant policy
`one_literal_per_numeric_base_no_siblings`, trip counts, and sha256s. You are
the validator; Agent A is the producer — do not take its report as evidence.

Keep the `SyntheticRandom` family in a **separate manifest and separate results
table**. Duty unions carry `target_fleet = k` as ground truth because the
industrial schedule ran one bus per duty; synthetic instances have no comparator.
The two families must never share a table.

---

## 7. Conventions

1. Report **executed output**, never readiness.
2. If something cannot be verified, say so and leave it unverified rather than
   inferring it.
3. Every claim of exactness must name what it is exact **for** — a stated route
   space at a stated grid, never the real-world problem.
4. RAW and KNOWN arms never share a row. KNOWN injects the industrial partition
   and is a plumbing control, not algorithmic recovery.
5. Finish with a full-suite run and an independent re-review.
