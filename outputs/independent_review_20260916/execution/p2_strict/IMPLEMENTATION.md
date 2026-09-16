# P2 item 11 / F4 — stricter C5 prefix experiment

**Submitted:31 CG jobs and31 dependent MIPs.** The research claim remains **UNRESOLVED** until these experiments finish: does the chain5 k31 LP still reach a route weight near30 after adding the selected physical restrictions?

The frozen input order is exactly chain5's existing order. Each prefix is divided into local-route buses (`133…`,18E2) and route21 buses (`134…`,18E1), following the original Partille group separation. Each group retains all columns from its preceding prefix, then adds current singleton routes. No original GIRO schedule is injected. When the added duty belongs to one group, the other group's completed result is reused without repeating its solve.

| Setting | Experiment |
|---|---|
| 18E1 battery / minimum energy |236.44 /35.466 kWh|
| 18E2 battery / minimum energy |239.01 /35.8515 kWh|
| Initial energy |Full respective battery|
| Depot PARX |60kW|
| Other stations |Constant240kW|
| Ending energy |At least the same15%reserve; no65%end-duty target|
| Shared station capacity |Not imposed|
| Objective |100000 per route + flat electricity +5 per charging start|
| Master |Set covering|
| Group mixing in service trips |Prohibited|
| Setup, minimum charge time, nonlinear power, idle draw, charger-to-group compatibility |Not imposed|

The18E2 value239.01 follows the existing documented physics profile and workbook scaling (257×93%); the PDF rounds it to239. This combined experiment is not a complete GIRO operational model and does not identify any one constraint's individual effect.

**Why independent group solves are valid here:** trip coverage and routes are disjoint by vehicle group; shared capacity and cross-group fleet limits are absent, and costs add. The union of the two selected solutions therefore covers the prefix, and the sum of their certified LP optima is the segregated model's LP optimum. Integer fleet and proof status must be summed/reported separately. A group with an unfinished certificate leaves the union uncertified.

There are31 distinct CG component cases, forming two true dependency chains, plus one dependent MIP per component. `manifest.json` supplies the group-result pair needed at every global k. At k31 the union has716 trips. This is a different experiment from an unconstrained mixed-fleet solve; compare to the earlier C5 baseline with that distinction explicit.

## Implementation and checks

Code is in isolated worktree `.codex-work/review-strict-chain-20260916`, branch `codex/review-strict-chain-20260916`, commit `50ceb6c095a580f79f87b53bef536cac31f81963`, based on station-specific-power driver `309d98d266ebaf6b7e99543a67f8f2be5736874a`.

The new inheritance adapter checks parent status/pool/input hashes, execution commit, tariff, reference data and physical parameters. It remaps local trip indices through `Ordered_Trip_ID`, checks inherited trip attributes did not change, physically replays every inherited route at the child's parameters, refreshes identity metadata and resets the new run's iteration counter. It preserves all distinct inherited routes, without the earlier512 cap. It refuses missing inputs, changed physics/tariffs or failed physical replay.

The driver now accepts explicit maximum station waiting time. This experiment uses1560minutes to match the original chain graph; the older capacity driver's220-minute default remains unchanged for its existing callers. The driver still adds **one exact best column per iteration**, whereas the older baseline chain driver enriches the pool with30. Therefore runtime and integer-pool differences cannot be attributed solely to the physical restrictions.

Validation:6 new inheritance tests and9 existing driver tests pass. A real Gurobi/event-network one-trip→two-trip smoke run under the strict18E1 parameters converges in both stages and imports its parent column; `smoke/verification.json` records exact commands and code pin. This is an implementation check, not a research result for C5. Earlier `smoke/p*`/`smoke/c*` files precede the final wait-setting option; use only `parent_final`/`child_final` and their verification for final-code checks.

## Deployment plan

Read the current cluster resource policy before submission. Use default partition, exclude `scaglione-compute-01`, preserve prior-group `afterok` CG dependencies, and let each component MIP depend only on its own CG. Every eligible case may run; two simultaneous CG chains are a mathematical dependency limit, not a resource throttle. Suggested CG allocation:8CPU,48GB,6h; internal CG limit4h, leaving graph/build/wrapper margin. MIPs use8CPU,24GB,75min allocations with1h internal two-stage solve. The driver checkpoints pools, but this prepared wrapper starts a fresh attempt after preemption; do not claim it resumes search automatically.

`worker.sh ROOT CASE MODE` invokes `worker.py`, which checks the frozen input/code, retains per-attempt logs/commands, writes canonical outputs only after success, and preserves CG/MIP separation. The idempotent `submit.py` validates the native smoke, frozen code/input/tooling hashes, records each accepted job, and refuses to retry unresolved submission intents. Root authorized submission after item10 was accepted; all31 CG components and31 MIPs are queued. Initial components should still be checked for graph memory and raw-duty feasibility before treating later scientific results as established. If an earlier component reaches a budget limit but returns a valid pool, the next prefix can inherit it; this does not turn the parent's result into a certificate.

The source constraint map is `outputs/model_fairness_audit_20260913/giro_requirements_audit.md`; the executable group profiles are `src/giro_partille_physics.py`. Source manifest/master/input hashes, commands and all per-prefix mappings are in this directory. No earlier results, Google Docs or Slides were changed.

## Submission receipt — 16 September

All62 jobs were accepted after P2 item10. The first independent CGs are **341259** (18E1, global k1) and **341263** (18E2, global k3); both were running at the first inspection. Every other CG has exactly its previous same-group CG dependency. Every MIP depends only on its own CG, so it may run while the next CG proceeds. Pending dependencies therefore express required data, not an artificial throttle.

`jobs.json` records every command and initial scheduler inspection. `submission_verification.json` checks all62 node exclusions, default partition,8 CPUs, time limits and dependency edges. No startup errors appeared in the two initial workers; the first E1 case reached its Gurobi LP solves. Native preflight341140 completed successfully in4 seconds with a clean pin and licensed Gurobi12.0.3. One initial submission preflight caught a missing deployed `prepare.py`; it stopped before submitting anything, then passed after the hash-matching file was copied.

Read-only monitoring: run remote `collect_remote.py` for compact raw-result hashes, model/certificate status, iteration counts and LP/pricing times. `cases/<case>/cg.json` and `mip.json` are published only after their stage returns successfully; every attempt retains its own log and command. Preemption retries use distinct job/restart directories and restart search; scheduler completion remains separate from scientific certification.
