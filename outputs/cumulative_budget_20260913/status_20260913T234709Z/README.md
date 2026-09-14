# Research status — 13 September, 19:49 EDT

Largest verified integer match: **18 buses**. These are baseline covering runs with inherited columns, 240 kWh batteries, 240 kW charging and a fee of 5 per charging start. Shared station capacity and a terminal-SOC floor are absent. Fleet proofs apply to the saved pools; physical replay checks individual routes. Charging optimality is a separate question.

| Chain | Largest target matched | Integer buses | CG minutes at this k |
|---|---:|---:|---:|
| 1 | 17 | 17 | 83.3 |
| 2 | 16 | 16 | 44.3 |
| 3 | 18 | 18 | 48.2 |
| 4 | 17 | 17 | 77.7 |
| 5 | 18 | 18 | 50.5 |
| 6 | 17 | 17 | 55.7 |

CG minutes include this k’s route import and CG. Earlier k values, original graph construction and MIP are separate. The source of each row is in [chain_reach.csv](chain_reach.csv). A CG certificate at a larger k is not an integer result.

Scheduler accounting later in this same collection recorded completed output for w4_k18. The detailed result was absent when the campaign section was read, so it is awaiting scientific verification and is not promoted into the table above. [Recorded completion and output hashes](late_scheduler_results.json).

Completed extension MIPs that have not matched the target:

| Case | Target | Buses found | Pool fleet bound | Fleet proved in pool? |
|---|---:|---:|---:|---|
| C2, k=17 | 17 | 18 | 17.0 | no |
| C2, k=18 | 18 | 19 | 18.0 | no |
| C3, k=19 | 19 | 20 | 19.0 | no |
| C3, k=20 | 20 | 21 | 20.0 | no |

An open fleet gap leaves target attainment unresolved. It is not proof that the pool requires the extra bus. [Every completed extension MIP and its source](extension_mips.csv).

## Fresh runs given the accumulated warm-chain time

24/24 fresh CG runs with the primary allowance have pricing certificates. 23 corresponding fresh MIPs have finished: **6 target matches, 4 proved pool limits above target, and 13 unresolved fleet gaps**. All 24 matched warm-reference MIPs reach their targets with finite-pool fleet proofs. The register retains any distinct larger-allowance continuations separately; this table never mixes them with primary results.

| Case | Fresh CG min | Fresh buses | Fresh pool fleet bound | Fleet proved? | Warm buses | Meaning |
|---|---:|---:|---:|---|---:|---|
| C1, k=5 | 20.6 | 5 | 5 | yes | 5 | target matched |
| C1, k=8 | 51.1 | 9 | 9 | yes | 8 | proved pool limit above target |
| C1, k=10 | 80.0 | 11 | 10 | no | 10 | fleet gap open |
| C1, k=15 | 290.7 | pending | pending | pending | 15 | MIP pending |
| C2, k=5 | 11.6 | 5 | 5 | yes | 5 | target matched |
| C2, k=8 | 18.0 | 9 | 8 | no | 8 | fleet gap open |
| C2, k=10 | 45.9 | 12 | 10 | no | 10 | fleet gap open |
| C2, k=15 | 259.7 | 17 | 15 | no | 15 | fleet gap open |
| C3, k=5 | 5.1 | 5 | 5 | yes | 5 | target matched |
| C3, k=8 | 13.3 | 9 | 8 | no | 8 | fleet gap open |
| C3, k=10 | 30.0 | 11 | 10 | no | 10 | fleet gap open |
| C3, k=15 | 76.7 | 18 | 15 | no | 15 | fleet gap open |
| C4, k=5 | 11.3 | 5 | 5 | yes | 5 | target matched |
| C4, k=8 | 47.3 | 9 | 9 | yes | 8 | proved pool limit above target |
| C4, k=10 | 59.4 | 11 | 10 | no | 10 | fleet gap open |
| C4, k=15 | 246.5 | 19 | 15 | no | 15 | fleet gap open |
| C5, k=5 | 8.0 | 6 | 6 | yes | 5 | proved pool limit above target |
| C5, k=8 | 25.9 | 9 | 9 | yes | 8 | proved pool limit above target |
| C5, k=10 | 33.5 | 11 | 10 | no | 10 | fleet gap open |
| C5, k=15 | 164.4 | 16 | 15 | no | 15 | fleet gap open |
| C6, k=5 | 6.1 | 5 | 5 | yes | 5 | target matched |
| C6, k=8 | 12.8 | 8 | 8 | yes | 8 | target matched |
| C6, k=10 | 27.2 | 11 | 10 | no | 10 | fleet gap open |
| C6, k=15 | 219.8 | 20 | 15 | no | 15 | fleet gap open |

**How to interpret this:** a proved pool limit means longer MIP search on the unchanged pool cannot meet the target. An unresolved gap does not prove that the target is absent from the pool. Overall MIP TIME_LIMIT can occur after fleet optimality was proved, during charging optimization. The certificate, fleet proof, target attainment and physical checks remain separate.

Pricing certificates concern the tested graph and reduced-cost tolerance. A weighted LP objective and total fractional route weight are different quantities; the latter is not automatically a fleet-only lower bound. Historical code revisions and hardware varied, so the time comparison is retrospective. Shared endpoints for the larger allowance are aliases, not independent searches.

[Exact comparison, budgets and source hashes](comparison.csv). Collector snapshot: `outputs/post_meeting_20260910/monitor/20260913T234709Z.json`; SHA-256 `68fba9e4d25fa0305d3a8cfa94a1a72c94727fc9ea3f1dedec56d459b0cb58be`. The original snapshot and remote artifacts remain authoritative.
