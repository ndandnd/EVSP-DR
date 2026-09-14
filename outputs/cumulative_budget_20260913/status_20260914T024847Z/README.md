# Research status — 13 September, 22:51 EDT

Largest verified integer match: **19 buses**. These are baseline covering runs with inherited columns, 240 kWh batteries, 240 kW charging and a fee of 5 per charging start. Shared station capacity and a terminal-SOC floor are absent. Fleet proofs apply to the saved pools; physical replay checks individual routes. Charging optimality is a separate question.

| Chain | Largest target matched | Integer buses | CG minutes at this k |
|---|---:|---:|---:|
| 1 | 18 | 18 | 234.3 |
| 2 | 16 | 16 | 44.3 |
| 3 | 18 | 18 | 48.2 |
| 4 | 18 | 18 | 68.9 |
| 5 | 18 | 18 | 50.5 |
| 6 | 19 | 19 | 61.2 |

CG minutes include this k’s route import and CG. Earlier k values, original graph construction and MIP are separate. The source of each row is in [chain_reach.csv](chain_reach.csv). A CG certificate at a larger k is not an integer result.

Extension CG has 25 pricing certificates among 27 collected endpoints. [Every CG endpoint, stopping reason and last reduced cost](extension_cg.csv).

CG stopped before convergence:

| Case | CG minutes | Stopping reason | Last minimum reduced cost |
|---|---:|---|---:|
| C4, k=19 | 239.1 | wall_limit | -0.003185 |
| C5, k=19 | 239.3 | wall_limit | -0.088463 |

These runs have no pricing certificate. Their restricted-master objectives are not certified full-model lower bounds. A saved pool can still be used by the already scheduled MIP and next chain step.

Completed extension MIPs that have not matched the target:

| Case | Target | Buses found | Pool fleet bound | Fleet proved in pool? |
|---|---:|---:|---:|---|
| C2, k=17 | 17 | 18 | 17.0 | no |
| C2, k=18 | 18 | 19 | 18.0 | no |
| C2, k=19 | 19 | 20 | 19.0 | no |
| C2, k=20 | 20 | 21 | 20.0 | no |
| C3, k=19 | 19 | 20 | 19.0 | no |
| C3, k=20 | 20 | 21 | 20.0 | no |
| C3, k=21 | 21 | 22 | 21.0 | no |
| C5, k=19 | 19 | 20 | 19.0 | no |
| C6, k=18 | 18 | 19 | 18.0 | no |

An open fleet gap leaves target attainment unresolved. It is not proof that the pool requires the extra bus. [Every completed extension MIP and its source](extension_mips.csv).

## Fresh runs given the accumulated warm-chain time

24/24 fresh CG runs with the primary allowance have pricing certificates. 24 corresponding fresh MIPs have finished: **6 target matches, 4 proved pool limits above target, and 14 unresolved fleet gaps**. All 24 matched warm-reference MIPs reach their targets with finite-pool fleet proofs. The register retains any distinct larger-allowance continuations separately; this table never mixes them with primary results.

| Case | Fresh CG min | Fresh buses | Fresh pool fleet bound | Fleet proved? | Warm buses | Meaning |
|---|---:|---:|---:|---|---:|---|
| C1, k=5 | 20.6 | 5 | 5 | yes | 5 | target matched |
| C1, k=8 | 51.1 | 9 | 9 | yes | 8 | proved pool limit above target |
| C1, k=10 | 80.0 | 11 | 10 | no | 10 | fleet gap open |
| C1, k=15 | 290.7 | 18 | 15 | no | 15 | fleet gap open |
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

[Exact comparison, budgets and source hashes](comparison.csv). Collector snapshot: `outputs/post_meeting_20260910/monitor/20260914T024847Z.json`; SHA-256 `16e21b239989c4277f13c3220a3f3ab3ad9f24c32e50b5daa5194c007d59d44e`. The original snapshot and remote artifacts remain authoritative.
