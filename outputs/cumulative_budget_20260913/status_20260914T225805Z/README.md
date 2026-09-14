# Research status — 14 September, 19:04 EDT

Largest verified match in the original one-hour chain MIPs: **24 buses**. Separate longer-MIP results are reported independently. These are baseline covering runs with inherited columns, 240 kWh batteries, 240 kW charging and a fee of 5 per charging start. Shared station capacity and a terminal-SOC floor are absent. Fleet proofs apply to the saved pools; physical replay checks individual routes. Charging optimality is a separate question.

| Chain | Largest target matched | Integer buses | CG minutes at this k |
|---|---:|---:|---:|
| 1 | 23 | 23 | 239.4 |
| 2 | 22 | 22 | 86.3 |
| 3 | 24 | 24 | 92.5 |
| 4 | 23 | 23 | 219.6 |
| 5 | 24 | 24 | 239.6 |
| 6 | 24 | 24 | 171.6 |

CG minutes include this k’s route import and CG. Earlier k values, original graph construction and MIP are separate. The source of each row is in [chain_reach.csv](chain_reach.csv). A CG certificate at a larger k is not an integer result.

Extension CG has 45 pricing certificates among 58 collected endpoints. [Every CG endpoint, stopping reason and last reduced cost](extension_cg.csv).

CG stopped before convergence:

| Case | CG minutes | Stopping reason | Last minimum reduced cost |
|---|---:|---|---:|
| C1, k=19 | 239.3 | wall_limit | -0.065741 |
| C1, k=20 | 239.6 | wall_limit | -0.008585 |
| C1, k=21 | 239.9 | wall_limit | -0.000737 |
| C1, k=23 | 239.4 | wall_limit | -0.073378 |
| C1, k=24 | 239.6 | wall_limit | -0.002991 |
| C4, k=19 | 239.1 | wall_limit | -0.003185 |
| C4, k=21 | 239.4 | wall_limit | -0.007782 |
| C4, k=25 | 239.3 | wall_limit | -0.155227 |
| C5, k=19 | 239.3 | wall_limit | -0.088463 |
| C5, k=21 | 239.6 | wall_limit | -0.058611 |
| C5, k=22 | 239.4 | wall_limit | -0.013743 |
| C5, k=24 | 239.6 | wall_limit | -0.047206 |
| C6, k=25 | 239.6 | wall_limit | -0.063090 |

These runs have no pricing certificate. Their restricted-master objectives are not certified full-model lower bounds. A saved pool can still be used by the already scheduled MIP and next chain step.

Original one-hour extension MIPs that missed the target (fixed-budget control):

| Case | Target | Buses found | Pool fleet bound | Fleet proved in pool? |
|---|---:|---:|---:|---|
| C1, k=19 | 19 | 20 | 19.0 | no |
| C1, k=20 | 20 | 21 | 20.0 | no |
| C1, k=22 | 22 | 23 | 22.0 | no |
| C2, k=17 | 17 | 18 | 17.0 | no |
| C2, k=18 | 18 | 19 | 18.0 | no |
| C2, k=19 | 19 | 20 | 19.0 | no |
| C2, k=20 | 20 | 21 | 20.0 | no |
| C2, k=23 | 23 | 25 | 23.0 | no |
| C2, k=24 | 24 | 25 | 24.0 | no |
| C2, k=25 | 25 | 26 | 25.0 | no |
| C3, k=19 | 19 | 20 | 19.0 | no |
| C3, k=20 | 20 | 21 | 20.0 | no |
| C3, k=21 | 21 | 22 | 21.0 | no |
| C3, k=22 | 22 | 23 | 22.0 | no |
| C3, k=23 | 23 | 24 | 23.0 | no |
| C3, k=25 | 25 | 26 | 25.0 | no |
| C4, k=21 | 21 | 22 | 21.0 | no |
| C4, k=22 | 22 | 23 | 22.0 | no |
| C4, k=24 | 24 | 25 | 24.0 | no |
| C4, k=25 | 25 | 26 | 25.0 | no |
| C5, k=19 | 19 | 20 | 19.0 | no |
| C6, k=18 | 18 | 19 | 18.0 | no |
| C6, k=23 | 23 | 24 | 23.0 | no |
| C6, k=25 | 25 | 26 | 25.0 | no |

These rows preserve the original one-hour searches; separate longer-MIP treatments may subsequently recover their targets. An open fleet gap is not proof that the pool requires the extra bus. [Every original extension MIP and its source](extension_mips.csv).

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

[Exact comparison, budgets and source hashes](comparison.csv). Collector snapshot: `outputs/post_meeting_20260910/monitor/20260914T225805Z.json`; SHA-256 `45fcc45515507d942983916e339b29e4ca0e358fc32a0045afd7d452bcb85c09`. The original snapshot and remote artifacts remain authoritative.
