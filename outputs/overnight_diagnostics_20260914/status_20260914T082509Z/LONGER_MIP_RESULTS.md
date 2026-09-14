# Longer MIP searches on unchanged pools

These are the selected unresolved one-hour cases from the frozen overnight manifest. Each rerun allows up to three hours for fleet search and 3½ hours total, compared with 30 minutes and one hour originally. It starts a new Gurobi tree. All 23 ordered MIP-pool hashes and input hashes match their original runs; no new CG columns enter this comparison.

| Pool source | Completed cases | Targets matched | Proved extra bus required in pool | Fleet gap still open |
|---|---:|---:|---:|---:|
| Inherited chain pools | 9 | 9 | 0 | 0 |
| Fresh CG pools | 14 | 0 | 3 | 11 |

All nine inherited-pool reruns find and prove the target fleet. Seven finish that fleet proof within 30 minutes in this rerun, despite the original 30-minute fleet search missing it. Therefore extra elapsed time alone does not explain all recoveries. The configured time allowance changed; hardware, parallel search and timing effects are not isolated. The result establishes that the target solutions were already present in those original pools.

None of the fourteen fresh-pool reruns reaches its target. C3k8 proves nine necessary, C5k10 proves eleven, and C6k10 proves eleven, each within its saved pool. Their fleet stages take about 71.8, 77.8 and 46.1 minutes. Eleven other gaps remain open; these are not proofs of missing target solutions. C2k10 improves twelve to eleven and C6k15 improves twenty to eighteen, still above target.

The populations have different sizes and were selected because they previously missed targets. These counts do not estimate general success probabilities or a causal warm-versus-fresh advantage. Individual-route replay passes. Shared station capacity and terminal-SOC constraints remain absent. Fleet proofs are finite-pool claims; charging optimality is separate.

| Case | Target | Original buses | Longer-search buses | Pool fleet bound | Fleet proved in pool? | Fleet-stage minutes |
|---|---:|---:|---:|---:|---|---:|
| c1_k10_fresh_longmip | 10 | 11 | 11 | 10 | no | 180.0 |
| c1_k15_fresh_longmip | 15 | 18 | 18 | 15 | no | 180.1 |
| c2_k08_fresh_longmip | 8 | 9 | 9 | 8 | no | 180.0 |
| c2_k10_fresh_longmip | 10 | 12 | 11 | 10 | no | 180.0 |
| c2_k15_fresh_longmip | 15 | 17 | 17 | 15 | no | 180.1 |
| c3_k08_fresh_longmip | 8 | 9 | 9 | 9 | yes | 71.8 |
| c3_k10_fresh_longmip | 10 | 11 | 11 | 10 | no | 180.0 |
| c3_k15_fresh_longmip | 15 | 18 | 18 | 15 | no | 180.0 |
| c4_k10_fresh_longmip | 10 | 11 | 11 | 10 | no | 180.0 |
| c4_k15_fresh_longmip | 15 | 19 | 19 | 15 | no | 180.0 |
| c5_k10_fresh_longmip | 10 | 11 | 11 | 11 | yes | 77.8 |
| c5_k15_fresh_longmip | 15 | 16 | 16 | 15 | no | 180.0 |
| c6_k10_fresh_longmip | 10 | 11 | 11 | 11 | yes | 46.1 |
| c6_k15_fresh_longmip | 15 | 20 | 18 | 15 | no | 180.0 |
| w2_k17_extension_longmip | 17 | 18 | 17 | 17 | yes | 25.2 |
| w2_k18_extension_longmip | 18 | 19 | 18 | 18 | yes | 10.0 |
| w2_k19_extension_longmip | 19 | 20 | 19 | 19 | yes | 17.4 |
| w2_k20_extension_longmip | 20 | 21 | 20 | 20 | yes | 21.0 |
| w3_k19_extension_longmip | 19 | 20 | 19 | 19 | yes | 45.3 |
| w3_k20_extension_longmip | 20 | 21 | 20 | 20 | yes | 19.8 |
| w3_k21_extension_longmip | 21 | 22 | 21 | 21 | yes | 47.4 |
| w5_k19_extension_longmip | 19 | 20 | 19 | 19 | yes | 16.7 |
| w6_k18_extension_longmip | 18 | 19 | 18 | 18 | yes | 13.5 |

[Exact results, timings and matched pool hashes](longer_mip_comparison.csv).
