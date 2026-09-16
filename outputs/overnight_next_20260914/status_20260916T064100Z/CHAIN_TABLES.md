# Original one-hour chain MIPs

Each cell is the actual integer number of buses found. A dash means no verified MIP result in this collection; it does not mean failure. Separate longer searches are excluded.

| GIRO target | Chain 1 | Chain 2 | Chain 3 | Chain 4 | Chain 5 | Chain 6 |
|---|---:|---:|---:|---:|---:|---:|
| 16 | 16 | 16 | 16 | 16 | 16 | 16 |
| 17 | 17 | 18 | 17 | 17 | 17 | 17 |
| 18 | 18 | 19 | 18 | 18 | 18 | 19 |
| 19 | 20 | 20 | 20 | 19 | 20 | 19 |
| 20 | 21 | 21 | 21 | 20 | 20 | 20 |
| 21 | 21 | 21 | 22 | 22 | 21 | 21 |
| 22 | 23 | 22 | 23 | 23 | 22 | 22 |
| 23 | 23 | 25 | 24 | 23 | 23 | 24 |
| 24 | 24 | 25 | 24 | 25 | 24 | 24 |
| 25 | 25 | 26 | 26 | 26 | 26 | 26 |
| 26 | 26 | 26 | 26 | 26 | 26 | 26 |
| 27 | 28 | 27 | 27 | 27 | 29 | 27 |
| 28 | 31 | 28 | 29 | 29 | 34 | 28 |
| 29 | 30 | 30 | 29 | 29 | 35 | 30 |
| 30 | 31 | 31 | 30 | 39 | 37 | 31 |
| 31 | — | — | — | — | — | — |
| 32 | — | — | — | — | — | — |

## CG and proof status for the latest completed cases

| Case | CG minutes | CG stop | Fractional route weight | Integer buses | Saved-pool fleet bound | Fleet proved in pool? |
|---|---:|---|---:|---:|---:|---|
| C1, k=26 | 239.9 | time limit | 26.000000 | 26 | 26.000000 | yes |
| C2, k=26 | 239.6 | time limit | 26.000000 | 26 | 26.000000 | yes |
| C3, k=26 | 113.2 | converged | 26.000000 | 26 | 26.000000 | yes |
| C4, k=26 | 239.7 | time limit | 26.000000 | 26 | 26.000000 | yes |
| C5, k=26 | 239.8 | time limit | 26.000000 | 26 | 26.000000 | yes |
| C6, k=26 | 119.9 | converged | 26.000000 | 26 | 26.000000 | yes |
| C1, k=27 | 239.4 | time limit | 27.000000 | 28 | 27.000000 | no |
| C2, k=27 | 239.9 | time limit | 27.000000 | 27 | 27.000000 | yes |
| C3, k=27 | 118.3 | converged | 27.000000 | 27 | 27.000000 | yes |
| C4, k=27 | 239.9 | time limit | 27.000000 | 27 | 27.000000 | yes |
| C5, k=27 | 239.7 | time limit | 26.000000 | 29 | 26.000000 | no |
| C6, k=27 | 239.9 | time limit | 27.000000 | 27 | 27.000000 | yes |
| C1, k=28 | 239.4 | time limit | 28.000000 | 31 | 28.000000 | no |
| C2, k=28 | 239.6 | time limit | 28.000000 | 28 | 28.000000 | yes |
| C3, k=28 | 239.9 | time limit | 28.000000 | 29 | 28.000000 | no |
| C4, k=28 | 239.9 | time limit | 28.000000 | 29 | 28.000000 | no |
| C5, k=28 | 239.8 | time limit | 27.000000 | 34 | 27.000000 | no |
| C6, k=28 | 239.9 | time limit | 28.000000 | 28 | 28.000000 | yes |
| C1, k=29 | 239.6 | time limit | 29.000000 | 30 | 29.000000 | no |
| C2, k=29 | 239.2 | time limit | 29.000000 | 30 | 29.000000 | no |
| C3, k=29 | 239.4 | time limit | 29.000000 | 29 | 29.000000 | yes |
| C4, k=29 | 239.9 | time limit | 29.000000 | 29 | 29.000000 | yes |
| C5, k=29 | 239.9 | time limit | 28.000000 | 35 | 28.000000 | no |
| C6, k=29 | 94.8 | converged | 29.000000 | 30 | 29.000000 | no |
| C1, k=30 | 240.0 | time limit | 30.000000 | 31 | 30.000000 | no |
| C2, k=30 | 239.2 | time limit | 30.000000 | 31 | 30.000000 | no |
| C3, k=30 | 239.6 | time limit | 30.000000 | 30 | 30.000000 | yes |
| C4, k=30 | 239.7 | time limit | 29.000000 | 39 | 29.000000 | no |
| C5, k=30 | 239.6 | time limit | 29.000000 | 37 | 29.000000 | no |
| C6, k=30 | 239.7 | time limit | 30.000000 | 31 | 30.000000 | no |
| C3, k=31 | 239.2 | time limit | 31.000000 | — | — | — |
| C5, k=31 | 197.5 | converged | 30.000000 | — | — | — |

CG minutes include import and CG at this k; graph preparation, earlier k values and MIP are separate. The weighted LP objective and fractional route weight come from the final pool re-solve. An uncertified restricted-master objective is not a full-model lower bound. A saved-pool fleet proof concerns only those columns; charging proof is separate.

These are baseline covering runs: inherited full pools, 240 kWh/240 kW, no reserve, shared-capacity constraint or ending-SOC floor, and a fee of 5 per charging start. Individual-route replay and duplicate-removal validation have separate columns in the source CSV.

[All values, units, proof flags, job IDs and source hashes](all_chain_extension_results.csv).
