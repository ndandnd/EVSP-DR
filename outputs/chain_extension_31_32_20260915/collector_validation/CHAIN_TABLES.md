# Original one-hour chain MIPs

Each cell is the actual integer number of buses found. A dash means no verified MIP result in this collection; it does not mean failure. Separate longer searches are excluded.

| GIRO target | Chain 1 | Chain 2 | Chain 3 | Chain 4 | Chain 5 | Chain 6 |
|---|---:|---:|---:|---:|---:|---:|
| 31 | — | — | — | — | — | — |
| 32 | — | — | — | — | — | — |

## CG and proof status for the latest completed cases

| Case | CG minutes | CG stop | Fractional route weight | Integer buses | Saved-pool fleet bound | Fleet proved in pool? |
|---|---:|---|---:|---:|---:|---|

CG minutes include import and CG at this k; graph preparation, earlier k values and MIP are separate. The weighted LP objective and fractional route weight come from the final pool re-solve. An uncertified restricted-master objective is not a full-model lower bound. A saved-pool fleet proof concerns only those columns; charging proof is separate.

These are baseline covering runs: inherited full pools, 240 kWh/240 kW, no reserve, shared-capacity constraint or ending-SOC floor, and a fee of 5 per charging start. Individual-route replay and duplicate-removal validation have separate columns in the source CSV.

[All values, units, proof flags, job IDs and source hashes](all_chain_extension_results.csv).
