# Decomposition results — 14 September, 14:35 EDT

All 46 production MIPs were submitted after nine verified pool constructions. At collection, **45 had finished; the all-nine union remained running**. The native validation fixture is excluded from those counts.

These are alternative route pools for **one 750-trip input with a GIRO target of 32 buses**. Every completed MIP proved its minimum fleet within its selected pool and passed whole-parent route replay. None of the completed pairs beat the best of its two contributing partitions.

| Source partition | Buses in its individual pool | Fleet proved within that pool? |
|---|---:|---|
| 1 | 35 | Yes |
| 2 | 35 | Yes |
| 3 | 36 | Yes |
| 4 | 35 | Yes |
| 5 | 35 | Yes |
| 6 | 36 | Yes |
| 7 | 34 | Yes |
| 8 | 37 | Yes |
| 9 | 35 | Yes |

Across the 36 pair unions: 8 cases at 34 buses, 25 cases at 35 buses, 3 cases at 36 buses. Across the nine individual controls: 1 cases at 34 buses, 5 cases at 35 buses, 2 cases at 36 buses, 1 cases at 37 buses.

## Why another matched batch is justified

The first selector retained up to 512 columns per component: the known integer solution, followed by long routes and cost-per-trip ranking. The source audit found that it omitted 3,187 of 3,413 routes with positive weight in the 36 source LP solutions. Thus this treatment did not preserve the source fractional solutions.

A separate treatment retains all positive LP routes plus the integer witness, then fills the remaining places to the same 512. Mandatory routes fit within that cap in all 36 components (maximum 193). Its pool size, source inputs, solver code, integer warm starts and two-/four-hour MIP allowances remain matched. This tests a route-selection policy; it does not establish a general causal explanation for all earlier integer gaps.

The independent simultaneous-trip bound is 29. No parent CG is run and no full-model LP certificate is claimed. A selected-pool optimum above 32 cannot show that 32 buses are impossible in the full model. Shared charger capacity and a terminal-SOC floor are absent from this baseline.

[Exact MIP values, pool bounds, job IDs and source hashes](decomposition_results.csv). [Collector and normalizer checks](normalizer_validation.json). [Full campaign and source manifests](../../decomposition_pool_union_20260914/README.md).
