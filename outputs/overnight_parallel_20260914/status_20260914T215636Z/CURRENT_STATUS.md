# Verified update — 14 September, 18:02 EDT

All 36 small-seed CG runs have endpoints: **35 pricing certificates and one time limit**. Twenty-three MIPs are complete: seven target matches, eleven proved pool limits above target, and five open fleet gaps.

## Eleven completed seed-method pairs

| Case | Integer-route seeds: buses | LP-selected seeds: buses | Pool fleet bounds: integer / LP | CG minutes: integer / LP |
|---|---:|---:|---|---|
| c1_k08 | 9 | 9 | 9 / 9 | 74.0 / 74.2 |
| c2_k08 | 8 | 9 | 8 / 9 | 17.9 / 13.1 |
| c2_k10 | 11 | 11 | 10 / 10 | 38.8 / 51.4 |
| c3_k10 | 11 | 11 | 10 / 10 | 28.6 / 27.7 |
| c3_k15 | 15 | 17 | 15 / 15 | 66.7 / 70.3 |
| c4_k08 | 9 | 9 | 9 / 9 | 52.6 / 49.6 |
| c4_k10 | 11 | 11 | 11 / 11 | 56.2 / 67.5 |
| c5_k08 | 8 | 9 | 8 / 9 | 26.4 / 26.3 |
| c5_k10 | 10 | 11 | 10 / 11 | 29.4 / 40.0 |
| c6_k08 | 8 | 8 | 8 / 8 | 9.1 / 10.2 |
| c6_k10 | 10 | 11 | 10 / 11 | 20.3 / 22.0 |

Integer-route seeds find fewer buses in five completed pairs and tie in six; seven other pairs remain unfinished. In this table the fleet minimum is proved when the incumbent equals the corresponding pool fleet bound. The C3 k15 integer-seeded pool supports 15 buses with a fleet proof; the LP-seeded MIP found 17 with bound 15, so that pool may still support 15. The new C1 k8, C4 k8 and C4 k10 pairs miss with both seed methods, and each pool is proved to require the extra bus. Small seeds therefore do not consistently reproduce the separate full-pool inheritance results.

Seed coverage differs between the two methods, and prior computation must be included in end-to-end accounting. These are interim selected-case comparisons, not a general success rate. C1 k15 LP-selected CG is the only uncertified seed endpoint:239.0minutes, last reduced cost−1.060160. Its saved-pool MIP remains a separate search. [Every seed result and source hash](seed_results.csv).

## Fresh pools after approximately four hours

| Case | Columns | Target | Buses found | Pool fleet bound | Fleet proved? | MIP minutes |
|---|---:|---:|---:|---:|---|---:|
| c1_k15_prefix_mip | 67,294 | 15 | 19 | 15 | no | 210.1 |
| c2_k15_prefix_mip | 64,225 | 15 | 17 | 15 | no | 210.1 |
| c4_k15_prefix_mip | 71,847 | 15 | 19 | 15 | no | 210.0 |

All three MIPs finish with open fleet gaps after three-hour fleet / 3½-hour total allowances, equal to the current small-seed MIP allowances. Selected routes pass individual replay; admission rejected/repaired no columns. Their prefixes represent pre-insertion iteration boundaries at239.881,239.917and239.990minutes, not exact wall-clock CG snapshots or pricing certificates. Corresponding C1/C2/C4 k15 small-seed MIPs remain pending, so that comparison is not yet complete. [Construction bindings, inputs and result hashes](prefix_control_results.csv).

## Larger chains and queue

New original C5 k24 matches24buses, with a finite-pool fleet proof and individual-route replay. Its CG remained capped at239.6minutes with last reduced cost−0.047206. Largest original matches across chains1–6 are23,22,24,23,24,24; the separate longer C2k25 result remains25. The new C4k25 CG also hit its time limit:239.3minutes, last reduced cost−0.155227.

Extension totals:57CGendpoints,45certificates and12caps;56original one-hour MIPs,33target matches and23misses. Separate longer searches recover15misses, leaving8unresolved targets. Baseline physics use covering,240kWh/240kW and charging-start fee5, without shared station capacity or a terminal SOC floor. [Every original chain result](../../cumulative_budget_20260913/status_20260914T215636Z/README.md).

Queue:34running (4CG,16MIP,14graph),38solver dependencies,14conditional graph checks and33held historical tasks. No invalid dependency, new execution failure or new confirmed preemption appeared. Two k26CGs are running. The two all-nine decomposition MIPs remain running; their completed pair results are unchanged. Capacity results and the documented cached13407preemption are unchanged; do not interpret its stale worker flag as a live job.

Register/workbook:2,935artifact-stage rows across63source groups, all six supplements retained. Core and new seed/prefix endpoint values were checked against source. Current Doc tables were updated in place; figure tabs, reference material and Slides were preserved. No new campaigns, solver changes, cancellations or retries were introduced.

Source:20260914T215636Z, SHA-2561d33ef956bb17f9af5556fa05e3b17ca4df47104b09c0f101bce89d01c80e402; collection completed18:02:52EDT after376seconds. Scheduler arrivals during collection remain separate from published endpoints.
