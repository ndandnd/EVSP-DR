# Completed controlled comparisons — 13 September, 02:35 EDT

All 24 paired allocations finished. Forty-two CG arms reached pricing certificates and produced 42 physically replayed MIP incumbents. Six original-scanning/full-pool arms exhausted their two-hour CG budgets during import, before any completed CG iteration; their MIPs were intentionally skipped. These are budget-capped results, not execution failures.

| Single change | Measured reduction in total CG time | Evidence |
|---|---:|---|
| Index the inherited-route replay, retaining the same 512 routes | 12.4–18.1% | Six matched pairs |
| Omit unused LP incidence construction, keeping indexed full inheritance | 9.3–14.6% | Six matched pairs |
| Inherit the full pool instead of 512 routes, using indexed replay in both | 40.6–64.5% | Six matched pairs; also changes integer quality |

For the 12 implementation-only pairs, imported sequence order and normalized pool hashes, CG iteration counts, final column counts and certified weighted LP objectives agree (objective tolerance1e-4). Timings use the solver’s reported `wall_s`, which includes graph loading/import and CG but subtracts telemetry overhead. They exclude prior graph construction, pre-arm authentication and the subsequent MIP. Three selected inputs and two reversed execution orders per contrast provide descriptive evidence, not population-level statistical significance. Reductions from different comparisons must not be added.

## Extra columns improve integer quality even at the same LP optimum

| Input | CG iterations:512 → full | MIP buses:512 → full | Proof for smaller pool |
|---|---:|---:|---|
| Chain1,k8 | 950 →177 | 9 →8 | Nine proved within pool |
| Chain4,k10 | 945 →222 | 11 →10 | Eleven proved within pool |
| Chain3,k15 | 891 →280 | 17 →15 | Seventeen is timed incumbent; lower bound15 |

Both execution orders give these outcomes. All full-pool fleets have finite-pool proofs. The certified weighted LP objectives agree to1e-4 between the 512 and full arms. Thus the first two cases directly demonstrate missing useful integer combinations in the restricted column pool. The third demonstrates better attained quality within the budget, without proving that its smaller pool cannot attain15.

Original arc scanning did not finish full-pool import within120minutes in any of six arms. Indexed full-pool CG reached certificates in9.1–20.8minutes in their paired counterparts. Do not report the capped originals as measured convergence times or invent a point speedup.

Settings are covering,240kWh/240kW,2.5kWh/5minute event graph,flat tariff,start fee5,no reserve/end-SOC floor/shared capacity. Eight CPUs per paired allocation; CG7200s and MIP3600s per arm. This isolates baseline replay/LP setup, not a capacity-pricing algorithm change. [Every pair and hashes](pairs.csv), [validation](validation.json), [source collection](collection.json).
