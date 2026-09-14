# First compact-pool endpoints — 14 September, 19:35 EDT

These are the first five completed MIPs, selected by completion time. They are not the final success rate or evidence that one seed method wins overall.

| Case | Target | Buses found | Fleet proved in its pool | CG minutes |
|---|---:|---:|---|---:|
| c3_k08_core512 | 8 | 8 | yes | 5.50 |
| c5_k08_core512 | 8 | 8 | yes | 6.47 |
| c6_k08_core512 | 8 | 8 | yes | 4.53 |
| c6_k08_core | 8 | 8 | yes | 8.71 |
| c6_k10_core512 | 10 | 10 | yes | 8.13 |

All five source CGs have pricing certificates on the tested event graph. Every selected schedule passes individual-route replay; source pool and result hashes match and no routes were rejected/repaired in MIP admission. Shared capacity and a terminal SOC floor are absent. These fleets are proved only within the corresponding saved pools.

The core retains previous-k integer routes and all positive-weight LP routes; core512 retains that core and fills to512 distinct trip sets. Neither imports a current-k solution.

[Exact identities and hashes](first_completed_mips.csv) · [Targeted collection](launch_collection_latest.json). The full workbook remains dated19:04; these later results are recorded here pending the next full collection.
