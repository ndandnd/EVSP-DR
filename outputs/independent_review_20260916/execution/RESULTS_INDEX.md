# Results and their scope

Use this index rather than interpreting job completion as proof.

| Evidence | F-number | What it establishes |
|---|---|---|
| [102-row chain table](audited_chain_results.csv) | F1/F3/F4/F9 | Original outcomes plus numerical event-model lower bounds, separate GIRO replay counts, and exact-source empty-driving dispatch checks |
| [All 128 dispatch replays](f1/per_case.csv) | F1 | Each passenger trip assigned once; unchanged physical routes and charging costs; no shared-capacity claim |
| [Route-overlap comparison](f2/README.md) | F2 | Observed Jaccard difference; no causal identification |
| [Bound derivation](f3/README.md) | F3 | Correct route-mass/cost envelope, matched pricing iterations and numerical qualifications |
| [GIRO duties](f4/README.md) | F4 | Original charging schedules versus reoptimized fixed-trip schedules, under explicitly different checks |
| [Three charging comparators](f6/comparison.csv) | F6 | One selected five-duty instance; tariff-specific costs and physical caveats |
| [Register repair](f9/README.md) | F9 | Provenance corrections with stable IDs and unchanged original scientific result fields |
| [Longer MIP comparisons](p1/README.md) | F2/F4/F5 | Frozen pools, seeds, budgets; new conclusions require verified completed endpoints |
| [k15 tariff-aware CG](p2/dr_mincharge/README.md) | F6/F7 | Newly submitted matched synthetic/real-price experiments; real-price numbers stay internal |
| [Stricter chain 5](p2_strict/README.md) | F4 | Separate vehicle groups, depot power and reserve sensitivity; not all GIRO constraints |
| [Random intermediate trip groups](p2_random/README.md) | F5 | A control without duty-based grouping; intermediate stage number is not a fleet target |
| [Full Partille](p3_full/README.md) | F8 | Fresh 40-duty / 948-trip campaign; graph and 48-hour CG budgets separated |
| [Frölunda ladder](p3_frolunda/README.md) | F8 | Declared input conversion, first verified endpoint, later stages and their own proof scopes |

[Execution order and verdicts](README.md) · [Machine-readable ledger](ledger.json) · [Monitoring instructions](MONITORING.md)
