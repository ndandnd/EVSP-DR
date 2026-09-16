# Current chain results — 20260916T144440Z

Actual covering fleet counts; the original budget and separately selected longer searches remain distinct.

| Treatment | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---:|---:|---:|---:|---:|---:|
| Largest matched target: original | 26 | 28 | 31 | 29 | 26 | 28 |
| Largest matched target: including longer | 30 | 30 | 31 | 29 | 31 | 31 |
| 31: original one-hour MIP | 34 | 32 | 31 | 40 | 36 | 32 |
| 31: best including completed longer MIPs | 34 | 32 | 31 | 37 | 31 | 31 |
| 32: original one-hour MIP | — | 34 | 33 | 34 | 37 | 33 |
| 32: best including completed longer MIPs | — | 34 | 33 | 34 | 37 | 33 |

Baseline: 240 kWh batteries, 240 kW charging, charge-start fee 5, set covering; no reserve, shared charger-capacity constraint or terminal-SOC floor. These are not full-GIRO-feasibility results. Individual route replay passes, but duplicate-trip removal and shared capacity remain unvalidated for these large fleets.

Longer searches use the same ordered pool and initializer policy; hash checks passed. They have 12,600 total solver seconds, including 10,800 for the fleet stage. They start a new tree; hardware and search timing may differ. Recovering a target with unchanged columns demonstrates that those columns suffice for covering at that target. It does not establish an executable schedule or global optimality.

A saved-pool fleet proof applies only to the recorded columns. A CG pricing certificate concerns its modeled graph and objective. Uncertified RMP endpoints are not full-model lower bounds. Fractional route weight is not the weighted objective. Charging optimality is separate from fleet optimality.

See [all original chain counts and CG stopping reasons](CHAIN_TABLES.md), [full original endpoint data](all_chain_extension_results.csv), [longer-search results](longer_gap_results.csv), and [source/pool validation](longer_gap_validation.json). A larger target can have a smaller recorded fleet because pools and time-limited searches differ; largest matches are not monotone cutoffs.
