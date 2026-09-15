# Evening experiments — verified results

Snapshot: 2026-09-15T00:03:59.334580+00:00. Counts include only published endpoints.

**Compact seeds: 20/36 CG endpoints, 20 certified; 14/36 MIP endpoints, 14 matching target.**

The core preserves the previous integer solution and every positive-weight LP route. The other treatment keeps that core and fills to 512 distinct trip sets. Both use the same input, code and budgets. Prior computation is recorded separately.

| Case | Core routes | Buses: core | Buses: 512 | CG minutes: core | CG minutes: 512 |
|---|---:|---:|---:|---:|---:|
| C1, k=8 | 95 | pending | 8 | 38.5 | 31.4 |
| C2, k=8 | 83 | 8 | pending | 15.5 | 14.2 |
| C3, k=8 | 82 | 8 | 8 | 10.2 | 5.5 |
| C4, k=8 | 74 | pending | pending | pending | 36.4 |
| C5, k=8 | 102 | 8 | 8 | 14.0 | 6.5 |
| C6, k=8 | 39 | 8 | 8 | 8.7 | 4.5 |
| C1, k=10 | 119 | pending | pending | pending | pending |
| C2, k=10 | 132 | pending | pending | 38.9 | 29.6 |
| C3, k=10 | 126 | pending | 10 | 21.8 | 15.7 |
| C4, k=10 | 116 | pending | 10 | pending | 24.1 |
| C5, k=10 | 112 | 10 | 10 | 21.6 | 17.9 |
| C6, k=10 | 99 | 10 | 10 | 12.5 | 8.1 |
| C1, k=15 | 208 | pending | pending | pending | pending |
| C2, k=15 | 216 | pending | pending | pending | pending |
| C3, k=15 | 198 | pending | pending | pending | pending |
| C4, k=15 | 213 | pending | pending | pending | pending |
| C5, k=15 | 192 | pending | pending | pending | pending |
| C6, k=15 | 234 | pending | pending | pending | pending |

14/14 available MIPs have a fleet proof within their own pools; all pass individual-route replay. At k=15, 0/12 MIPs are published. This is a completion-selected subset; unfinished cases prevent an overall success-rate or timing claim. A CG certificate applies to the recorded discretized weighted model and tolerance. Fractional route weight is not a fleet-only lower bound.

Baseline physics: covering, 240 kWh / 240 kW, charging-start fee 5, without shared charger capacity or a terminal-SOC floor. CG allows four hours; MIP allows three hours of fleet search within 3.5 hours total. Stage two constrains fleet to be no greater than the incumbent and minimizes charging-related cost.

[All values, exact stopping reasons and source hashes](compact_seed_results.csv).

The nine longer searches on unchanged large-chain pools have 0 published endpoints. The fixed-state capacity calls are recorded as single-call diagnostics, never automatically as CG or MIP results. The two earlier wrapper failures and corrected attempts remain separate.
