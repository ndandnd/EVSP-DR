# Evening experiments — verified results

Snapshot: 2026-09-15T14:15:09.672397+00:00. Counts include only published endpoints.

**Compact seeds: 36/36 CG endpoints, 36 certified; 36/36 MIP endpoints, 33 matching target.**

The core preserves the previous integer solution and every positive-weight LP route. The other treatment keeps that core and fills to 512 distinct trip sets. Both use the same input, code and budgets. Prior computation is recorded separately.

| Case | Core routes | Buses: core | Buses: 512 | CG minutes: core | CG minutes: 512 |
|---|---:|---:|---:|---:|---:|
| C1, k=8 | 95 | 8 | 8 | 38.5 | 31.4 |
| C2, k=8 | 83 | 8 | 8 | 15.5 | 14.2 |
| C3, k=8 | 82 | 8 | 8 | 10.2 | 5.5 |
| C4, k=8 | 74 | 8 | 8 | 48.0 | 36.4 |
| C5, k=8 | 102 | 8 | 8 | 14.0 | 6.5 |
| C6, k=8 | 39 | 8 | 8 | 8.7 | 4.5 |
| C1, k=10 | 119 | 10 | 10 | 93.8 | 49.0 |
| C2, k=10 | 132 | 10 | 10 | 38.9 | 29.6 |
| C3, k=10 | 126 | 10 | 10 | 21.8 | 15.7 |
| C4, k=10 | 116 | 10 | 10 | 43.8 | 24.1 |
| C5, k=10 | 112 | 10 | 10 | 21.6 | 17.9 |
| C6, k=10 | 99 | 10 | 10 | 12.5 | 8.1 |
| C1, k=15 | 208 | 16 | 16 | 168.4 | 147.6 |
| C2, k=15 | 216 | 15 | 15 | 161.2 | 159.6 |
| C3, k=15 | 198 | 17 | 15 | 50.9 | 46.5 |
| C4, k=15 | 213 | 15 | 15 | 166.5 | 132.3 |
| C5, k=15 | 192 | 15 | 15 | 99.4 | 100.2 |
| C6, k=15 | 234 | 15 | 15 | 98.5 | 77.1 |

35/36 available MIPs have a fleet proof within their own pools; all pass individual-route replay. At k=15, 12/12 MIPs are published. This is a completion-selected subset; unfinished cases prevent an overall success-rate or timing claim. A CG certificate applies to the recorded discretized weighted model and tolerance. Fractional route weight is not a fleet-only lower bound.

Baseline physics: covering, 240 kWh / 240 kW, charging-start fee 5, without shared charger capacity or a terminal-SOC floor. CG allows four hours; MIP allows three hours of fleet search within 3.5 hours total. Stage two constrains fleet to be no greater than the incumbent and minimizes charging-related cost.

[All values, exact stopping reasons and source hashes](compact_seed_results.csv).

The nine longer searches on unchanged large-chain pools have 9 published endpoints. The fixed-state capacity calls are recorded as single-call diagnostics, never automatically as CG or MIP results. The two earlier wrapper failures and corrected attempts remain separate.
