# Current chain tables

Verified 11 September 2026, 23:51 EDT. Green means integer fleet matches GIRO target under the stated model; it does not mean all GIRO constraints are enforced. Orange means extra buses. Grey means no completed integer result.

## Covering with inherited previous-k columns

| Target buses | Chain 1 | Chain 2 | Chain 3 | Chain 4 | Chain 5 | Chain 6 |
|---|---|---|---|---|---|---|
| 2 | 🟩 2 | 🟩 2 | 🟩 2 | 🟩 2 | 🟧 3 | 🟩 2 |
| 3 | 🟩 3 | 🟩 3 | 🟩 3 | 🟩 3 | 🟩 3 | 🟩 3 |
| 4 | 🟩 4 | 🟩 4 | 🟩 4 | 🟩 4 | 🟩 4 | 🟩 4 |
| 5 | 🟩 5 | 🟩 5 | 🟩 5 | 🟩 5 | 🟩 5 | 🟩 5 |
| 6 | 🟩 6 | 🟩 6 | 🟩 6 | 🟩 6 | 🟩 6 | 🟩 6 |
| 7 | ⬜ — | 🟩 7 | 🟩 7 | 🟩 7 | 🟩 7 | 🟩 7 |
| 8 | ⬜ — | 🟩 8 | 🟩 8 | 🟩 8 | 🟩 8 | 🟩 8 |
| 9 | ⬜ — | ⬜ — | 🟩 9 | 🟩 9 | 🟩 9 | 🟩 9 |
| 10 | ⬜ — | ⬜ — | 🟩 10 | ⬜ — | 🟩 10 | 🟩 10 |

k2 starts from singletons; k3 onward inherits columns. Chain1 k7 timed out during initialization and blocks k8–10. Chain2 k9 exhausted its budget importing columns, has no final LP and its MIP export failed; k10 is still initializing. Chain4 k10 timed out during initialization. Chain3 k10 additionally includes validated routes from a fresh solver solution, not GIRO seeds. No warm k11–15 campaign has completed. All displayed warm fleets are proved within their saved pools; this does not automatically prove the unrestricted model optimum or charging optimum.

## Fresh covering, independent initialization at each size

| Target buses | Chain 1 | Chain 2 | Chain 3 | Chain 4 | Chain 5 | Chain 6 |
|---|---|---|---|---|---|---|
| 2 | 🟩 2 | 🟩 2 | 🟩 2 | 🟩 2 | 🟧 3 | 🟩 2 |
| 3 | 🟩 3 | 🟩 3 | 🟩 3 | 🟩 3 | 🟩 3 | 🟩 3 |
| 4 | 🟩 4 | 🟩 4 | 🟩 4 | 🟩 4 | 🟩 4 | 🟩 4 |
| 5 | 🟩 5 | 🟩 5 | 🟩 5 | 🟩 5 | 🟧 6 | 🟩 5 |
| 6 | 🟩 6 | 🟧 7 | 🟩 6 | 🟩 6 | 🟧 7 | 🟩 6 |
| 7 | 🟧 8 | 🟩 7 | 🟧 8 | 🟧 8 | 🟧 8 | 🟩 7 |
| 8 | 🟧 9 | 🟧 9 | 🟧 9 | 🟧 9 | 🟧 9 | 🟩 8 |
| 9 | 🟧 11 | 🟧 11 | 🟧 11 | 🟧 10 | 🟧 10 | 🟧 10 |
| 10 | 🟧 11 | 🟧 11 | 🟧 11 | 🟧 11 | 🟧 11 | 🟧 11 |
| 11 | 🟧 13 | 🟧 12 | 🟧 13 | 🟧 13 | 🟧 12 | 🟧 12 |
| 12 | 🟧 14 | 🟧 13 | 🟧 14 | 🟧 14 | 🟧 13 | 🟧 14 |
| 13 | 🟧 15 | 🟧 15 | 🟧 15 | 🟧 15 | 🟧 14 | 🟧 15 |
| 14 | 🟧 17 | 🟧 17 | 🟧 16 | 🟧 18 | 🟧 16 | 🟧 17 |
| 15 | 🟧 18 | 🟧 17 | 🟧 18 | 🟧 19 | 🟧 16 | 🟧 19 |

All 84 fresh cases completed; these are integer incumbents with differing MIP proof status. Both fresh and inherited campaigns use set covering and 240kWh/240kW baseline physics. Individual route replay passes; duplicate removal and shared station capacity are not certified by these tables.

## Separate station-capacity / depot-speed pilot

| Case | Baseline | PARX60 only | Capacity only | Capacity + PARX60 |
|---|---:|---:|---:|---:|
| k1 duty13408 | 1 | 1 | 1 | 1 |
| k1 duty13406 | 1 | 1 | 1 | 1 |
| k2, 23 trips | 2 | 2 | 3 | 3 |
| k3, 35 trips | 3 | 3 | 16 | 16 |

All16 cells have MIP outcomes. Capacity-constrained selected schedules pass station sweeps; six recovered CG cases are uncertified. k3 capacity runs added only3 columns in8h; one pricing call took7.15h versus0.006s for its LP. Sixteen is a saved-pool result, not a proved physical requirement. Baseline k2/k3 schedules violate the one-space station limit.

This is not almost every GIRO constraint: it has 240kWh batteries, opportunity-station counts and/or PARX60kW; opportunity charging remains constant240kW, reserve is0%, and no65% return-SOC floor is imposed. The aggregate return-energy tariff experiment is a separate five-bus,350kW cohort and is not combined with capacity. Nonlinear charging and driver rules are not included.

Sources: outputs/post_meeting_20260910/monitor/20260912T035212Z.json; fresh_covering_complete84.csv; capacity_deadline5_completed/records.json. Detailed provenance/proof fields remain in the experiment workbook.
