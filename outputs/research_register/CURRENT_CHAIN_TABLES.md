# Current chain tables

Integer tables rechecked 12 September 2026, 01:27 EDT. Green means integer fleet matches GIRO target under the stated model; it does not mean all GIRO constraints are enforced. Orange means extra buses. Grey means no completed integer result.

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

k2 starts from singletons; k3 onward inherits columns. The earlier full-pool treatment timed out during initialization at Chain1 k7 and Chain4 k10; Chain2 k9 exhausted its import budget without a final LP. The new bounded-import treatment has now completed pricing-certified CG for all three cases; their new MIPs are not completed in this snapshot. Chain3 k10 additionally includes validated routes from a fresh solver solution, not GIRO seeds. No warm integer k11–15 result is available yet. The new bounded-import treatment has pricing-certified CG at chain3 k11–12 and chains5–6 k11; those are LP results, not integer matches. All displayed warm fleets are proved within their saved pools; this does not automatically prove the unrestricted model optimum or charging optimum.

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

Sources: outputs/post_meeting_20260910/monitor/20260912T052658Z.json; fresh_covering_complete84.csv; capacity_deadline5_completed/records.json. Detailed provenance/proof fields remain in the experiment workbook.

Latest extension and decomposition evidence: [01:27 EDT verified results](../overnight_extension_20260912/RESULTS_20260912T052658Z.md). Two component groups each proved eight buses within their pools; no combined 32-duty solution is claimed.
