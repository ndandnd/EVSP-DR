# Paired charge-start-fee comparison

Source collection: `outputs/zero_charge_start_fee_20260913/status_20260913T053436Z/fee_results.json`

| Case | k | Buses 0 | Buses 5 | Δ buses | Starts 0 | Starts 5 | Δ starts | Elec. 0 | Elec. 5 | Δ elec. | CG cert. 0/5 | Fleet proof 0/5 | Physical 0/5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| w1_k05 | 5 | 5 | 5 | 0 | 49 | 16 | 33 | 145.367 | 150.598 | -5.23072 | yes/yes | yes/yes | yes/yes |
| w1_k10 | 10 | — | — | — | — | — | — | — | — | — | no/yes | —/— | —/— |
| w1_k15 | 15 | — | — | — | — | — | — | — | — | — | no/no | —/— | —/— |
| w2_k05 | 5 | 5 | 5 | 0 | 35 | 10 | 25 | 113.851 | 143.417 | -29.5662 | yes/yes | yes/yes | yes/yes |
| w2_k10 | 10 | — | — | — | — | — | — | — | — | — | yes/yes | —/— | —/— |
| w2_k15 | 15 | — | — | — | — | — | — | — | — | — | no/yes | —/— | —/— |
| w3_k05 | 5 | 5 | 5 | 0 | 27 | 7 | 20 | 89.1059 | 91.5648 | -2.45887 | yes/yes | yes/yes | yes/yes |
| w3_k10 | 10 | — | — | — | — | — | — | — | — | — | yes/yes | —/— | —/— |
| w3_k15 | 15 | — | — | — | — | — | — | — | — | — | yes/yes | —/— | —/— |
| w4_k05 | 5 | 5 | 5 | 0 | 51 | 12 | 39 | 151.77 | 166.518 | -14.7479 | yes/yes | yes/yes | yes/yes |
| w4_k10 | 10 | — | — | — | — | — | — | — | — | — | yes/yes | —/— | —/— |
| w4_k15 | 15 | — | — | — | — | — | — | — | — | — | no/yes | —/— | —/— |
| w5_k05 | 5 | 5 | 5 | 0 | 60 | 17 | 43 | 191.169 | 213.899 | -22.73 | yes/yes | yes/yes | yes/yes |
| w5_k10 | 10 | — | — | — | — | — | — | — | — | — | no/yes | —/— | —/— |
| w5_k15 | 15 | — | — | — | — | — | — | — | — | — | no/yes | —/— | —/— |
| w6_k05 | 5 | 5 | 5 | 0 | 34 | 7 | 27 | 98.1401 | 100.932 | -2.79198 | yes/yes | yes/yes | yes/yes |
| w6_k10 | 10 | 10 | — | — | 87 | — | — | 279.241 | — | — | yes/yes | yes/— | yes/— |
| w6_k15 | 15 | — | — | — | — | — | — | — | — | — | yes/yes | —/— | —/— |

All deltas are fee 0 minus fee 5. Process success, CG pricing certification, finite-pool MIP scope, fleet proof, and physical replay remain separate CSV fields.

Charging totals are reported for selected routes before duplicate-trip removal. The baseline model does not validate cross-route charger capacity. A missing value is shown as an em dash and is never treated as zero.
