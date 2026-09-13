# Paired charge-start-fee comparison

Source collection: `outputs/zero_charge_start_fee_20260913/status_20260913T073532Z/collection.json`

| Case | k | Buses 0 | Buses 5 | Δ buses | Starts 0 | Starts 5 | Δ starts | Elec. 0 | Elec. 5 | Δ elec. | CG cert. 0/5 | Fleet proof 0/5 | Physical 0/5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| w1_k05 | 5 | 5 | 5 | 0 | 49 | 16 | 33 | 145.367 | 150.598 | -5.23072 | yes/yes | yes/yes | yes/yes |
| w1_k10 | 10 | 10 | 10 | 0 | 121 | 41 | 80 | 387.333 | 393.457 | -6.12391 | yes/yes | yes/yes | yes/yes |
| w1_k15 | 15 | — | — | — | — | — | — | — | — | — | no/no | —/— | —/— |
| w2_k05 | 5 | 5 | 5 | 0 | 35 | 10 | 25 | 113.851 | 143.417 | -29.5662 | yes/yes | yes/yes | yes/yes |
| w2_k10 | 10 | 10 | 10 | 0 | 90 | 41 | 49 | 284 | 315.593 | -31.5931 | yes/yes | yes/yes | yes/yes |
| w2_k15 | 15 | 15 | 15 | 0 | 158 | 79 | 79 | 501.334 | 515.64 | -14.3059 | yes/yes | yes/yes | yes/yes |
| w3_k05 | 5 | 5 | 5 | 0 | 27 | 7 | 20 | 89.1059 | 91.5648 | -2.45887 | yes/yes | yes/yes | yes/yes |
| w3_k10 | 10 | 10 | 10 | 0 | 67 | 25 | 42 | 221.056 | 225.958 | -4.90246 | yes/yes | yes/yes | yes/yes |
| w3_k15 | 15 | 15 | 15 | 0 | 119 | 51 | 68 | 389.509 | 398.826 | -9.31657 | yes/yes | yes/yes | yes/yes |
| w4_k05 | 5 | 5 | 5 | 0 | 51 | 12 | 39 | 151.77 | 166.518 | -14.7479 | yes/yes | yes/yes | yes/yes |
| w4_k10 | 10 | 10 | 10 | 0 | 107 | 28 | 79 | 319.902 | 340.512 | -20.6103 | yes/yes | yes/yes | yes/yes |
| w4_k15 | 15 | — | 15 | — | — | 55 | — | — | 520.761 | — | yes/yes | —/yes | —/yes |
| w5_k05 | 5 | 5 | 5 | 0 | 60 | 17 | 43 | 191.169 | 213.899 | -22.73 | yes/yes | yes/yes | yes/yes |
| w5_k10 | 10 | 10 | 10 | 0 | 112 | 24 | 88 | 344.63 | 381.749 | -37.1182 | yes/yes | yes/yes | yes/yes |
| w5_k15 | 15 | — | 15 | — | — | 57 | — | — | 548.238 | — | yes/yes | —/yes | —/yes |
| w6_k05 | 5 | 5 | 5 | 0 | 34 | 7 | 27 | 98.1401 | 100.932 | -2.79198 | yes/yes | yes/yes | yes/yes |
| w6_k10 | 10 | 10 | 10 | 0 | 87 | 27 | 60 | 279.241 | 292.699 | -13.4585 | yes/yes | yes/yes | yes/yes |
| w6_k15 | 15 | 15 | 15 | 0 | 158 | 70 | 88 | 500.911 | 497.75 | 3.16071 | yes/yes | yes/yes | yes/yes |

All deltas are fee 0 minus fee 5. Process success, CG pricing certification, finite-pool MIP scope, fleet proof, and physical replay remain separate CSV fields.

Charging totals are reported for selected routes before duplicate-trip removal. The baseline model does not validate cross-route charger capacity. A missing value is shown as an em dash and is never treated as zero.
