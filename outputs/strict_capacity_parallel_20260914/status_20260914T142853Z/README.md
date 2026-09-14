# Charging-constraint pilot: verified results

Source collection started 2026-09-14T14:28:54.209053+00:00. Missing results are pending, not infeasible.

| Case | CG minutes | Pricing certified | Matched MIP buses | Fleet proved in pool | Charging-related cost | Shared capacity check |
|---|---:|---|---:|---|---:|---|
| k1_13408_flat_combined_reference | 3.0 | yes | 1 | yes | 36.536 | yes |
| k1_13408_flat_combined_prefix_memo | 2.5 | yes | 1 | yes | 36.536 | yes |
| k1_13408_peak12_combined_reference | pending | pending | pending | pending | pending | pending |
| k1_13408_peak12_combined_prefix_memo | pending | pending | pending | pending | pending | pending |
| k2_flat_240r0_baseline | 64.3 | yes | 2 | yes | 79.272 | no |
| k2_flat_240r0_capacity | pending | pending | pending | pending | pending | pending |
| k2_flat_240r0_parx60 | 65.9 | yes | 2 | yes | 79.272 | no |
| k2_flat_240r0_combined | pending | pending | pending | pending | pending | pending |
| k2_flat_236p44r15_baseline | 30.6 | yes | 2 | yes | 91.712 | no |
| k2_flat_236p44r15_capacity | pending | pending | pending | pending | pending | pending |
| k2_flat_236p44r15_parx60 | 42.8 | yes | 2 | yes | 91.712 | no |
| k2_flat_236p44r15_combined | pending | pending | pending | pending | pending | pending |

All four completed k2 controls match two buses and have pricing certificates. Their matched one-hour MIPs prove fleet and charging objectives within their respective saved pools. Shared capacity was disabled: each selected solution has two simultaneous connections at station 2190L, where the documented limit is one. These results do not establish feasibility with station capacity enforced. Those four k2 treatments remain separate.

The 236.44-kWh treatment also applies a 35.466-kWh (15%) reserve. It changes battery and reserve together. All cases use constant charging power and no 65% terminal target. Individual route feasibility in this dedicated solver is by construction; it is not a separate continuous replay audit.

Capacity/combined k2 cases receive 220 minutes of CG versus 110 for baseline/PARX-only. The matched MIPs equalize only final integer-search allowances. These physics cells are feasibility pilots, not isolated runtime-causal estimates. Reference-versus-cached-pricing pairs have matching settings; these are single runs, not controlled runtime repetitions.

Charging-related cost includes electricity and the modeled charging-start fee. A pool proof is distinct from the CG pricing certificate and from shared-capacity validation.

[Editable result table and source hashes](results.csv).
