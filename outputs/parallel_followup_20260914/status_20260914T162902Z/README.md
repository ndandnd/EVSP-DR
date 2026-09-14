# Longer searches on unchanged saved pools

| Case | Target | Prior buses | New buses | Fleet proved in pool | Fleet-stage minutes | Meaning |
|---|---:|---:|---:|---|---:|---|
| c1_k08_c200_longmip | 8 | 9 | 9 | no | 180.1 | fleet gap open |
| c2_k08_c200_longmip | 8 | 9 | 9 | no | 180.1 | fleet gap open |
| c2_k08_complementary_longmip | 8 | 9 | 9 | no | 180.0 | fleet gap open |
| c4_k08_c200_longmip | 8 | 9 | 9 | yes | 153.8 | target absent from this pool |
| c4_k08_complementary_longmip | 8 | 9 | 9 | yes | 133.2 | target absent from this pool |
| c5_k08_c200_longmip | 8 | 9 | 9 | no | 180.1 | fleet gap open |
| c5_k10_c200_longmip | 10 | 11 | 11 | no | 180.1 | fleet gap open |
| c5_k10_complementary_longmip | 10 | 11 | 11 | no | 180.0 | fleet gap open |
| c6_k10_c200_longmip | 10 | 11 | 11 | no | 180.0 | fleet gap open |
| c6_k10_complementary_longmip | 10 | 12 | 11 | yes | 121.0 | target absent from this pool |
| w2_k25_longmip | 25 | 26 | 25 | yes | 87.2 | target matched |
| w4_k19_resumed_pool_longmip | 19 | 20 | 19 | yes | 46.8 | target matched |
| w4_k22_longmip | 22 | 23 | 22 | yes | 32.7 | target matched |
| w6_k23_longmip | 23 | 24 | 23 | yes | 19.3 | target matched |

All 14 pairs have matching ordered-pool and input hashes; selected routes pass individual replay. These searches add no CG columns. Fleet search has at most three hours within 3½ hours total; charging uses the remaining time. A fleet proof above target establishes target absence only from that finite pool. Open gaps do not establish absence.

Across the original extension controls, 15 of 21 misses now have a verified target solution from the same pool. The other 6 remain unresolved. The continued C4 k19 pool is a separate treatment; its recovery does not add another original-control recovery.

The four recovered targets in this batch have fleet-proof times 87.2, 46.8, 32.7 and 19.3 minutes. C6 k23 therefore recovered within the original 30-minute fleet allowance; extra allocated time alone cannot explain every changed outcome. Hardware and parallel-search timing remain uncontrolled. Charging-cost optimality is separate from fleet proof.

Baseline physics omit shared station capacity and a terminal-SOC floor. No branch-and-price or full-model integer proof is claimed.

[Editable before/after values and source hashes](results.csv); [original gap recovery map](original_gap_recoveries.csv).
