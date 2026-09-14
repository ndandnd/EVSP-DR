# Capacity-pricing boundary launch — 2026-09-14

Eight independent, matched k1 diagnostics are running. Each duty 13405–13408 has a reference and prefix-memo capacity-pricing cell with flat prices, 240 kWh initial energy, reserve 0, 13,200 seconds of CG, a 600-second diagnostic finite-pool MIP, and a four-hour allocation.

## Immutable execution identity

- Driver commit: `309d98d266ebaf6b7e99543a67f8f2be5736874a` (tracked-clean detached checkout checked by every worker).
- Manifest SHA-256: `05f726731d98dc7763d33d0ad78394a45ada1bbf1c838783f0503291a9c17ead`.
- Freeze SHA-256: `3064f22abff20ab97adf567da07a2a0f22d376471c3490f26150733a4e1ed3f7`.
- Jobs record SHA-256: `e070bc4002c5e487dda46ee22c5fb86b1472d4b11f615f2a9a38d0777ca6c0c9`.
- Objective constants are frozen as bus cost 100000, charge-start cost 5, energy-cost premium 1, and artificial cost 500000.
- All four duty CSVs, flat prices, reference dictionary, and deadhead table are hash-bound in `manifest.json`; observed hashes matched before launch.

## Validation

- Five focused local tests passed, including the eight-cell factorial and all 16 actual driver parser invocations.
- Native smoke job 189147 completed on `snavely-cpu-02` with exit 0. Gurobi 12.0.3 used one thread; the actual CG master had 7,813 rows and 26 columns. The deliberately 15-second CG stopped without a certificate (`cg_wall_limit`), and the saved-pool two-stage MIP completed OPTIMAL. This validates the intended incomplete-CG-to-usable-pool path and cluster license.
- The existing strict-capacity `pilot` adapter accepted the production root. At the postlaunch snapshot it reported eight running attempt-progress rows and zero native records, as required before any stage has completed.

## Jobs

| index | case | job | initial state |
|---:|---|---:|---|
| 0 | k1_13405_flat_capacity_reference | 189164 | RUNNING |
| 1 | k1_13405_flat_capacity_prefix_memo | 189165 | RUNNING |
| 2 | k1_13406_flat_capacity_reference | 189166 | RUNNING |
| 3 | k1_13406_flat_capacity_prefix_memo | 189167 | RUNNING |
| 4 | k1_13407_flat_capacity_reference | 189168 | RUNNING |
| 5 | k1_13407_flat_capacity_prefix_memo | 189169 | RUNNING |
| 6 | k1_13408_flat_capacity_reference | 189170 | RUNNING |
| 7 | k1_13408_flat_capacity_prefix_memo | 189171 | RUNNING |

All jobs use the default partition, one CPU, 24 GB, `--export=NONE`, `--no-requeue`, and exclude `scaglione-compute-01`. The scheduler returned a unique numeric ID for every submission; no reconciliation or retry was needed.

## Interpretation

A pricing deadline does not certify the full CG LP and leaves terminal exact minimum reduced cost null. A completed MIP proves only its saved finite pool. Selector equivalence requires matching exact terminal status and normalized routes; censored progress alone is descriptive. The driver reports route feasibility by construction and its station-capacity audit; this campaign adds no independent route replay. Because k1 has one selected route, these runs diagnose pricing behavior rather than multi-route charger contention.

The existing adapter is reused with `kind=pilot`. The remote collector and `Register.capacity_speed` now register `capacity_pricing_boundary_20260914`. A focused normalization of the four completed duty-13408 production endpoints produced two certified CG rows and two separate OPTIMAL finite-pool MIP rows with the expected k1 capacity metadata; `normalizer_validation.json` preserves those assertions. No full collection or register rebuild was run, and no document was changed.

Fresh prior strict-k2 evidence is saved separately in `strict_capacity_k2_audit_20260914T1659Z.json` and `.md`.
