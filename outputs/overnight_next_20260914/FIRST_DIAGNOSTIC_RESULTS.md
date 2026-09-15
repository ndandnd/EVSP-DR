# First overnight diagnostic results

Collection: 2026-09-15T01:53:59.251180+00:00. All nine support-only controls have completed; the26 paired augmentation MIPs remain running.

| Chain, target | Target buses | Buses needed using only final LP support | Fleet minimum proved in this pool |
|---|---:|---:|---|
| c1_k08 | 8 | 9 | yes |
| c1_k10 | 10 | 12 | yes |
| c2_k08 | 8 | 10 | yes |
| c3_k08 | 8 | 10 | yes |
| c4_k08 | 8 | 10 | yes |
| c4_k10 | 10 | 13 | yes |
| c5_k08 | 8 | 10 | yes |
| c5_k10 | 10 | 12 | yes |
| c6_k10 | 10 | 12 | yes |

All nine pass native individual-route replay with zero rejected/repaired columns. The full frozen donor pools match their targets; the restricted support-only pools do not. Thus at least some columns with zero weight in these donors' final LP solutions are necessary for target-fleet recovery. This statement concerns these saved pools and these selected cases. It does not identify every useful inactive column or prove general algorithmic performance. No new CG or full-model pricing certificate is produced by filtering a pool.

[Collection and result hashes](launch_collection.json), [normalizer validation](normalizer_validation.json), [frozen campaign](../lp_support_pool_diagnostic_20260914/README.md).
