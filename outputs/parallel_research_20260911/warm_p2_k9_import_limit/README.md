# Warm chain2 k9 import limit

## Warm chain 2, k=9 — import exhausts CG budget; freeze fails

Snapshot 20260911T221530Z: job810332 completed at scheduler level after08:09:19, but CG stop_reason is wall_limit and final/final_lp are null. Import reoptimized and accepted42,732 predecessor columns using8workers in29,324.84s (488.75min). Overall saved wall time29,337.41s (488.96min). There is no new LP endpoint or pricing certificate. Unlike P1k7, imported columns were saved and inherited by running child k10 job810333.

Freeze810974 failed after3s; its exact error is `no usable terminal source for k09_p2: continuation: source retains artificials for k09_p2; baseline: source retains artificials for k09_p2`. This rejection is not evidence of actual positive artificials: the source final LP fields are absent. MIP810975 is pending DependencyNeverSatisfied and has not optimized. Do not bypass validation or relabel absent LP fields as zero. Recovery needs an explicit validated terminal-RMP/pool export from the saved journal, or a tested continuation, with no fabricated pricing certificate. No blind retry was submitted.

This corroborates an inherited-column initialization bottleneck independently of the capacity-pricing issue. The mathematical feasibility of the full k9 model has not been disproved. Source path, input/provenance and predecessor hashes are in cg_record.json.
