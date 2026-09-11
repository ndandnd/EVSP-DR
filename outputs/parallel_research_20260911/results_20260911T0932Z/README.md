# 05:32 EDT results — 11 September 2026

Fresh set-covering runs now match the target on all six chains at **k=3 and k=4**. Each is optimal within its saved pool, with individual selected-route replay passing. At k=2, chain 5 still needs three buses in its pool; this is not a proof that the full routing model needs three.

| Target | Chain 1 | Chain 2 | Chain 3 | Chain 4 | Chain 5 | Chain 6 |
|---|---:|---:|---:|---:|---:|---:|
| 2 | 2 | 2 | 2 | 2 | 3 | 2 |
| 3 | 3 | 3 | 3 | 3 | 3 | 3 |
| 4 | 4 | 4 | 4 | 4 | 4 | 4 |
| 6 | 6 | 7 | 6 | 6 | 7 | 6 |

These fleet values are proved optimal only within their respective saved pools. Shared station capacity and duplicate-trip removal are separate validation requirements. The exact results, times, source paths and hashes are in [the editable CSV](fresh_covering_results.csv).

Warm chain 2 now also matches k=7: CG 164.60 minutes; MIP 43.77 minutes. Both MIP stages reached optimality within the saved pool; individual replay passed, with 11 overcovered trips. No full-model integer proof is claimed.

## Pipeline and default-partition trial

The earlier dependency repair has produced 13 additional fresh MIP results. At the snapshot, the default trial had 65 started attempts: 36 completed and 29 running, with zero recorded preemptions. Running attempts are censored; these heterogeneous durations do not establish a one-hour survival rate. No new FAILED, TIMEOUT, OUT_OF_MEMORY or PREEMPTED state appeared in the queried main arrays. Two fresh CG cases and all six capacity-budget-extension CG cases remain running.

The large MIP array throttle was raised from 20 to 30 after verifying only nine small-array cases remained. This removes an avoidable split-array limit: all remaining large and small cases together number at most 39, below the intended combined 50. Job IDs, resources, solver budget, source pools and dependencies are unchanged. See [scheduler audit](concurrency_rebalance.json).

After the concurrency change, 37 default MIPs were verified running and only two remained pending on their CG prerequisites. The dated queue sample is `queue_after_rebalance.json`.
