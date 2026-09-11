# 04:31 EDT results — 11 September 2026

Warm chain 5 now matches target k=8 with **eight buses**. Both MIP stages proved optimal within its 29,421-column pool. The CG source reports a pricing certificate for the conservative expanded-grid model. Selected routes passed individual physical replay; nine trips are overcovered. Duplicate removal and shared station capacity are not validated by this result.

| Measurement | Time |
|---|---:|
| CG reported total | 214.69 minutes |
| Inherited-column import telemetry | 210.88 minutes |
| MIP total | 10.37 minutes |

Import dominates elapsed CG time. Phase timers can overlap, so their sum is not a disjoint runtime decomposition. Inheritance improves the fleet result here compared with the fresh nine-bus pool result, but it does not establish a speed improvement.

Default MIP reliability remains 23 completed started attempts and zero recorded preemptions. Access is healthy. No new failed, timed-out, out-of-memory or preempted allocations appeared in the queried main arrays.

## Dependency repair

The snapshot showed fresh CG tasks completed in accounting while their corresponding freeze jobs still had unfulfilled `aftercorr` dependencies. This delayed their downstream MIPs. The repair retains all existing job IDs, resources and scientific settings. Completed prerequisites are discharged only after checking successful accounting exit, final CG status and its column journal; unfinished cases use explicit `afterok` task dependencies. MIPs depend on their own freeze task. Exact before/after scheduler records and source hashes are retained in `dependency_repair.json`.

Original snapshot and solver evidence: [evidence.json](evidence.json). The register and Google Doc use the 04:31 snapshot; the dependency repair is a subsequent scheduler action, not a new scientific result.

Verification after repair: 25 default MIPs and 27 freeze jobs were running. Three freeze tasks still correctly waited for their running CG predecessors. See [dated queue verification](dependency_verification.json).
