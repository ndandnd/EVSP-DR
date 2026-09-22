# Scoped operations check — 21 September, 23:55 UTC

One scoped SSH collection succeeded. The previous successful operations snapshot was older than90minutes. No jobs, source pins, Docs, Slides or shared register entries were changed.

**Meaningful new result:** both53-trip and90-trip strict representation benchmarks completed successfully. Each compares original explicit, new explicit and new packed construction/pricing sequentially on the same node, using identical input hashes, event lattices and five deterministic dual vectors. All15generated routes per case pass the recorded individual-route replay. Completion/result/manifest hashes were independently reconciled.

| Trips | Implementation | Graph build (s) | Process peak RSS (GiB) | Mean pricing call (s) | Retained arcs |
|---:|---|---:|---:|---:|---:|
| 53 | original explicit | 840.232 | 9.122 | 260.950402 | 15,216,549 |
| 53 | new explicit | 361.782 | 9.122 | 261.144790 | 15,216,549 |
| 53 | new packed | 296.171 | 0.311 | 0.266991 | 5,203,229 |
| 90 | original explicit | 1356.740 | 26.192 | 519.786416 | 44,104,803 |
| 90 | new explicit | 616.835 | 26.193 | 499.200627 | 44,104,803 |
| 90 | new packed | 525.018 | 0.644 | 0.246363 | 14,964,253 |

Compared with original explicit, packed construction is2.837×/2.584× faster for53/90trips; peak process memory is29.350×/40.693× lower; mean tested pricing calls are977.375×/2,109.839× faster. The intermediate new-explicit arm isolates the deferred tie-key change: graph construction is2.322×/2.200× faster, with the same explicit graph metrics. All compared reduced costs agree exactly in the saved numeric values.

53-trip job668478 completed in1h09m12s on luxlab-cpu-02;90-trip job668479 completed in2h07m43s on jingjie-cpu-18. Slurm batch MaxRSS is9,581,712KiB and27,482,024KiB respectively; these batch peaks differ from the per-process implementation peaks above. Both allocations respected their16GiB/48GiB requests. Different cases ran on different nodes, so cross-size raw timing comparisons are not hardware controlled.

**Scope:** this extends the earlier26-trip fixed-dual benchmark to two larger cases. It is not a CG convergence, integer-fleet, target-attainment or full-GIRO dispatch result. These runs retain the same strict single-factor239.01kWh battery/35.8515kWh reserve/PARX60kW physics without shared capacity. No capacity-pricing acceleration claim follows. There are only five deterministic pricing queries per implementation and one allocation per case.

[Editable benchmark table](benchmark_results.csv) · [Exact metrics and source hashes](benchmark_results.json) · [All verification checks](findings.json) · [Collected source artifacts](artifacts/).

## Existing campaigns

- **45 jobs running in this scope:**44baseline graph builds and strict k17CG668430. All99pending stages retain genuine input dependencies:96baseline solver stages, k17MIP, k19CG and k19MIP. The48baseline case dependency audit and reserved-node exclusions pass.
- All44baseline graphs remain in arc construction. Latest source-row counts cover11.94–43.49% of rows, with observed process peaks8.086–24.666GiB. Source rows have unequal cost; these percentages are not elapsed-time completion estimates. No new complete graph/CG/MIP endpoint exists.
- The six earlier baseline graph preemptions remain the entire observed interruption history:84.6minutes lost, all requeued attempts running. No new failed, timed-out or out-of-memory allocation appears. CANCELLED batch-step records belong to those already-accounted preemptions, not new job failures.
- Strict k17 is277trips/10reference18E2duties within global prefix17. At collection it was3h53m58s into its4hCGbudget, actively solving restricted LPs; no final endpoint existed. Its last sampled LP log is not a pricing certificate or scientific endpoint. k19 still correctly waits for k17CG, and both MIPs wait only for their own CG.

[Graph progress samples](graph_progress.json) · [Scoped queue](squeue.txt) · [Dated accounting including requeues](sacct.txt) · [Raw snapshot](snapshot.json).

No immediate recovery action is indicated by this scope. Root owns the global queue and any next submissions. Existing source pins, true dependencies and historical holds remain preserved.
