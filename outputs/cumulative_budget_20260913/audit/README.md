# Retrospective cumulative-budget audit

The panel is fixed to all six existing chains at k5, k8, k10 and k15: 24 inputs, without filtering on outcomes. `targets.json` is the preparation interface; `budgets.csv` is the compact table. All ancestry is traced through actual `inherited_event_pool_audit.source_status` edges. The six chains each run consecutively from k2 to k15. There are 84 unique statuses, 78 consumed parent edges, no missing statuses, no same-k repeats, and every current parent hash matches the hash consumed by its child.

Every k2 root uses singletons, no validated seed route file, and no inherited parent. All 84 statuses report pricing certificates. Parent MIPs are absent from the ancestry and contribute zero to the CG budgets. Earlier sources use `ecb60c154a9a5db385e3a573949ec9fd0a737af3` (47 stages), followed by `e091a4dba549510238507ef5e5367abea958bd30` (37 stages). This supports a retrospective budget comparison, not an isolated causal claim about implementation speed.

## Budget definitions

Primary allowance is `ceil(sum(wall_s))` over every native CG ancestor including the target. Each chain prefix has 4, 7, 9 or 14 distinct stages at k5, k8, k10 or k15. Native wall ranges are approximately 900–12,193 seconds at k5, 6,273–37,824 at k8, 21,198–72,589 at k10, and 37,927–83,321 at k15. The 24 prospective primary allowances sum to 222.232 hours; overlapping historical prefixes must not be treated as additional unique historical experiments.

The source excerpts in `timing_source_evidence.json` establish that wall time starts before problem construction and includes graph loading, import and CG. Phase telemetry overhead is subtracted where that implementation supports it. All 84 cases have cache hits; their original graph construction therefore occurred outside the reported CG wall time. No native statuses were resumed.

Sensitivity adds the recorded external graph construction of **ancestors only**, excluding the common target graph. It does not add graph-load time a second time. The 24 sensitivity allowances sum to 350.05 hours. `cache_original_build_s` measures the graph constructor; it excludes pickle serialization, pickle hashing, and pre-graph problem construction in the graph-producing invocation. Accordingly this is a measured graph-construction sensitivity, not a complete end-to-end historical preparation accounting. Those missing costs are not silently assigned zero. Target graph construction remains separately recorded as common cost. If a fresh primary run certifies early, its sensitivity endpoint can be shared; a capped fresh run may continue its own same-instance checkpoint to the larger cumulative limit.

## CPU evidence

`accounting.json` preserves submission mappings and raw `sacct` evidence, explicitly bounded to 8–14 September 2026 to avoid recycled historical job IDs. All 84 mapped CG allocation records are COMPLETED, use eight allocated CPUs, and have zero restarts. `actual_cpu_s` sums allocation-level Slurm TotalCPU; `allocated_cpu_s` sums CPUTimeRAW. Batch and extern steps are not added again. The first is measured CPU usage and the second allocation exposure; neither is inferred from CG wall time. External graph allocations and intermediate MIPs are excluded from both fields.

For k15, cumulative measured CPU hours across C1–C6 are 67.72, 82.58, 113.40, 93.23, 159.53 and 50.36; allocation CPU hours are 127.63, 129.52, 133.28, 122.88, 185.87 and 85.10. Matching elapsed budget and eight-CPU allocation does not guarantee matching actual CPU consumption.

## Artifacts and limitations

All 24 input files were hashed on Unicorn and matched their status provenance. Cache manifest files were hashed and retained in the ancestry evidence. Their large pickle hashes are producer-recorded, not independently rehashed by this audit; preparation must verify each selected pickle and establish revision compatibility before reuse. Journals were not read; consumed journal hashes are preserved from solver ancestry audits.

Historical MIP references, hashes, fleet/proof and physical-validation fields are recorded for 23 targets. C3k10's expected original warm MIP artifact is absent and remains explicitly missing; no substitute result is silently chosen. The other 23 recorded MIPs attain their target fleets. CG certificate, finite-pool MIP proof and physical replay remain separate fields.

Reproduction: run `collect_ancestry.py` and `collect_accounting.py` read-only on Unicorn, saving their stdout as `ancestry.json` and `accounting.json`; then run local `build_targets.py`. This directory is input/evidence preparation only and submits no jobs.
