# EVSP–DR Scaglione resource policy

Standing user instruction, confirmed 9 September 2026.

Reserve scaglione-compute-01 for GPU users. GPU jobs also need CPU resources; CPU-only jobs must not exhaust that node. Every EVSP–DR Scaglione CPU/MIP submission must include `--exclude=scaglione-compute-01`. Note that scaglione-cpu-01 is a different node and is not the reserved GPU node.

Use the remaining Scaglione CPU machines for MIP computations, subject to the scheduler and other users. Do not interpret the request to reduce usage as a blanket shutdown of all Scaglione work. Run column generation on default_partition where suitable; save supported checkpoints for preemption. A Gurobi incumbent checkpoint does not preserve its branch-and-bound tree.

Do not release held historical campaigns automatically. Verify node exclusion in scontrol after submission. This policy remains in force until the user explicitly changes it.

Clarification, 10 September 2026: scaglione-compute-01 also belongs to default_partition. The exclusion is a physical-node reservation, not just a partition-specific rule. Include --exclude=scaglione-compute-01 in ALL EVSP–DR CPU-only submissions, including default-partition CG, and verify the effective exclusion after submission.

## Default-partition concurrency — user instruction, 10 September 2026

For independent EVSP–DR column-generation cases on default_partition, use a default array concurrency of 50, or the number of cases if fewer than 50 exist. Do not impose arbitrary caps of 2, 16 or 24. A smaller cap requires a documented cluster restriction or concrete measured bottleneck (per-job RAM, aggregate node RAM, I/O, license service or other resource), with the reason recorded in the campaign manifest. More than 50 is allowed when justified by workload size and resource evidence.

Request realistic CPU and RAM per job; Slurm controls actual admission and reserves these resources. Raising the array throttle does not raise an individual job's memory limit. Preserve true data dependencies: a chain that imports the previous k's columns is sequential by design, not artificially throttled. If several independent chains exist, they may proceed in parallel.

This supersedes the earlier pilot default of 16. It does not release held historical jobs, change Scaglione MIP limits, or relax the scaglione-compute-01 exclusion. Verify actual array throttles and node exclusions after submission. At this audit, the Cornell Unicorn documentation and inspected user/QOS fields contain no explicit concurrency restriction below 50; this is not a claim that all system/parent-account limits are absent.

## Overnight default-partition MIPs — 11 September 2026 UTC

The user authorizes default_partition for the one-hour saved-pool MIPs, accepting preemption and lost branch-and-bound search. Pending fresh-covering MIPs may migrate there; retain exact scientific settings and true input dependencies. Keep per-attempt job IDs, scheduler states, timing/resource/priority samples and output hashes in the MIP preemption study. No automatic requeue for wrappers unable to restart safely; interrupted retries require unique attempt outputs and preserved prior records. Preemption risk is measured, not presumed zero. Existing Scaglione MIPs remain usable. scaglione-compute-01 remains excluded everywhere and held historical jobs stay untouched.

## Default MIPs — user clarification 11 September evening

Use default_partition for new table-filling MIPs. User accepts preemption and requeue; use restart-safe output paths and retain attempt statistics. Gurobi tree search restarts. Exclude scaglione-compute-01 and preserve held jobs. Overnight extension worker uses explicit --requeue and job/restart output directories.
