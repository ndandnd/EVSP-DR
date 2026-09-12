# Cluster resource policy

**Independent CG arrays on default_partition: default concurrency 50, or all cases if fewer exist.** A lower throttle requires a recorded cluster-policy limit or measured resource reason. True previous-k dependencies remain sequential.

This replaces the earlier defaults of 2 and 16. The current capacity/speed pilot has only 16 cells, so increasing its limit beyond 16 does not create additional work. Live array 772080 was changed to 50 and verified; source execution and per-job memory were unchanged.

| Setting checked on 10 September | Observation |
|---|---|
| User association and normal QOS | Inspected concurrency/resource-limit fields have no explicit values below 50 |
| Cluster array index limit | MaxArraySize 1001; this is not a concurrency limit |
| Resource allocation | select/cons_tres with CR_CORE_MEMORY; CPU and memory requests enter scheduling |
| Pilot per-job RAM | 24GiB requested; completed sampled jobs used about 0.43–3.55GiB MaxRSS |
| Observed interrupted pilot tasks | TIMEOUT, not OUT_OF_MEMORY |
| Reserved physical node | scaglione-compute-01 excluded from every CPU-only job |
| Scaglione MIPs | Existing allocation/concurrency policy retained; held 537227 remains held |

The inspected fields do not prove that no parent-account or system limit exists. Slurm may still queue jobs for priority or available resources. An array limit is not an individual job’s memory allowance; raising it cannot fix a job that exceeds its own request. Correct per-job requests and measured memory/I/O behavior guide any exception.

Cornell’s [Unicorn guide](https://it.coecis.cornell.edu/researchit/using-the-unicorn-cluster/) requires accurate resource requests and describes the default partition. Its [resource FAQ](https://it.coecis.cornell.edu/researchit/using-the-unicorn-cluster/unicorn-faqs/) explains scheduler admission. Neither inspected page states a 50-job prohibition. Slurm’s [array documentation](https://slurm.schedmd.com/job_array.html) defines the optional percent throttle.

Evidence: ../post_meeting_20260910/capacity_speed/concurrency50_cluster_limits.json and concurrency50_update.json. Durable policy: /home/nc437/ladder-lite/SCAGLIONE_RESOURCE_POLICY.md and local ../../AGENTS.md. Future pilot launcher default is 50 in commit 9422455; commit 37d26e7 records the requested concurrency and task-count rule in submission metadata.

## Overnight default-partition MIPs — 11 September 2026 UTC

The user authorizes default_partition for the one-hour saved-pool MIPs, accepting preemption and lost branch-and-bound search. Pending fresh-covering MIPs may migrate there; retain exact scientific settings and true input dependencies. Keep per-attempt job IDs, scheduler states, timing/resource/priority samples and output hashes in the MIP preemption study. No automatic requeue for wrappers unable to restart safely; interrupted retries require unique attempt outputs and preserved prior records. Preemption risk is measured, not presumed zero. Existing Scaglione MIPs remain usable. scaglione-compute-01 remains excluded everywhere and held historical jobs stay untouched.

## Default MIPs — user clarification 11 September evening

Use default_partition for new table-filling MIPs. The user accepts preemption and requeue; preserve unique per-restart outputs and count lost attempts. Do not claim Gurobi tree resume. New overnight workers have restart-safe MIP output paths and explicit --requeue. The compute-01 exclusion and held-job protections remain.
