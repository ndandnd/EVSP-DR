# Parent graph preparation timed out — 13 September, 05:38 EDT

**The shared graph for the 32-duty, 750-trip parent problem did not finish within its 12½-hour allocation.** Job 42509 ended at 05:14:40 EDT with scheduler state TIMEOUT. No saved parent cache or parent CG result exists. Slurm cancelled all 20 dependent jobs before they started.

| Stage | Observed outcome |
|---|---|
| Shared parent graph/cache preparation | Timed out after 12 h 32 min 22 s; limit was 12 h 30 min |
| Ten parent CG jobs | Cancelled before starting because the shared cache job did not succeed |
| Ten parent MIPs | Cancelled before starting; no new MIP result |
| Existing component and recombined schedules | Preserved; this timeout does not invalidate them |

This was **not preemption, an out-of-memory event or a Gurobi license failure**. The native license preflight passed. The job requested two CPUs and 128 GiB; the batch step recorded a maximum RSS of 26,603,696 KiB, about 25.4 GiB. Adding memory is not supported as the immediate remedy by this evidence.

The scheduler reports top-level ExitCode 0:0 together with TIMEOUT. That exit-code field alone is not success: the batch step was cancelled with signal 15, and stderr explicitly records the time limit. All dependent jobs have zero elapsed time and no start timestamp. Their cancellations must not count as MIP preemptions or solver failures after optimization began.

## What is known about the bottleneck

The command was preparing the shared event-network cache using `--event-network-cache-only`, lazy event arcs, 2.5-kWh SOC steps and 5-minute event blocks. Telemetry contains only the session-start record. There is no cache file, completed network-build phase or CG iteration. The evidence places the bottleneck in preparation before CG, but does not locate the internal operation responsible. It could include graph construction or cache-related work; the existing log is not detailed enough to distinguish them.

Source is `a0e0bb7681c8451e3cbbbfa06aef390026d9af4b`. The parent input hash is `4367335166098c6c50fb283b1cd3307a72720ea0b70fff4567b085af9a37e66e`. Physics and objective remain the baseline covering model with 240 kWh / 240 kW, flat tariffs, no shared capacity and no terminal floor. Original job/dependency records remain in `graph_recovery_retry2_20260912`; its borrowed source roots must be preserved.

The next useful diagnostic is instrumentation of graph/cache preparation before another full-size attempt, including elapsed time, progress and whether partial work can be saved. Merely repeating the same build would risk losing another 12½ hours. No replacement was submitted by this monitor.

## Current queue and evidence

At the 05:38 collection, no active EVSP–DR jobs remained. The 33 held historical tasks were untouched. The completed baseline chain, controlled algorithm and charge-start-fee results are unchanged; no new scientific result arrived. Unicorn access is working.

[Raw accounting, launch command, stderr and file hashes](graph_failure_evidence.json) and [compact execution audit](graph_parent32_timeout_20260913.json) retain the failure evidence. Cancelled CG jobs are 42512, 42514, …, 42530; cancelled MIPs are 42513, 42515, …, 42531. The existing preemption study records their cancellations separately from preemptions.
