# Live overnight queue — 14 September, 16:13 EDT

99 EVSP–DR allocations are running on the default partition. Another 48 solver stages wait for actual inputs; 18 conditional graph-timeout checks wait separately. The held historical array contains 33 tasks and is excluded. No invalid dependency, missing predecessor or use of scaglione-compute-01 was found. No scheduler setting or job was changed during this check.

| Experiment | Running allocations | What it resolves |
|---|---:|---|
| Previous-k integer routes versus LP-selected routes | 7 CG + 22 MIPs | Which small inherited route set helps the integer solution? |
| Combine decomposed route pools | 40 MIPs | Can routes from different partitions improve the 32-duty fleet? |
| Fresh pools at about four hours | 3 MIPs | Compare fresh and seeded pools under similar CG allowances. |
| Remaining chain steps through k=25 | 3 CG + 1 MIP | Finish the current larger-instance ladder. |
| Graphs for k=26–28 | 18 | Prepare the next three targets in each of six chains. |
| Charging-capacity diagnostics | 5 combined workflows | Identify difficult duties and compare pricing implementations. |

The five capacity allocations include CG and a short MIP; this census does not claim their exact internal phase. Counts describe allocated jobs, not measured CPU utilization. Every unresolved dependency has a recorded predecessor. A chain's k+1 CG requires k's saved columns; each final MIP requires its own CG output. Independent experiments already provide substantial parallelism, so do not duplicate these campaigns to raise the queue count.

Source: snapshot.json at 2026-09-14T20:13:51.925793+00:00, with raw squeue/sacct, source manifests and graph progress. summary.json retains every job and dependency edge. The scheduler observations are later than the scientific results collection ending at 16:09 EDT.
