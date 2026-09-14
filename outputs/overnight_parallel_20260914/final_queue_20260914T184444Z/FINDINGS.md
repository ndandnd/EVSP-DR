# Final overnight queue audit

At **18:44:44 UTC on14September2026**,109EVSPjobs were RUNNING:

| Work | Running |
|---|---:|
| CG |21:12seed-content,3extension,6capacity-boundary |
| MIP |70:46second decomposition wave,1first decomposition wave,18seed,3prefix controls,2extension |
| Graph construction |18 |
| Conditional graph-recovery gates |0running;18pending |

**55research solver jobs wait on true dependencies, plus18operational recovery gates.** No job was pending for resources or priority. Every unfulfilled dependency parent is present in the fresh queue; no DependencyNeverSatisfied or missing-parent condition was observed. The33tasks in historical held array537227remain held and excluded from these counts. No other-project queue rows were observed. No EVSP CPU job used reservedscaglione-compute-01. This audit submitted, changed or cancelled nothing.

The first decomposition campaign has45of46production MIPs completed and1running; all46second-wave production MIPs are running. Each campaign's native fixture is validation, excluded from production counts. Seed-content work has24CGscompleted/12running and6MIPscompleted/18running/12waiting. All3prefix-control MIPs are running.

All18large graphs are advancing; none has newly completed. Their source-node progress ranges32.7–53.1% after approximately6.07hours, with newest progress records younger than56seconds. Per-case timestamps and counters are in the JSON. The conditional timeout gates remain available if the original builds exhaust their captured deadlines.

Running allocations total536CPUs and4416GiB of **requested memory**, not measured RAM use. Raw batched accounting contains some historical/reused array IDs; no historical resource/failure conclusions are drawn from those undated matches. Current allocations and dependency-parent checks use the fresh queue. Capacity phase was independently confirmed from worker/log metadata at18:47:12UTC, with all6still in CG.

Evidence: [summary](summary.json), [raw queue](squeue.txt), [raw accounting](sacct.txt), [captured manifests/rosters/graph metadata](snapshot.json), [capacity phase](capacity_phase.json), [read-only probe](final_queue_probe.py), [summarizer](summarize_final_queue.py).
