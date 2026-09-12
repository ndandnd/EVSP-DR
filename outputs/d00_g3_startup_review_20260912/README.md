# d00_g3 startup audit and proposed recovery

No cluster jobs were submitted, cancelled, retargeted or altered during this review. Original CG array 949623_3/concrete949644 remains a timeout; dependent MIP 949624_3 remains DependencyNeverSatisfied.

## Evidence

The 325-trip input SHA is `cf3d15335300725f7d5db443be5886720c4a901a00cc348b1cc96b70ae306841`. Recorded code is `a29992196acb74d02b8c7891be4061718889999f`, event/lazy, 2.5 kWh SOC,5-minute grid,240 kWh battery/240 kW charging,zero reserve,flat tariff,cover/Gurobi/singletons,30 columns per iteration,rc tolerance1e-4, 14400 s scientific wall budget. Slurm requested 2 CPUs / 32 GB / 4 h 45 min; elapsed 4 h 47 min before timeout. No cg.json, journal or recorded LP/pricing event exists.

The session_start is emitted after input problem construction/provenance and before tariff loading/event network construction. Hence the evidence bounds failure to startup after that point; it does not identify an exact instruction. Eager packed graph construction is the leading explanation, supported by neighboring builds, but no failed-process stack was captured. Slurm reports approximately 4.83 GiB MaxRSS, with no OOM status; this does not justify increasing RAM. Its 0.231 s TotalCPU is incomplete/unhelpful for this terminated child and cannot establish an idle hang.

| Comparator | Nodes | Arcs | Graph-build seconds |
|---|---:|---:|---:|
| d00_g0 | 8,038 | 19,702,399 | 1311.2 |
| d00_g1 | 10,873 | 35,522,176 | 2336.0 |
| d00_g2 | 17,554 | 94,577,843 | 8992.2 |
| d02_g1 | 19,492 | 115,869,028 | 7492.1 |
| d03_g3 | 19,945 | 120,208,918 | 6513.3 |

## Code inspection

`EventExpandedNetwork._build_arcs` walks each trip/SOC source and materializes packed arc arrays even in lazy mode. `_charge_candidates` nests station, destination and target SOC loops, caching selected windows. These loops neither consult the wall budget nor the termination flag installed by run_cg. The graph is not checkpointed mid-construction; completed-cache persistence occurs only after construction returns. Thus increasing checkpoint frequency or requeueing cannot recover this lost startup work. Flat tariffs already use the direct earliest-window shortcut, so the new capacity dual-prefix selector is not a remedy for this builder.

## Tested diagnostic and recovery gate

`diagnostic_graph.py` observes the existing builder without changing its traversal or records: progress counts and RSS every 60 seconds, stack samples every 120 seconds during arc building. The timer is deliberately armed only inside _build_arcs, after preflight/provenance subprocesses. An initial broader timer probe timed out; those test artifacts are retained. The final wrapper passed an unmocked 8-trip cache build using solver/event sources identical to a299. Instrumented and uninstrumented graph cache bytes have identical SHA-256; 697 progress records and stack samples were verified. See smoke_result.json. This is a local wrapper test, not a full-instance recovery.

`recovery_plan.json` gives a concrete PROPOSED, UNSUBMITTED15-minute diagnostic using exactly the registered 325-trip input and scientific settings, 2 CPUs / 32 GB / default partition,required physical-node exclusion, 20-minute allocation and an external process-group watchdog. The bounded profile is designed to establish whether source/arc counts advance and which loops dominate; it is not a blind rerun until the same 4 h 45 min timeout. It saves a complete cache only if the builder finishes. There is no partial-graph resume capability.

After profiling, choose a tested graph-construction change or an explicitly extended cache-preparation attempt. Do not automatically grant a fresh 14400 seconds after expensive cache building: the original wall budget includes startup. A longer cache+CG recovery belongs to a separate budget stratum, while the original row remains censored. A genuinely budget-matched run must include all setup/cache work in its 14400 seconds. Preserve both artifacts and distinguish cached reuse from time from scratch.

Leave the blocked historical MIP dependency intact. Once a new attempt produces a valid, hashed source status and journal, any replacement MIP needs a new output directory and a dependency on that validated source. CG certification, pool-MIP proof, physical validation and GIRO target attainment remain separate.
