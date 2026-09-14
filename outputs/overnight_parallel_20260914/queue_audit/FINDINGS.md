# Focused queue and graph audit

At16:29:02Z the snapshot had34running allocations and86pending:53true dependency waits and33held historical tasks. Running groups:18new graph builds,4extensionCGs,1C1k15c200CG,1extensionMIP,6unionMIPs and4strict-capacity combined allocations. This is not an empty queue.

At16:50Z all18new graphs were actively writing progress (3–59s old), after4.15h. All60old extension graphs are published. Current old-chain CG bottlenecks areC1k23,C4k24,C5k24,C6k25. New CGs correctly wait for their own graph and preceding-kCG; MIPs wait only for ownCG. Frozen52dependencyedges verified; no invalid edges identified. One compact sacct succeeded; parent squeue timeouts still mean controller responsiveness is uncertain, distinct from running work.

Completed k23–25 graphs demonstrate acceleration: at4.15h they were24–58% throughsources but finished5.3–10.8h. Totalduration was61–74% of naive sourcefraction extrapolation. Traversal is sorted(trip,level), so do not assume a strictly chronological suffix. Historicalshape scaling suggests8.2–12.3h totals for new cases; w1k28 andw4k27 are highestwatchdog risks. Ranges in graph_estimates.json are heuristic historicalvariation, not confidenceintervals.

No partialgraph checkpoint exists: the network builds in memory before atomic pickle/manifest publication. A wallcap loses that construction. Keep current frozen runs unchanged. Only afteractualfailure, consider targeted16hwatchdog/16.5hallocation fresh retry with explicit dependencyrepair. Changing Slurm timelimit alone would not bypass the current12hwatchdog.

64GiBgraph requests are not shown excessive: completedgraphs reached35.5GiB and newgraphs are larger, with memorystillgrowing. Current9.4–12.4GiB is not finalpeak. All18graphs already admitted; admission is not their bottleneck. No adequate completed-CGMaxRSS evidence here supports cutting96GiB. Heldhistory untouched; no mutations or newjobs.
