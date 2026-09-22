# 22 September check — 15:58 UTC

**Four completed k33 MIPs now use 34–38 buses.** All retain open finite-pool gaps; none has matched target 33. Five completed k33 CGs and three new k34 CGs reached their four-hour limits without pricing certificates. These are current endpoints, not evidence of an algorithmic size cutoff.

| Chain / target | Trips | CG minutes, this stage | Fractional route weight | Integer fleet / pool bound |
|---|---:|---:|---:|---|
| C1 / 33 | 785 | 239.3 | 33 | 36 / 33, open |
| C2 / 33 | 787 | 239.9 | 32 | 36 / 32, open |
| C3 / 33 | 770 | 239.3 | 33 | 34 / 33, open |
| C4 / 33 | 785 | 239.1 | 32 | 38 / 32, open |
| C6 / 33 | 796 | 239.9 | 32 | MIP running at snapshot |

C5 still awaits its graph; the preempted build has a running replacement. Fractional weights are final restricted-master values, not certified full-model lower bounds. Separate graph builds for the table took 9.56–15.10 hours; CG minutes exclude earlier sequential prefixes. At k34, C1/C3/C4 have fractional weights 33/34/33 and running MIPs. Their last pricing reduced costs remain negative.

Both MIP stages timed out in all four completed k33 cases. Native selected-route replay passes baseline 240/240 physics, but duplicate cleanup and shared charger capacity remain unvalidated. Extra trip assignments are 315/225/118/285 for C1/C2/C3/C4. Source provenance retains the CG dirty-checkout flag. No new independent whole-pool physical replay was performed by this collector.

[New endpoint tables, full Gurobi logs and audit](operations/README.md) · [Earlier C1/C3/C4 CGs and C3 MIP](../monitor_20260922T115605Z/operations/README.md).

The audit passes 286 of 287 checks; all required source, coverage, log and dependency checks pass. The disclosed numerical discrepancy is C2's saved positive-only LP support: its reconstructed objective exceeds the saved scalar by 0.037537633, with route-weight difference 3.75160e−7. The recorded bound violation is within the existing 1e−6 solver tolerance. The evidence retains both values, without changing tolerances or claiming certification.

**Cluster:** one scoped check at 15:58:38 UTC found 20 running jobs (11 graphs, five CGs, four MIPs), 33/44 completed graphs and 75 genuine solver dependencies. Seven graphs finished since the previous check. No new failed allocation, broken dependency or additional preemption required recovery. The ten graph preemptions remain a cumulative count. Held historical jobs, V2G work and existing scientific settings were preserved.

**Strict graph reuse:** native job **768638 completed** in 35 scheduler seconds. All 8,343 inherited records match the saved k19 initial pool after normalizing only the checkpoint ID; all 8,397 initial routes pass physical replay, including independent schedule/cost/metadata checks on the 54 new singletons. The gate used 29.075 seconds and peaked at 499,560 KiB in its child process. No graph or solver ran. [Source pins, independent review and native receipt](../strict_graph_reuse/production_gate/). This does not prove that freshly optimizing each singleton would reproduce the saved optimum. Full-size graph preparation and cold/reload comparison must pass before a new strict CG; k20 remains unsubmitted.

The earlier native-preflight README now correctly states a 1,560-minute waiting bound, matching its executed source. Its former 220-minute statement was a prose error; the source pin, tests and results are unchanged. [Corrected scope](../strict_graph_reuse/native_preflight/README.md).

**Next validation running:** graph-only job **772820** started on joachims-cpu-02 at 16:32:55 UTC (phase start verified at 16:35:58), 2 CPUs / 16 GiB / eight hours on the default partition, GPU node excluded. It builds one unchanged 331-trip graph, saves it, then uses a separate process to verify the reload, two diagnostic pricing queries and all 331 native singleton calls that form the 8,397-column initial pool. Wrapper `3bb32c1a` is pushed; model code remains clean `fedf4214`. Independent review passed 27 package checks, five focused tests and three mock restart checks. This is a submitted test, not a new solver result. [Exact settings, source hashes, restart policy and job receipt](../strict_graph_reuse/production_gate/full_graph_proposal/README.md).

Completed matrix-setting and capacity-compression tests were not rerun or recollected. Their [25-cell MIP comparison](../monitor_20260922T075504Z/mip_structure/README.md) and [capacity representation results](../charging_column_structure/RESULTS.md) remain the source for the sparsity conclusions. Quiet four-hour monitoring remains unchanged.

**Publication verified:** 70/70 checks pass. The current Doc's existing endpoint table and strict-validation paragraph, and weekly slides 42 and 10 with source notes, are updated in place. Other tables, figures, historical content and effective deck styles are preserved. [Before/after exports, source checks and visual review](publication/README.md).
