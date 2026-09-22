# 22 September check — 11:57 UTC

**New endpoints:** three k33 CGs hit their four-hour limits without pricing certificates. The first k33 MIP, C3, ends at 34 buses with finite-pool bound 33; target 33 is not attained. C1/C4 MIPs were still running at the snapshot. [Exact values, timings, native logs and 234 checks](operations/README.md).

| Chain / target | Trips | CG minutes, this stage | Final fractional route weight | Integer fleet / pool bound |
|---|---:|---:|---:|---|
| C1 / 33 | 785 | 239.3 | 33 | MIP running at snapshot |
| C3 / 33 | 770 | 239.3 | 33 | 34 / 33, open |
| C4 / 33 | 785 | 239.1 | 32 | MIP running at snapshot |

Fractional weights are restricted-master values, not certified full-model lower bounds. Separate graph builds took 9.56–12.31 hours, excluded from this table's CG minutes; earlier sequential prefixes are also excluded. C3's native route replay passes baseline 240/240 physics, but the selected cover has 118 extra trip assignments across 95 trips. Shared capacity and duplicate removal are not validated. Source provenance retains the CG `git_dirty=true` flag; it is not relabelled clean. Different application/Slurm memory-accounting scopes are documented before any future resource sizing.

**Cluster:** one scoped collection at 11:57:12 UTC found 25 running jobs: 18 graph builds, five CGs and two MIPs. Graphs ready: 26/44, up from 12. All 85 pending solver dependencies are genuine; no new failed allocation or broken dependency needed repair. The existing ten cumulative graph preemptions remain recorded. No baseline jobs were duplicated or changed, and held historical/V2G work was preserved.

**Strict graph reuse:** source-reviewed implementation [fedf4214](https://github.com/ndandnd/EVSP-DR/commit/fedf421461f94727e6b1292a0e7789ab76ed8587) now provides atomic preparation, hash-bound load-only reuse and separate preparation/CG timing. All 13 new local tests and adjacent regression suites pass. The one new cluster job, **741034**, passed all 13 tests on NFS in 18 scheduler seconds, using 1 CPU / 4 GiB and excluding the GPU node. [Native verification, logs and exact scope](../strict_graph_reuse/native_preflight/README.md). This does not rebuild the 331-trip graph or validate the complete production initial pool; no new strict CG, MIP or k20 was submitted.

Next: verify complete k17→k19 inherited-pool equivalence under the explicit audited parent-commit gate; then prepare/reload the full same-physics graph and measure memory/time before CG. The old k19 zero-pricing result remains preserved. Do not silently migrate old same-instance checkpoint IDs, charge graph preparation as free, or mutate running source pins.

The completed 25 MIP trials, capacity-representation pilot, strict MIP668434 and k15 supplemental jobs remain final and were not recollected. [Previous completed tests](../monitor_20260922T075504Z/README.md). Quiet four-hour monitoring remains unchanged.

**Publication:** current Doc extension paragraph now contains the editable k33 table; one strict-graph sentence records native validation. Weekly slide 42 adds the same endpoint table, and slide 10's final sentence records the cache test. Original figures, tables, source footer and historical decks remain preserved. [Before/after publication verification](publication/README.md).
