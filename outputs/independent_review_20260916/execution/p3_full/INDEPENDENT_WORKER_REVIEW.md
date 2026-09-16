# Independent worker review of P3 item 13

Reviewed the prepared local `campaign.py`, `manifest.json`, `validation.json`, `prepare_remote.py`, `worker.sub` and `full40_c1.csv` without modifying them or submitting jobs.

**No blocking execution/data-identity mismatch found.** Checks:

- Manifest hashes match all three frozen worker/tool files.
- Full input hash matches the manifest; 948 unique ordered trip IDs and contiguous row IDs are present.
- Original GIRO mapping gives 40 distinct duties, including `13316m` and `13324muw`. This is the frozen C1-compatible 40-duty variant, not all 987 rows from 42 original labels.
- CG arguments are fresh singleton initialization with no inherited-pool or GIRO-column arguments.
- Graph budget is 86,400 seconds. CG budget is 172,800 seconds, its process watchdog is 176,400 seconds, and allocation is 50 hours.
- MIP budget is 12,600 seconds total, including a 10,800-second fleet stage; process watchdog is 15,300 seconds and allocation is 4.5 hours.
- Every submission uses default partition, requeue, and excludes `scaglione-compute-01`. True graph → CG → MIP dependencies are retained.
- Resume checks pinned code and input identity, copies journal/checkpoint artifacts to a new attempt, and retains earlier attempts. This does not claim resumable MIP search trees.
- The stale template `prepare()` body is disabled by the CLI; `prepare_remote.py` created the audited fresh manifest.

Nonblocking reporting/operational points:

1. Add explicit total/fleet MIP budgets to `scientific_settings`, so readers do not need to inspect worker arguments. The actual worker uses the intended budgets.
2. The 24 GB MIP request is inherited from smaller pools. A full40 pool may exceed it; monitor actual preparation memory and do not mistake OOM or pre-optimization timeout for a bad mathematical result. No measured full40 footprint exists yet.
3. The graph process watchdog equals its nominal 24-hour budget, while its allocation has one extra hour. It may stop an unfinished graph cleanly; this is a graph-budget limit, not evidence of pricing failure.
4. Preserve distinction between the 48-hour CG budget and preceding graph computation. Record wasted attempts if preempted, rather than reporting only the final attempt's duration.
5. The full40 baseline retains uniform 240 kW, zero reserve and unlimited shared charging capacity; it is not a full-GIRO-constraints test.

This audit did not execute the full-instance graph, CG or MIP; resource sufficiency remains empirical.
