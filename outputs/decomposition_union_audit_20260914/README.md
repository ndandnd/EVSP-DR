# 32-duty decomposed-pool union audit

The most useful additional overnight experiment is a **complete pairwise union
of the nine finished 4×8 partitions**, followed by one all-partition union. It
uses the component CG work already paid for and never builds the failed
750-trip parent pricing graph.

The unanswered question is whether routes generated under different
eight-duty boundaries are complementary enough to recover a feasible 32-bus
schedule for the common parent. The earlier joined schedules used one
partition at a time and produced 34–37 buses among the nine complete
partitions. Their parent CG attempts all stalled in graph construction, so they
never tested recombination across partitions.

## Available evidence

- The source campaign planned 40 group CGs. Thirty-nine have journals and
  completed MIPs; only `d00_g3` is absent.
- Partitions 1–9 each have all four components. Their component fleet sums are
  35, 35, 36, 35, 35, 36, 34, 37, and 35.
- All 39 available component pools passed the existing physical admission
  path with zero rejected or repaired columns. Thirty-three group CGs carry
  pricing certificates; six stopped at their wall limit but remain usable
  finite pools.
- The journals contain 1,555,157 recorded columns and 6.58 GB in total. Reading
  every journal separately for every pair would waste shared-filesystem I/O.
- Parent graph job 42509 ran 45,142 seconds and timed out before CG, with no
  published cache. It used 25.37 GiB against 128 GiB requested, so the observed
  failure was time in graph construction rather than memory pressure.

The parent input SHA-256 is
`4367335166098c6c50fb283b1cd3307a72720ea0b70fff4567b085af9a37e66e`.
Source group CG commit is
`a29992196acb74d02b8c7891be4061718889999f`; the validated exact pool-MIP
commit is `871d057e1067411f09581e37d78f7c1ca43f68bb`.

## Proposed 46-allocation campaign

Run nine independent construction jobs, one for each complete partition. Each
job streams its four component journals once, maps local `count_trip_id` and
integer `route_nodes` through `Ordered_Trip_ID` into the parent index, and
emits a parent-index pool. Force all routes selected by each source MIP into
the pool, then fill to 512 routes per component using the already documented
ordering: longest sequence, cheapest cost per trip, stable parent trip IDs.
This caps a partition pool at 2,048 routes while retaining its known feasible
schedule.

After construction, launch:

- all 36 unordered two-partition union MIPs; and
- one descriptive all-nine union MIP.

The 36 pair jobs form a complete complementarity matrix rather than a chosen
set of favorable pairs. Each pair has the same maximum pool size of 4,096.
The all-nine pool can contain at most 18,432 routes and is a separate ceiling
experiment. The second wave supplies 37 parallel solver jobs, within the
requested 30–50 range.

Suggested bounded resources are 2 CPUs, 8 GB, and one hour for each
construction; 4 CPUs, 24 GB, and two hours for each pair MIP with one hour
reserved for stage 1; and 4 CPUs, 24 GB, and four hours for the all-nine MIP.
For scale, the validated union campaign spent up to 210 minutes on pools of
118,068–241,369 columns; the proposed matched pair pools are capped at 4,096.
Use the default partition, no requeue, individual `afterok` dependencies, and
exclude `scaglione-compute-01` throughout.

## Implementation path

Reuse the index mapping and selected-sequence checks in
`outputs/queue_recovery_20260912/graph_worker.py`, and the SQLite-backed
deduplication, atomic construction, and hash gates in
`outputs/parallel_followup_20260914/union_design/union_logic.py` and
`union_worker.py`. The remapper must update both `trips` and integer entries in
`route_nodes`; charging station/time payloads remain unchanged. Deduplicate by
the sorted parent trip set, retaining the lower-cost complete payload.

Run the resulting parent descriptors through the detached exact
`run_exact_pool_mip.py` at commit `871d057e…`. Require the existing parent
physical replay, zero rejected or repaired columns, at-least-once coverage,
and source/status/journal/MIP/mapping hashes before publishing an endpoint.
Construction records are workflow artifacts, not CG results.

## Proof scope

A 32-bus incumbent would show baseline-model feasibility from the available
decomposed columns. A result above 32 would not show that 32 is impossible.
No union has a parent pricing certificate, even when every contributing group
CG is certified. MIP bounds apply only to the admitted finite pool. The model
scope remains covering, 240 kWh, 240 kW, reserve zero, event time, 2.5 kWh SOC
steps, five-minute blocks, flat prices, bus coefficient 100,000 and charge
start fee 5, without shared charger capacity or a terminal floor. Duplicate
trip removal and global charger capacity are outside this experiment.

New random group CGs would duplicate eight existing seeded-random partitions.
New geographic partitions should wait until the pair matrix shows whether
partition diversity matters; the earlier geography screen found about 32% of
candidate connections crossed its two service groups, so it did not justify a
disconnected decomposition.

Exact machine-readable counts, paths, hashes, resources, and interpretation
are in [`audit.json`](audit.json). No job was submitted and no existing
campaign file was changed by this audit.
