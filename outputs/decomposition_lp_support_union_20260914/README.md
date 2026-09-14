# LP-support preserving decomposition union campaign

This is a matched selector treatment for the same 750-trip, 32-duty parent as
`decomposition_pool_union_20260914`. An audit found that the first 512-route
component selector retained only 226 of 3,413 positive final child-RMP routes;
all 36 components omitted support. Every component's union of source integer
witness routes and positive child-RMP routes fits under the unchanged cap of
512 (maximum 193 before fill).

Nine data-only construction jobs verify the same frozen child CSVs, CG statuses,
journals, source MIPs, model hashes, and real trip attributes. They remap full
route payloads and ID-dependent realization hashes. Each component forces both
the source integer witness trip sets and every positive final child-RMP trip set,
then fills to 512 by the first campaign's length, cost-per-trip, stable-ID order.
The constructor recomputes each child fractional cover, objective, route weight,
and artificial total, then verifies after remapping that coverage remains at
least one and retained route costs are unchanged or lower for each trip set.
The integer witness alone remains the covering MIP warm start.

The 46 production MIPs exactly match the first wave: nine same-partition
controls and 36 pair unions get 7,200 seconds total with 3,600 seconds for fleet;
the all-nine union gets 14,400 seconds with 7,200 seconds for fleet. Every job is
independent once its source pool exists, requests 4 CPU and 24 GB on
`default_partition`, excludes `scaglione-compute-01`, and does not automatically
requeue. A preempted manual retry must create a new attempt and retain the old
record.

These are treatment comparisons on one parent instance. Positive child-RMP
support is a feasible fractional witness for each disconnected child problem.
It is not a parent CG certificate or parent full-model lower bound, including
for the six time-limited child CG endpoints. A physically replayed 32-bus
incumbent proves baseline feasibility; a result above 32 does not prove 32
impossible. The exact MIP at commit
`871d057e1067411f09581e37d78f7c1ca43f68bb` replays all admitted routes on the
whole parent without constructing the full parent event network.

Collector entry point:

    python3 /home/nc437/ladder-lite/decomposition_lp_support_union_20260914/collect.py --root /home/nc437/ladder-lite/decomposition_lp_support_union_20260914
