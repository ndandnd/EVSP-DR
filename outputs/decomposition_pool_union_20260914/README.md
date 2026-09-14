# Parent mapped decomposition pool union campaign

This campaign asks whether route diversity across nine completed decompositions
of the same 750 trip, 32 duty parent can recover a 32 bus schedule without the
12.5 hour parent graph construction that previously timed out. The 46 MIPs are
treatment comparisons on one instance: nine same partition controls, all 36
partition pairs, and one all nine union. They are not 46 independent instances.

Each of nine construction jobs verifies the frozen child CSV, CG status,
column journal, and source MIP; checks every child's physical trip attributes
against the parent by `Ordered_Trip_ID`; maps both `trips` and integer
`route_nodes`; retains the entire route payload; and rebinds ID dependent
realization hashes. It always keeps the source MIP routes, then fills each
component to 512 by route length, cost per trip, and stable parent IDs. If the
mandatory set exceeds 512, it retains the full mandatory set and records the
actual count. Published construction endpoints are `cases/pNN/pool.json` and
`completion.json`; they have `optimization_run=false`, no CG certificate, no
full model LP bound, and `parent_graph_constructed=false`.

The nine controls and 36 pairs receive 7,200 seconds each: 3,600 seconds for
fleet, then the remaining time for charging cost with fleet at most the best
incumbent. The all nine solve receives 14,400 seconds with 7,200 seconds for
fleet. Each solve starts from the included partition with the smallest source
fleet count, breaking ties by saved expanded grid cost. The exact commit
`871d057e1067411f09581e37d78f7c1ca43f68bb` replays every pool and warm start
route on the whole parent and refuses rejected or repaired routes. It does not
construct the full event network. A physically admitted 32 bus incumbent proves
baseline feasibility. A result above 32 is finite pool evidence and does not
prove that 32 is impossible.

All jobs use `default_partition` and exclude `scaglione-compute-01`.
Construction requests are 2 CPU, 8 GB, one hour. All 46 independent solver
requests are eligible together (below the policy cap of 50), use 4 CPU and
24 GB, and do not requeue. The 2 hour solver jobs have 2.5 hour allocations;
the 4 hour solve has a 4.5 hour allocation for source union, replay, and safe
publication overhead. Gurobi is forced to four solver threads while BLAS helper
libraries are restricted to one thread.

The immutable manifest is `manifest.json`. `jobs.json` records each exact Slurm
submission. Results are collected with:

```bash
python3 /home/nc437/ladder-lite/decomposition_pool_union_20260914/collect.py \
  --root /home/nc437/ladder-lite/decomposition_pool_union_20260914
```

The adapter emits `construction`, `mip`, and standard `records` entries with
manifest, result, journal, construction, and source union hash gates. It emits
no CG records because this campaign performs no CG.

Jobs do not automatically requeue. If a MIP is preempted, its immutable attempt
remains evidence; a manual retry must use a new Slurm job and attempt token. The
Gurobi branch and bound tree is not restartable. `case_metadata.json` binds every
case to the 32 duty parent for the standard diagnostic normalizer.
