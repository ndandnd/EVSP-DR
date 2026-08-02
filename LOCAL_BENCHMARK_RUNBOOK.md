# Local randomized Goal-1 benchmark runbook

This is the conservative macOS path for testing the repaired pricing search
before spending hours on larger instances. It uses SciPy/HiGHS for every
restricted-master LP, a model-derived minimum-path-cover initializer, the flat
price file, and at most two pricing processes at once. Gurobi is not used unless
a saved column pool is later sent to the separate final integer master.

## What these instances do and do not prove

`Par_VehicleDetails_Updated.csv` combines historical block data, including two
weekday-variant groups. The generator samples base `VehicleTask` groups
uniformly without replacement. If base 13316 or 13324 is drawn, it chooses one
literal variant and never mixes that base's variants in the same instance.

These are deterministic **synthetic rediscovery/scaling tests**. They have not
been verified as coherent single-day schedules and therefore cannot establish
GIRO full-day parity. The tracked `Practice_10bus.csv` (the first ten block IDs)
remains the hard regression reference: those blocks contain unusually many
trips, so it is deliberately less representative than a uniform random sample.

## One-time generation

Use the Python environment that has pandas and SciPy. On this Mac the known
working interpreter has been:

```bash
PYTHON=/Users/nathan.cho/.pyenv/versions/3.10.6/bin/python
"$PYTHON" src/generate_random_goal1_instances.py
```

The default seed is `20260802`. This creates five replicates each at 10, 15,
and 20 sampled base tasks under:

```text
data/random_goal1_instances/seed_20260802/
```

`manifest.json` and `manifest.csv` record the seed, replicate, selected base and
literal tasks, trip count, source SHA256, and every generated CSV's SHA256.
Rerunning the command is idempotent. It refuses to replace different same-name
bytes unless `--force` is explicit.

The generated inputs and all result directories are ignored by Git. That keeps
large/long-lived column pools out of source control; it does **not** mean they
are disposable. Never use `git clean` to collect or manage results. Back up the
whole chosen `src/results/local_goal1/<batch-tag>/` directory.

Audit the structural LP route-weight bounds before choosing a numerical target:

```bash
"$PYTHON" src/audit_structural_route_bound.py \
  Practice_10bus.csv \
  random_goal1_instances/seed_20260802/Practice_SyntheticRandom_15bus_s20260802_r04.csv \
  --json-out /tmp/goal1_structural_bounds.json
```

This reports both peak concurrency and a maximum reachability antichain, plus
the exact trip IDs forming the antichain certificate. The antichain bound is
valid for fractional master variables; a direct minimum path-cover count alone
is not always such a bound when the compatibility graph is not transitively
closed.

## Required pricing integration

Before execution, the maintained runner must expose:

```text
--master_backend scipy
--matching
--queue_order reduced_cost_bound
--max_charge2trip 1560
```

The launcher checks those options and fails loudly rather than silently falling
back to the historical time-first heap or 220-minute station-to-trip window.
If the repaired bound-priority mode receives a different final name, update the
two runner-facing constants near the top of
`src/run_local_goal1_batch.py` and/or pass `--queue-order FINAL_NAME`.

Commit the exact tested code before a reportable 30-minute or 3-hour batch. The
runner records the commit and a deterministic worktree fingerprint.
`--allow-dirty` is available only for a clearly provisional diagnostic.

## Stage 0: review a dry run

The launcher does nothing unless `--execute` is present. The most useful first
batch is replicate 4 plus the hard first-ten reference:

```bash
MANIFEST=data/random_goal1_instances/seed_20260802/manifest.json
"$PYTHON" src/run_local_goal1_batch.py 5m \
  --manifest "$MANIFEST" \
  --replicates 4 \
  --include-hard-reference
```

This selects three cases. The hard first-ten and random 10-r04 cases both have
peak concurrency equal to 10, so a feasible ten-route MATCHING seed already
closes their fleet-count question. Random 15-r04 has a weaker time-overlap bound
of `P=14` for 337 trips, but a computed 15-trip antichain in the current
reachability graph strengthens the fractional lower bound to 15. MATCHING
supplies 15 feasible routes, while GREEDY supplies 17. It is therefore a useful
17-to-15 rediscovery test, not a valid target for a result below 15. It remains
a synthetic, unverified-day stress test.

The 10- and 15-bus cases are interleaved. Threading variables for OpenMP,
OpenBLAS, MKL, Accelerate/vecLib, NumExpr, and BLIS are fixed at one. The
launcher accepts only one or two workers and defaults to two; do not raise this
just because the machine reports 18 cores. Two label searches can already use
substantial portions of the 48 GB RAM.

## Gate 1: five minutes

```bash
"$PYTHON" src/run_local_goal1_batch.py 5m \
  --manifest "$MANIFEST" \
  --replicates 4 \
  --include-hard-reference \
  --execute
```

This is a plumbing and early-search gate, not a convergence verdict. Confirm:

- every process exits cleanly and leaves a run directory, pricing CSV, and
  checkpoint/final pool;
- MATCHING begins with `Artificial_Trips_Before_Add == 0` and reports its path
  count, retry provenance, and peak-concurrency lower bound;
- DP columns and LP movement are reported separately. Negative reduced cost is
  useful search evidence, but a zero master step can be legitimate under dual
  degeneracy;
- two concurrent jobs stay comfortably within memory. If memory pressure or
  swap becomes material, stop and rerun with `--max-workers 1`.

Live output is prefixed by case and copied to non-overwritten launcher logs.
Pressing Ctrl-C terminates active child processes but does not delete outputs.

## Controlled heap comparison

Heap order can drastically change which labels are expanded, so compare it on
the same GREEDY master rather than comparing unrelated initializers or duals.
Run each of these only after committing the code, and use distinct batch tags:

```bash
for ORDER in time reduced_cost reduced_cost_bound; do
  "$PYTHON" src/run_local_goal1_batch.py 5m \
    --manifest "$MANIFEST" \
    --sizes 15 \
    --replicates 4 \
    --initializer greedy \
    --queue-order "$ORDER" \
    --max-workers 1 \
    --batch-tag "heap_${ORDER}_5m_$(git rev-parse --short HEAD)" \
    --execute
done
```

Run these sequentially so hardware contention is not another experimental
factor. Compare initial/final route weight, master objective, artificial variables, unique
incidence patterns admitted, best reduced cost, longest returned routes, label
expansions/evictions, and timeouts. A heap is useful only if its columns improve
the reoptimized master; a more negative route at one degenerate dual is not by
itself a win.

## Gate 2: thirty minutes

Only after Gate 1 is healthy:

```bash
"$PYTHON" src/run_local_goal1_batch.py 30m \
  --manifest "$MANIFEST" \
  --sizes 15 \
  --replicates 4 \
  --execute
```

At this gate, inspect each pricing CSV over time rather than only its last row.
The per-iteration CSV records the master *before* that row's newly priced
columns are added. Its key fields are `Artificial_Trips_Before_Add`,
`Artificial_Total_Before_Add`, `LP_Route_Weight_Before_Add`,
`Peak_Trip_Concurrency`, `Master_Obj_Before_Add`, `Best_RC`, `Cols_Added`, the
timeout/truncation fields, actual labels expanded and evicted, and search time.
Use `Final_LP_Artificial_Trips`, `Final_LP_Artificial_Total`, and
`Final_LP_Route_Weight` from `colgen_summary_*.json` for the post-run state. A
negative-reduced-cost known GIRO duty that is model-feasible but still never
found is evidence of pricing-search failure, not a reason to buy more MIP time.
Queue exhaustion is an exhaustive restricted-pricing result only when neither a
time limit nor a label-cap eviction occurred.

## Gate 3: three hours

Run the three-hour stage only for configurations that survived Gate 2 and show
useful DP progress:

```bash
"$PYTHON" src/run_local_goal1_batch.py 3h \
  --manifest "$MANIFEST" \
  --sizes 15 \
  --replicates 4 \
  --execute
```

If those are healthy, expand `--replicates` to `all` for the five 10-bus and
five 15-bus replicates. Keep at most two workers. If the 30-minute runs add no
useful columns, do not automatically multiply that failure into ten three-hour
runs; return to the known-column oracle, queue/bound instrumentation, and
feasibility diagnostics first.

## Success rule and interpretation

Every report must first compute the peak number `P` of simultaneously active
trips. Since a feasible route can serve at most one of those trips,
`LP_Route_Weight >= P` for every feasible master solution. The unconditional
Goal-1 checks are therefore:

```text
Final_LP_Artificial_Trips == 0
P <= Final_LP_Route_Weight
```

The first condition says the restricted master has a real-column cover. A
feasible `N`-route seed in the same model separately proves an `N`-route upper
bound. If it also has `P == N`, the fleet count is certified:
asking for an LP route weight below `N` is mathematically impossible. A value
strictly below `N` is meaningful only when every valid structural lower bound
is also below `N`. Report the exact
`Final_LP_Route_Weight` directly; `LP_Obj / BUS_COST` is larger because it also
contains charging cost and is not a fleet-count metric. Never relabel a
still-running restricted-master objective as a full-master lower bound.

MATCHING is not cheating: it uses only the active model graph, computes a maximum
bipartite matching, and independently realizes the resulting paths under the
time/SOC/charging rules. It never reads historical block membership. If no tried
maximum matching is resource-feasible as an exact minimum path cover, the
initializer deterministically cuts its relaxed paths into the fewest contiguous
resource-feasible routes. Check `Seed_Matching_Provenance`: exact runs report
`is_exact_minimum_path_cover: true`; repaired runs report
`resource_repair_mode: contiguous_split`, the relaxed and final path counts, and
the number of added splits. GREEDY is also model-derived and remains available
as a control:

```bash
"$PYTHON" src/run_local_goal1_batch.py 5m \
  --manifest "$MANIFEST" \
  --replicates 1,2 \
  --initializer greedy
```

The seed routes are candidates, not fixed buses, and the LP may replace or
fractionally combine them. CHEAT, which imports historical block membership, is
intentionally absent here.

## Why 20 buses are deferred

The launcher defaults to only 10 and 15. A 20-bus command is rejected unless
both `--sizes 20` and `--include-20` are explicit:

```bash
"$PYTHON" src/run_local_goal1_batch.py 5m \
  --manifest "$MANIFEST" \
  --sizes 20 \
  --replicates 1,2 \
  --include-20
```

Leave this as a dry run until multiple 10/15 instances show real covers and the
reported route weights are interpreted against their concurrency bounds. Scale
pricing only when the shorter gates show a reason to optimize beyond the
matching seed; do not multiply a truncated or zero-step search into many
three-hour runs.
