# Pricing and column-generation review

## Scope and evidence

This is a read-only review of the pinned baseline commit `a29992196acb74d02b8c7891be4061718889999f` in `/private/tmp/evsp-algorithm-review-source-20260912`, centered on `src/exact_pricer_expanded.py`, `src/event_pricer_network.py`, and the restricted-master adapters. I also reviewed the bounded capacity branch at commit `253588e9b22d68fcbc67cb56bc3eb30cbb0e16b6` in `.codex-work/capacity-timeout-checkpoint-20260911`, as requested. No solver source, register, cluster state, or experiment artifact was changed.

The relevant unit suites passed under `python3 -m unittest -q tests.test_event_pricer_network`: 12 baseline tests in 2.961 seconds and 14 capacity-branch tests in 3.347 seconds. `pytest` was unavailable in the local interpreter. These are small correctness tests, not performance measurements.

The strongest measured evidence is the completed capacity pilot in `outputs/parallel_research_20260911/capacity_deadline5_completed/records.json`: the two k3 iteration-3 pricing calls took 25,722.648 and 24,797.474 seconds with a 38-column pool and one nonzero capacity dual, while the corresponding master LP was about 0.00563 seconds. The recommendations below state mechanisms and validation criteria; they do not assert unmeasured speedups.

## Findings, ordered by priority

### 1. Capacity-aware pricing repeatedly solves the same interval problem in the innermost arc loop

**Severity: critical performance blocker. Confidence: high.**

In capacity commit `253588e9`, `_charge_window_options` explicitly inserts every one-minute capacity boundary and its duration-shifted counterpart, then evaluates and sorts all feasible starts (`src/event_pricer_network.py:130-177`). `_capacity_adjusted_arc` invokes that enumeration for every charge arc, constructs a dictionary for each candidate action, constructs a `frozenset` of all occupied minute rows, and performs one dictionary lookup per occupied minute (`src/event_pricer_network.py:266-312`). The shortest-path pass calls this routine inside the source-node and outgoing-arc loops (`src/event_pricer_network.py:919-952`). The capacity runner deliberately builds an explicit graph (`src/run_capacity_speed_event_cg.py:224-236`) and supplies the sparse capacity duals on every iteration (`src/run_capacity_speed_event_cg.py:377-389`).

The resulting work is approximately proportional to

`number of charge arcs × feasible start minutes × occupied minutes`,

plus repeated tariff integration and object allocation. That mechanism agrees with the observed jump from seconds to roughly seven hours when just one capacity row dual becomes nonzero.

The first safe acceleration should preserve the present breakpoint theorem and replace evaluation:

1. Define one shared integer-span helper for the exact half-open row convention currently implemented at `src/event_pricer_network.py:180-200`.
2. Build per-station prefix sums of capacity duals once per CG iteration, so the dual contribution of `[start,end)` is computed in O(1), without materializing row tuples or a `frozenset`.
3. Memoize the best dynamic window within that iteration by a complete key including `(station, arrival, latest, energy, charge_power, capacity_grid)`, tariff, objective and the current dual generation. Many physical arcs share those values. The cache must be cleared or generation-keyed whenever the dual vector changes; a network-lifetime cache would silently reuse stale reduced costs.
4. As a later, separate experiment, investigate range minima for each station and charge duration over the capacity-boundary lattices (`integer boundary` and `integer boundary - duration`), combined with arrival/latest and tariff-boundary candidates. Prove that the union retains every potentially optimal current breakpoint and the tie policy before replacing enumeration. This is not required for the initial prefix-sum/memoization change.
5. Add an immediate `if not capacity_duals: return cost, action`. The current combined branch at lines 273-280 still constructs capacity rows even when the sparse dual dictionary is empty.

Required invariant: for randomized small windows, compare the optimized selector against the current exhaustive `_charge_window_options` implementation for the tuple `(adjusted cost, start, end)` under sparse valid master duals, including fractional arrival, fractional duration, exact minute endpoints, hourly tariff changes, base-station aliases, and 60/240 kW power. The runner's independent reduced-cost replay at `src/run_capacity_speed_event_cg.py:399-409` should remain mandatory.

### 2. “Lazy” baseline arcs are packed, but still exhaustively generated and stored

**Severity: high. Confidence: high.**

At baseline commit `a2999219`, `_build_arcs` loops over every trip/SOC node and calls both direct and charge arc builders (`src/event_pricer_network.py:314-354`). Charge generation nests station, destination, and every higher target SOC level (`src/event_pricer_network.py:423-494`). `arc_mode="lazy"` changes each retained arc into packed endpoint/cost/recipe arrays; it does not generate successors on demand. The test explicitly expects lazy and explicit graphs to have the same arc count (`tests/test_event_pricer_network.py:235-251`).

This is already a useful memory optimization and its NumPy relaxation is already implemented (`src/event_pricer_network.py:812-867`), but it does not change the asymptotic graph-construction or stored-arc count. Paper text and work plans should call it a **packed-arc representation**, not a lazy graph algorithm.

Safe next steps are:

- Compute forward reachability from depot and reverse reachability to depot over conservative time/SOC bounds before emitting outgoing arcs. Do not create states that cannot lie on any source-to-sink route.
- Generate outgoing transitions only for states reached in the forward pass, with a deterministic cache if repeated CG iterations need them.
- Factor trip-to-station, charge, and station-to-trip transitions rather than precomputing every trip/SOC-to-trip/SOC combination. This must preserve the exact same source-to-sink path costs and station choices.

Required invariant: on small instances, compare the complete reachable state/arc projection, minimum reduced cost, trip sequence, expanded-grid cost, and physical replay against the existing explicit oracle for random dual vectors. The current explicit-versus-packed tests only prove representation equivalence after both have built the same exhaustive arc set.

### 3. Multi-column pricing is not k-best enumeration and performs expensive realization before incidence deduplication

**Severity: high convergence opportunity. Confidence: high.**

The implementation correctly documents that `sink_predecessor_route_batch` returns the global shortest path plus the best prefix for distinct sink predecessors and is “not k-shortest-path enumeration” (`src/event_pricer_network.py:869-879`; CLI help at `src/exact_pricer_expanded.py:3199-3203`). It sorts sink arcs, reconstructs and physically realizes a candidate, and only then forms its `frozenset(trips)` to discard duplicate incidence (`src/event_pricer_network.py:905-947`). A different negative route that shares the same terminal predecessor but has the second-best prefix cannot enter the batch.

There is no certificate defect: the first route is the exact global shortest path, and only that route is used for the stopping test (`src/exact_pricer_expanded.py:2590-2642`). The limitation can, however, cause many more master/pricing iterations and waste realization work on duplicate incidences.

Two safe improvements are separable:

- Reconstruct only parent trip IDs (or maintain a compact ancestry/incidence signature) before `_walk`; skip already-seen incidences before tariff-block construction and physical replay.
- Implement exact k-best paths in the DAG as an enrichment layer, retaining the current global shortest path as the sole certificate. Deduplicate by master column identity and realize only retained paths. A bounded k-label-per-node dynamic program or a DAG deviation-path algorithm is suitable; benchmark both batch diversity and total CG wall time.

`--column-selection complementary` is already implemented (`src/event_pricer_network.py:921-999`) and tested (`tests/test_event_pricer_network.py:75-96`). It reranks the same restricted sink-predecessor candidate set, so it is not a substitute for k-best generation and should not be presented as a new algorithmic contribution.

### 4. The stopping certificate needs an independent numerical audit and a strict scope label

**Severity: medium-to-high for claims. Confidence: high.**

The restricted-master adapters require solver optimality and independently check primal row/bound feasibility (`src/master_lp_scipy.py:287-365`; `src/master_lp_gurobi.py:324-405`). They do not independently check the returned duals against all current route and artificial-variable reduced costs, nor record a primal-dual objective gap. The driver then declares reduced-cost optimality whenever the priced minimum is at least `-rc_eps` (`src/exact_pricer_expanded.py:2635-2642`; default `rc_eps=1e-4` at line 3221). With route costs around 100,000, this relies on cancellation of similarly scaled numbers.

Before a conference-paper certificate is accepted, recompute and persist:

- minimum reduced cost over every restricted-pool route under the returned dual;
- artificial reduced costs (`BIG_M - dual`) and cover-dual sign restrictions;
- primal objective, dual objective, and their gap;
- the exact-pricer minimum reduced cost;
- the tolerances used, ideally after scaling all costs by a common factor near the bus fixed cost.

For an epsilon-feasible dual, report a tolerance-adjusted bound rather than silently treating it as exact arithmetic. Also distinguish “augmented Big-M master has no improving route” from “real-route master is feasible”: baseline `certified = best is not None` does not require zero artificials (`src/exact_pricer_expanded.py:2639-2641`). A full real-route LP claim should additionally require artificial total below a declared tolerance.

The model scope is narrower than the continuous operational problem. SOC is floored to a 2.5 kWh grid and the saved master cost is the conservative expanded-grid cost; metadata correctly says continuous-cost pricing is not certified (`src/exact_pricer_expanded.py:1164-1169`; `src/event_pricer_network.py:667-724`). The baseline has no shared charger-capacity rows, no terminal return-SOC floor, and no hard recharge-count cap (module statement at `src/exact_pricer_expanded.py:15-19`). These are formulation choices or relaxations, not accelerations. A valid phrasing is “optimal LP over the stated conservative expanded event/SOC route space,” with capacity and terminal policy stated explicitly.

### 5. Capacity dual signs and schedule identity are correct, but the proof test surface is too narrow

**Severity: medium assurance gap. Confidence: high on code reading.**

The capacity master creates minimization rows of the form usage `<= count` (`src/run_capacity_speed_event_cg.py:260-267`), so Gurobi capacity duals are nonpositive. Pricing subtracts their sum from route cost (`src/event_pricer_network.py:273-304`), which correctly makes a congested schedule more expensive. Different charging schedules have different master columns in the capacity runner: `route_key` includes stations, start/end, and energy (`src/run_capacity_speed_event_cg.py:168-176`). Baseline incidence-only deduplication (`src/exact_pricer_expanded.py:1418-1463`, `2677-2680`) is valid only because the baseline master has no schedule rows and retains the cheapest objective coefficient for an identical trip-incidence column.

The capacity test demonstrates that one synthetic dual pattern moves one charge window (`tests/test_event_pricer_network.py:164-197`). It does not prove global pricing exactness across paths, SOC states, stations, tariffs, and fractional interval endpoints. Add a brute-force oracle test that enumerates every path and every current breakpoint on random tiny acyclic instances, then compares minimum reduced cost and selected capacity rows. Include two routes with the same trip incidence but different occupied rows to prevent accidental reintroduction of baseline deduplication.

### 6. Fixed-sequence replay needs an adjacency index, with all parallel physical alternatives preserved

**Severity: medium for warm-start cost. Confidence: high.**

`fixed_sequence_record` scans every outgoing arc for every frontier state and filters for the requested next trip (`src/event_pricer_network.py:1019-1061`). Bounded warm import therefore pays for repeated irrelevant scans across up to hundreds of sequences.

Index outgoing arcs by `(source_node, successor_trip_or_sink)` once. The value must remain a list of all matching target-SOC/station alternatives. Collapsing to one edge per next trip would be incorrect because the fixed sequence still needs dynamic programming over SOC and charging location. For the capacity branch, preserve schedule alternatives needed by capacity duals; do not reuse a static cheapest-action index as capacity-aware pricing. Validate byte-identical fixed-sequence records for the inherited sequence set before measuring import time.

### 7. Gurobi iterations build a SciPy incidence matrix that the persistent master does not use

**Severity: low-to-medium. Confidence: high.**

Every nonempty iteration constructs a sparse incidence matrix (`src/exact_pricer_expanded.py:2456-2478`). `_solve_master` ignores that argument for the Gurobi backend and synchronizes route dictionaries directly (`src/exact_pricer_expanded.py:2410-2435`). The persistent Gurobi master and lower-cost coefficient rewrites are already implemented (`src/master_lp_gurobi.py:221-291`).

Skip sparse-matrix construction for Gurobi; compute `incidence_nnz = sum(len(route["trips"]) ...)` for telemetry and retain full matrix construction for SciPy. `GurobiRestrictedMaster._normalize_route` already validates nonempty, unique, known trips (`src/master_lp_gurobi.py:204-219`), so removing the redundant matrix does not remove route validation. Preserve the telemetry schema because downstream profiling may expect the incidence phase.

### 8. Two small correctness/measurement guards should accompany optimization work

**Severity: low. Confidence: high.**

- The shortest-path proof assumes every retained edge is forward in `self.topo`, but graph construction does not assert it (`src/event_pricer_network.py:260-275`, `314-335`). Add a construction/cache-load invariant `rank[source] < rank[target]` for all edges. This converts malformed timing input from a silent missed relaxation into an explicit error.
- Event batch telemetry measures `pricing_shortest_path` from `started`, then measures `pricing_extra_columns` from the same unchanged timestamp (`src/event_pricer_network.py:888-900`, `1000-1016`). The second duration includes the shortest-path time again, so sums of phase durations double count it. Reset the timer after the first callback; this changes telemetry only.

## Implementation order and optional extensions

The integrated [review](REVIEW.md) and [work orders](WORK_ORDERS.md) set the schedule. The first tranche combines certificate/timing guards, indexed replay, removal of unused incidence construction and the capacity prefix-sum/memoization kernel. Compare the saved k3 dual before any long capacity rerun.

Reachability filtering, factored transitions, range-minimum queries and true k-best enrichment are conditional extensions. They require separate equivalence arguments and should not delay the paper if the smaller changes suffice. Audit the existing complementary and diversification modes before implementing a new enumerator.

The deadline/checkpoint behavior in capacity commit `253588e9` is already present (`src/event_pricer_network.py:30-37`, `913-957`; `src/run_capacity_speed_event_cg.py:373-393`) and should be retained while profiling. It prevents another lost eight-hour pass, but it is operational protection rather than a pricing acceleration.
