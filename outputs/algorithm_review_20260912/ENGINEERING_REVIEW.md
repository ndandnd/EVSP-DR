# EVSP–DR engineering efficiency review

Review scope: pinned source `a29992196acb74d02b8c7891be4061718889999f` at `/private/tmp/evsp-algorithm-review-source-20260912`, with the actual pool-MIP runner reviewed at `871d057e1067411f09581e37d78f7c1ca43f68bb` from the original repository. This is a read-only code review. No cluster work, solver run, or code change was performed. The repository instructions in `AGENTS.md` and the execution record in `outputs/research_register/EXECUTION_ISSUES_20260910.md` were used as scope and correctness constraints.

This subreview concentrates on large baseline campaigns; capacity-pricing findings and the overall priority order are in the main review. The 10 September execution record establishes that inherited full-pool replay can consume the whole allocation before the first LP. The bounded 12 September campaign already limits inheritance to 512 sequences, 900 seconds, and 8 workers. Separate runtime evidence shows that large baseline cases spend substantial time in pricing, incidence construction, and master solves, whereas fsync is small. The recommendations below preserve the distinction between a bounded seed pool and a pricing certificate.

## Ranked findings

### Measurement prerequisite — Phase telemetry has overlapping pricing timers

**Evidence.** In `src/event_pricer_network.py:888-899`, `sink_predecessor_route_batch` starts one timer before the shortest path and reports `pricing_shortest_path`. It then reports `pricing_extra_columns` at `:1000-1017` using the same timer, so the latter includes the shortest-path duration. The frozen-monitor example supplied with this review records shortest path 203.55 s and extra columns 421.33 s; the exclusive extra-column time is about 217.78 s. This inflates phase sums and can misdirect optimization work.

**Action.** Use one timer per phase, or report an explicit inclusive/exclusive field. Keep the route-generation behavior unchanged. Add a telemetry regression asserting that exclusive child phases do not sum above the enclosing pricing phase and that the total still matches wall time within the measured instrumentation overhead.

**Complexity and effort.** No model or asymptotic change. About 0.5–1 day. This is already instrumented but not semantically correct; do not compare old phase totals with corrected totals without relabeling them.

### P1 — Inherited replay is still unbounded by the core CLI defaults

**Evidence.** `src/exact_pricer_expanded.py:464-470` reads and collects the complete source journal, `:494-510` materializes every replay item, and `:525-537` starts the fork pool. Every item runs `fixed_sequence_record`, physical route validation, and block-hash validation in `:404-428`. The CLI defaults at `:3164-3165` are `inherit_max_columns=0` and `inherit_time_limit_s=0`; the worker default at `:3269-3275` is one. The old warm-chain record reports 42,732 unique columns, 29,324.84 s (0.686 s/sequence), no LP, and no certificate. The same record reports peak RSS 3,765.7 MB while the cached event graph contained 119,168,287 arcs and 1,906,692,592 packed arc bytes.

**Assessment.** The campaign wrapper in the 12 September bounded treatment already passes 512/900/8 and explicitly labels the result as a partial, uncertified seed. That is the correct operational policy. The unsafe part is that the reusable core still makes unlimited replay the default, so a caller can recreate the eight-hour initialization failure accidentally.

**Action.** Make unbounded inheritance an explicit opt-in (for example, require a named unlimited mode), and make the bounded values visible in the status identity and launch manifest. Preserve completed validated replays after a deadline and retain `inherited_duals=false`, `inherited_basis=false`, and `inherited_lp_certificate=false`. Do not describe a bounded import as a warm LP or pricing continuation.

**Complexity and effort.** Current import is O(R × replay-cost) time and O(U × full-record-payload + graph) memory, where R is source journal records and U is unique incidences. The policy hardening is about 1 day. It is already implemented at the campaign layer; the core default and provenance guard remain to be hardened.

### P1 — Stream a compact inheritance index instead of retaining full route records

**Evidence.** `src/exact_pricer_expanded.py:453-470` loads all JSONL records and then `load_column_pool` at `:1418-1463` retains complete route dictionaries, including charging stops and continuous block payloads, even though inheritance later uses only the ordered `trips` sequence and source `cost` at `:494-510`. The bounded selector at `:513-518` sorts all replay items before taking the cap.

**Action.** Add an inheritance-specific streaming reader that validates the same journal identity and route-incidence rules while maintaining only the cheapest `(ordered_sequence, cost)` per incidence. Select the bounded top set with a bounded heap or compact metadata list, then replay those sequences. Keep the general `load_column_pool` path unchanged for normal resume and MIP preparation.

**Complexity and memory.** Journal parsing remains O(R); replay remains O(B × replay-cost) for a cap B. Memory changes from full route payloads to O(U × average sequence length), plus the event graph. The exact reduction depends on the stored charging-block payload and must be measured on one immutable journal; no numeric speedup should be promised. About 2–3 days.

**Correctness gate.** On a toy and one archived journal, compare source hash, unique-incidence count, selected sequence/cost list, accepted/rejected reason counts, and child records against the current bounded implementation. The compact reader must reject the same malformed or interior-corrupt JSONL cases as `durable_io.read_jsonl_records`.

### P1 — Index fixed-sequence replay transitions by successor trip

**Evidence.** `src/event_pricer_network.py:1019-1061` implements `fixed_sequence_record`. For each required successor it scans every outgoing arc from every current frontier node (`:1029-1044`) and then filters by the successor trip. In the lazy representation those source arcs are already stored in sorted contiguous slices (`:496-506` and `:314-353`). Most replay arcs cannot lead to the known next trip, so the scan repeats avoidable work for every inherited sequence.

**Action.** Build a non-authoritative lookup from `(source_node, successor_trip)` to the relevant arc offsets or a compact range over sorted targets. Use it only in fixed-sequence replay; retain the current all-arc iterator as a debug fallback and for any ambiguous equal-cost transition. Parent pointers should replace `edges + [(source, target)]` list copying in `:1040-1043` if profiling shows long sequences make that allocation visible.

**Complexity.** The current fixed-sequence replay is proportional to the sum of outgoing degrees over each frontier and successor step, plus repeated path-list copying. An indexed lookup approaches the number of arcs matching the requested successor, with O(number of replay states) auxiliary memory. The exact benefit depends on arc degree and frontier width; measure it on the archived 42,732-column case rather than extrapolating from the 0.686 s/sequence average.

**Correctness gate and effort.** For small explicit/lazy graphs, compare cheapest path cost, ordered actions, expanded and continuous schedules, and rejection reasons for every fixed sequence. Verify that equal-cost tie ordering remains deterministic or record the intentional tie-policy change. About 3–5 days. This is a replay optimization only; it does not transfer a saved schedule in place of child-graph reoptimization.

### P1 — Avoid rebuilding incidence for the production Gurobi master

**Evidence.** In `src/exact_pricer_expanded.py:2456-2477`, every CG iteration constructs a SciPy sparse incidence matrix. The production branch at `:2410-2427` sends Gurobi through `GurobiRestrictedMaster.sync_routes`, which does not consume that matrix. Runtime evidence for the large baseline cohort attributes 1,019.06 s to incidence construction and 5,588.27 s to master attempts; this is material at scale even though it was negligible in the small capacity pilot.

**Action.** Make incidence construction conditional on the SciPy backend or on telemetry that explicitly requests the matrix. For Gurobi, report rows/columns and an incrementally maintained nonzero count without constructing a SciPy matrix. Retain a debug assertion that the maintained count equals a sampled full construction before production use.

**Complexity.** Current work is O(I × nnz) incidence construction over I iterations, plus allocations proportional to the current pool. The Gurobi path can reduce this to O(I × Δnnz) bookkeeping, where Δnnz is newly added or rewritten route incidence. This is a safe representation optimization because Gurobi's persistent model owns the constraints and columns.

**Correctness gate and effort.** Compare each LP objective, artificial total, route weights, trip duals, row/bound violations, pool order/hash, and final certificate on a toy and a matched medium case. About 3–5 days.

### P1 — Synchronize only changed persistent-master columns

**Evidence.** `src/master_lp_gurobi.py:221-291` normalizes the entire route list and scans the entire persistent prefix on every `sync_routes` call. This is done from `exact_pricer_expanded.py:2415-2427` on every iteration. The implementation correctly allows a lower-cost rewrite of an existing same-incidence column at `:250-263`; replacing that correctness check with blind append-only logic would be unsafe.

**Action.** Keep a route identity/cost version in the CG pool and add an incremental synchronization path: append only new columns and update objective coefficients only for explicitly changed same-incidence columns. Preserve the prefix identity checks for a debug or audit mode. This should be paired with the incidence change above, not treated as a reason to loosen route-cost validation.

**Complexity.** Current synchronization is O(I × P) route normalization/prefix checks, where P is the pool size. The target is O(I × ΔP + W), where W is the number of cheaper rewrites. Gurobi model update cost remains to be measured. About 3–5 days.

**Correctness gate.** Require identical route order, incidence, objective coefficients, duals, LP objective, and certificate status between full synchronization and incremental synchronization. Include a test that rediscovers an existing ordered sequence with a lower cost and verifies that the persistent objective coefficient is reduced exactly once.

### P1 — The documented Gurobi dual fallback is not active for production stall handling

**Evidence.** The stall branch in `src/exact_pricer_expanded.py:2778-2793` says it will switch to interior-point duals. However, `method_order` is initialized to `("gurobi",)` for the Gurobi backend at `:1552-1555`; the replacement at `:2784-2788` executes only when `master_backend == "scipy"`. Thus a Gurobi degenerate stall repeats the same production method and then stops uncertified. Separately, `src/master_lp_gurobi.py:324-329` forces Method 1 on every solve.

**Action.** Treat this as a certification/recovery issue before treating it as a speed issue. Add a controlled, explicitly configured alternate-dual solve for Gurobi. Until it is validated, retain the uncertified stop and correct the misleading message; do not disable termination or conflate duplicate-column stalls with the separate marginal-returns rule. Method 0 is primal simplex, not automatic selection. Benchmark any method change on matched pools, and do not assume barrier with default crossover supplies an interior dual solution.

**Correctness gate and effort.** The alternate solve must return a valid optimal RMP, preserve route costs and pool identity, and only change the dual vector used for the next pricing pass. Add a regression where duplicate-incidence pricing stalls under one dual solution but continues under the alternate. About 2–4 days, plus a matched benchmark.

### P2 — Path realization is performed before incidence deduplication

**Evidence.** `src/event_pricer_network.py:869-943` walks every eligible sink predecessor. `_walk` calls `_record` (`:727-754`), and `_record` runs continuous realization, cost reconstruction, and physical validation (`:663-725`) before `sink_predecessor_route_batch` checks `frozenset(route["trips"])` in `:932-934`. A candidate that maps to an already-seen trip incidence still pays the full realization cost. Complementary selection can also inspect up to `candidate_multiplier × limit` candidates.

**Action.** Split path reconstruction into a cheap action/sequence form and a validated record form. Reconstruct enough metadata to compute the path reduced cost, deduplicate by ordered sequence/incidence according to the current pool semantics, and only call `_record` for candidates that survive the batch selection. Keep the exact best path as the first validated candidate.

**Complexity.** Saves one physical replay and block serialization for each discarded duplicate candidate; worst-case pricing remains O(E) for shortest path plus O(K × record-cost) for K retained routes. No speedup should be claimed until candidate discard rates and record times are measured.

**Correctness gate and effort.** Compare explicit and lazy event modes on small instances, including same-incidence different-order routes, tariff boundaries, and failed continuous realizations. Require identical retained route incidences, reduced costs, expanded-grid costs, continuous block hashes, and certificate behavior. About 3–5 days.

### P2 — Event-network construction spends CPU on JSON tie keys

**Evidence.** During construction, `src/event_pricer_network.py:276-292` calls `json.dumps(action, sort_keys=True)` for every candidate in `_add` and again when sorting each source's retained rows. The large cached graph has 119.2M arcs; a fresh `d00_g0` component reports 1,311.19 s of network build versus 331.91 s for its pricing batches. The current implementation already deduplicates arcs and uses packed lazy storage, so this is a construction constant-factor target rather than a change to the pricing recurrence.

**Action.** Replace JSON tie strings with a fixed primitive canonical key containing the fields that determine the desired tie order. Keep the retained cost/incidence semantics and document whether route tie ordering is part of reproducibility. Profile before and after; a different equal-cost path is acceptable only if the project accepts changed column identity and retains the same reduced-cost certificate.

**Complexity and memory.** Current tie handling is O(candidate count × serialized-action size) plus transient JSON allocations. A primitive tuple is O(candidate count) with much smaller constants. About 3–5 days including an explicit-mode oracle comparison. No speedup is promised before measurement.

### P2 — Narrow recipes can reduce packed arc buffers by 12.5%

**Evidence.** Lazy arcs use `array("I")` for targets and recipes and `array("d")` for costs at `src/event_pricer_network.py:314-347`. That is 16 bytes per arc for the three packed buffers. The observed 119,168,287-arc graph therefore uses exactly 1,906,692,592 packed bytes, matching the recorded cache metric. Recipe values are `1 + station_position × len(grid) + exit_level` at `:367-374`; the reviewed production grids are far below 65,535.

**Action.** Add a schema-versioned narrow recipe representation (`H`) when the maximum encoded recipe is proven below 65,535, and retain `I` otherwise. Store the selected dtype in the cache manifest and reconstruct the matching NumPy view in `__setstate__` (`:217-235`).

**Memory estimate.** Narrowing only recipes from 4 to 2 bytes saves 2 bytes per arc, about 238.3 MB for 119.2M arcs, or 12.5% of the packed buffers. Narrowing both targets and recipes is not recommended without proving node-ID bounds; targets can exceed 65,535. The process RSS will not fall by the same amount because Python graph metadata and caches remain.

**Correctness gate and effort.** Assert max recipe and node bounds before writing; compare cache hash/metrics, explicit-mode paths, lazy-mode paths, and selected action reconstruction. About 2–4 days.

### P2 — MIP pool preparation performs several full passes and is safe to streamline after CG

**Evidence.** The actual runner at commit `871d057e1067411f09581e37d78f7c1ca43f68bb` hashes status/journal, parses the journal through `load_pool`, then hashes the sources again around `src/run_exact_pool_mip.py:1952-2056`. Strict preparation then replays and validates each route in `:494-850`, including persisted block validation, deterministic expanded-path realization, continuous cost reconstruction, and cost/hash checks. This is intentionally fail-closed, but it repeats O(file-bytes) hashing and retains complete route records before producing the accepted pool.

**Action.** Consider a streaming parser that computes the digest while validating records, then performs one immutable post-read stat/hash check. In strict preparation, retain the existing deterministic realization and input-hash checks; only discard superseded duplicate records earlier. This is a lower priority than large CG pricing/master work. The 871 fix must remain: its `merge_validated_partition_start` changes preserve `expanded_grid_charging_stops` and saved continuous blocks (`git show 871d057e...`, around lines 1192–1326) and verifies the expanded-grid cost. Removing those fields to save serialization would silently change master cost semantics.

**Complexity and effort.** Current I/O is multiple O(file-size) passes plus O(P × physical-replay) strict preparation. A streaming pass reduces transient memory and some I/O, not the required route-validation work. About 2–4 days. Validate identical source hashes, accepted/repaired/rejected sets, ordered-pool hash, and MIP objective semantics.

### P2 — Pool deduplication is safe only under the current no-cross-route-capacity MIP

**Evidence.** Both exact-CG and MIP pool loaders retain the cheapest record by `frozenset(record["trips"])` (`src/exact_pricer_expanded.py:1418-1463`; actual runner `:198-260` and `:266-277`). The MIP model has only per-trip cover/partition rows at `:2345-2355`; it has no shared charging-capacity constraints. Under that model, a more expensive route with the same incidence is dominated in the master objective, regardless of its ordered path.

**Action.** Keep this representation for the present fleet/charging-only model, but encode the scope in the pool schema and refuse or disable incidence-only deduplication when cross-route charger capacity or any path-dependent coupling is added. A route identity for such a model must include the ordered trip sequence and the capacity-resource profile, not only the trip set.

**Complexity and effort.** Current deduplication is O(P × average route length) time and O(U) retained records. Adding a scope guard is about 1 day. This is a correctness gate, not a promised speedup.

### P3 — Final replay can reuse the already-built arc map; durable fsync is not a speed target

`validate_injected_route` builds an adjacency map when `arc_map` is absent (`src/run_exact_pool_mip.py:350-355`), while `charging_stop_arrivals` rebuilds a similar map at `:451-458`. Strict pool preparation already passes one map (`:584-635`), but final selected-route replay can call the default path (`:1598-1606`) and rebuild it. Pass a shared map through final replay and charging-arrival helpers after a regression on route validation. This is a small O(|adjacency|) setup saving, likely negligible for a handful of selected buses; effort 1 day.

The durable journal fsyncs are deliberate at `src/durable_io.py:24-33` and are needed for preemption-safe checkpoints. Runtime evidence reports about 4.42 s of fsync for the large baseline cohort, so relaxing durability is not justified as a speed plan.

## Two-month feasible implementation plan

The schedule below is an engineering option inventory. The main review limits implementation to the highest-value items and reserves time for paper experiments and writing.

**Weeks 1–2: measurement and semantics.** Correct the phase telemetry labels and add counters for candidate paths, duplicate incidences, record realization time, pool synchronization time, incidence construction time, and journal bytes. Build a fixed toy oracle with explicit event arcs and a medium immutable pool. Record source/input hashes and commit IDs for every comparison. No production algorithm change should be judged from the currently double-counted phase totals.

**Weeks 2–3: bounded inheritance hardening.** Make unlimited replay explicit, implement the compact streaming inheritance index, and preserve the current 512/900/8 campaign policy. Verify selected-sequence hashes, accepted/rejected reasons, child route records, and the absence of inherited certificates. This directly prevents another full-pool initialization loss while preserving the scientific interpretation.

**Weeks 3–5: production master plumbing.** Conditionalize incidence construction for Gurobi and add incremental persistent-master synchronization. In parallel, implement and test the Gurobi degenerate-stall fallback; until then retain the honest uncertified stop and correct its misleading log. Compare complete LP trajectories and duals on matched cases before allowing a default change.

**Weeks 4–6: event representation work.** Profile JSON tie-key construction and candidate path realization. If counters show material cost, implement primitive tie keys and deferred record realization behind a feature flag. Use explicit/lazy equivalence tests and retain the current best-route certificate semantics.

**Weeks 6–7: memory and MIP I/O.** Prototype narrow recipe buffers on the largest cached graph and measure peak RSS, cache bytes, load time, and route replay behavior. Then consider streaming MIP journal parsing and shared arc maps. Keep the 871 expanded-grid path preservation and cost checks unchanged.

**Week 8: matched validation and paper freeze.** Run a small exact oracle set, one medium baseline set, and one largest feasible case with the same inputs and physics. Report measured wall/RSS changes, pool hashes, LP objective/dual differences, certificate scope, finite-pool MIP scope, and physical replay results separately. Any change without a matched measurement or a completed correctness gate should stay out of the paper implementation.

## Recommendation order

The immediate engineering order is: fix telemetry accounting; enforce bounded inheritance in the reusable entry point; remove Gurobi-only incidence construction; make persistent synchronization incremental; resolve the Gurobi stall fallback; then profile and address path realization and event-network construction. Recipe narrowing and MIP streaming are worthwhile memory/I/O work, but they should follow the measured large-case profile. The runtime evidence does not support spending the first weeks on fsync reduction or small-pilot LP tuning.
