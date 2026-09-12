# Implementation work orders

These are review deliverables for a subsequent implementation pass. No solver change or new cluster campaign was executed by this review. Each work order should have its own branch, implementation commit, tests, input hashes and measured comparison. Keep the current experiments as controls. Do not change scientific defaults while implementing a representation-only optimization.

## Common contract

Use baseline `a29992196acb74d02b8c7891be4061718889999f`, capacity `253588e9b22d68fcbc67cb56bc3eb30cbb0e16b6`, and MIP `871d057e1067411f09581e37d78f7c1ca43f68bb` as the respective starting references; never mix their model assumptions implicitly. Preserve input, deadhead and tariff hashes; SOC/time discretization; initial/reserve/terminal energy; shared-capacity settings; objective; covering/partitioning sense; initialization; and proof scope. Existing experiments and held historical jobs remain untouched.

Report code revision, changed files, exact checks, failures, runtime/RSS, pool identities, and restoration/reversion procedure. Solver-optimal duals and selected columns need not be unique: distinguish an intentional tie-policy or simplex-method change from a wrong objective, infeasible dual or invalid certificate. Performance comparisons must state cold versus cache-hit execution and include initialization and validation time. Any cluster benchmark must follow the standing resource policy and receive a distinct experiment record; this document is not a submission manifest.

## WO-0 — Make profiling and stop descriptions trustworthy

**Suggested owner:** Luna. **Scope:** event batch timing and Gurobi method/stop metadata. About one focused implementation day plus review.

`event_pricer_network.py` starts the enrichment timer before shortest-path execution, so raw counters overlap. Add a batch-inclusive timer and genuinely exclusive shortest-path/enrichment counters. Retain or version old fields so historical charts cannot silently mix meanings. Record actual backend method rather than only the outer adapter's `highs-ds` label. Correct the Gurobi duplicate-stall message until a tested alternate-dual implementation exists.

Add counters for candidate incidence extraction, full route realization, duplicate skips, pool synchronization, solver optimization, result extraction and cold graph generation. Keep instrumentation inexpensive and measure its overhead.

**Acceptance:** controlled fake-clock or bounded timing fixture demonstrates counter boundaries; unchanged route/pricing behavior; old artifacts remain readable; no new full-model certificate is inferred from a corrected log. Do not weaken fsync or remove durable status publication.

## WO-1 — Remove unused Gurobi incidence construction

**Suggested owner:** Luna, with Sol reviewing master invariants. **Scope:** first a small matrix-construction change; delta synchronization is a second commit. Approximately one to three days for the initial change and checks.

In `exact_pricer_expanded.py`, make the ordinary/final/diversification incidence construction conditional on the backend actually requiring it. In the persistent Gurobi branch, supply `None` or a typed backend-specific input and obtain shape/nonzero telemetry without constructing the SciPy matrix. Preserve the SciPy path as an oracle. Verify all callers, including error/fallback and final-resolve paths.

Then benchmark a separate incremental synchronization path in `master_lp_gurobi.py`: stable column IDs, appended incidences and explicit cheaper-cost replacements. Batch `X`/dual retrieval where supported. Keep full prefix checking as a debug verification mode. Do not remove protection against route mutation or accidental incidence replacement.

**Acceptance:** equal model coefficients, objective and artificial totals within stated tolerances; valid primal/dual residuals; same certificate decision on small and medium cases; preserved cheaper same-incidence updates; no SciPy matrix allocation in ordinary Gurobi iterations. Report synchronization and optimizer times separately. Benchmark Method 0 versus Method 1 only in a subsequent treatment with its own identity; do not bundle it into the representation comparison.

## WO-2 — Index inherited fixed-sequence replay

**Suggested owner:** Luna for indexing; Sol for equivalence review. **Scope:** `event_pricer_network.py:fixed_sequence_record`. Approximately two to four days.

Verify that each lazy source slice is sorted by target and that target trip/SOC IDs occupy contiguous blocks. Use binary searches or a compact successor index to scan only arcs reaching the required next trip or sink. Preserve costs, all feasible alternatives and the existing tie rule. Keep the old iterator as a selectable reference for tests. Parent-pointer reconstruction can be a later independent optimization.

Preserve the current campaign cap of 512 selected sequences and 900 seconds of replay; do not silently reinterpret old unbounded experiments. A compact streaming importer may follow: validate/hash the full journal while retaining only the metadata needed for deduplication and selection. Ensure source corruption still fails closed.

**Acceptance:** exhaustive fixed-sequence checks on tiny explicit/lazy graphs, including infeasible transitions, tight SOC, time/tariff boundaries and equal-cost choices; identical cheapest cost and valid realized schedule. Compare the selected-sequence list and accepted/rejected results against current bounded import on an immutable pool. Record the performance gain at fixed sequences before testing changed warm selection.

**Separate algorithm treatment:** protect a validated predecessor integer incumbent and LP-support routes before filling the remaining cap with long/cheap/diverse routes. Map stable trip IDs and validate every carried route. Record accepted route hashes; a deadline with unordered workers can change which candidates finish. No dual, basis or certificate is inherited merely by retaining columns.

## WO-3 — Capacity-window evaluation with prefix sums and memoization

**Suggested owner:** Sol, with an independent mathematical check. **Scope:** capacity commit only, especially `_capacity_adjusted_arc` and `_charge_window_options`. Approximately three to five days, contingent on the oracle tests.

Build per-station prefix sums of the current capacity-dual rows. Reproduce `conservative_capacity_rows`' exact interval/index convention using interval endpoints, rather than creating a row frozenset for every candidate. Cache immutable breakpoint lists and energy costs. Cache the best adjusted window within an iteration under a complete identity; invalidate when any relevant dual, station/power, tariff, grid, physical window or objective changes.

Avoid sorting every feasible window if a running lexicographic minimum returns the same cost/start/end tie choice. Preserve cooperative deadlines inside any long generation/evaluation step and keep atomic checkpoints. The optimization must retain every current breakpoint; changing the capacity grid or charge discretization is outside scope.

**Acceptance:** compare old/new per-arc adjusted cost and selected interval over randomized dual vectors, fractional arrivals, exact minute boundaries, very short charges, long charges and station aliases. Include zero/missing duals and changed-dual cache invalidation. Enumerate tiny full paths and compare global pricing minima; independently recompute the complete route reduced cost from master coefficients. Compare a saved hard-case pricing dual before launching any eight-hour rerun. A speedup claim requires that paired measurement.

## WO-4 — Covering fleet-only bound and certificate guard

**Suggested owner:** Sol; mathematical review required before reporting results. **Scope:** adapt existing `certify_fleet_lp_bound.py` and objective modes. Schedule after WO-1/WO-2 rather than blocking them.

Make master sense and the supported physics explicit. Reject unsupported capacity/terminal features until their master rows, duals and pricing terms are implemented. Reuse `fleet-only` pricing with coefficient one per real route. Implement and document a valid tolerance-adjusted bound, such as the dual-scaling construction in `MATHEMATICAL_NOTES.md`, using an exact pricing minimum or a rigorous pricing lower bound. Never use a heuristic route value as that lower bound.

**Acceptance:** tiny complete-route LP/MIP oracle under both covering and partitioning; input mismatch rejection; positive/negative near-zero reduced costs; no artificials; integer-safe rounding; equality of the reported bound scope and actual route model. Stop on target attainment only when the validated incumbent matches the conservative bound. Charging at fleet cap K is a separate subsequent objective with its fleet-row dual included.

## WO-5 — Screen existing enrichment and literature accelerations

**Suggested owner:** Sol for experiment design, Luna for isolated tooling. **Scope:** existing `complementary` selector and `diversify_rounds` first. No full branch-and-price rewrite.

At equal end-to-end budgets, compare current reduced-cost selection with complementary selection and a small, prespecified diversification treatment. Preserve the exact best pricing route. Measure time to a validated fleet incumbent and charging cost, not only LP iterations or the number of columns. Audit the existing `run_exact_dive.py`/`run_peel_and_price.py` model contracts before reusing them with current event physics.

Only if these controls leave a clear gap, prototype sparse/heuristic pricing with full exact fallback and original-dual re-evaluation, following the Parmentier paper. A completion-bound or k-best implementation must retain a proof or exact oracle for the claimed route space. Keep each extra heuristic's CPU and pool/MIP overhead in the comparison.

**Acceptance:** no false certificate after heuristic failure; every added route is feasible and correctly priced; no unsafe incidence-only deduplication of resource-distinct schedules; independent held-out chains; all timeouts and unsuccessful target cases retained.

## Paper freeze

Implement only the first tranche with a clear measured benefit. Freeze the selected solver and method settings before the final grid. Keep the remainder as future work if it cannot be validated promptly. The review's full list is an option set, not a requirement to implement every item before submission.
