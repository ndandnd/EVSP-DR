# EVSP–DR algorithm and implementation review

The best near-term strategy is to retain the current column-generation architecture, remove redundant work in its hot paths, and test a small number of established acceleration methods under unchanged physical assumptions. The evidence supports different priorities for the uncapacitated baseline, inherited initialization and shared-capacity extension. It does not support an immediate solver rewrite or a broad parameter sweep.

The first implementation tranche should contain three independent changes: capacity-window prefix sums and memoization; indexed fixed-sequence replay; and removal of redundant incidence construction in the persistent Gurobi path. Pair these with reliable timing and certificate checks. Then evaluate fleet-first pricing/bounds and column-pool enrichment. These recommendations are mechanisms and testable hypotheses; no end-to-end speedup from them has yet been measured.

## Scope and evidence

The baseline source is commit `a29992196acb74d02b8c7891be4061718889999f`, the actual saved-pool MIP runner is `871d057e1067411f09581e37d78f7c1ca43f68bb`, and the capacity deadline/checkpoint branch is `253588e9b22d68fcbc67cb56bc3eb30cbb0e16b6`. The active baseline uses event pricing, 2.5 kWh SOC discretization, 5-minute event support, 240 kWh batteries, 240 kW charging, covering rows, zero reserve, no shared charger capacity or return-energy floor, and route cost 100000 plus electricity and charging-start fees. Other experiments deliberately have different physics and must remain separate.

The runtime analysis freezes collector snapshot `20260912T052658Z.json`; [provenance](review_provenance.json) records source-file hashes, and [runtime evidence](RUNTIME_EVIDENCE.md) records all 84 certified fresh-covering cases plus 14 certified overnight cases available in that snapshot. Active jobs subsequently continue changing. The existing event-pricer suites passed on both pinned branches (12 baseline and 14 capacity tests), and the runtime extractor reconciled all 98 selected records against the frozen snapshot. These are limited correctness/evidence checks, not a proof of the entire algorithm or a speed benchmark. This review did not alter production solver files or operate the cluster.

| Regime | Observed evidence | Implication |
|---|---|---|
| Fresh covering, k=15, six cases | Median wall 16,324.94 s; pricing batch 9,595.16 s; master attempts 5,588.27 s; incidence construction 1,019.06 s; journal/iteration fsync 4.42 s | Both pricing and master plumbing merit attention. Weakening durability is a poor trade. |
| Cold overnight component d00_g0 | Wall 1,662.77 s, graph construction 1,311.19 s | Cached-CG timing does not measure first-use scalability. |
| Old warm P2 k=9 | 42,732 inherited columns replayed in 29,324.84 s; no LP endpoint | Import can consume the whole budget before optimization. |
| Capacity-only k=3 recovery | One pricing call 25,722.65 s, corresponding LP 0.00563 s; final 38-column pool | Optimizing the tiny master will not fix this regime. The sixteen-bus pool solution is not evidence that sixteen buses are necessary. |
| Chain-3 k=8 fresh versus inherited | Same LP endpoint; saved-pool integer optima nine versus eight buses | LP convergence and integer pool usefulness are separate targets. |

The k=15 numbers are separate medians, so they need not sum. The raw counter `pricing_extra_columns` includes shortest-path time; the analysis subtracts `pricing_shortest_path` when describing enrichment alone. The graph-build counter includes cache loading on a hit. Sources: [runtime extraction](runtime_evidence.json), [execution issues](../research_register/EXECUTION_ISSUES_20260910.md), and [capacity outcome](../parallel_research_20260911/capacity_deadline5_completed/README.md).

## Ranked changes

| Priority | Change | Why it fits | Required acceptance evidence |
|---|---|---|---|
| First | Capacity-dual prefix sums; memoized charging-window evaluation within an iteration | Removes repeated occupancy-set construction and interval summation from the demonstrated multi-hour pricing call | Identical adjusted arc costs, selected windows and exact pricing values over boundary/dual fixtures; invalidation on dual changes |
| First | Index fixed-sequence arcs by the required next trip | Current replay scans unrelated outgoing arcs despite knowing the trip sequence | Same cheapest sequence cost and valid action trace, including ties and infeasible sequences |
| First | Skip SciPy incidence construction for Gurobi; then incremental pool synchronization and bulk solution extraction | The persistent Gurobi model already owns its columns; the extra sparse matrix is unused | Same LP/model coefficients, artificials, dual checks and certificate; preserve cheaper same-incidence replacements |
| First, enabling | Fix timing labels and expose actual backend method; add dual/certificate checks | Prevents misleading profile sums and inaccurate fallback descriptions | Disjoint/explained timing totals; honest stop reasons; independent primal/dual residuals |
| Next | Deterministic, compact warm import that protects incumbent and LP-support routes | The 512/900 s cap solves runaway replay operationally but does not guarantee that useful integer combinations survive | Input/selection hashes, stable mapping, validated feasible seed routes, separate bounded-treatment label |
| Next | Fleet-only pricing and a valid early fleet bound | Aligns Phase 1 with the fleet target and can avoid unnecessary weighted-LP tailing-off | Covering-aware model contract; exact pricing or pricing lower bound; conservative integer-bound comparison |
| Next | Test existing complementary selection and diversification; only then add sparse pricing with exact fallback or deeper enrichment | Could reduce calls and improve saved-pool integer quality without abandoning the exact oracle | Equal-budget comparisons; original-dual reduced-cost checks; full exact fallback before certification |
| Conditional | Prune unreachable states, reduce graph-construction serialization, compile the packed DAG kernel | Potentially addresses cold-build and large pricing costs | Same named graph/path model, deterministic tie policy, measured build/runtime/RSS |

**Capacity pricing is the strongest algorithm-specific opportunity.** The deployed capacity branch already includes charger duals when choosing charging windows. Its expensive implementation enumerates the breakpoint windows repeatedly for each charging arc, constructs occupied-row sets and sums their duals. For consecutive occupied rows, prefix sums make each interval-dual query constant time. Reuse immutable breakpoint/energy-cost data, and memoize window minimization only under a complete key including the physical window, station/power, tariff/grid and current dual-vector version. Preserve the current breakpoint set and conservative occupancy convention. Do not replace it with a single tariff-cheapest window or coarsen capacity time bins as an unnoticed speed change. [Capacity source](https://github.com/ndandnd/EVSP-DR/blob/253588e9b22d68fcbc67cb56bc3eb30cbb0e16b6/src/event_pricer_network.py#L266).

**Warm replay has an exact representation improvement available.** `fixed_sequence_record` optimizes charging for a fixed sequence, but loops through all outgoing arcs before checking the known successor. Since trip/SOC nodes form known blocks, a verified index can select the relevant arc range directly. Preserve the existing minimum-cost and tie behavior before replacing copied path lists with parent pointers. A second, separate treatment could import a saved charging schedule by mapping and validating it, using reoptimization only when needed; this changes seed treatment and must not masquerade as identical replay. [Replay source](https://github.com/ndandnd/EVSP-DR/blob/a29992196acb74d02b8c7891be4061718889999f/src/event_pricer_network.py#L1019).

**The Gurobi master is already persistent.** Creating another persistent wrapper would duplicate existing work. The immediate waste is outside it: every iteration builds a SciPy incidence matrix that the Gurobi branch ignores. The wrapper also rescans the route prefix, and accesses individual variable values. Remove the redundant matrix first, then benchmark delta synchronization and bulk attributes independently. A primal-simplex versus dual-simplex comparison after column insertion is reasonable, but changing the method is not a guaranteed improvement. Gurobi documents reuse of prior solution information after compatible model changes. [CG call path](https://github.com/ndandnd/EVSP-DR/blob/a29992196acb74d02b8c7891be4061718889999f/src/exact_pricer_expanded.py#L2410); [Gurobi warm-start guidance](https://doc.gurobi.com/projects/optimizer/en/current/features/warmstart.html).

The Gurobi duplicate-column stall path also needs an honest fallback: it prints that it is switching to interior-point duals, but changes the method order only for SciPy. Implement an explicit supported alternate-dual path and revalidate its solution, or keep the current uncertified stop with an accurate message. This does not turn an uncertified result into a false certificate today; it can cause avoidable premature stopping. [Stall handling](https://github.com/ndandnd/EVSP-DR/blob/a29992196acb74d02b8c7891be4061718889999f/src/exact_pricer_expanded.py#L2777).

## Mathematical boundaries

The current event pricer is a shortest path on a discretized DAG. Its batch of additional routes is a sink-predecessor enrichment heuristic, not true k-shortest-path enumeration. The global best route supports its pricing certificate; the extra routes primarily affect convergence and the eventual integer pool. Complementary selection and post-CG dual perturbation already exist and should be tested before commissioning a replacement.

Cheapest-by-trip-incidence deduplication is valid only when those routes have identical coefficients in every master constraint. It is appropriate for the current uncoupled baseline, but charger occupancy, terminal energy and V2G injection profiles can distinguish otherwise identical trip sets. The capacity driver already uses schedule-sensitive keys. Preserve that distinction in any compression, indexing or shared code.

The existing fleet-certification helper hardcodes partitioning rows and reconstructs a narrower model contract. It is useful source code, but is not a drop-in certificate for current covering or capacity results. The [mathematical notes](MATHEMATICAL_NOTES.md) derive a valid fleet-only dual bound and specify what must change before using it. A weighted RMP objective, fractional route count, certified fleet-only LP bound and finite-pool MIP bound must continue to be reported separately.

## Literature to borrow from

The detailed [literature review](LITERATURE_REVIEW.md) maps papers to assumptions, code and implementation gates. The most useful immediate reading is:

| Resource | Useful transfer | Limit |
|---|---|---|
| Parmentier, Martinelli and Vidal, *Electric Vehicle Fleets: Scalable Route and Recharge Scheduling through Column Generation* | Backward completion bounds, sparse pricing, stabilization and diving; author implementation available | Different recharge/model assumptions; its reported speedups are not estimates for this code |
| Zhang et al., *On the role of time-of-use electricity price in charge scheduling for electric bus fleets* | Closely matched tariff/partial-charge/capacity-aware pricing structure | Capacity duals are already implemented here; transfer efficient evaluation and comparison design |
| de Vos, van Lieshout and Dollevoet, *Electric Vehicle Scheduling with Capacitated Charging Stations and Partial Charging* | Network/discretization design and price-and-branch versus truncated-CG comparison | Grid and model assumptions determine what its gaps certify |
| Pessoa et al., *Automation and Combination of Linear-Programming Based Stabilization Techniques in Column Generation* | Adaptive dual smoothing to reduce oscillation and calls | Always certify against the correct original dual/model |
| Desaulniers, *Managing large fixed costs…*, and Bärmann et al., *Lexicographic column generation…* | Fleet-first/lexicographic alternatives to an arbitrarily large route coefficient | Existing helper code still needs covering and model-identity adaptation |

Primary sources: [Parmentier paper](https://arxiv.org/abs/2104.03823), [Zhang paper](https://onlinelibrary.wiley.com/doi/10.1111/mice.13134), [de Vos paper](https://arxiv.org/abs/2207.13734), [Pessoa paper](https://pubsonline.informs.org/doi/10.1287/ijoc.2017.0784), [Desaulniers paper](https://www.sciencedirect.com/science/article/pii/S0305054805002066), [Bärmann paper](https://link.springer.com/article/10.1007/s00186-025-00905-3).

The strongest verified code resource is [ElectricalVSP-ColumnGeneration](https://github.com/axelparmentier/ElectricalVSP-ColumnGeneration): the author repository identifies an MIT license and contains C++17 code, with CPLEX and Boost dependencies. It is a source of algorithms and implementation patterns, not a drop-in Python/Gurobi replacement. Preserve attribution and license notices for copied code. Other papers with visible code but no verified license remain idea references rather than code-copy candidates.

## Experiments and paper direction

Use a staged experiment, not the Cartesian product of every idea. First compare each representation-only change against the frozen implementation on tiny exact fixtures and saved pricing duals. Then use a compact diagnostic set: the hard capacity k=2/k=3 cases, the problematic warm prefixes, and the existing fresh six-chain cases at k=5,10,15. Retain failed/censored cases and use identical inputs, physics, tariff, graph, hardware class and wall/CPU budgets. A faster implementation may generate more columns within a wall budget; that is a useful outcome, but differs from an identical-trajectory microbenchmark.

After screening, run only the successful combined implementation over the existing 84-case grid. Report end-to-end cold time, warm/cache-hit time, initialization, shortest path, enrichment, master synchronization/optimization, peak RSS, time to validated target fleet, charging objective and proof status. Include a tuning/held-out split of the six chains so settings are not chosen and evaluated on exactly the same cases. For heuristic variants, record random seeds and accepted warm-pool hashes; parallel deadline selection is not deterministic merely because the preselection sort is deterministic.

Reserve roughly two weeks for focused implementation and correctness gates, two weeks for diagnostic/held-out experiments, and the remaining month for the final result grid, tariff/capacity sensitivity and writing. Work on the paper and comparison definitions from the start. Do not spend all eight weeks implementing the full engineering inventory in the appendix.

A viable central question is: **under matched fleet, terminal-energy and charging-resource assumptions, when does joint duty/charging optimization improve a fixed-duty charging baseline, and how quickly can a validated solution with a meaningful bound be obtained?** TOU-aware electric-bus optimization is already in the literature. The contribution would need to be the particular validated method, controlled computational evidence and operational findings, rather than merely observing that optimizing electricity cost beats ignoring it. The synthetic peaks at 08:00,12:00,18:00 remain useful controlled checks. Add a real or explicitly justified tariff scenario and charging-flexibility sensitivity only after the comparison is fair and the solver is stable.

The simultaneous V2G review confirms that scenario-template reuse and sparse/persistent masters are useful there. DR already has a persistent master, so that transfer primarily reinforces avoiding reconstruction. Conversely, DR's trip-incidence-only column identity does not transfer to V2G energy profiles. Full stochastic/V2G integration, a complete branch-and-price tree, and a new continuous-state/DDD solver are sensible later directions; they should not be dependencies of this conference submission.

## Deliverables

- [Implementation work orders and acceptance gates](WORK_ORDERS.md)
- [Pricing and algorithm audit](PRICING_REVIEW.md)
- [Engineering audit](ENGINEERING_REVIEW.md)
- [Mathematical notes](MATHEMATICAL_NOTES.md)
- [Literature and code resources](LITERATURE_REVIEW.md)
- [Frozen runtime evidence](RUNTIME_EVIDENCE.md), [machine-readable cases](runtime_evidence.json), [source pins](review_provenance.json)

No existing run is superseded merely by a proposed efficiency change. Deployments and new results require their own execution identities and register entries. The active baseline and capacity campaigns remain the controls.
