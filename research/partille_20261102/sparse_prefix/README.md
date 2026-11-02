# Partille sparse-prefix comparator

For each of six nested chains, the k8 policy solves k5 then k8 inside one allocation. The k10 policy solves k5, k8 and k10 inside its own allocation. The latter pays again for its own predecessors. This is a sparse-prefix approximation to historical every-k initialization.

The matched fresh control and sparse-prefix comparator both use simple dive, 240 kWh batteries, 240 kW opportunity charging, 60 kW at PARX, a fee of 5, the same flat tariff, covering trip rows, aggregate terminal target zero, and the same station capacity map. Rates and flat tariffs here describe the algorithm benchmark. The separate five-bus study uses different charging physics. Each target's fleet cap is its registered time-only floor plus four.

## What crosses the boundary

Only trip sequences from the immediately preceding finite-pool MIP selection are eligible. The receiver authenticates the source instance, selected file, summary, validation, model bytes, shared inputs and terminal/plug/physics settings. Every parent trip is matched by its stable source ID, endpoints, start/end times and energy before translating to row indices in the extended instance. A predecessor cover with duplicate service may supply individual sequences. It never provides an inherited incumbent or a target hit.

The receiving event graph regenerates a path along each sequence. It recalculates grid and continuous charging costs, charge blocks, return energy and conservative one-minute occupancy. Physical validation uses zero arrival grace. The master receives these regenerated columns and newly constructed singletons, including the new trips. Failure to authenticate or replay a predecessor stops the case and preserves its receipts. A predecessor with no MIP selection ends the policy as a scientific miss. The legacy route-import guard continues to reject raw inherited schedules under station-specific power.

## Equal-compute comparison

Each target policy gets one 20-hour allocation with eight CPUs and 48 GB. Per-phase limits total 10,800 seconds of CG, 5,400 of simple dive, and 3,600 of finite-pool MIP. Two-stage nominal shares are 1/3 and 2/3. Three-stage shares are 1/6, 2/6 and 3/6. A later stage's limit cannot exceed the remaining allowance after measured previous expenditure. Unused nominal limits do not roll forward. The receipt reports allocated limits, actual solver phase duration, replay and graph time, stage wall time, and the elapsed wall time already spent on all ancestors. End-to-end time includes input checks, graph construction, replay, validation, both MIP phases and all preceding stages. Native solver cleanup can exceed a requested phase limit. Measured totals and per-phase compliance are reported separately.

For the primary equal-compute comparison, an interrupted or requeued arm is censored unless consumption across all attempts can be established within the original limits. Apply this same rule to fresh and prefix jobs. Preserve a restarted scientific endpoint as secondary. The prefix worker refuses a fresh computational budget on restart. An actual aggregate phase-limit breach also excludes the case from the budget-matched primary comparison.

## Interpretation

A primary target hit requires an independently replayed selection with exactly-once source service, valid physical energy/chronology and shared plugs, and fleet at the registered bound. A valid covering solution with duplicate service is recorded separately. Weighted CG LP certificates, when issued, apply only to the specified event-graph master. MIP bounds apply to the attained finite route pool. A gate release or Slurm completion alone establishes no scientific hit.

The deployment checks real nesting for every k5→k8 and k8→k10 pair on Unicorn. It reads the cluster resource policy, checks both pilot release IDs and the frozen solver pin, installs the receiver patch in an isolated sparse checkout and uses a locked submission ledger plus Slurm history to prevent duplicate submissions. Both arrays are gated with afterok. Their six cases have full array concurrency. No owner checkout or historical job is modified.

The source patch is against `3a0ab6a8ab50e4af7d5a7be7688d5ffa1fbbf78c`. Run `test_sparse.py` with `PREFIX_SOLVER` pointing to the isolated patched checkout. Tests use synthetic data. Licensed trip inputs, columns, route-level witnesses and detailed solver logs stay on Unicorn.
