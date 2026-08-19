# Post-ladder research decision plan (2026-08-19)

## Scope

This is a decision plan, not a new algorithm implementation. It accepts or
rejects the 138-task flat-tariff scale ladder, separates model-space effects
from runtime effects, and authorizes only conditional follow-up work.

The evidence statements below come from tracked artifacts at reviewed base
`1d80402d79d1cbb4b786b780f7287c12b02d3621`. No Slurm query, Gurobi run, or
scientific solve was used to prepare this plan.

## Verified route-space facts

The controlling artifact is
`data/scale_ladder/known_membership_preflight.json`, SHA-256
`5124534373e8d3aff981c55891b8f7ed321fdf1efe96c8bbfd093d957c1b94c8`.
It binds:

- schema `evsp-dr-scale-ladder-membership-preflight-v1`;
- duty schema `evsp-dr-scale-ladder-known-membership-v1`;
- instance manifest SHA-256
  `a7ef8b77351440a8d7873b949891663ca7b28f135d366d4c6b003d09ca84839a`;
- historical flat tariff SHA-256
  `1f51f2e1f6ca303838ebaaf6272a28ff2d6bbee97146cb04d330e10f191f8200`
  in every cell.

Direct enumeration gives:

1. All 22 known partitions are continuously feasible.
2. All 22 have
   `known_partition_in_primary_expanded_space=false`. This means at least one
   duty in every known partition is absent from the 15-kWh/10-minute expanded
   route space. It does **not** mean every duty is absent and does not prove
   that no different target-fleet partition exists.
3. The frozen k40 cell is `k40_s2`. Exactly 9 of its 40 duties are
   primary-grid representable and fixed-sequence pricing-certified:
   `13307`, `13309`, `13311`, `13320`, `13323`, `13401`, `13406`, `13412`,
   and `13414`. The remaining 31 are not representable on that grid.

Per-cell membership context:

| Cell | Duties | Primary-grid representable duties | Fixed-sequence certified on any tested grid | Whole known partition in primary space |
|---|---:|---:|---:|---|
| k02_s1 | 2 | 0 | 2 | No |
| k02_s2 | 2 | 1 | 2 | No |
| k02_s3 | 2 | 0 | 2 | No |
| k03_s1 | 3 | 1 | 3 | No |
| k03_s2 | 3 | 0 | 3 | No |
| k03_s3 | 3 | 2 | 3 | No |
| k05_s1 | 5 | 1 | 4 | No |
| k05_s2 | 5 | 1 | 5 | No |
| k05_s3 | 5 | 1 | 5 | No |
| k08_s1 | 8 | 1 | 1 (no adaptive grid at k>5) | No |
| k08_s2 | 8 | 1 | 1 (no adaptive grid at k>5) | No |
| k08_s3 | 8 | 2 | 2 (no adaptive grid at k>5) | No |
| k13_s1 | 13 | 3 | 3 (no adaptive grid at k>5) | No |
| k13_s2 | 13 | 2 | 2 (no adaptive grid at k>5) | No |
| k13_s3 | 13 | 1 | 1 (no adaptive grid at k>5) | No |
| k20_s1 | 20 | 3 | 3 (no adaptive grid at k>5) | No |
| k20_s2 | 20 | 2 | 2 (no adaptive grid at k>5) | No |
| k20_s3 | 20 | 5 | 5 (no adaptive grid at k>5) | No |
| k30_s1 | 30 | 5 | 5 (no adaptive grid at k>5) | No |
| k30_s2 | 30 | 9 | 9 (no adaptive grid at k>5) | No |
| k30_s3 | 30 | 9 | 9 (no adaptive grid at k>5) | No |
| k40_s2 | 40 | 9 | 9 (no adaptive grid at k>5) | No |

“Certified on any tested grid” is not a cross-scale comparison for k>5:
adaptive 5/2.5/1-kWh and 1-kWh/5-minute tests were intentionally restricted
to k2/k3/k5.

### Duty 13411

Tracked k5-r1 and frozen k40-r2 inputs map duty `13411` to the same ordered
`Ordered_Trip_ID` sequence:

`[15, 28, 41, 54, 67, 80, 93, 106, 119, 132, 145, 158, 167, 176]`

Its canonical sequence SHA-256 is
`2fbf856bd5955eb8ce2bbad32f135400ec9dbc46d755cc471555eccd46aab39b`.
Solver-local row indices differ by instance, so they must not be compared:
the failed transition is local `46->53` in k5-r1 and `447->504` in k40-r2;
both are the ordered-trip transition `106->119`.

The membership artifact establishes:

- `known_partition_continuously_feasible=true`;
- no primary expanded-grid path;
- for k5-r1, no path at 15, 5, 2.5, or 1 kWh with 10-minute blocks, followed
  by no path at 1 kWh/5-minute;
- `first_feasible_soc_step=null`,
  `first_feasible_block_min=null`, and
  `fixed_sequence_pricing_certified=false`;
- recorded reason
  `no fixed-duty transition 46->53;blocked_through_1kwh_5min`.

The immediate technical reason is exact but limited: after trip 106, the DP
has no surviving direct or station/charge state that can reach trip 119 while
meeting its time, floored SOC, reserve, and block-alignment tests. Continuous
re-realization does have a feasible path. The artifact does **not** identify
which of SOC flooring, block timing, station reachability, or their interaction
is the root cause. That is why a transition oracle is required below; this
plan does not invent a more specific explanation.

The current full Tier-1 GIRO40 runner would abort. It calls
`optimize_fixed_duty` at exactly 15 kWh/10 minutes for every route and tariff;
`src/run_fixed_giro_tariff_response.py` raises immediately when any route
returns `feasible=false` (the failure path at lines 195–215). Duty 13411 is
one such frozen k40 route. The runner does not currently downgrade that cell
to unavailable or invoke an adaptive grid.

## What the current LP does and does not certify

`src/exact_pricer_expanded.py` passes each route's full `cost` into
`solve_restricted_master_lp`. That cost includes bus fixed cost and charging
terms. `route_weight` is only the sum of route variables; it is not the
optimized objective.

Therefore, route weight from the current bus-plus-charging-cost master is
**not formally a minimum-fleet LP lower bound**. Exact reduced-cost
certification proves optimality for the combined cost master in its expanded
route space. A cheaper charging mix can trade against route weight. A
fleet-only or rigorously lexicographic exact-CG phase is required before
claiming a minimum-fleet LP lower bound.

## Gate 1: accept the 138-task campaign

The 138 scientific/diagnostic tasks are accepted only as one hash-bound
evidence package:

- 22 membership preflights;
- 21 known-partition seed preparations;
- 23 primary 15-kWh/10-minute exact-CG runs;
- 30 small-grid sensitivity CG runs;
- 21 RAW finite-pool MIPs;
- 21 KNOWN-PARTITION augmented diagnostic MIPs.

The two infrastructure probes and activation controller are separate
infrastructure records, not part of 138.

All of the following must pass:

1. **Approval and scheduler gate**
   - immutable `approved-plan.json` hash equals `campaign.json` approval;
   - exact reviewed Git commit/code hashes, Python/package identity, flat
     tariff hash, instance hashes, seeds, physics, budgets, and task mapping;
   - both exact environment probes are compatible and hash-validated;
   - activation/gate/array identities and dependency contracts match;
   - scientific gate has durable `COMPLETED`, `ExitCode=0:0` evidence;
   - `submitted=true` is backed by exact scheduler verification, not a cached
     string.
2. **Task completeness**
   - exactly the planned 138 unique task keys;
   - one worker completion per task with matching plan, job key, instance,
     tariff, grid, algorithm role, and artifact hashes;
   - no completion is accepted merely because an allocation ended;
   - intended checkpoint times absent after solver termination are explicit
     censored rows, not fabricated observations.
3. **CG integrity**
   - journal prefix/hash and status/snapshot chain validate;
   - every normalized iteration comes from the bound telemetry/status;
   - master objective, route weight, artificial mass, reduced cost, columns,
     timing, certification, and stop reason retain their distinct meanings;
   - primary and sensitivity grids remain separate experimental roles.
4. **MIP integrity**
   - source pool status/journal hashes equal the paired CG completion;
   - RAW contains no external known routes or start;
   - KNOWN-PARTITION records base pool identity, added route identity, and
     accepted start separately;
   - every incumbent is an exact partition and every selected route passes the
     required physical replay;
   - checkpoint/final provenance and censoring validate.
5. **Publication**
   - deterministic normalizer completes without overwrite;
   - `artifact_inventory.csv` covers every accepted source and normalized
     artifact by hash;
   - rerunning normalization from the same package is byte-deterministic.

Any failed gate makes the corresponding cell `incomplete/censored`; it is not
silently dropped from scale summaries.

## Gate 2: per-cell classification

Classification is precedence ordered and must retain a separate
`known_partition_in_primary_expanded_space` flag.

1. **certified feasible**
   - all provenance/completion gates pass;
   - artificial mass is zero;
   - exact pricing terminates certified for the stated objective and grid.
   - The row must say `combined_cost_master_certified`; it must not call route
     weight a minimum-fleet lower bound.
2. **certified route-space infeasible**
   - an exact fleet-only CG or target-constrained global expanded-space oracle
     proves that no target-fleet solution exists under the named grid/physics;
     or a fixed-sequence membership oracle proves the specifically named known
     duty/partition is outside that space.
   - Current combined-cost CG alone cannot assign target-fleet
     `certified route-space infeasible`. For the present artifacts, the valid
     statement is narrower: all 22 **known partitions** are certified outside
     the primary route space.
3. **feasible but pricing-censored**
   - provenance passes and artificial mass is zero;
   - a feasible restricted master exists;
   - exact pricing lacks a certificate because of wall limit, preemption, or
     another explicit censoring reason.
4. **incomplete/censored**
   - missing/invalid completion, positive or unknown artificial mass, broken
     journal/snapshot provenance, environment failure, unclassified stop, or
     missing required output.

Target route weight, target incumbent, and known-partition membership are
orthogonal columns, not classification shortcuts.

## RAW and KNOWN-PARTITION MIP interpretations

### RAW

RAW asks what can be assembled integrally from exactly the generated CG pool.

- Target incumbent found: that finite pool contains a target partition.
- Finite-pool optimum/bound above target, or exact target-constrained
  finite-pool infeasibility: pool composition is insufficient for target.
- Target not found with a lower bound still below/equal target: MIP search is
  censored; pool insufficiency is not proved.
- No incumbent plus weak bound: both assembly and search remain unresolved.

RAW never proves full expanded-route-space or physical-world infeasibility.

### KNOWN-PARTITION

KNOWN-PARTITION changes the finite pool by adding exactly the selected GIRO
duties and assigning their validated start. It is an
integral-assembly/feasibility diagnostic, not algorithmic recovery.

- Accepted target start/incumbent confirms the MIP plumbing and expected
  partition after external augmentation.
- It does not show that CG generated those duties. This distinction is
  especially important because all 22 known partitions have at least one
  route outside the primary grid.
- Failure to recover the injected exact target partition indicates invalid
  augmentation/start, physical validation, formulation, or a severe solver
  lifecycle issue; it is not evidence that the target partition does not
  exist.
- RAW and KNOWN bounds/trajectories must never be merged because their column
  sets differ.

## Required normalized decision tables

1. `ladder_cell_decision.csv`: one row per primary cell with classification,
   target, objective scope, route weight, artificial mass, certificate,
   stop/censor reason, wall time, and all identity hashes.
2. `known_membership_cell.csv`: the six membership fields, representable duty
   count, duty count, and artifact hash.
3. `known_membership_duty.csv`: duty ID, ordered-trip identity, continuous
   feasibility, every tested grid outcome, first feasible grid, and exact
   failed transition/reason.
4. `cg_iteration_long.csv` and `cg_run_summary.csv`: existing normalized
   schemas with primary/sensitivity role explicit.
5. `mip_checkpoint_long.csv` and `mip_run_summary.csv`: RAW and KNOWN kept
   separate, including pool/start identities and censoring.
6. `pool_composition_summary.csv`: generated columns, unique trip sets,
   singleton/known/new counts, coverage, duplicate identities, and target
   finite-pool oracle result when run.
7. `decision_cause_matrix.csv`: one row per cell and four Boolean/three-state
   cause columns described below.
8. `artifact_inventory.csv`: hashes, sizes, roles, producer completion, and
   missing/censored status.
9. `conditional_rerun_plan.csv`: only approved cells, exact triggering
   evidence, reuse/resume source, budget, and the question the rerun resolves.

## Required figures

- primary-grid LP route weight and artificial mass versus time/iteration,
  faceted by scale and replicate;
- sensitivity-grid curves in a separate panel/style, never pooled with the
  primary comparison;
- known-duty primary-grid representability heatmap and representable-duty
  fraction versus scale;
- RAW versus KNOWN incumbent/bound trajectories with different pool-scope
  labels;
- per-cell four-cause decision matrix;
- censoring/completion map showing absent observations explicitly.

Every route-weight figure carries the combined-cost-master disclaimer.

## Decision matrix

| Observed evidence | Grid/model nonrepresentability | Pricing runtime | Missing pool composition | MIP search difficulty | Decision |
|---|---|---|---|---|---|
| Known duty fails membership but continuous replay passes | Yes for that named sequence/grid | Not implicated | Possible downstream consequence | Not implicated | Do not call CG slow; use sensitivity or transition oracle only if decision-relevant. |
| Zero artificials, exact pricing certified, target route weight not reached | Known partition flag may be yes; target fleet remains unproved | No | Possible | Not yet tested | Run fleet-only CG only if a minimum-fleet claim is needed. |
| Zero artificials, pricing wall-limited | Unknown | Yes | Unknown | Not yet tested | Resume exact journal only for decision-frontier cells. |
| RAW target-constrained finite-pool MIP proves infeasible | No global conclusion | CG may be certified or censored | Yes for this pool | No | Inspect/generate missing composition; do not merely extend MIP. |
| RAW has target-capable bound but no target incumbent at limit | No conclusion | Separate CG field | Unknown | Yes/likely | Run target-constrained finite-pool feasibility MIP for that pool. |
| KNOWN finds target immediately, RAW does not | Known routes externally repair pool | Separate | Demonstrated RAW composition gap if RAW infeasibility is proved; otherwise suggestive | RAW may still be search-censored | Keep algorithmic and diagnostic claims separate. |
| KNOWN cannot use an exact validated target start | Not a scientific route-space result | Not a pricing result | Augmentation may be invalid | Solver/formulation/lifecycle fault | Stop; repair evidence path, no scientific rerun. |

## Designed follow-up oracles (not implemented)

### 1. Fleet-only exact CG

- Use the same expanded feasibility network and trip constraints.
- Set each real route's primary objective coefficient to exactly one bus.
- Keep artificials at a separately dominant, proven penalty.
- Price reduced cost `1 - sum(trip duals)` with exact pricing.
- Optimize charging cost only in a second lexicographic phase after the
  fleet-weight optimum is fixed, or use it solely as deterministic tie-break
  metadata with a proved noninterfering scale.
- Certificate output must bind route space, grid, artificial mass, fleet-only
  objective, reduced-cost tolerance, and exact stop.

Only this result may be called the minimum-fleet LP lower bound for the named
expanded space.

### 2. Target-constrained finite-pool feasibility MIP

For a fixed, hash-bound pool, binary route variables satisfy exact trip
partition equalities and `sum(route variables) <= target_fleet`. Use a constant
objective (or a deterministic tie-break only after feasibility) and request an
exact feasibility/infeasibility conclusion.

Its certificate scope is `target_feasibility_in_named_finite_pool`; it is not a
global route-space certificate. Run it first on RAW. Run the matching
KNOWN-PARTITION pool only as a positive-control sanity check.

### 3. Duty 13411 transition/root-cause oracle

Build one immutable diagnostic for ordered-trip transition `106->119`:

1. bind the duty sequence and both k5/k40 local-index maps;
2. replay the continuous feasible witness and expose arrival time/SOC;
3. enumerate all direct and modeled-station alternatives at each tested grid;
4. report every rejection predicate separately: reachability arc, station
   arrival, first/last charge block, power/energy gain, SOC flooring,
   successor energy, and deadline;
5. compare against an exact continuous-time/continuous-SOC transition model;
6. emit either a discrete witness or a minimal rejection set with hashes.

This oracle must diagnose one transition. It must not launch another scale
sweep or silently change campaign physics.

## Minimal conditional reruns

1. Do not rerun any cell whose immutable completion and classification gates
   pass.
2. For `incomplete/censored`, resume the existing durable journal only when its
   plan/code/environment/instance/tariff/grid hashes match. Never restart from
   an unbound pool.
3. For `feasible but pricing-censored`, resume only cells whose unresolved
   certificate changes a scale-threshold conclusion (the first transition and
   one bracketing cell), not all larger scales.
4. Run fleet-only CG only on those decision-frontier cells where the paper
   needs a minimum-fleet LP claim.
5. Run target-constrained finite-pool feasibility only where RAW cannot
   distinguish pool composition from branch-and-bound search.
6. Run the duty-13411 transition oracle once. Any broader grid change requires
   a separate reviewed scientific design.
7. Never duplicate the reviewed k40 MIPs; ingest them only after exact
   instance/tariff/physics/pool/hash validation.

The next decision is made from the normalized cause matrix, not from scale
alone.
