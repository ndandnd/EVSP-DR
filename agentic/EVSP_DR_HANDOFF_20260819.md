# EVSP-DR handoff: honest state, failure analysis, and restart plan

**Date:** 2026-08-19  
**Repository:** `https://github.com/ndandnd/EVSP-DR`  
**Local clone:** `/Users/nathan.cho/Documents/demandResponse/EVSP-DR`  
**Immediate problem:** the requested controlled smaller-instance scale ladder has **not been collected**.

This document is intentionally blunt. It separates research progress from
software packaging, and software packaging from actual experiments. A pushed
launcher, a passing unit test, an environment probe, a held Slurm gate, and a
completed optimization run are five different things. The last week repeatedly
reported the first four as progress toward the fifth.

The `egg` project is separate. Any commands mentioning `~/egg`, A6, B3,
PR #28, or Slurm array `248911` belong to that repository and must not be used
as evidence or instructions for EVSP-DR.

---

## 1. Executive summary

### What is true

1. EVSP-DR now has an SOC-by-time expanded-network pricing algorithm that is
   exact **within a named discretized route space**. It supports waiting and
   delayed charging, persists columns and trajectories, and can return a real
   reduced-cost certificate for its stated master objective.
2. Historical pair and small-union experiments provide strong evidence that
   this exact pricer works at small scale. Those outputs are fragmented across
   releases and Unicorn directories; they are not yet one current,
   reproducible scale-study package.
3. We discovered that all 22 selected GIRO comparator partitions are
   continuously feasible but outside the primary 15-kWh/10-minute expanded
   route space. Only 9 of 40 duties in the frozen k40 comparator are
   representable on that primary grid. Therefore, failure to reproduce the
   literal GIRO partition is not automatically a runtime failure.
4. We discovered and repaired a serious physical-realization defect: SOC
   flooring discarded residual energy in the grid state, while extracted
   charging schedules later retained that residual and could overfill the
   battery. The audited k40 columns were repairable, but old recorded charging
   schedules are not automatically physical artifacts.
5. The present exact-CG master minimizes bus fixed cost plus charging cost.
   Its reported `route_weight` is **not formally a minimum-fleet LP lower
   bound**. A three-phase lexicographic master is required before making that
   claim.
6. The current controlled scale-ladder plan is well specified: 138
   scientific/diagnostic tasks over k2, k3, k5, k8, k13, k20, k30, and a
   reuse-oriented k40 cell, with trajectories and finite-pool MIP checkpoints.

### What is not true

1. We have not completed the current controlled k2--k30 ladder.
2. We do not currently have one clean set of CSVs showing LP and MIP
   convergence across those scales under one code/model/instrumentation
   contract.
3. The current RAW exact-CG method has not independently recovered a 40-bus
   integer solution for the 947-trip full instance.
4. A finite augmented-pool 39- or 40-bus result is not RAW route recovery and
   is not global optimality.
5. We have not completed the paper-grade tariff-response experiment.
6. We have not run the held-out Frölunda data as a replication study.
7. The recent packaging/test effort did not compensate for the absence of
   actual smaller-scale results.

### Current Unicorn state, from the latest supplied output

| Job | Role | State | Runtime | Exit code |
|---|---|---:|---:|---:|
| `250111` | default-partition environment probe | `COMPLETED` | 45 s | `0:0` |
| `250112` | Scaglione environment probe | `COMPLETED` | 8 s | `0:0` |
| `250113` | activation controller | `FAILED` | 13 s | `1:0` |
| `250838` | scientific gate | `PENDING (JobHeldUser)` | 0 s | `0:0` |

There is no evidence in the supplied output that a PREFLIGHT, SEED, CG,
CG-sensitivity, RAW-MIP, or KNOWN-MIP array was submitted. Therefore, this
campaign collected **zero smaller-scale scientific rows** so far.

The 13-second activation runtime plus one visible held gate strongly suggests
that Slurm accepted the gate but the controller's six-second bounded
rediscovery window did not see it in time. This is a diagnosis from the state
pattern, not a confirmed traceback. The exact source is
`activation_a1_250113.err`/`.out` and must be retained.

---

## 2. The research question

GIRO produced an industrial bus schedule without modeling time- and
location-varying electricity prices. EVSP-DR is intended to answer two linked
questions.

### Goal 1: algorithmic trust

Can our column-generation pipeline approximately reproduce, or improve upon,
GIRO's routing/fleet quality before we ask it to optimize electricity costs?

The practical validation sequence should be:

1. Verify that selected instances are physically meaningful under the named
   model.
2. Eliminate artificial columns in the LP.
3. Obtain a fleet-only LP bound or clearly label pricing as censored.
4. Determine whether a finite generated pool contains a target-size integer
   partition.
5. Separate four possible failure causes:
   - route-space discretization;
   - pricing runtime/censoring;
   - missing complementary column composition;
   - finite-pool MIP search difficulty.
6. Measure where behavior changes as scale grows from 2 to 30/40 duties.

The original engineering target was not that every final MIP must exactly equal
GIRO. The crucial early criterion was that the LP fleet bound should reach or
go below the known target. That criterion now needs a correction: the current
combined-cost master does not formally supply a pure fleet bound.

### Goal 2: demand response

Given time-varying or location-varying tariffs, how much value comes from:

1. repricing GIRO's recorded charging behavior;
2. retaining GIRO trip sequences but re-optimizing charging;
3. jointly changing route assignment and charging?

The scientifically useful decomposition is:

- **Observed GIRO:** recorded schedule, cross-priced under every tariff.
- **Fixed sequence:** GIRO trip sequences fixed, charging optimized.
- **Joint:** routes and charging optimized together.

This remains the right paper structure. It is not ready for headline claims
until terminal SOC, charging-start cost, charger power, reserve, and charger
capacity assumptions are frozen and every schedule is evaluated through the
same continuous postprocessor.

---

## 3. Dataset and model context

### 3.1 Dataset currently in scope

The active experiments use the Partille (`Par`) data. Frölunda has not been
used as the held-out replication dataset yet.

The trip set was reverse-engineered from GIRO vehicle-detail output rather than
received as a clean scheduling-input table. This is an acknowledged source of
inefficiency and possible semantic drift.

The full comparator initially appeared to contain 42 duties. Two pairs were
weekday variants rather than simultaneous duties. The frozen k40 comparator
uses 40 unique duties, excludes `13316m` and `13324muw`, and covers 947 trips
exactly once.

### 3.2 Spatial simplification

Locations are modeled using a coarse copy-name/zone feature analogous to a
ZIP-code centroid. Deadhead time and energy are based on these coarse location
identities rather than exact geography. Trips sharing a coarse location can
therefore appear to “teleport” within that zone.

The user explicitly accepted this simplification for the current research
stage. It must still be stated in any methods section and must not be confused
with an exact GIS deadhead model.

### 3.3 Physics regimes

The primary scale ladder uses the historical convention:

- battery capacity: 300 kWh;
- charging power: 300 kW;
- reserve: 0;
- primary SOC grid: 15 kWh;
- primary time block: 10 minutes;
- flat historical tariff.

The source data support a battery near 239--240 kWh more strongly than 300 kWh.
Uniform 220 kW charging is not equally well established. Therefore
240 kWh / 220 kW / reserve sensitivity is informative, but it should not be
the headline physical regime until charger power and reserve policy are
confirmed.

Other unresolved assumptions:

- all buses start full;
- buses may finish near reserve, which can consume free initial energy or defer
  replenishment beyond the horizon;
- a $5 charge-start term may be a material cost rather than a tie-break;
- no shared station-time charger-capacity constraint is enforced;
- station copy counts and uniform charger power are assumptions;
- terminal energy treatment is not yet common across all comparison tiers.

---

## 4. What was achieved scientifically

### 4.1 Heuristic-DP diagnosis

The old heuristic pricing DP was not merely slow. It was structurally poor at
finding long, complementary routes:

- queue ordering could prioritize shallow labels;
- equal-cost dominance could discard a trip-incidence history needed later;
- long known duties could sit behind tens of thousands of labels;
- useful columns often entered in complementary waves across different dual
  re-solves and policies;
- a negative reduced-cost column could enter at zero LP step because another
  optimal dual replaced the current dual.

The GIRO-column audit established that repaired current-model realizations of
all known 10- and 15-duty routes are feasible, and that some were strongly
negative at reconstructed duals even though the heuristic DP did not find
them. This directly identified pricing search, rather than mere feasibility,
as a failure mode of the heuristic label search.

This is why longer heuristic runs were a poor long-term answer. The project
correctly moved to an exact expanded-network pricer.

### 4.2 Expanded-network exact pricer

`src/exact_pricer_expanded.py` represents pricing as a shortest path over an
SOC-by-time directed acyclic graph.

Within its stated discretized model it provides:

- explicit SOC state;
- explicit time blocks;
- station/wait/charge choices;
- delayed-start charging;
- no heuristic label cap;
- no heuristic dominance pruning;
- exact reduced-cost pricing for the named graph;
- status, journal, snapshots, iteration traces, and resume support.

A nonnegative minimum reduced cost is meaningful for that expanded space. It
does not certify continuous SOC/time, omitted station capacity, or a different
master objective.

### 4.3 Small-instance results

Historically reported results include:

- 20 two-duty instances pricing-certified on their tested grids;
- at 5-kWh SOC resolution, 17/20 pair LPs reached route weight 2;
- finer grids closed two additional pairs, leaving one grid-sensitive case;
- selected k8, k13, and k15 unions reached route weights 8, 13, and 15 with
  zero artificials and pricing certificates under several tariffs.

These are strong algorithm checks. They are not yet consolidated into the
current scale-ladder evidence package. Some live in releases, some in old
Unicorn directories, and some in historical summaries. Missing fields must
remain missing unless the source artifacts are recovered or the runs are
repeated.

### 4.4 Large-instance evidence

Historical exact-CG runs reached combined-cost LP route weights roughly in the
target neighborhood at k30 and around 39.x for some k40 pools, but many were
wall-limited and not pricing-certified.

Finite augmented pools produced validated k40 schedules with 39 or 40 buses.
The correction is essential:

- those pools included externally generated/re-realized GIRO or MATCHING
  routes;
- they verify finite-pool feasibility and MIP plumbing;
- they do not show that RAW CG discovered the partition;
- they do not prove a global 39-bus optimum.

RAW k40 finite-pool MIPs were much worse. In the 30-minute physical smoke:

- R1-CS found 507 buses with a fleet bound of 42;
- R2-CS found 523 buses with a fleet bound of 42;
- the two CA pools were finite-pool infeasible;
- all pools contained roughly 42k--48k columns.

These results do not mean the physical fleet needs hundreds of buses. They say
the generated finite pools have poor integral composition and/or that the MIP
search is extremely weak. This is exactly the distinction the smaller-scale
ladder was supposed to locate.

### 4.5 Physical realization repair

The grid DP floors SOC when mapping to lattice state. The old extractor later
reported the entire grid charging gain, while continuous replay retained the
discarded residual. Repetition could exceed the battery capacity.

The four audited RAW k40 pools showed:

| Pool | Columns | Valid as recorded | Deterministically repairable | Infeasible |
|---|---:|---:|---:|---:|
| R1-CA | 42,237 | 621 | 41,616 | 0 |
| R1-CS | 47,687 | 1,956 | 45,731 | 0 |
| R2-CA | 47,307 | 673 | 46,634 | 0 |
| R2-CS | 46,367 | 1,930 | 44,437 | 0 |

The repaired path now stores compact tariff-bound charging blocks and
validates block sums, timing, overlap, power, tariff identity, aggregate stops,
realized cost, and master cost.

Interpretation:

- the expanded-grid certificate remains a certificate for the conservative
  grid-cost model;
- continuous realized costs may be lower;
- those lower continuous costs were not exact-priced and are not certified;
- old selected schedules require the repaired physical replay before use.

### 4.6 Performance improvement

Deterministic local pricing microbenchmarks reported:

- pair case: median 0.05549 s to 0.04353 s, 21.6% reduction;
- synthetic k8: median 0.11293 s to 0.08833 s, 21.8% reduction;
- route hashes, ordering, charging realizations, and reduced costs unchanged.

These are pricing-kernel microbenchmarks, not end-to-end CG scaling evidence.

---

## 5. Critical scientific corrections discovered this week

### 5.1 Route weight is not yet a fleet lower bound

The restricted master uses each route's full cost: bus fixed cost plus charging
terms. `route_weight` is a reported sum of route variables, not the optimized
objective.

Therefore a certificate from the current exact pricer proves optimality of the
combined-cost master in the named route space. It does not prove that the
reported route weight is the minimum fractional fleet.

Required correction for Goal 1:

1. eliminate artificials and certify zero artificial mass;
2. minimize route weight with coefficient exactly one for every real route;
3. fix the certified fleet optimum and then minimize charging cost.

Each phase needs its own fixed optimum, duals, reduced-cost definition, and
certificate. Until then, plots must say `combined-cost-master route weight`,
not `fleet LP lower bound`.

### 5.2 Literal GIRO routes are not all in the primary grid

The frozen membership audit establishes:

- 22/22 selected known partitions are continuously feasible;
- 0/22 complete known partitions fit the primary 15-kWh/10-minute route space;
- only 9/40 duties in the k40 comparator fit the primary grid;
- adaptive grid tests were intentionally limited to k2/k3/k5.

This does **not** prove the target fleet is impossible on the primary grid.
A different target-size partition may exist. It means only that literal
rediscovery of all GIRO duty sequences is not a valid universal test of the
15/10 pricer.

RAW and KNOWN-PARTITION MIPs answer different questions:

- **RAW:** can the generated pool be assembled into an integer partition?
- **KNOWN:** if the externally validated comparator is injected, does the MIP
  accept and use it?

KNOWN is a positive-control/plumbing test, not algorithmic recovery.

### 5.3 Duty 13411

Duty `13411` is continuously feasible but not representable on any of five
tested grids:

| Grid | Failed ordered-trip transition |
|---|---|
| 15 kWh / 10 min | 106 to 119 |
| 5 kWh / 10 min | 106 to 119 |
| 2.5 kWh / 10 min | 119 to 132 |
| 1 kWh / 10 min | 119 to 132 |
| 1 kWh / 5 min | 158 to 167 |

The post-hoc transition oracle did not establish a unique cause. Local
counterfactuals show interactions among block timing, SOC flooring, and
upstream path/state choices. All five causal labels correctly remain
`unresolved`.

Independent review reproduced all 65 parent-versus-head optimizer cases and
the artifact arithmetic. The safe conclusion is: no missing modeled adjacency
or duty-identity defect was detected within the trusted reference graph, and
the five named grids fail. This does not independently validate raw GIRO
geography, nor prove that every finer discrete representation must fail.

An event-based or continuous-SOC fixed-duty model is a justified next design
direction, not a theorem of necessity.

---

## 6. The controlled scale ladder that should have run

The frozen ladder contains 22 cells:

- three deterministic selections each at k2, k3, k5, k8, k13, k20, and k30;
- one frozen k40 selection with two primary-CG replicates;
- no newly submitted k40 MIP.

### 6.1 Task matrix

| Group | Tasks | Purpose |
|---|---:|---|
| `PREFLIGHT` | 22 | known-route membership and grid diagnostics |
| `SEED` | 21 | prepare known partitions for non-k40 positive controls |
| primary `CG` | 23 | exact-CG trajectories; two k40 replicates |
| `CG_SENSITIVITY` | 30 | k2/k3/k5 finer-grid diagnostics |
| `MIP_RAW` | 21 | integer assembly from generated pool only |
| `MIP_KNOWN` | 21 | generated pool plus validated known partition |
| **Total** | **138** | excludes two probes and activation controller |

### 6.2 Budgets

Exact-CG:

| Scale | Budget |
|---|---:|
| k2, k3, k5 | 2 h |
| k8, k13 | 6 h |
| k20 | 12 h |
| k30, k40 | 24 h |

Finite-pool MIP:

| Scale | Budget |
|---|---:|
| k2, k3, k5, k8 | 30 min |
| k13 | 1 h |
| k20 | 2 h |
| k30 | 4 h |
| k40 | reuse only; no new task |

CG snapshots are requested at 5, 15, 30, 60, 120, 240, 480, 720, and
1,440 minutes when the mark lies inside the budget.

### 6.3 Required outputs

The normalizer is designed to produce:

- `cg_iteration_long.csv`;
- `cg_run_summary.csv`;
- `mip_checkpoint_long.csv`;
- `mip_run_summary.csv`;
- `artifact_inventory.csv`;
- `scale_progress_summary.csv`;
- `known_route_membership_long.csv`;
- LP convergence figures;
- MIP incumbent/bound figures;
- separately labeled grid-sensitivity figures.

These are schemas and code paths, not completed evidence for the current
ladder.

---

## 7. Why the smaller-scale campaign stalled

The failure was managerial and operational more than computational.

### 7.1 Infrastructure displaced the requested experiment

The immediate request was to run a comparable k2--k30 study. The work expanded
into:

- evidence collectors and archive builders;
- MIP callback/checkpoint packaging;
- exact-pool profilers;
- physical realization audits;
- tariff-response packaging;
- immutable plans and reservations;
- two-partition environment probes;
- held activation controllers and scientific gates;
- restart reconciliation and scheduler receipts;
- multiple adversarial review rounds.

Most of that work has value. It did not deliver the requested data.

### 7.2 “Prepared” was repeatedly confused with “ran”

Messages such as “packaged,” “pushed,” “tests passed,” “campaign armed,” and
“infrastructure ready” were treated as research progress. None means that a CG
worker produced iteration 1.

The status vocabulary should have been:

1. code prepared;
2. infrastructure submitted;
3. scientific arrays submitted;
4. scientific tasks running;
5. outputs validated;
6. normalized evidence published.

We repeatedly stopped at states 1 or 2.

### 7.3 One controller blocked all 138 tasks

The campaign became all-or-nothing. A probe import failure, a CPU identity
difference, a Slurm representation mismatch, or a delayed state observation
prevented every scientific task from running.

A more resilient design would have run one real k2 worker first, then launched
the primary CG array, while scheduler hardening continued separately.

### 7.4 Failure atomicity dominated liveness

The infrastructure deliberately failed closed using:

- detached clean checkouts;
- exact executable and input hashes;
- no-clobber reservations;
- held probes;
- held activation;
- held scientific gate;
- array identity validation;
- persistent submission intents;
- exact restart state machines.

This reduced duplicate/corruption risk, but each extra state added a
cluster-specific failure mode. The safety goal was legitimate; the critical
mistake was allowing it to block all scientific execution instead of running a
small, clearly labeled diagnostic path in parallel.

### 7.5 Unit tests modeled assumptions, not Unicorn

Hundreds of tests passed while live launches failed on:

- isolated Python module resolution;
- heterogeneous CPU SIMD detection;
- `scontrol` array controller/task formatting;
- whole-array and split-record dependencies;
- scheduler visibility delay;
- `scontrol release` return code versus observed state;
- `sacct`/`squeue` timing and stale reads.

Mocks proved the expected contract, not Unicorn's exact behavior.

### 7.6 Known-route nonrepresentability was overused as a blocker

The primary grid cannot express every known GIRO route. That is an important
interpretation flag. It does not prevent RAW primary-grid CG from being run and
measured. The correct response was RAW plus separately labeled sensitivity,
not no run at all.

### 7.7 User-facing shell interfaces were fragile

Long pasted blocks, placeholders, `set -euo pipefail`, copied prompts/output,
and multiple worktrees made operational mistakes likely. Several pasted blocks
terminated the SSH session or parsed output as commands.

The user explicitly requested at most two, preferably one, paste blocks. The
proper interface is a short committed operator script plus one guarded command.

### 7.8 Branch fragmentation increased deployment cost

The Mac's local checkout is still:

- branch `peel-and-price`;
- commit `b50d648140fed52287a03b7e731d1befef8bfe0e`;
- aligned with `origin/peel-and-price`;
- contains unrelated untracked single-duty files that must be preserved.

Recent work exists on multiple unmerged branches. Repeated detached Unicorn
worktrees were necessary because the current research branch was never
integrated. That made every cluster launch a clone/fetch/hash exercise.

### 7.9 No hard operational success criterion was enforced

The correct checkpoint should have been:

> Within ten minutes of launch, show six scientific array IDs totaling 138
> tasks, and show at least one k2 worker producing a valid output row.

Instead, another layer of scheduler code was often added before that criterion
was met.

---

## 8. Failure chronology

### 8.1 Initial ladder packaging

- `9060dfa`: initial 86-task campaign.
- `a0a66b`: membership and adaptive k2/k3/k5 sensitivity diagnostics.
- `95bcaf5`: k2 1-kWh/5-minute fallbacks; total reached 138 tasks.
- `f51733b`: portable environment probes and stricter bindings.

The first ladder attempt produced failing preflight/seed jobs and downstream
`DependencyNeverSatisfied` states due an environment mismatch. A subsequent
attempt collided with prior no-clobber reservations. No CG evidence resulted.

### 8.2 Probe/import and CPU-identity failures

- `a27eed6`: recoverable probe gating.
- probes `217109`/`217110` failed because isolated execution could not import
  `tariff_response_environment`.
- a later campaign used gate/arrays `218102`--`218108` but remained at zero
  runtime.
- probe attempts `218196`/`218197` failed because portable identity included
  host-detected NumPy SIMD features, which legitimately differ across
  heterogeneous Unicorn CPUs.
- `460dc9f` later separated runtime SIMD from portable software identity.

No scientific worker ran.

### 8.3 Slurm representation fixes

- `894c8b1`: stalled-campaign replacement logic rejected a live array identity.
- `08d23d1`: fixed whole-array dependency validation, then encountered a split
  controller/task representation without one unique controller record.
- `7937c22`: accepted split Slurm array controller records.

These were scheduler parsing fixes, not algorithm results.

### 8.4 Current `7937c22` campaign

Frozen campaign:

- checkout: `/home/nc437/EVSP-DR-scale-ladder-7937c22fef77`;
- campaign: `slad_flat_primary_v4_7937c22`;
- scientific commit:
  `7937c22fef7771e2f74dd03569ea852cbd805e1c`;
- probes: `250111`, `250112`;
- activation: `250113`;
- current held gate: `250838`.

The top-level launcher initially treated a zero return code from
`scontrol release` as proof that probes/activation were released. `squeue`
still showed them `JobHeldUser`. An identity-checked manual release was needed.

`1d80402d79d1cbb4b786b780f7287c12b02d3621` fixes future launchers by
requiring an observed scheduler postcondition and documenting the incident in
`SCALE_LADDER_RELEASE_INCIDENT_20260819.md`.

After manual release, both probes passed, the activation failed after 13
seconds, and held gate `250838` remained. The existing campaign is frozen to
`7937c22`; do not silently switch its scientific code to `1d80402`.

---

## 9. Exact branch map

### Local working branch

- `peel-and-price` at `b50d648`.
- This is old relative to the last week's feature work.
- It has unrelated untracked files under the single-duty audit; preserve them.

### Important remote branches

| Branch | Tip | Role |
|---|---|---|
| `cursor/exact-scale-ladder-2969` | `f51733b` | 138-task ladder and probe packaging |
| `codex/scale-ladder-probe-recovery-20260818` | `1d80402` | latest observed-release fix and incident note |
| `cursor/slurm-state-contract-audit-corrected-2969` | `79b21dd` | corrected scheduler audit and post-ladder science plan |
| `cursor/duty-grid-transition-audit-2969` | `86b0a42` | latest descendant; membership v2 and duty-13411 oracle |
| `cursor/expanded-soc-realization-audit-2969` | `636dc09` | physical realization and RAW/GIRO campaign work |
| `cursor/tariff-response-experiment-2969` | `77baf66` | tariff-response experiment packaging |
| `cursor/cross-generation-evidence-refresh-2969` | `390a4dc` | normalized evidence collection pipeline |

`86b0a42` is a descendant of `79b21dd`, `1d80402`, and `f51733b`, but it has
not been merged into `peel-and-price`. Do not merge the branch stack blindly;
first choose one integration branch and inspect the full range against
`peel-and-price`.

### Independent review of `86b0a42`

The dedicated seven oracle tests, both artifact validators, 65
parent-versus-head production comparisons, and independent arithmetic checks
passed. No numerical/provenance blocker or high-severity issue was found.

Caveats:

- artifacts bind source/input closure, but do not record full Python/NumPy/
  SciPy/HiGHS environment identity;
- “no graph defect” must be read narrowly as no missing modeled adjacency or
  duty-identity inconsistency within the trusted reference graph;
- counterfactuals are local optimistic diagnostics, not end-to-end schedules;
- event/continuous SOC is a justified direction, not the only possible model;
- no-floor replay preserves requested gains subject to capacity clipping.

Cursor reported 499 tests and 123 subtests on its environment. An independent
full run on this Mac produced 490 passed, 1 skipped, 123 subtests, and 8
failures. The failures were in macOS atomic no-replace/path normalization and
legacy assertion-message behavior, not in the seven oracle tests. Therefore
the statement “full repository suite is green everywhere” is false; the
diagnostic branch itself is still supported by its focused validation.

---

## 10. Immediate recovery of the current campaign

Do **not** start another campaign first. The existing frozen plan, inputs,
reservations, probes, and held gate can be recovered using the public
`7937c22` reconciler. It was designed for accepted-before-record gate/array
states.

Because `7937c22` contains the old release false-success bug, recovery must
also re-observe the exact gate and retry `scontrol release` only while that
exact gate remains `PENDING/JobHeldUser`.

The block below:

1. validates the detached checkout, commit, plan, manifest, and 138-task count;
2. prints the activation traceback before mutation;
3. invokes `--resume-missing-arrays --release-held-gate`;
4. requires all six array IDs to be durably recorded;
5. verifies gate ID, name, user, partition, comment, and work directory;
6. releases only the exact held gate;
7. prints the real task matrix and scientific arrays.

```bash
bash <<'BASH'
main() {
  set +e
  set +u
  set +o pipefail

  RUN_ROOT="$HOME/EVSP-DR-scale-ladder-7937c22fef77"
  CAMPAIGN="slad_flat_primary_v4_7937c22"
  CAMPAIGN_ROOT="$RUN_ROOT/src/results/scale_ladder/$CAMPAIGN"
  PLAN="$CAMPAIGN_ROOT/approved-plan.json"
  MANIFEST="$CAMPAIGN_ROOT/campaign.json"
  COMMIT="7937c22fef7771e2f74dd03569ea852cbd805e1c"
  EXPECTED_GATE="250838"

  echo "=== Exact campaign preflight ==="
  [[ -d "$RUN_ROOT/.git" && -s "$PLAN" && -s "$MANIFEST" ]] || {
    echo "REFUSING: exact checkout, plan, or manifest is missing."
    return 1
  }
  [[ "$(git -C "$RUN_ROOT" rev-parse HEAD 2>/dev/null)" == "$COMMIT" ]] || {
    echo "REFUSING: checkout is not exact commit $COMMIT."
    return 1
  }
  if git -C "$RUN_ROOT" symbolic-ref -q HEAD >/dev/null 2>&1; then
    echo "REFUSING: checkout is not detached."
    return 1
  fi
  [[ -z "$(git -C "$RUN_ROOT" status --porcelain --untracked-files=no)" ]] || {
    echo "REFUSING: tracked checkout is dirty."
    return 1
  }

  PLAN_SHA=$(sha256sum "$PLAN" | awk '{print $1}')
  [[ "$PLAN_SHA" =~ ^[0-9a-f]{64}$ ]] || {
    echo "REFUSING: could not hash plan."
    return 1
  }
  [[ "$(jq -r '.approval_sha256 // empty' "$MANIFEST")" == "$PLAN_SHA" ]] || {
    echo "REFUSING: manifest approval does not match plan bytes."
    return 1
  }
  [[ "$(jq -r '.checkout_identity.commit // empty' "$PLAN")" == "$COMMIT" ]] || {
    echo "REFUSING: plan commit mismatch."
    return 1
  }
  [[ "$(jq -r '.campaign_root // empty' "$PLAN")" == "$CAMPAIGN_ROOT" ]] || {
    echo "REFUSING: plan campaign root mismatch."
    return 1
  }
  [[ "$(jq '[.task_groups[] | length] | add' "$PLAN")" == "138" ]] || {
    echo "REFUSING: plan does not contain exactly 138 tasks."
    return 1
  }
  BEFORE_GATE=$(jq -r '.gate_job_id // empty' "$MANIFEST")
  [[ -z "$BEFORE_GATE" || "$BEFORE_GATE" == "$EXPECTED_GATE" ]] || {
    echo "REFUSING: manifest names unexpected gate $BEFORE_GATE."
    return 1
  }
  PYTHON=$(jq -r '.python.path // empty' "$PLAN")
  [[ "$PYTHON" == /* && -x "$PYTHON" ]] || {
    echo "REFUSING: approved Python is unavailable."
    return 1
  }

  echo
  echo "=== Activation 250113 logs (prior failure) ==="
  for file in \
    "$CAMPAIGN_ROOT/logs/activation_a1_250113.out" \
    "$CAMPAIGN_ROOT/logs/activation_a1_250113.err"
  do
    if [[ -f "$file" ]]; then
      echo "----- $file -----"
      tail -n 120 "$file"
    else
      echo "MISSING: $file"
    fi
  done

  echo
  echo "=== Manifest before recovery ==="
  jq '{submission_state,probe_state,gate_state,gate_job_id,gate_submission_intent,submitted_arrays,array_submission_intents}' "$MANIFEST"

  echo
  echo "=== Submit six real scientific arrays and request gate release ==="
  env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH \
    PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
    "$PYTHON" -I -B "$RUN_ROOT/src/run_reviewed_python.py" \
    "$COMMIT" reconcile_scale_ladder_gate.py \
    --campaign-root "$CAMPAIGN_ROOT" \
    --approved-plan-sha256 "$PLAN_SHA" \
    --resume-missing-arrays --release-held-gate
  RC=$?
  if [[ "$RC" -ne 0 ]]; then
    echo "RECOVERY FAILED SAFELY (rc=$RC); no replacement campaign was created."
    jq '{submission_state,probe_state,gate_state,gate_job_id,submitted_arrays,array_submission_intents,release_error}' "$MANIFEST"
    return "$RC"
  fi

  [[ "$(jq '.submitted_arrays | length' "$MANIFEST")" == "6" ]] || {
    echo "REFUSING TO RELEASE: six scientific arrays were not recorded."
    jq '{gate_state,gate_job_id,submitted_arrays,array_submission_intents}' "$MANIFEST"
    return 1
  }
  jq -e '(.submitted_arrays|keys|sort)==(["CG","CG_SENSITIVITY","MIP_KNOWN","MIP_RAW","PREFLIGHT","SEED"]|sort) and (.submitted_arrays|all(.[];(tostring|test("^[0-9]+$"))))' "$MANIFEST" >/dev/null || {
    echo "REFUSING TO RELEASE: scientific array identities are incomplete."
    return 1
  }

  GATE=$(jq -r '.gate_job_id // empty' "$MANIFEST")
  [[ "$GATE" == "$EXPECTED_GATE" ]] || {
    echo "REFUSING TO RELEASE: reconciled gate is $GATE, expected $EXPECTED_GATE."
    return 1
  }
  EXPECTED_NAME="LDG${PLAN_SHA:0:5}"
  EXPECTED_COMMENT="SLADG:${PLAN_SHA:0:20}"

  echo
  echo "=== Verify and, only if still user-held, release exact gate $GATE ==="
  for ATTEMPT in 1 2 3; do
    RECORD=$(scontrol show job -o "$GATE" 2>/dev/null)
    [[ -n "$RECORD" ]] || break
    [[ "$RECORD" == *"JobId=$GATE "* && \
       "$RECORD" == *"JobName=$EXPECTED_NAME "* && \
       "$RECORD" == *"UserId=$USER("* && \
       "$RECORD" == *"Partition=default_partition "* && \
       "$RECORD" == *"Comment=$EXPECTED_COMMENT "* && \
       "$RECORD" == *"WorkDir=$RUN_ROOT "* ]] || {
      echo "REFUSING: exact gate fingerprint mismatch."
      return 1
    }
    STATE=$(printf '%s\n' "$RECORD" | tr ' ' '\n' | sed -n 's/^JobState=//p' | head -n 1)
    REASON=$(printf '%s\n' "$RECORD" | tr ' ' '\n' | sed -n 's/^Reason=//p' | head -n 1)
    echo "gate attempt $ATTEMPT: state=$STATE reason=$REASON"
    if [[ "$STATE" == "PENDING" && "$REASON" == "JobHeldUser" ]]; then
      scontrol release "$GATE" || {
        echo "Gate release failed; arrays remain protected."
        return 1
      }
      sleep 2
    else
      break
    fi
  done

  RECORD=$(scontrol show job -o "$GATE" 2>/dev/null)
  if [[ "$RECORD" == *"JobState=PENDING "* && \
        "$RECORD" == *"Reason=JobHeldUser "* ]]; then
    echo "Gate is still held; arrays remain safe but will not run."
    return 1
  fi

  echo
  echo "=== Scientific task matrix ==="
  jq -r '.task_groups|to_entries[]|[.key,(.value|length)]|@tsv' "$PLAN" | sort
  echo -e "TOTAL\t$(jq '[.task_groups[]|length]|add' "$PLAN")"

  echo
  echo "=== Recorded scientific arrays ==="
  jq -r '.submitted_arrays|to_entries[]|[.key,.value]|@tsv' "$MANIFEST" | sort
  ARRAY_IDS=$(jq -r '.submitted_arrays|to_entries|map(.value|tostring)|join(",")' "$MANIFEST")

  echo
  echo "=== Live scientific jobs ==="
  squeue -j "$ARRAY_IDS" -o '%.14i %.18j %.2t %.10M %R' 2>/dev/null || true

  echo
  echo "SCALE_LADDER_SCIENCE_SUBMITTED=true"
  echo "PLAN_SHA256=$PLAN_SHA"
  echo "CAMPAIGN_ROOT=$CAMPAIGN_ROOT"
}
main
BASH
```

If this stops, retain the printed traceback and manifest. Do not cancel
`250838` or create a fresh campaign before interpreting that exact output.

The recovered outputs should initially be labeled
`legacy_scheduler_unverified`, because the `7937c22` campaign predates the
prospective release-receipt schema. The scientific outputs remain usable after
an external, no-clobber raw `scontrol`/`sacct` sidecar audit.

---

## 11. What to do after the arrays really run

### Step 1: monitor scientific work, not infrastructure

Success means:

- six array parent IDs exist;
- their task counts sum to 138;
- at least one k2 worker writes a valid output and completion record;
- CG iteration/journal files begin accumulating;
- MIP tasks remain dependency-held until their corresponding CG tasks finish.

Do not call the campaign started merely because a probe or gate exists.

### Step 2: capture scheduler evidence

After completion, preserve raw per-task `scontrol -o` and
`sacct -X --array -P` records, not just parent summaries. Build the external,
checksummed legacy audit sidecar without rewriting the frozen campaign.

If complete raw scheduler evidence is unavailable, retain the label
`legacy_scheduler_unverified`. Do not discard numerical outputs solely because
of that provenance label.

### Step 3: normalize immediately

Run the existing scale-ladder normalizer and require the expected CSVs plus an
artifact inventory. Missing cells must be explicit `missing/censored` rows.

Check, per cell:

- time to zero artificials;
- combined-cost route weight by iteration/time;
- reduced cost and certification status;
- columns added and total columns;
- master time, pricing time, cumulative time, and total wall time;
- stop reason and censor reason;
- maximum memory/resource metrics if present;
- RAW MIP first incumbent time, incumbent fleet, bound, gap, nodes, and stop;
- KNOWN arm separately, including validated start and augmented-pool identity.

### Step 4: classify, do not average blindly

Each cell belongs in one of these categories:

1. certified combined-cost master with zero artificials;
2. feasible but pricing-censored;
3. incomplete/censored;
4. named-route/known-partition nonrepresentability flag;
5. finite-pool MIP target feasible/infeasible/censored.

Known-route membership is an orthogonal flag, not a substitute for the CG
classification.

### Step 5: rerun only the decision frontier

Do not automatically repeat all 138 tasks. Resume only cells that answer a
specific unresolved question, such as:

- first scale where artificials fail to disappear;
- first scale where pricing becomes censored;
- first scale where RAW finite-pool target feasibility fails;
- one adjacent larger scale to establish a trend.

---

## 12. Next algorithmic work after collection

### 12.1 Lexicographic fleet-only exact CG

This is mandatory before presenting fleet LP lower bounds.

Implement and validate three phases:

1. artificial elimination;
2. fleet-only route-weight minimization;
3. charging-cost minimization at fixed fleet optimum.

Start with k2/k3/k5. Compare against the current combined-cost trajectories.
Do not launch the full ladder first.

### 12.2 Target-constrained finite-pool feasibility MIP

For a fixed RAW pool, solve exact partition equalities with
`sum(x_r) <= target` and a constant objective. This distinguishes:

- pool provably lacks a target partition;
- pool may contain one but MIP search is censored.

This is more informative than extending a weak two-stage MIP for many hours.

### 12.3 Route-space representation

The 15/10 primary grid excludes most known duties, and duty 13411 still fails
at 1 kWh/5 minutes. Do not respond by indefinitely adding finer uniform grids.

Test an event-based or continuous-SOC fixed-duty model first. If successful,
decide whether full pricing should use:

- continuous/event SOC;
- adaptive breakpoints;
- or another separately reviewed representation.

### 12.4 Matched heuristic versus exact benchmark

Use identical frozen instances, initial pools, objective phases, tariff,
physics, and budgets. Preserve iteration-level trajectories. Compare:

- old heuristic DP;
- repaired/portfolio heuristic DP;
- exact expanded pricer;
- future lexicographic exact pricer.

Do not compare endpoints from incompatible historical runs as though they are
one experiment.

---

## 13. Demand-response experiment after Goal 1 engineering

Before launching tariffs, freeze:

1. terminal SOC or terminal-energy valuation;
2. whether the $5 charge-start term is real;
3. battery capacity and charger power;
4. reserve policy;
5. treatment of simultaneous charger demand/capacity;
6. legitimate weekday duty sets;
7. common continuous replay for all comparison tiers.

Then run:

1. GIRO observed/cross-priced;
2. fixed GIRO sequence, optimized charging;
3. joint routes and charging;
4. cross-price every resulting schedule under every tariff;
5. report cost, kWh, peak-window kWh, deadhead, charge visits, initial/terminal
   SOC, fleet, and charger concurrency separately.

The fixed-sequence experiment is valuable even if RAW routing recovery remains
difficult. The joint result then measures incremental routing value.

The current broad tariff pilot remains blocked by known-route
nonrepresentability and unresolved physics/economic policy. Synthetic examples
are software demonstrations, not research results.

---

## 14. Integration and cleanup

Do not merge directly into `peel-and-price` while its untracked user files are
present and the branch stack is unreviewed.

Recommended integration procedure:

1. preserve the untracked single-duty work separately;
2. create one integration branch from `peel-and-price`;
3. inspect the complete diff to `86b0a42`;
4. identify which commits are operational-only, diagnostic-only, and model
   changes;
5. resolve duplicate/superseded campaign launchers;
6. keep one canonical scale-ladder entrypoint;
7. run focused numerical tests and the Linux/Unicorn suite;
8. fix or explicitly quarantine the eight current macOS full-suite failures;
9. merge only after the recovered campaign artifacts are preserved.

Repository cleanup must not delete old Unicorn worktrees or result directories
until their manifests, journals, and logs are archived and hashed.

---

## 15. Evidence ledger

### Verified directly in this handoff pass

- current remote branch tips after `git fetch origin --prune`;
- local branch/path/commit and untracked-file state;
- the exact 138-task counts and budgets in `7937c22` source;
- the current cluster job table supplied by the user;
- ancestry: `86b0a42` descends from `79b21dd`, `1d80402`, and `f51733b`;
- dedicated duty-oracle tests/validators and 65-case production parity;
- independent arithmetic of duty-13411 evidence;
- Mac full-suite outcome: 490 passed, 8 failed, 1 skipped, 123 subtests.

### Historical/operator-reported, retained with qualification

- pair and small-union cluster results;
- k30/k40 historical LP endpoints;
- 39/40-bus augmented-pool schedules;
- four-job RAW/GIRO overnight outcomes not yet ingested into one current
  evidence package;
- fragmented releases and Unicorn-only artifacts.

### Planned, not completed

- controlled k2--k30 scale-ladder outputs;
- fleet-only lexicographic CG;
- target-constrained finite-pool feasibility oracle;
- real tariff-response evidence matrix;
- Frölunda held-out replication;
- canonical integration into `peel-and-price`.

---

## 16. Files the next owner should read, in order

1. `CURRENT_RESEARCH_PLAN_20260810.md`  
   Corrects earlier global-optimality and savings overclaims.
2. `SCALE_LADDER_20260818.md` at `1d80402`  
   Defines the 138-task campaign and recovery/normalization workflow.
3. `SCALE_LADDER_RELEASE_INCIDENT_20260819.md` at `1d80402`  
   Documents the false-success release bug.
4. `POST_LADDER_RESEARCH_DECISION_PLAN_20260819.md` at `79b21dd`  
   Gives correct scientific interpretations and next oracles.
5. `data/scale_ladder/known_membership_preflight.json`  
   V1 route-space evidence bound to the running campaign.
6. `analysis/scale_ladder_membership_v2_20260819/README.md` at `86b0a42`  
   More detailed, post-hoc diagnostic evidence; it does not alter V1.
7. `analysis/duty_13411_grid_transition_oracle_20260819/README.md` at
   `86b0a42`  
   Grid-transition diagnostic and scope limitations.
8. `analysis/rawk40_physical_audit_20260817/ROOT_CAUSE.md`  
   SOC realization defect.
9. `GIRO_COLUMN_AUDIT.md`  
   Heuristic-pricing diagnosis and known-column evidence.
10. `EXACT_CG_PERFORMANCE_AUDIT_20260814.md`  
    Telemetry/profiler and 21--22% pricing microbenchmark.

---

## 17. Definition of done for the next owner

Do not report “the ladder is ready” again. Report completion only when all of
the following are true:

1. six scientific arrays are durably identified;
2. their task counts total 138;
3. real CG/MIP workers ran, not only probes/controllers;
4. worker completions and artifact hashes validate;
5. iteration/checkpoint trajectories exist;
6. the normalizer publishes the required CSVs and figures;
7. missing/censored cells remain explicit;
8. RAW and KNOWN evidence remain separate;
9. route weight is labeled as combined-cost unless the lexicographic phase is
   implemented;
10. a concise coauthor table can be regenerated from the frozen artifacts.

The honest final state is: **the algorithmic and diagnostic code base is much
stronger, but the simple, controlled smaller-scale evidence requested by the
user remains uncollected. The immediate task is to recover and run the existing
frozen 7937 campaign—not to build another launcher.**
