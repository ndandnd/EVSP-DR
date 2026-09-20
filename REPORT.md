# Bounded integer-directed column generation — diving-with-pricing pilot

**Worktree** `.codex-work/diving-pricing-20260919`, base `6830caa2`.
Status: **code complete, locally tested end-to-end, not run on the benchmark.**
No job submitted, no `Docs/`/`Slides` touched, no other worktree touched,
nothing committed.

---

## 1. Question and mechanism

Can integer-complementary routes be generated **without** a sequential/GIRO
witness?

The witness study
(`outputs/independent_review_20260916/advisor_witness_columns_20260917/README.md`)
was read for source identities only. Its finding sets the target: the routes
missing from a fresh k=8 pool carry reduced costs up to **53** under that
pool's own certified duals, while the pool already holds 381–670 columns at
|rc| ≤ 1e-4. They are LP-*suboptimal*, so no reduced-cost enrichment of a
fresh CG run will ever produce them. Only a change in the duals will — and
fixing part of the solution is what changes them.

The pilot implements exactly that loop:

```
solve the restricted master over the frozen fresh pool (+ fleet row)
  -> fix one promising fractional route  (lb = ub = 1)
  -> reprice with the resulting duals, adding what the pricer returns
  -> repeat; backtrack or restart deterministically on failure
```

**This mechanism is demonstrated, not assumed.** `DiveMechanismTests` builds
a six-trip instance whose frozen pool LP is *certified* (minimum reduced cost
zero over the entire route universe), fractional at route weight 3.000 with
zero artificials — the same signature as the real fresh k=8 pools — and whose
integer optimum in-pool is 4. The two completing routes sit at reduced cost
**+60** under those certified duals. Plain CG cannot reach them. One fixing
moves the duals and the pricer emits them; the dive closes at 3 buses.

---

## 2. Design

### Restricted master (`DiveMaster`)

| | |
|---|---|
| rows | `cover[t]: Σ_{t∈r} x_r + a_t ≥ 1` per trip; `fleet: Σ_r x_r ≤ 8` |
| objective | `min Σ c_r x_r + M Σ a_t`, `M = BIG_M_PENALTY` (production's own artificial penalty, so the pricing objective stays the true combined cost) |
| variables | `x_r ≥ 0`, **no upper bound**; fixings are `lb = ub = 1` |
| determinism | `Threads=1`, `Method=1`, explicit `Seed` |

Feasibility is handled by big-M artificial coverage rather than a separate
Phase I — the same construction the production restricted master uses, which
keeps the pricing objective correct without a second objective mode. Because
the artificials are unbounded above, the LP is feasible at every dive node,
so an "infeasible" verdict is never a solver status but an explicit,
*certified* condition (see §3).

**One bug found and fixed during development.** The first implementation
bounded route variables at `x_r ≤ 1`. That bound is redundant for covering
(clipping any `x_r` to one preserves `Ax ≥ 1`, relaxes the fleet row and
lowers the cost) but it is *not* harmless: a column sitting at an upper bound
may carry a negative reduced cost at optimality, which destroys the pricing
certificate. Observed directly — the fleet dual settled at −499,000 on a
degenerate vertex and pricing could never certify. Removing the bound fixed
it. The reasoning is recorded in the class docstring.

### Pricing (`EventPricer`) — and a latent defect that had to be routed around

Reduced cost is `rc = c_r − Σ_{t∈r} α_t − μ`, with `μ` the fleet-row dual.

The network exposes a `route_dual` argument that looks like the right place
for `μ`. **It is not usable.** In `event_pricer_network.py`
`_min_reduced_cost_route_lazy`, the non-fast branch applies
`arc_costs − BUS_COST_KX` at the source arc for *every* objective except
`artificial-elimination`/`fleet-only` — including `combined-cost` — whereas
the explicit-arc path subtracts `BUS_COST_KX` only for `charging-cost`. So
`objective="combined-cost", route_dual≠0` under `--event-arc-mode lazy` (the
mode every campaign uses) shifts every reduced cost by 1e5. Production never
triggers it because production has no fleet row.

The pilot therefore prices with `α` only, on the verified fast path, and
applies the `μ` shift itself. This is exact, not an approximation: `μ` enters
every column identically, so the argmin is unchanged and the node certificate
is the network's own minimum minus `μ`. `μ ≤ 0` is asserted, not assumed.
`test_network_route_dual_argument_is_unsafe_in_lazy_mode` pins the 1e5
divergence as a regression witness. **This defect is in production code and I
did not change it** — it is latent there, and fixing it is out of scope for
this pilot. It should be reported separately.

### Performance and input binding

Two changes after a review pass, both verified on the 86-trip end-to-end run
(97.3 s → 58.4 s, identical result: 3 nodes, 19,726 columns, 3-bus cover):

* `DiveMaster.solve()` accumulated the row-violation check by scanning every
  column for every trip — O(trips × columns), i.e. 194 × 40,000 per LP on the
  real pools. It now accumulates over the LP *support* only (exact: nonbasic
  columns are exactly zero) and reads `X`/`RC`/`Pi` through batched
  `model.getAttr`. `set_fixed` likewise touches only the columns whose bounds
  actually change instead of sweeping all 40,000 per node.
* The cache identity is checked against the frozen *status* provenance, so a
  wrong `--data-dir` would pass every cache check and still replay routes
  against a different instance. `resolve_options` now re-hashes the instance
  CSV, price CSV, `Ref_dict.csv` and `par_ref_dhd.csv` against
  `provenance.*_sha256` and refuses on mismatch or on a missing recorded
  hash — the binding `prep_witness.py` does. Verified to fire on a truncated
  instance.

### Dive node outcomes

| outcome | condition | action |
|---|---|---|
| `node_integral` | certified, artificials ≈ 0, all `x ∈ {0,1}` | incumbent, stop |
| `node_fractional` | certified, artificials ≈ 0, fractional | branch |
| `node_infeasible` | **certified**, artificials > 1e-6 | backtrack |
| `node_uncertified` | node time / pricing-iteration cap / budget / degenerate stall | branch if coverage complete, else backtrack — **never** called infeasible |

`node_infeasible` means *infeasible under these fixings and this fleet cap*.
The manifest carries `global_certificate: null` unconditionally and every node
record carries `"scope": "dive node only; never a global certificate"`.

### Branching and backtracking

Deterministic ranking by rule; `max_alternatives` (default 3) alternatives are
retained per level. Backtracking tries a **different candidate at the same
level** — it never sets `ub = 0` on a column, which would make the RMP
unable to express the exclusion and leave pricing unable to certify.
Restarts walk a fixed rule sequence (`max_value`,
`max_value_long_route`, `max_value_cheap_seat`), keeping all generated
columns. A dive whose root never branched declines to restart
(`root_not_branchable`) rather than repeat identical work.

### Artifacts

Every generated route is physically replayed through
`run_exact_pool_mip.validate_injected_route` *before* it is written. Output
mirrors `prep_witness.py` exactly: the original journal is copied
**byte-for-byte** and new records appended; `cg.json` is the source status with
only `columns_journal` swapped plus a `diving_augmentation` block. Source
hashes are taken on entry and re-taken on exit; a mismatch is a non-zero exit.

The final pool MIP is **not** reimplemented — the augmented pool is handed to
the trusted `src/run_exact_pool_mip.py`.

**Which MIP commit.** The campaign and the witness study both ran
`run_exact_pool_mip.py` from `871d057e`. That commit has no `--seed`
(`git show 871d057e:src/run_exact_pool_mip.py | grep -c '"--seed"'` → 0;
`--seed` arrives in this worktree's base `6830caa2`). Pointing the `.sub`
files at `871d057e` while passing `--seed` would have killed every cluster
MIP job on argparse — caught in review, not in the local smoke, which used
the worktree's own runner. Both arms therefore run the pool MIP from the
**pinned execution checkout**, which is a deliberate, recorded departure from
replaying the campaign argv verbatim: a seeded control/treatment pair is
otherwise impossible.

---

## 3. Files

| file | role |
|---|---|
| `src/diving_pricing_pilot.py` | the runner: `DiveMaster`, `EventPricer`, `DivePilot`, source guards, publication, CLI |
| `src/diving_cache_identity.py` | identity-verified cache load with an explicit, audited commit bridge |
| `tests/test_diving_pricing_pilot.py` | 49 tests (below) |
| `scripts/research/diving_pricing_20260919/cases_k08.json` | frozen case identities, hashes, cache identities, graph-build seconds |
| `scripts/research/diving_pricing_20260919/make_manifest.py` | manifest generator + validator (submits nothing) |
| `scripts/research/diving_pricing_20260919/control_mip.sub` | control arm |
| `scripts/research/diving_pricing_20260919/treatment_dive.sub` | treatment arm (cache load + dive + MIP in one budget) |
| `scripts/research/diving_pricing_20260919/collect.py` | comparison table + status README |
| `scripts/research/diving_pricing_20260919/PREREGISTRATION.md` | the preregistration |

No existing file was modified.

---

## 4. Tests run

```
cd .codex-work/diving-pricing-20260919
python3 -m unittest tests.test_diving_pricing_pilot
```

**49 tests, all passing** (Gurobi 12.0.1, Python 3.12, macOS).

Neighbouring suites re-run to confirm nothing regressed (no existing file was
modified; `git status` shows only new untracked files):

```
python3 -m unittest tests.test_event_pricer_network tests.test_run_exact_pool_mip \
                    tests.test_event_pricer_gates tests.test_master_lp_gurobi \
                    tests.test_diving_pricing_pilot
# Ran 97 tests in 130.2s — OK (skipped=5, pre-existing)
```

| requirement | tests |
|---|---|
| reduced costs vs direct recomputation | `rc_true == cost − Σα − μ` recomputed from the replayed record, in **both** arc modes and at three `μ` values; argmin invariance under the shift; the lazy-mode `route_dual` 1e5 divergence pinned |
| fixed route contributions | Gurobi `var.RC` equals `c − Σ Pi_row·a − Pi_fleet` for every column, with and without fixings; fixing collapses the covered rows' duals; bounds released on backtrack; over-cap fixing refused; `μ ≤ 0` |
| phase-I infeasibility vs pool shortage | shortage repaired by pricing and *not* labelled infeasible; a genuinely uncoverable node (4 trips, 1 bus) labelled `node_infeasible` with node-only scope; manifest asserts no global certificate |
| no witness input | refuses `witness_augmentation`, `validated_seed_routes_sha256`, `inherited_event_pool_status_sha256`, non-RAW treatment, and warm/witness-looking paths; pattern accepts `c1_k08`/`base`/`cg.json` |
| budget handling | zero budget stops before any node; node limit bounds the search; pricing-iteration cap yields `node_uncertified` |
| source immutability | augmented journal starts with the original bytes verbatim; source status and journal byte-identical after publication |
| physical validation | every generated route passes `validate_injected_route`; cost equals `expanded_grid_cost`; a route with a bogus node is refused |
| synthetic fixture where added columns enable the integer target | the deficient-pool fixture, **and** the `DiveMechanismTests` fixture reproducing the witness study's rc-positive-at-the-certified-root signature |
| determinism | repeated dives produce identical node outcomes and identical generated column order |
| cache identity | exact identity loads; commit difference refused without the flag; physics difference refused *with* it; tampered pickle refused; method audit passes against both real producer commits |
| input binding | truncated instance refused by sha256; missing recorded provenance hash refused; matching inputs accepted and recorded |
| cache-bridge negative cases | a stubbed `_build_arcs` is detected in `differing_methods`; a load whose only identity gap is `git_commit` is refused once a graph-critical method differs; the warm-named cache paths are shown to exist, to still be refused as *column* sources, and to be recorded as an explicit exemption |

**End-to-end, on a real instance** (`Practice_Custom_TwoDuty_13301_13302.csv`,
86 trips, real `EventExpandedNetwork`, real cache write/load, real Gurobi):
the CLI dove to depth 2 in 58 s, generated 19,726 physically-replayed columns,
found a 3-bus integer cover, published the augmented pool, and
`src/run_exact_pool_mip.py --cover --two-stage` then returned
`buses=3, fleet_bound=3.0, fleet_proven=True, physical_replay_validated=True`
on it, against `buses=86` on the unaugmented source. Source hashes unchanged.
This validates the plumbing and the deployment path; it is **not** a
scientific result — the control pool there was singletons-only.

`make_manifest.py`, `collect.py` and both `.sub` files were syntax- and
run-checked locally (the `.sub` files with `bash -n`; the Python tools
executed against a dry work directory).

---

## 5. Graph-build accounting — the budget cannot be 60 minutes

Measured builds for these cases
(`outputs/cumulative_budget_20260913/audit/budgets.csv`,
`target_external_graph_build_s`):

| case | graph build | residual of a 3,600 s budget | 60-min treatment incl. build viable? |
|---|---:|---:|---|
| c1_k08 | 7,279 s (121 min) | −3,679 s | no |
| c3_k08 | 3,295 s (55 min) | 305 s | no |
| c4_k08 | 4,170 s (70 min) | −570 s | no |
| c5_k08 | 2,520 s (42 min) | 1,080 s | no |

A 60-minute treatment that *includes* a fresh graph build is impossible for
all four cases. Reported, not worked around. Two arms are preregistered and
both are to be run:

* **Arm A (primary, shared-prerequisite).** Both arms get 3,600 s of real
  compute. The build is a shared prerequisite — the frozen fresh pool the
  control MIP consumes could not exist without it — so charging it to one arm
  only would be arbitrary. Cache **load** is measured and charged to the
  treatment. Treatment split: dive ≤ 2,400 s (cache load included), per-node
  time cap 150 s, MIP `max(1200, 3600 − elapsed)`. The split is set from
  evidence: the witness study's augmented MIPs proved 8 in 5.5–93 s once the
  columns existed, so the final MIP needs far less than half the budget while
  the dive needs many fully-priced nodes.
* **Arm B (secondary, matched adjusted budget).** Both arms charged
  `3600 + graph_build_s`. The control spends it as real MIP time
  (10,879 / 6,895 / 7,770 / 6,120 s); the treatment is debited the build while
  running its 3,600 s. The build is fully accounted and the comparison is
  conservative against the treatment.

Both arms' per-stage wall times land in `timing.json`. The cache load is never
excluded.

### Cache reuse is identity-verified, and it needs an explicit bridge

The k=8 caches were built at `e091a4db` (c1) and `ecb60c15` (c3/c4/c5). The
production loader compares the entire identity dict including `git_commit`, so
they cannot be loaded at `6830caa2` without a bridge. The bridge requires all
other identity fields equal, the recorded pickle sha256, the recomputed
`network.metrics()` (nodes, arcs, arc mode, packed bytes, event-lattice
sha256) equal to the producer's, **and** byte-identical source for all 19
graph-critical `EventExpandedNetwork` methods. **Verified passing against both
producer commits at `6830caa2`** (`identical True, missing [], differing []`).
This mirrors, programmatically, the manual audit already recorded in
`outputs/cumulative_budget_20260913/cache_compatibility.json`. Default is
refusal; `--cache-commit-bridge` is opt-in and the audit is written into the
run manifest.

⚠️ **If root pins a commit other than `6830caa2`, re-run the audit** — it may
refuse, in which case the case is blocked, not silently rebuilt.

---

## 6. Deployment

Prerequisites on Unicorn: the frozen `cumulative_budget_20260913` case
directories, the four `network.pkl` caches with their manifests, a pinned
detached checkout of this branch under
`/home/nc437/ladder-lite/execution/<commit>/`, the trusted MIP runner at
`871d057e`, and `/home/nc437/evsp_env/bin/python`.

```bash
WORK=/home/nc437/ladder-lite/diving_pricing_20260919
CODE=/home/nc437/ladder-lite/execution/<pinned-commit>

# 1. manifest + validation (submits nothing; must print "validation: passed")
$PY $CODE/scripts/research/diving_pricing_20260919/make_manifest.py \
    --work $WORK --code $CODE

# 2. two control jobs (one per budget arm) + ONE treatment job per case
#    = 8 control + 4 treatment = 12 jobs
for CASE in c1_k08 c3_k08 c4_k08 c5_k08; do
  for ARM in arm_a arm_b; do
    sbatch --partition=default_partition --exclude=scaglione-compute-01 \
           --cpus-per-task=8 --mem=24G --time=4:00:00 \
           --job-name=dp_${CASE}_control_${ARM} \
           --output=$WORK/logs/%x_%j.out --error=$WORK/logs/%x_%j.err \
           $CODE/scripts/research/diving_pricing_20260919/control_mip.sub \
           $WORK $CASE $ARM
  done
  sbatch --partition=default_partition --exclude=scaglione-compute-01 \
         --cpus-per-task=8 --mem=96G --time=2:30:00 \
         --job-name=dp_${CASE}_treatment \
         --output=$WORK/logs/%x_%j.out --error=$WORK/logs/%x_%j.err \
         $CODE/scripts/research/diving_pricing_20260919/treatment_dive.sub \
         $WORK $CASE
done

# 3. collect
$PY $CODE/scripts/research/diving_pricing_20260919/collect.py --work $WORK
```

Local test command (no cluster, no instance data needed):

```bash
cd .codex-work/diving-pricing-20260919 && python3 -m unittest tests.test_diving_pricing_pilot -v
```

The treatment runs **once per case** and is paired against both control rows:
the arms differ only in the control's budget and in whether the graph build is
charged, never in what the treatment does. Running it twice would be four
redundant 96 GB jobs and would contradict "one seed, no repeats".

Arm A control wall requests only need ~1.2 h; Arm B's c1 control needs
10,879 s of MIP time, so `--time=4:00:00` covers the largest case. Treatment
jobs need 3,600 s plus cache-load headroom; 2.5 h is generous. Concurrency is
the default 50 (well above the 12 jobs here); no throttle is imposed. Jobs are
independent — no dependencies.

---

## 7. Remaining limits — read before trusting anything

1. **No benchmark run happened.** The k=8 instances live under
   `data/scale_ladder/`, which is not present in this checkout, and the graph
   caches are multi-GB cluster artifacts. Every number about c1/c3/c4/c5 in
   this report is a *recorded* number from prior campaigns, not a measurement
   from this pilot. Nothing here claims the method works on the benchmark.
2. **The dive is a heuristic.** It produces incumbents and columns. The only
   fleet proofs come from `run_exact_pool_mip.py` on a supplied pool, and
   those are pool-scoped. Nothing in the pilot certifies the full model.
3. **`node_uncertified` nodes are branched on.** When a node's pricing
   allowance runs out with complete coverage, the dive branches anyway. Those
   branch choices rest on non-certified duals. Recorded per node; never used
   to claim infeasibility.
4. **Root convergence is assumed cheap.** The benchmark's roots start from a
   certified fresh pool, so the added fleet row should re-certify in few
   iterations. The local smoke test started from singletons and needed 400
   pricing rounds at the root. If a real root hits
   `node_pricing_iter_limit`, `--max-pricing-iters` needs raising and the
   budget split revisiting.
5. **`--skip-cache-hash` exists** to avoid rehashing a multi-GB pickle. It is
   off by default and the `.sub` files do not pass it. If a cache-load
   measurement forces it on, the manifest records
   `pickle_sha256_verified: false` and the run's identity is weaker.
6. **Arm A's fairness rests on a judgement call** — that the graph build is a
   shared prerequisite. Arm B exists precisely because that judgement is
   contestable. Report both; do not report Arm A alone.
7. **The production `route_dual`/lazy-arc defect is unfixed** (§2). It does
   not affect production today and does not affect this pilot, which routes
   around it. It should be raised as its own issue.
8. **c3_k08's pool bound is 8.0-unproven in the campaign record**; the
   witness-study control MIP later proved 9 at 1,758 s. The stronger claim
   depends on that later run, which is noted in the preregistration.
9. **`max_alternatives`, `max_nodes` and the restart-rule list are
   unvalidated guesses.** They were chosen to bound the search, not because
   any evidence supports those values. The dive/MIP time split is evidence-
   based (§5) but the node caps are not.
10. **Dive determinism holds only up to wall-clock effects.** `node_time_s`,
    the wall limit and the budget checks are time-based, so two runs on
    different hardware may diverge in where they stop. The determinism test
    forces those caps off. Report the observed stop reasons rather than
    assuming bit-reproducibility.
11. **Three of the four graph caches are stored under `nested_warm_*`
    directories** (c3/c4/c5; c1's is under `cases/w1_k08/`). Those are
    storage locations of a *graph*, not warm column pools — an
    `EventExpandedNetwork` pickle is a function of the instance and physics
    only and contains no routes, and its identity is verified field by field
    against the fresh pool's own provenance. The cache path is deliberately
    exempt from the warm/witness path refusal that guards column sources;
    the exemption is recorded in the run manifest
    (`cache_path_warm_named`, `cache_contains_columns: false`) rather than
    left silent, and those paths are still refused if offered as a column
    source. Full path table in `PREREGISTRATION.md`. No warm or witness
    *column* is read anywhere in this pilot.
12. **`--timelimit` was verified to bound the whole two-stage solve**
    (stage 2 gets `max(0, timelimit − stage1_runtime_s)`), which is what the
    treatment's `max(1200, 3600 − elapsed)` and Arm B's wall requests assume.
    If that changes in a future MIP-runner revision, both budgets break.
