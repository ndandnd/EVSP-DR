# Preregistration — diving-with-pricing pilot, fresh k=8 (19 Sep 2026)

Fixed before any case is run. Anything not written here is a deviation and
must be reported as one.

## Question

Can integer-complementary routes be generated **without** access to a
sequential/GIRO witness solution?

The witness study
(`outputs/independent_review_20260916/advisor_witness_columns_20260917`)
established that the routes missing from a fresh k=8 pool are *LP-suboptimal*
under that pool's own certified duals — reduced cost up to 53 — while the pool
already holds 381–670 columns at |rc| ≤ 1e-4. Reduced-cost enrichment
therefore cannot find them. Fixing part of the solution changes the duals,
which is the only mechanism on the table. This pilot tests exactly that, with
no witness input of any kind.

## Benchmark

Four frozen fresh k=8 pools from campaign `cumulative_budget_20260913`
(identities pinned in `cases_k08.json`; the campaign's **base/fresh** arm
only — `mip_warm` is never referenced):

| case | trips | fresh MIP buses | pool fleet bound | graph build (s) |
|---|---:|---:|---:|---:|
| c1_k08 | 194 | 9 | 9.0 (proven) | 7,279 |
| c3_k08 | 154 | 9 | 8.0 in-campaign; 9 proven by the witness-study control MIP | 3,295 |
| c4_k08 | 188 | 9 | 9.0 (proven) | 4,170 |
| c5_k08 | 167 | 9 | 9.0 (proven) | 2,520 |

All four fresh CG runs are rc-certified at route weight 8.000. A globally
feasible 8-bus solution is known to exist for each (from the sequential arm),
but **no 8-bus solution, warm pool, witness column or inherited pool is an
input to either arm of this experiment.** The known target is used only to
read the result afterwards.

## Arms (paired, same frozen pool)

Both arms consume the identical, byte-verified fresh `cg.json` + journal.

* **Control** — `run_exact_pool_mip.py` on the **unmodified** frozen fresh
  pool. `--cover --two-stage --threads 8 --mipgap 1e-4 --seed 20260919`.
* **Treatment** — identity-verified event-network cache load → diving with
  pricing (`src/diving_pricing_pilot.py`) → `run_exact_pool_mip.py` on the
  **augmented** pool (= every original column, byte-for-byte, plus whatever
  the dive generated). Same MIP flags and seed.

One deterministic seed (`20260919`) per case, four cases. No repeats.
**One treatment run per case**, paired against both control rows: the two
budget arms differ only in what the control is given and in how the graph
build is charged, never in what the treatment does.

Both arms run the pool MIP from the **pinned execution checkout**, not from
the historical campaign commit `871d057e`. That commit predates `--seed`
(added in `6830caa2`), so a seeded control/treatment pair is impossible with
it; `run_exact_pool_mip.py` is otherwise the same trusted script. This is a
deliberate, recorded departure from replaying the campaign argv verbatim.

## Outcomes

* **Primary** — `buses` and `fleet_proven` from each arm's final pool MIP.
* **Secondary** — whether the dive itself reached an 8-bus integer cover, and
  the reduced cost each generated column had at generation.
* **Diagnostic** — columns generated, dive nodes, node outcomes, stop reasons,
  and the dive's `min_reduced_cost_with_fleet_dual` trajectory.

A pool fleet bound is a statement about the supplied pool only. No node
outcome inside the dive is ever a global optimality or infeasibility claim.

## Budget and the graph-build problem

**A 3,600 s treatment that includes a fresh event-graph build is impossible
for this benchmark.** Recorded builds
(`outputs/cumulative_budget_20260913/audit/budgets.csv`,
`target_external_graph_build_s`) are 7,279 / 3,295 / 4,170 / 2,520 s. For c1
and c4 the build alone exceeds 60 minutes; for c3 and c5 it would leave 305 s
and 1,080 s for the dive *and* the final MIP. This is reported, not worked
around, and both arms below are run.

* **Arm A — shared-prerequisite accounting (primary).** Both arms get 3,600 s
  of real compute. The graph build is declared a shared prerequisite: the
  frozen fresh pool the control MIP consumes could not exist without it, so
  charging it to one arm only would be arbitrary. The cache **load** is
  measured and charged to the treatment. Split inside the treatment's
  3,600 s: dive wall limit **2,400 s** (includes the cache load), per-node
  time cap **150 s**, final MIP `max(1200, 3600 − elapsed)` with stage 1 at
  half of that. Rationale for the split: the witness study's augmented MIPs
  proved 8 in 5.5–93 s once the columns existed, so the final MIP needs far
  less than half the budget, while a dive toward depth 8 needs many
  fully-priced nodes. Fixed here, before any case is run.
* **Arm B — graph-charged accounting (secondary, the matched adjustment).**
  Both arms are charged `3600 + graph_build_s` for that case. The control
  spends it as real MIP time (10,879 / 6,895 / 7,770 / 6,120 s); the
  treatment is *debited* the build while still running its 3,600 s. The build
  is thus fully accounted and the comparison is conservative against the
  treatment.

**`--timelimit` bounds the whole two-stage solve.** Verified in
`run_exact_pool_mip.py`: stage 2 receives
`remaining_s = max(0, timelimit − stage1_runtime_s)`, so `--timelimit` is a
total, not a per-stage, budget. The treatment's
`MIP = max(1200, 3600 − elapsed)` therefore really does keep the treatment
inside 3,600 s, and Arm B's control wall request only has to cover
`3600 + graph_build_s`.

Both arms' per-stage wall times are measured and written to `timing.json`;
the cache load is never excluded from the treatment's budget.

## Graph cache reuse

The k=8 caches were produced at commits `e091a4db` (c1) and `ecb60c15`
(c3/c4/c5). The production loader compares the whole identity dict including
`git_commit`, so the pilot cannot load them under its own commit without an
explicit bridge. The bridge (`src/diving_cache_identity.py`) requires:
every other identity field equal, the recorded pickle sha256, the recomputed
`network.metrics()` (node count, arc count, arc mode, packed arc bytes,
event-lattice sha256) equal to the producer's, **and** byte-identical source
for all 19 graph-critical `EventExpandedNetwork` methods. Verified passing
against both producer commits at worktree HEAD `6830caa2`; the audit is
re-run by `make_manifest.py` and recorded per case. Default is refusal;
`--cache-commit-bridge` must be passed explicitly. If root pins a different
commit the audit must be re-run and may refuse — then the case is blocked,
not silently rebuilt.

**Three of the four caches are stored under warm-chain directories** — root
will grep for "warm", so it is stated here rather than discovered:

| case | cache path | producer |
|---|---|---|
| c1_k08 | `full_pool_recovery_20260912/cases/w1_k08/network.pkl` | `e091a4db` |
| c3_k08 | `nested_warm_chain_p3_k2_10_20260909_8830a34/network_cache/M__k08_p3__event_2p5_event5.pkl` | `ecb60c15` |
| c4_k08 | `nested_warm_multichain_p1246_k2_10_20260910_ecb60c1/p4/network_cache/M__k08_p4__event_2p5_event5.pkl` | `ecb60c15` |
| c5_k08 | `nested_warm_chain_p5_k2_10_20260910_ecb60c1/network_cache/M__k08_p5__event_2p5_event5.pkl` | `ecb60c15` |

These are *storage locations of a graph*, not warm column pools. An
`EventExpandedNetwork` pickle is a function of the instance and physics only
and contains **no routes**; every identity field is verified against the
fresh pool's own provenance. The cache path is therefore deliberately exempt
from the warm/witness path refusal that guards *column* sources, and the run
manifest records that exemption explicitly
(`cache_path_warm_named`, `cache_contains_columns: false`). Those same paths
are still rejected if offered as a column source. No warm or witness
**column** is read anywhere in this pilot.

## Input binding

The cache identity is checked against the frozen *status* provenance, so a
wrong `--data-dir` would pass every cache check and still replay routes
against a different instance. The runner therefore re-hashes the instance
CSV, the price CSV, `Ref_dict.csv` and `par_ref_dhd.csv` on disk against the
frozen pool's `provenance.*_sha256` and refuses on any mismatch or any
missing recorded hash, as `prep_witness.py` does.

## Refusal conditions

The run is refused (not degraded) if any of these hold:

* the frozen `cg.json` or journal sha256 differs from the manifest, before or
  after the job;
* the source status carries `witness_augmentation`,
  `validated_seed_routes_sha256`, `inherited_event_pool_status_sha256` or
  `diving_augmentation`, or `column_pool_treatment != "RAW"`, or is not
  rc-certified;
* any input path component matches `warm|witness|giro|seed_routes|mip_warm`;
* the cache identity differs in any field other than `git_commit`, its pickle
  hash or metrics differ, or a graph-critical method differs;
* a generated route fails `validate_injected_route`;
* an on-disk model input does not match the frozen pool's recorded sha256.

## Reporting rules

* Report both budget arms. Report the graph build in seconds alongside every
  treatment result.
* Report control and treatment `buses`/`fleet_proven` for all four cases,
  including failures, and the dive stop reason for every case.
* "The dive found 8" and "the final MIP proved 8 on the augmented pool" are
  reported separately.
* Dive determinism holds up to wall-clock effects: `node_time_s`, the wall
  limit and the dive's own budget checks are time-based, so two runs on
  different hardware may diverge. The determinism test forces those caps off.
  Report the observed stop reasons rather than assuming reproducibility.
* If the dive generates no useful column in any case, report that as the
  answer to the question — it is an informative negative.
