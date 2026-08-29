# EVSP-DR Goal 1 proof audit and Goal 2 scale-study registry

Snapshot date: 2026-08-30.

This directory is the machine-readable starting point for the publication-scale
study.  It separates mathematical proof scopes, immutable instance identity,
sampling design, and run-level measurements.  The companion workbook is
`outputs/goal2-research-registry-20260830/EVSP_DR_goal1_goal2_research_registry.xlsx`.

## Proof vocabulary

The industrial GIRO duty count is a comparator.  It is not assumed optimal.  It
can become an upper bound only after its duty partition is replayed successfully
under the same route representation and physical parameters as the optimization
model.

For a fixed instance, route representation, and physics:

* `L_model` is the optimum of the LP over every route allowed by that named
  representation.  It requires certified exact pricing, not merely a stopped CG
  run.
* `I_model` is the integer fleet optimum of that same mathematical
  representation.  It is not the unknowable optimum of physical reality and it
  is not automatically the GIRO count.
* `I_pool` is the exact integer optimum over one finite column pool produced by a
  CG trajectory.
* `I_timed` is the best physically validated finite-pool incumbent found before
  a stated time limit.

For fleet minimization, the scopes obey

`L_model <= I_model <= I_pool <= I_timed`.

An `I_model` proof normally uses a sandwich: integerize the certified full-model
LP lower bound, construct and physically replay a same-model integer witness,
and show that the lower and upper fleet bounds coincide.  Thus a GIRO witness at
three buses and a certified LP value of 2.4 proves `I_model = 3`; a certified LP
value of five with only a 12-bus witness leaves `I_model` in `[5, 12]`.

`240/240` denotes a 240 kWh battery and 240 kW maximum constant charging power.
The current envelope also fixes zero reserve, flat tariff, and unlimited charger
count.  These parameters determine route feasibility.  Results from the older
300 kWh / 300 kW campaign are therefore retained as a separate historical
experiment.

## Current Goal 1 audit

The committed 240/240 Panel A contains nine instance cells and six distinct
route representations, hence 54 rows.  All 54 full-representation LP bounds are
certified.  The full integer model is proved in 39/54 rows, and the finite RAW
pool integer optimum is proved in 42/54 rows.  Those denominators are not
replicates of one algorithm: they mix six mathematical representations.

For the preferred event representation, `I_model` is proved in 9/9 cells while
`I_pool` is proved in 7/9.  The two open event-pool rows are `k05_s2` and
`k05_s3`; this does not reopen the already completed full-model proof.

The historical 300/300 uniform 15 kWh / 10 minute model proves `I_model` in 7/9
cells.  Its unproved rows are `k05_s1` and `k05_s3`.  These historical counts
must not be combined numerically with the current 240/240 counts.

## Goal 2 sampling design

The existing registry contains 40 validated duty-union instances at target duty
counts 2, 3, 5, 8, 13, 20, 30, and 40.  Target duty count is an instance-size
label and industrial comparator, not a claim about the optimized fleet.

The primary medium-scale study will focus on targets 8, 10, 13, and 20.  To
avoid outcome-driven selection, new rows must be selected before optimization:

1. Draw six fixed-seed probability-sample instances at each medium target from
   a declared candidate universe, stratified by trip-count tercile and direct
   compatibility-density class.
2. Select six additional feature-space stress instances at each medium target
   using a deterministic maximin rule.  These diagnose mechanisms and are never
   pooled into the probability-sample mean.
3. Retain the six existing target-8, target-13, and target-20 rows as a separately
   labelled legacy cohort.  Generate the target-10 cohort from scratch.
4. Treat targets 30 and 40 as upper-scale case studies, without population-mean
   claims.

The planned medium study has 66 rows: 18 each at targets 8, 13, and 20 and 12 at
target 10.  The 48 new selections will expand the full registry from 40 to 88
instances once generated and validated.

Trip count alone is not an adequate hardness descriptor.  Selection and
analysis should include trip count, direct-compatibility or deadhead density,
station-only bridge fraction, layover slack, service energy per duty, coarse-grid
representability, maximum simultaneous-trip lower bound, and a time-only
minimum-path-cover lower bound.  Compatibility-graph degree distribution,
components, and entropy should be added before freezing the new sample.

## Reporting rules

Record time to the first validated solution at or below the comparator separately
from time to the certified LP bound.  Record pricing time, restricted-master LP
time, validation time, iteration count, column count, peak memory, exact-pricing
state count, and final-pool MIP time.  Never average successful runtimes while
silently dropping censored runs.  Report certification rate and a fixed-cap or
restricted-mean survival-time summary alongside conditional timing summaries.

For every row, publish both the comparator gap and the proof scope:

* `I_model_lower`, `I_model_upper`, and `I_model_proven`;
* `I_pool_lower`, `I_pool_upper`, and `I_pool_proven`;
* `I_timed` and its time limit;
* whether the integer witness passed physical replay.

The first nine rows of `goal2_event_ladder_progress.csv` contain committed event
results.  All other rows are marked not run.  Live cluster outputs must not enter
the publication table until harvested, hash-identified, validated, and committed.

## Files

* `goal1_model_proof_registry.csv`: row-level current 240/240 proof scopes.
* `goal1_historical_primary_proofs.csv`: separate historical 300/300 proofs.
* `goal1_proof_summary.csv`: compact counts derived from the row-level tables.
* `goal2_instance_registry.csv`: immutable identity and pre-outcome features for
  the 40 existing validated instances.
* `goal2_event_ladder_progress.csv`: one event-representation progress row per
  existing instance.
* `goal2_sampling_plan.csv`: predeclared expansion counts and selection roles.
* `goal2_run_result_schema.csv`: required fields for subsequent run records.

Run `python3 scripts/research/validate_goal2_research_registry.py` after editing
these artifacts.

## Source commits

* Event-envelope results: `2683c775c2ba29018404c5f41ca2aff3266b5739`.
* Arc-flow historical results: `585f98c01fcba562bca8e28bde10f969bcd68b3d`.
* Validated ladder instances/features: `ff7fb2ba93cf13a31171e1e4aeb2d28dc8aeee20`.
