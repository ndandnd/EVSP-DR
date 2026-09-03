# Event-versus-uniform cluster workflow

These scripts execute and document the event-versus-uniform envelope experiment
defined in `analysis/event_uniform_envelope_20260821/plan.json`.

The operator runs only the top-level shell entry points.  Every Slurm worker
uses a detached checkout pinned to the submitting commit; scripts staged by
Slurm never infer the repository from `BASH_SOURCE`.

## Continue after the 2026-08-24 Panel A run

```bash
bash scripts/event_uniform_envelope/continue_after_panel_a.sh \
  "$HOME/ladder-lite/event_uniform_A_20260824_2dd2b4c"
```

This command:

1. writes `panel_a_summary.csv`, `panel_a_stage_counts.csv`,
   `stderr_inventory.csv`, `slurm_accounting.psv`, and checksums under the
   Panel A root;
2. hashes every immutable CG status and journal before submitting corrected
   RAW-pool MIP and target-feasibility arrays;
3. submits Panel B's 45 matched-wall uniform CG cells and 45 exact wall-boundary
   freezes when all nine Panel A event sources are certified.

It does not append to `records/*.csv`, inject known routes, or submit arc-flow
fallbacks.  Panel B integer jobs are deliberately a later step: their frozen
pool hashes must exist before the MIPs are submitted.

## Statistics contract

The normalized cell table is `panel_a_summary.csv`.  Each row records the
representation, CG certificate and stopping reason, LP value, reduced cost,
iterations, columns, wall time, peak RSS, DAG size, fleet-LP certificate,
finite-pool MIP result, target-feasibility result, and Slurm state/exit/RSS.

`panel_a_stage_counts.csv` is the compact stage-level table.  Raw artifacts
remain authoritative; the CSVs are indexes, not replacements.

## Corrected recovery and matched-wall boundary

The first recovery manifest used the Python CSV default CRLF line ending.
Bash therefore retained a carriage return on the final journal-hash field and
all 54 MIP workers correctly refused the mismatch.  In addition, the original
matched-wall freezer conservatively omitted insertions from the last durable
iteration.  Preserve those failed/v1 artifacts and run the reviewed v2 path:

```bash
bash scripts/event_uniform_envelope/recover_and_refreeze_v2.sh \
  "$HOME/ladder-lite/event_uniform_A_20260824_2dd2b4c" \
  "$HOME/ladder-lite/event_uniform_B_20260824_13596d0"
```

This writes an LF-only immutable manifest, reruns missing Panel A MIPs, retries
only the one missing target result with the reviewed event-tariff replay fix,
and creates new Panel B snapshots under `frozen_v2/`.  It pins solver/freezer
code to reviewed Agent E commit `44b6d5030a78ddca9c74f582d70ad87572e61794`
and accepts a later Agent E tip only when that commit remains an ancestor.

Once all 45 `frozen_v2/` snapshots validate, submit their integer comparison:

```bash
bash scripts/event_uniform_envelope/submit_panel_b_v2_integer.sh \
  "$HOME/ladder-lite/event_uniform_B_20260824_13596d0"
```

This runs the plan-specified Gurobi backend on Unicorn: 1,800-second,
eight-thread two-stage RAW-pool MIPs plus independently parallel target-fleet
feasibility solves.  The scientific solver limit is 1,800 seconds; the Slurm
wrapper allows 75 minutes for source validation, physical replay, and durable
serialization.  No routes are injected.

## Extended free-capacity campaigns

After the corrected Panel B integer arrays have been submitted, one command
launches three non-duplicative follow-ups:

```bash
bash scripts/event_uniform_envelope/submit_long_cluster_fill.sh \
  "$HOME/ladder-lite/event_uniform_A_20260824_2dd2b4c" \
  "$HOME/ladder-lite/event_uniform_B_20260824_13596d0"
```

It preserves the primary artifacts and stages independent copies for (a) the
two censored Panel A `uniform_2_1` cells, resumed to a 24-hour cumulative cap,
and (b) all 18 censored Panel B cells, resumed to a separate six-hour
certification cap.  The latter is a certification-tail diagnostic and does not
replace Panel B's matched-wall snapshots.  It also reproduces all 99 immutable
finite pools with explicitly eight-thread native HiGHS, when the Unicorn
environment passes a native-HiGHS preflight.  If the base environment lacks
the repository-pinned `highspy==1.15.1`, the launcher installs that single
wheel without dependencies into an isolated `$HOME/ladder-lite/vendor`
directory; it does not mutate `evsp_env`.  The import path, version, and vendor
file hashes are recorded under both panel roots.  Rerunning the launcher
recognizes active arrays and submits only artifacts that are still incomplete.
Native-HiGHS MIPs run on the non-preemptible `scaglione` partition.  Source CG
telemetry is archived separately because telemetry identity includes its output
path; each resumed copy starts a fresh canonical telemetry stream.  The
fail-closed repair and hashes are recorded in `telemetry_repair.csv`.

Once the queue is empty, normalize the results with:

```bash
bash scripts/event_uniform_envelope/audit_long_cluster_fill.sh \
  "$HOME/ladder-lite/event_uniform_A_20260824_2dd2b4c" \
  "$HOME/ladder-lite/event_uniform_B_20260824_13596d0"
```

The accessible statistics are `resume_summary.csv` inside each extended-CG
directory and `backend_reproduction.csv` at each panel root.  The latter gives
per-cell source hashes, solver status, fleet, bound, gap, proof scope, runtime,
RSS, physical validation, backend agreement, and Slurm accounting.

The two-hour native-HiGHS disagreement retries are normalized after completion
with `audit_highs_disagreement_retry.sh PANEL_A_ROOT PANEL_B_ROOT`.  It writes
`backend_retry7200.csv`, `backend_retry7200_unresolved.csv`, Slurm accounting,
and SHA-256 summary manifests under both panel roots.  The row-level CSV keeps
Gurobi, 30-minute HiGHS, and two-hour HiGHS status, incumbent, fleet bound,
proof, physical validation, runtime, identity hashes, and resource statistics
separate; Slurm `COMPLETED` is never treated as mathematical optimality.

`submit_highs_unresolved_retry28800.sh PANEL_A_ROOT PANEL_B_ROOT` consumes that
audited CSV and submits only safe rows that still lack an independent proof.
It refuses identity, configuration, physical-validation, accounting, or proven
fleet contradictions.  The eight-hour native-HiGHS jobs remain on `scaglione`,
use eight threads and 24 GiB each, write to a new immutable output directory,
and have a TSV job/configuration record under both panel roots.
After completion, `audit_highs_unresolved_retry28800.sh PANEL_A_ROOT
PANEL_B_ROOT` writes per-panel `backend_retry28800.csv` and unresolved subsets
with the Gurobi/30-minute/two-hour/eight-hour proof trajectory, identities,
runtime, memory, node, Slurm accounting, and SHA-256 summary manifests.
`submit_highs_unresolved_retry86400.sh PANEL_A_ROOT PANEL_B_ROOT` then selects
only safe eight-hour rows that remain unproven and launches fresh 24-hour
native-HiGHS solves on `scaglione`; proved rows are never resubmitted, and the
new immutable output directories and TSV job record preserve the experiment.

The two Panel B `uniform_2_1` CG rows that exhausted the six-hour cumulative
cap can be continued independently with `submit_panel_b_cg24h_tail.sh
PANEL_B_ROOT`.  This stages immutable children from the capped resume outputs,
preserves both generations of source evidence and telemetry, and runs only the
two pending cells on `default_partition` to a 24-hour cumulative scientific
cap.  `audit_panel_b_cg24h_tail.sh PANEL_B_ROOT` writes their row-level outcome,
iteration, column, runtime, memory, node, and Slurm statistics to
`cg_certification24h_13596d0/resume_summary.csv`.
The historical execution commit is validated against the immutable parent
resume plan; it is intentionally not required to be an ancestor of Agent E's
later development tip, because the reproducible cluster-workflow commit is a
sibling child of the reviewed Agent E base.

## 2026-08-30 overnight research fill

`audit_and_submit_overnight_20260830.sh` first audits the completed 24-hour
native-HiGHS pool retries and writes `backend_retry86400.csv` under each panel
root. It then submits only still-unproved, fully validated rows for a 48-hour
native-HiGHS solve on `scaglione`.

The same guarded command starts 18 new event-CG legacy medium probes: all six
validated duty unions at each of targets 8, 13, and 20. These use the reviewed
event solver commit `44b6d503...`, the validated input commit `ff7fb2ba...`,
240/240/zero-reserve physics, a 2.5 kWh event representation, and a 12-hour
scientific wall cap. Arrays use scale-specific memory requests of 32, 64, and
96 GiB on `default_partition`. The matrix, execution plan, job IDs, source
commits, and hashes are durable under
`$HOME/ladder-lite/medium_event_legacy_20260830_44b6d5`.
After the arrays finish, `audit_medium_event_legacy.sh CAMPAIGN_ROOT` writes
`medium_event_summary.csv` with row-level proof status, LP value, iterations,
columns, wall time, memory, event-DAG size, pricing/master/build telemetry, and
Slurm accounting.

`submit_event_extension_overnight.sh` adds a non-overlapping second wave: the
nine validated `r4`--`r6` duty unions at targets 2, 3, and 5, all three
target-30 cases, and the target-40 case. The small extension doubles the event
model's small-instance sample; targets 30 and 40 are explicitly boundary case
studies. They share the pinned event solver, immutable input manifest, physics,
telemetry, 12-hour cap, and CSV auditor described above. Memory requests are
16 GiB for the small extension, 128 GiB for target 30, and 192 GiB for target
40; Slurm may leave the boundary jobs pending until a suitable node is free.

If Slurm preempts the original medium index 12 (target 20) and extension index
12 (target 40), `submit_preempted_event_recovery.sh` stages each durable
checkpoint, column journal, iteration log, and telemetry stream under a new
immutable recovery root.  It then submits only those two cells with the
original 43,200-second cumulative scientific cap.  The source artifacts are
never modified, and the launcher refuses a certified source, a non-signal
checkpoint, changed input identity, or a mismatched solver commit.  This
recovery is safe while the other original array tasks remain active.
After both recovery jobs leave the queue,
`audit_preempted_event_recovery.sh` writes a row-level `resume_summary.csv`
and Slurm accounting under each recovery root, including cumulative and added
wall time, iterations, columns, peak RSS, node, stop reason, and outcome.

After the original medium and extension audits qualify their 12-hour
wall-limited cells, `submit_wall_capped_event_resume24h.sh` stages immutable
continuations and submits them by target scale with the established memory
policy.  The scientific cap is cumulative 86,400 seconds: these are 12h-to-24h
continuations, not fresh 24-hour repetitions.  The launcher requires the
committed solver identity, matching instance hashes, a completed Slurm state,
a validated configuration, and a source checkpoint at the parent wall cap.
`audit_wall_capped_event_resume24h.sh` writes the corresponding row-level CSVs
and Slurm accounting after all continuation jobs leave the queue.

## 2026-08-31 event-launch correction

The 2026-08-30 medium and extension launch worker omitted the explicit
`--time-model event` option.  The solver therefore wrote `time_model=uniform`,
despite the event-labeled matrix and output names.  Those roots are quarantined
as misconfigured uniform-lattice runs and must never be used as event evidence
or resumed by the event continuation tools.

The corrected launch contract is fail closed at both layers.  The launchers
export `EVSP_TIME_MODEL=event` and `EVSP_EVENT_ARC_MODE=lazy`; the batch worker
requires those exact values and passes both CLI options to the reviewed solver.
Fresh corrected roots default to
`medium_event_corrected_20260831_44b6d5` and
`event_extension_corrected_20260831_44b6d5`, with distinct `me31_*` and
`xe31_*` job names.  Their execution plans and `jobs.tsv` files record the
time-model and arc-mode identities explicitly.

When the corrected arrays are still active but the operator will be offline,
`queue_deferred_event_followup.sh` submits one small `afterany` controller job.
The controller waits for all six source arrays, writes both final audits, and
stages 12h-to-24h continuations only for rows whose immutable event/lazy status,
Slurm completion state, wall-cap status, input hash, journal, iteration log, and
telemetry all validate.  Certified rows and non-wall-cap failures are not
resubmitted.  The controller and every child array use immutable detached
checkouts and write machine-readable job records.  The generic continuation
worker now passes the saved time-model explicitly; uniform callers retain the
uniform default, while event callers must export the event/lazy identity.

`audit_highs_unresolved_retry172800.sh PANEL_A_ROOT PANEL_B_ROOT` is the
normalized 48-hour native-HiGHS auditor.  It joins the 48-hour artifacts and
Slurm outcomes to the validated 24-hour rows, writes
`backend_retry172800.csv` plus an unresolved subset in each panel root, and
classifies OOM or other non-completed tasks as execution errors rather than
silently treating a missing JSON artifact as an optimization result.
Rows that were intentionally selected for 48 hours after a missing or failed
24-hour artifact retain the selector's validated eight-hour fallback basis;
the output records `prior_validated_stage` so that fallback is never confused
with a valid 24-hour optimization result.

`submit_highs_oom_retry172800_mem48.sh` selects only normalized 48-hour rows
whose Slurm state is `OUT_OF_MEMORY`, whose exit is `0:125`, whose output
artifact is absent, and whose prior 24-hour or explicit eight-hour fallback
evidence remains validated.  It repeats the same 172800-second, eight-thread
native-HiGHS solve with 48 GiB on Scaglione and writes a separate output
directory and job record; completed or merely unproven rows are not rerun.

## 2026-09-02 k3/k4/k6/k7 transition-gap campaign

`data/scale_ladder/instances/transition_gap_20260902` freezes 36 new duty-union
instances before solver outcomes are observed: six fixed-seed probability rows
and three structural-stress rows at each of k3, k4, k6, and k7.  The stress
rows are selected from a committed 512-candidate universe per scale using the
pre-outcome criteria trip-heavy, service-energy-heavy, and tight scheduled
inter-trip gaps.  Existing six-selection duty sets are excluded.

Every one of the 42 source GIRO duty orders has an independently optimized and
replay-validated continuous charging schedule under 240 kWh, 240 kW,
zero reserve, and the flat tariff.  Each selected union therefore has a
physical k-route upper bound.  This is deliberately **not** described as an
event-lattice representability certificate or an optimum, and the known routes
are not injected into the RAW column-generation runs.

On Unicorn, after resolving the branch name with `git ls-remote` and pulling
the exact remote tip, launch the 36-cell 12-hour event/lazy campaign with:

```bash
bash scripts/event_uniform_envelope/submit_transition_gap_event.sh
```

It submits four nine-task arrays on `default_partition`, using 24 GiB for k3
and k4 and 32 GiB for k6 and k7.  The current four memory-only pool-MIP retries
run on `scaglione` and are not duplicated or made dependencies.  After every
transition array leaves the queue, write the row-level telemetry and Slurm CSV
with:

```bash
bash scripts/event_uniform_envelope/audit_medium_event_legacy.sh \
  "$HOME/ladder-lite/transition_gap_event_20260902_44b6d5"
```

No downstream fleet-only or integer stage is launched automatically.  The
Stage-1 audit is reviewed first; certified k3/k4 cells are then candidates for
the full L_model/I_model proof pipeline, while capped rows remain censored.

## 2026-09-03 small exact-recovery threshold fill

`data/scale_ladder/instances/small_threshold_20260903` adds 50 inputs at
targets 2, 5, 8, 9, and 10.  Each target has six fixed-seed probability rows
and four pre-outcome structural rows: trip-light, trip-heavy,
service-energy-heavy, and tight-gap.  This fills the missing threshold scales
without selecting “easy” cases after seeing solver outcomes.  Existing
scale-ladder duty sets are excluded, and the same independently certified
240/240 continuous-duty upper-bound contract used for the transition campaign
is preserved.  The known routes are not injected into raw CG.

Launch the 50-cell event/lazy campaign on Unicorn with:

```bash
bash scripts/event_uniform_envelope/submit_small_threshold_event.sh
```

It submits five ten-task arrays on `default_partition`, with 16, 24, 40, 48,
and 56 GiB at k2, k5, k8, k9, and k10 respectively.  Every task uses one CPU,
a 43,200-second scientific cap, a 12:15 Slurm limit, immutable inputs, and the
reviewed solver commit `44b6d503...`.  Audit after all arrays finish with:

```bash
bash scripts/event_uniform_envelope/audit_medium_event_legacy.sh \
  "$HOME/ladder-lite/small_threshold_event_20260903_44b6d5"
```

This is Stage 1 only.  Exact fleet recovery and finite-pool integrality are
different claims, so target-feasibility and integer-pool stages are held until
the Stage-1 audit identifies certified LP rows.
