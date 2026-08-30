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
